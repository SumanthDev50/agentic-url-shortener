"""Engine configuration with hard limits (engine-semantics section 20).

``EngineConfig`` is immutable and validated at construction: every numeric field must lie inside
``HARD_LIMITS`` and the approval floors (LOW 1, MEDIUM 1, **HIGH 2**) cannot be lowered, so a
configuration can make governance stricter or merely different but never silently weaker than the
floors. Anything that differs from the defaults is reported by :meth:`EngineConfig.posture` as
``DEGRADED: ...`` (recorded in ``RUN_CREATED`` and printed in every report).

``poll_interval_s`` is a pure wake-up cadence of the scheduler loop (no governance effect), so it is
the one field that is validated but not counted as a posture difference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from orchestrator.core.errors import ConfigError
from orchestrator.core.types import RiskLevel

ENGINE_SCHEMA_VERSION = 1
"""Part of every node fingerprint: bump when the meaning of a fingerprint or of a state field changes."""

IDENTITY_MODES = frozenset({"attested", "strict"})
MIN_APPROVALS: Mapping[RiskLevel, int] = MappingProxyType(
    {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
)
MAX_APPROVALS = 5

# Inclusive ``(low, high)`` bounds of every numeric field. Integers must be ``int`` (not ``bool``),
# floats must be finite.
HARD_LIMITS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "max_parallel": (1, 16),
        "max_total_attempts": (1, 1000),
        "max_attempts_per_node": (1, 50),
        "max_replans": (0, 50),
        "max_human_inputs": (0, 200),
        "max_remediation_loops": (0, 10),
        "max_clarification_rounds": (0, 10),
        "max_active_runtime_s": (0.001, 86400.0),
        "max_attempt_s": (0.001, 3600.0),
        "circuit_breaker_failures": (1, 20),
        "max_zombie_threads": (0, 8),
        "safe_stop_quiesce_s": (0.0, 60.0),
        "compensation_timeout_s": (0.001, 600.0),
        "max_overrides": (0, 5),
        "max_artifact_bytes": (1024, 16 * 1024 * 1024),
        "max_input_bytes": (1024, 1024 * 1024),
        "poll_interval_s": (0.0005, 5.0),
    }
)
_INT_FIELDS = frozenset(
    name for name, (low, high) in HARD_LIMITS.items() if isinstance(low, int) and isinstance(high, int)
)
_UNCOUNTED_FOR_POSTURE = frozenset({"poll_interval_s"})


class AutonomyMode(StrEnum):
    """How much the engine asks humans (engine-semantics 10.3)."""

    SUPERVISED = "supervised"  # every ApprovalSpec behaves as ``always``
    GUIDED = "guided"  # default: ``ApprovalSpec.threshold`` decides
    AUTONOMOUS = "autonomous"  # non-mandatory thresholds are raised to HIGH


def _default_approvals() -> Mapping[RiskLevel, int]:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


def _short(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


@dataclass(frozen=True)
class EngineConfig:
    """Budgets, limits and governance knobs of one engine process (never read from ``state.json``)."""

    max_parallel: int = 4
    max_total_attempts: int = 80
    max_attempts_per_node: int = 12
    max_replans: int = 5
    max_human_inputs: int = 20
    max_remediation_loops: int = 4
    max_clarification_rounds: int = 3
    max_active_runtime_s: float = 1800.0
    max_attempt_s: float = 600.0
    circuit_breaker_failures: int = 6
    max_zombie_threads: int = 2
    safe_stop_quiesce_s: float = 2.0
    compensation_timeout_s: float = 60.0
    max_overrides: int = 2
    max_artifact_bytes: int = 2097152
    max_input_bytes: int = 65536
    approvals_by_risk: Mapping[RiskLevel, int] = field(default_factory=_default_approvals)
    autonomy_mode: AutonomyMode = AutonomyMode.GUIDED
    identity_mode: str = "attested"
    approver_allowlist: frozenset[str] | None = None
    allow_scripted: bool = False
    poll_interval_s: float = 0.05
    rollback_on_safe_stop: bool = True

    def __post_init__(self) -> None:
        for name, (low, high) in HARD_LIMITS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number (got {_short(value)})")
            if name in _INT_FIELDS and not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer (got {_short(value)})")
            if not math.isfinite(value) or not low <= value <= high:
                raise ConfigError(f"{name} must be in [{low}, {high}] (got {_short(value)})")
            if name not in _INT_FIELDS:
                object.__setattr__(self, name, float(value))
        object.__setattr__(self, "approvals_by_risk", self._checked_approvals())
        try:
            mode = AutonomyMode(self.autonomy_mode)
        except ValueError:
            raise ConfigError(f"autonomy_mode must be one of {[m.value for m in AutonomyMode]}") from None
        object.__setattr__(self, "autonomy_mode", mode)
        if self.identity_mode not in IDENTITY_MODES:
            raise ConfigError(f"identity_mode must be one of {sorted(IDENTITY_MODES)}")
        object.__setattr__(self, "approver_allowlist", self._checked_allowlist())
        for name in ("allow_scripted", "rollback_on_safe_stop"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")

    def _checked_approvals(self) -> Mapping[RiskLevel, int]:
        raw = self.approvals_by_risk
        if not isinstance(raw, Mapping):
            raise ConfigError("approvals_by_risk must be a mapping of RiskLevel to int")
        merged = dict(_default_approvals())
        for level, count in raw.items():
            if not isinstance(level, RiskLevel):
                raise ConfigError(f"approvals_by_risk key is not a RiskLevel: {_short(level)}")
            merged[level] = count
        for level, count in merged.items():
            if isinstance(count, bool) or not isinstance(count, int):
                raise ConfigError(f"approvals_by_risk[{level.name}] must be an integer")
            if count < MIN_APPROVALS[level]:
                raise ConfigError(
                    f"approvals_by_risk[{level.name}] cannot be lowered below {MIN_APPROVALS[level]} "
                    f"(got {count})"
                )
            if count > MAX_APPROVALS:
                raise ConfigError(f"approvals_by_risk[{level.name}] must be at most {MAX_APPROVALS}")
        return MappingProxyType(merged)

    def _checked_allowlist(self) -> frozenset[str] | None:
        raw = self.approver_allowlist
        if raw is None:
            return None
        if isinstance(raw, str) or not all(isinstance(name, str) and name for name in raw):
            raise ConfigError("approver_allowlist must be a collection of non-empty names")
        names = frozenset(raw)
        if not names:
            raise ConfigError("approver_allowlist must not be empty (use None to disable it)")
        return names

    # -- reporting -----------------------------------------------------------------
    def posture(self) -> str:
        """``NOMINAL`` for the default governance settings, else ``DEGRADED: <what differs>``."""
        default = type(self)()
        differing: list[str] = []
        for f in fields(self):
            if f.name in _UNCOUNTED_FOR_POSTURE:
                continue
            mine, base = getattr(self, f.name), getattr(default, f.name)
            if mine != base:
                differing.append(f"{f.name}={self._render(f.name, mine)}")
        return "NOMINAL" if not differing else "DEGRADED: " + ", ".join(differing)

    @staticmethod
    def _render(name: str, value: object) -> str:
        if name == "approvals_by_risk" and isinstance(value, Mapping):
            return "{" + ",".join(f"{k.name}:{v}" for k, v in sorted(value.items())) + "}"
        if name == "approver_allowlist" and isinstance(value, frozenset):
            return f"restricted({len(value)})"
        return str(value.value) if isinstance(value, StrEnum) else str(value)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able snapshot (recorded in ``RUN_CREATED`` / ``state.config``; never trusted on load)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "approvals_by_risk":
                out[f.name] = {level.name: n for level, n in sorted(value.items())}
            elif f.name == "approver_allowlist":
                out[f.name] = sorted(value) if value is not None else None
            elif isinstance(value, StrEnum):
                out[f.name] = value.value
            else:
                out[f.name] = value
        return out
