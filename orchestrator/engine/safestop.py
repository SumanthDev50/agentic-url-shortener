"""Kill switch and safe-stop budgets (engine-semantics sections 9 and 14).

``KillSwitch`` is a *fail-closed* file switch: ``<runs_dir>/KILL`` (global) and ``<run_dir>/KILL``
(one run), both taken from process configuration, never from ``state.json``.

* Existence alone means *tripped*; the contents are never required to be valid. At most
  ``MAX_KILL_BYTES`` (1 KiB) are read and everything shown to a human is sanitised.
* ``FileNotFoundError`` is the only "not tripped" answer. Any other ``OSError`` (permission, I/O, a
  directory where a file should be, ...) is treated as tripped.
* The final path component is never followed: a symlink / reparse point or any other non-regular
  object counts as tripped and is not read.
* ``trip`` writes atomically (temp file + fsync + replace, which also replaces a pre-planted symlink
  instead of writing through it). ``clear`` is an *attributed* action: it needs a named actor, only
  removes a regular file (anything else is left in place, so the switch stays tripped) and returns
  who cleared what so the caller can put it into the audit chain.
* Limits: only the *final* path component is protected against links (a symlinked parent directory
  is followed, the runs directory itself may be a link); the switch is unauthenticated by design,
  so anyone who can create the file stops the run (fail closed) while only a named ``clear``
  removes it.

``SafeStopController`` evaluates the run-level budgets of section 9 against ``RunState.counters``
and keeps the per-agent circuit breaker state in ``RunState.counters`` too, so it survives a resume
and the controller itself holds nothing but the (process-supplied) ``EngineConfig``.

Every budget ``max_*`` is a *maximum*: the run is stopped when a counter exceeds it (so ``0`` is a
valid "none allowed" setting for replans or zombies and ``max_total_attempts`` retries really run).
Zombie threads additionally stop the run when they fill the cap and no node can progress.
Human inputs are *refused* beyond their budget (section 9), not safe-stopped: ``check`` ignores
them and ``human_input_allowed`` is the gate the scheduler asks before accepting an ``amend`` or
``answer``.
"""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.core.clock import SystemClock
from orchestrator.core.errors import ConfigError, StateError
from orchestrator.core.fsutil import atomic_write_bytes
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import canonical_json, loads_strict, parse_datetime
from orchestrator.core.state import Counters, RunState
from orchestrator.core.types import Clock, Epoch, FailureKind, KillSignal, NodeStatus
from orchestrator.engine.config import EngineConfig

SCOPE_GLOBAL = "global"
SCOPE_RUN = "run"
MAX_KILL_BYTES = 1024
UNKNOWN = "unknown"

TRIGGER_BUDGET = "budget"
TRIGGER_CIRCUIT_BREAKER = "circuit_breaker"
TRIGGER_WORKER_EXHAUSTION = "worker_exhaustion"

BREAKER_FAILURE_KINDS = frozenset({FailureKind.TRANSIENT, FailureKind.TIMEOUT, FailureKind.FATAL})
"""Only these failures count towards an agent's circuit breaker (engine-semantics section 9)."""

_SCOPES = (SCOPE_GLOBAL, SCOPE_RUN)
_BY_MAX = 64
_REASON_MAX = 200
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MAX_LISTED = 5


def _clean(value: object, limit: int) -> str:
    return sanitize_for_terminal(value, limit).strip()


# --------------------------------------------------------------------------- kill switch


def _lstat(path: Path) -> os.stat_result:
    """The one place the switch inspects its path (never follows a final symlink)."""
    return os.lstat(path)


def _same_file(inspected: os.stat_result, opened: os.stat_result) -> bool:
    if not inspected.st_ino or not opened.st_ino:
        return True  # the platform reports no file identity
    return (inspected.st_dev, inspected.st_ino) == (opened.st_dev, opened.st_ino)


def _read_head(path: Path, inspected: os.stat_result) -> bytes | None:
    """At most ``MAX_KILL_BYTES + 1`` bytes of the regular file that was just ``inspected``.

    The extra byte tells "exactly 1 KiB" from "larger". ``None`` means the path no longer is that
    file (swapped for a link, another file or a FIFO between the ``lstat`` and the ``open``).
    ``O_NOFOLLOW`` / ``O_NONBLOCK`` (where the platform has them) make such a swap fail or stay
    harmless, the identity comparison covers platforms without them (Windows).
    """
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(inspected, opened):
            return None
        chunks: list[bytes] = []
        remaining = MAX_KILL_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _is_link(st: os.stat_result) -> bool:
    attributes = getattr(st, "st_file_attributes", 0) or 0
    return stat.S_ISLNK(st.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _mtime(st: os.stat_result) -> datetime:
    try:
        return datetime.fromtimestamp(st.st_mtime, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return _EPOCH


def _parse_ts(value: object, fallback: datetime) -> datetime:
    if not isinstance(value, str):
        return fallback
    try:
        return parse_datetime(value).astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        return fallback


def _signal(scope: str, by: object, reason: object, ts: datetime) -> KillSignal:
    return KillSignal(
        by=_clean(by, _BY_MAX) or UNKNOWN,
        reason=_clean(reason, _REASON_MAX) or "kill switch file present",
        ts=ts,
        scope=scope,
    )


def _signal_from_content(scope: str, head: bytes, fallback_ts: datetime) -> KillSignal:
    """Contents are advisory: a JSON object may name ``by`` / ``reason`` / ``ts``; anything else is
    shown as (sanitised) free text. The scope always comes from *which* file was found."""
    try:
        document = loads_strict(head, max_bytes=MAX_KILL_BYTES)
    except ValueError:
        document = None
    if isinstance(document, Mapping):
        by, reason = document.get("by"), document.get("reason")
        return _signal(
            scope,
            by if isinstance(by, str) else UNKNOWN,
            reason if isinstance(reason, str) else "",
            _parse_ts(document.get("ts"), fallback_ts),
        )
    return _signal(scope, UNKNOWN, head.decode("utf-8", errors="replace"), fallback_ts)


def _probe(scope: str, path: Path) -> KillSignal | None:
    """The signal of one switch file, or ``None`` iff the file is positively absent."""
    try:
        st = _lstat(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:  # fail closed: we cannot tell, so assume tripped
        reason = f"kill switch path could not be inspected ({type(exc).__name__})"
        return _signal(scope, UNKNOWN, reason, _EPOCH)
    when = _mtime(st)
    if _is_link(st):
        reason = "kill switch is a symbolic link or reparse point (not followed)"
        return _signal(scope, UNKNOWN, reason, when)
    if not stat.S_ISREG(st.st_mode):
        return _signal(scope, UNKNOWN, "kill switch path is not a regular file", when)
    try:
        head = _read_head(path, st)
    except FileNotFoundError:
        return None  # cleared between the lstat and the open
    except (OSError, ValueError) as exc:
        reason = f"kill switch file could not be read ({type(exc).__name__})"
        return _signal(scope, UNKNOWN, reason, when)
    if head is None:
        return _signal(scope, UNKNOWN, "kill switch path changed while it was being read", when)
    if len(head) > MAX_KILL_BYTES:
        reason = f"kill switch file is larger than {MAX_KILL_BYTES} bytes (contents ignored)"
        return _signal(scope, UNKNOWN, reason, when)
    return _signal_from_content(scope, head, when)


def _encode(by: str, reason: str, scope: str, ts: datetime) -> tuple[bytes, str]:
    """Canonical JSON of a trip, shortened (reason only) until it fits ``MAX_KILL_BYTES``."""
    while True:
        data = canonical_json({"by": by, "reason": reason, "scope": scope, "ts": ts.isoformat()})
        if len(data) <= MAX_KILL_BYTES or len(reason) <= 1:
            return data, reason
        reason = reason[: len(reason) // 2]


@dataclass(frozen=True)
class KillClearance:
    """Outcome of an attributed ``KillSwitch.clear``: the removed trip and who removed it."""

    signal: KillSignal
    by: str
    reason: str
    ts: datetime


class KillSwitch:
    """Global (``global_path``, optional) and per-run (``run_path``) kill switch files."""

    def __init__(
        self,
        global_path: Path | str | None,
        run_path: Path | str,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._paths: dict[str, Path | None] = {
            SCOPE_GLOBAL: Path(global_path) if global_path is not None else None,
            SCOPE_RUN: Path(run_path),
        }
        self._clock: Clock = clock if clock is not None else SystemClock()

    def _configured(self) -> Iterator[tuple[str, Path]]:
        for scope in _SCOPES:
            path = self._paths[scope]
            if path is not None:
                yield scope, path

    def _path_for(self, scope: str) -> Path:
        if scope not in _SCOPES:
            raise ConfigError(f"kill switch scope must be one of {list(_SCOPES)}")
        path = self._paths[scope]
        if path is None:
            raise ConfigError(f"no {scope} kill switch path is configured")
        return path

    def signals(self) -> list[KillSignal]:
        """Every tripped switch, global first (so a masked run signal can still be acknowledged)."""
        found = (_probe(scope, path) for scope, path in self._configured())
        return [signal for signal in found if signal is not None]

    def check(self) -> KillSignal | None:
        """The first tripped switch (global before run) or ``None``. Never raises."""
        return next(iter(self.signals()), None)

    def trip(self, scope: str, by: str, reason: str) -> KillSignal:
        """Trip one switch atomically; returns the signal ``check`` will report for it."""
        path = self._path_for(scope)
        who = _clean(by, _BY_MAX)
        if not who:
            raise ConfigError("a kill switch must be tripped by a named actor (by)")
        why = _clean(reason, _REASON_MAX) or "no reason given"
        stamp = self._clock.now()
        data, why = _encode(who, why, scope, stamp)
        try:
            atomic_write_bytes(path, data)
        except OSError as exc:
            detail = type(exc).__name__
            raise StateError(f"could not trip the {scope} kill switch ({detail})") from exc
        return KillSignal(by=who, reason=why, ts=stamp, scope=scope)

    def clear(self, scope: str, by: str, reason: str = "") -> KillClearance | None:
        """Clear one switch on behalf of a named actor (never an anonymous ``rm``).

        Returns ``None`` when nothing was tripped. Only a regular file (or a link to one) is
        removed; a directory or any other object is left in place, so the switch stays tripped, and
        ``StateError`` explains why.
        """
        path = self._path_for(scope)
        who = _clean(by, _BY_MAX)
        if not who:
            raise ConfigError("clearing a kill switch must be attributed to a named actor (by)")
        signal = _probe(scope, path)
        if signal is None:
            return None
        try:
            st = _lstat(path)
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
                raise StateError(
                    f"the {scope} kill switch is not a regular file; inspect and remove it manually"
                )
            os.unlink(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            detail = type(exc).__name__
            raise StateError(f"could not clear the {scope} kill switch ({detail})") from exc
        return KillClearance(signal, who, _clean(reason, _REASON_MAX), self._clock.now())


# --------------------------------------------------------------------------- safe-stop budgets


@dataclass(frozen=True)
class SafeStopTrigger:
    """Why the run must be safe-stopped (``trigger`` is a ``SafeStopRecord.trigger`` value)."""

    trigger: str
    reason: str


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _agent_pairs(graph_document: object) -> dict[str, tuple[str, str | None]]:
    """``node id -> (agent, fallback_agent)`` read defensively from ``RunState.graph``."""
    nodes = graph_document.get("nodes") if isinstance(graph_document, Mapping) else None
    pairs: dict[str, tuple[str, str | None]] = {}
    for raw in nodes if isinstance(nodes, list) else ():
        if not isinstance(raw, Mapping):
            continue
        node_id, agent, fallback = raw.get("id"), raw.get("agent"), raw.get("fallback_agent")
        if isinstance(node_id, str) and isinstance(agent, str):
            pairs[node_id] = (agent, fallback if isinstance(fallback, str) and fallback else None)
    return pairs


def _stalled(state: RunState) -> bool:
    """A node is ready to dispatch but nothing is in flight or waiting for a human to act."""
    statuses = {node.status for node in state.nodes.values()}
    busy = {NodeStatus.RUNNING, NodeStatus.AWAITING_APPROVAL, NodeStatus.AWAITING_INPUT}
    return NodeStatus.READY in statuses and not statuses & busy


def _listed(names: list[str]) -> str:
    shown = ", ".join(sanitize_for_terminal(name, 60) for name in names[:_MAX_LISTED])
    return shown + (f", ... ({len(names)} in total)" if len(names) > _MAX_LISTED else "")


class SafeStopController:
    """Run-level budgets and per-agent circuit breaker (state lives in ``RunState.counters``)."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    # -- budgets ---------------------------------------------------------------------
    def check(self, state: RunState, active_runtime_s: float) -> SafeStopTrigger | None:
        """The first exhausted budget, in the order attempts, replans, runtime, zombies, breaker.

        ``active_runtime_s`` is the live total (human wait excluded); the larger of it and the
        persisted ``state.active_runtime_s`` counts, and a non-finite value counts as exhausted.
        """
        counters, limits = state.counters, self.config
        if counters.budgeted_attempts > limits.max_total_attempts:
            return SafeStopTrigger(
                TRIGGER_BUDGET,
                f"retry/fallback attempt budget exhausted "
                f"({counters.budgeted_attempts} used, limit {limits.max_total_attempts})",
            )
        if counters.replans > limits.max_replans:
            return SafeStopTrigger(
                TRIGGER_BUDGET,
                f"re-plan budget exhausted ({counters.replans} used, limit {limits.max_replans})",
            )
        runtime = self._runtime_trigger(state, active_runtime_s)
        if runtime is not None:
            return runtime
        return self._zombie_trigger(state) or self._breaker_trigger(state)

    def _zombie_trigger(self, state: RunState) -> SafeStopTrigger | None:
        """More zombies than the cap always stops the run. Zombies *at* the cap block dispatch (the
        scheduler's gate), so they stop it only when nothing else can progress (section 15)."""
        zombies, cap = state.counters.zombies, self.config.max_zombie_threads
        if zombies > cap:
            reason = f"{zombies} zombie attempt thread(s) exceed the cap of {cap}"
        elif zombies > 0 and zombies >= cap and _stalled(state):
            reason = (
                f"{zombies} zombie attempt thread(s) fill the cap and no other work can progress"
            )
        else:
            return None
        return SafeStopTrigger(TRIGGER_WORKER_EXHAUSTION, reason)

    def _runtime_trigger(self, state: RunState, live: float) -> SafeStopTrigger | None:
        values = (_as_float(live), _as_float(state.active_runtime_s))
        if not all(math.isfinite(v) for v in values):
            return SafeStopTrigger(
                TRIGGER_BUDGET, "active runtime is not a finite number (treated as exhausted)"
            )
        used, limit = max(values), self.config.max_active_runtime_s
        if used > limit:
            return SafeStopTrigger(
                TRIGGER_BUDGET,
                f"active runtime budget exhausted ({used:.1f}s used, limit {limit:.1f}s)",
            )
        return None

    def human_input_allowed(self, state: RunState) -> bool:
        """May another ``amend`` / ``answer`` be accepted? Beyond the budget inputs are refused."""
        return state.counters.human_inputs < self.config.max_human_inputs

    # -- circuit breaker ---------------------------------------------------------------
    def _open_agents(self, counters: Counters) -> set[str]:
        threshold = self.config.circuit_breaker_failures
        failing = {a for a, count in counters.breaker_failures.items() if count >= threshold}
        return set(counters.breaker_open) | failing

    def breaker_open(self, state: RunState, agent_id: str) -> bool:
        """Is the breaker of ``agent_id`` open (listed, or at / above the failure threshold)?"""
        return agent_id in self._open_agents(state.counters)

    def record_attempt(self, state: RunState, agent_id: str, kind: FailureKind | None) -> bool:
        """Account one finished attempt of ``agent_id``; ``kind=None`` means success.

        Success resets the agent's consecutive failures and closes its breaker. TRANSIENT, TIMEOUT
        and FATAL count; every other kind (gate / policy / approval rejections, autonomy
        violations, cancellations) leaves the breaker alone. An unknown kind string counts as FATAL.
        Returns ``True`` iff this call opened the breaker.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ConfigError("agent_id must be a non-empty string")
        counters = state.counters
        if kind is None:
            counters.breaker_failures.pop(agent_id, None)
            counters.breaker_open[:] = [a for a in counters.breaker_open if a != agent_id]
            return False
        failure = kind if isinstance(kind, FailureKind) else FailureKind.parse(str(kind))
        if failure not in BREAKER_FAILURE_KINDS:
            return False
        count = counters.breaker_failures.get(agent_id, 0) + 1
        counters.breaker_failures[agent_id] = count
        if count >= self.config.circuit_breaker_failures and agent_id not in counters.breaker_open:
            counters.breaker_open.append(agent_id)
            return True
        return False

    def reset_breaker(self, state: RunState, agent_id: str | None = None) -> None:
        """Close the breaker of one agent, or of every agent (safe-stop override, section 14)."""
        counters = state.counters
        if agent_id is None:
            counters.breaker_failures.clear()
            counters.breaker_open.clear()
            return
        counters.breaker_failures.pop(agent_id, None)
        counters.breaker_open[:] = [a for a in counters.breaker_open if a != agent_id]

    def _breaker_trigger(self, state: RunState) -> SafeStopTrigger | None:
        """An open breaker stops the run only when a READY node cannot be routed anywhere else:
        its primary agent is open and it has no fallback (or the fallback is open too), or it is
        already on its fallback and that one is open."""
        opened = self._open_agents(state.counters)
        if not opened:
            return None
        pairs = _agent_pairs(state.graph)
        stuck_nodes: list[str] = []
        stuck_agents: dict[str, None] = {}
        for node_id, node in state.nodes.items():
            pair = pairs.get(node_id)
            if node.status != NodeStatus.READY or pair is None:
                continue
            primary, fallback = pair
            if node.epoch == Epoch.FALLBACK:
                current, alternative = fallback or primary, None
            else:
                current, alternative = primary, fallback
            if current in opened and (alternative is None or alternative in opened):
                stuck_nodes.append(node_id)
                stuck_agents[current] = None
        if not stuck_nodes:
            return None
        return SafeStopTrigger(
            TRIGGER_CIRCUIT_BREAKER,
            f"circuit breaker open for agent(s) {_listed(list(stuck_agents))} with no usable "
            f"fallback (nodes: {_listed(stuck_nodes)})",
        )


__all__ = [
    "BREAKER_FAILURE_KINDS",
    "MAX_KILL_BYTES",
    "SCOPE_GLOBAL",
    "SCOPE_RUN",
    "TRIGGER_BUDGET",
    "TRIGGER_CIRCUIT_BREAKER",
    "TRIGGER_WORKER_EXHAUSTION",
    "KillClearance",
    "KillSwitch",
    "SafeStopController",
    "SafeStopTrigger",
]
