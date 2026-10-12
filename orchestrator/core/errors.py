"""Exception taxonomy for the orchestration layer.

The engine classifies every failure into a :class:`FailureKind` (see
``types.py``).  Agents and tools raise the exceptions below; the engine maps
them onto retry / fallback / rollback / safe-stop behaviour:

==========================  ===================  ==============================
Exception                   FailureKind          Engine reaction
==========================  ===================  ==============================
``TransientAgentError``     TRANSIENT            bounded retry with backoff
``AgentTimeout``            TIMEOUT              bounded retry with backoff
``AgentError``              as declared          retry iff ``retryable``
``AutonomyViolation``       AUTONOMY_VIOLATION   **safe-stop** (never retried)
``PolicyDenied``            POLICY_VIOLATION     retry with feedback, then fallback
``SafeStopRequested``       n/a                  safe-stop the whole run
==========================  ===================  ==============================
"""

from __future__ import annotations

from typing import Any


class OrchestratorError(Exception):
    """Base class for every error raised deliberately by this package."""


class ConfigError(OrchestratorError):
    """Invalid or missing configuration (never includes secret values)."""


class GraphError(OrchestratorError):
    """The workflow graph is structurally invalid (cycle, dangling edge, ...)."""


class PlanPatchError(GraphError):
    """A dynamic plan expansion was rejected by governance or validation."""


class StateError(OrchestratorError):
    """Run state is missing, corrupt or in a state that forbids the operation."""


class IntegrityError(OrchestratorError):
    """The audit chain or an artifact hash failed verification."""


class ApprovalError(OrchestratorError):
    """An approval operation was refused (self-approval, duplicate vote, ...)."""


class UndeclaredInputError(OrchestratorError, KeyError):
    """A node tried to read an artifact kind it did not declare in ``consumes``."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


class AgentError(OrchestratorError):
    """Raised by an agent to report a failed attempt.

    ``kind`` is a ``FailureKind`` *value string* (kept as ``str`` here so this
    module has no import dependency on ``types.py``).
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "transient",
        retryable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.details = details or {}


class TransientAgentError(AgentError):
    """Temporary failure (network blip, rate limit, flaky tool). Retryable."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, kind="transient", retryable=True, details=details)


class AgentTimeout(AgentError):
    """An attempt exceeded its time budget."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, kind="timeout", retryable=True, details=details)


class AgentCancelled(AgentError):
    """The attempt observed its cancel token (kill switch / safe-stop)."""

    def __init__(self, message: str = "attempt cancelled") -> None:
        super().__init__(message, kind="cancelled", retryable=False)


class AutonomyViolation(AgentError):
    """An agent tried to act outside its declared autonomy boundary.

    Always treated as CRITICAL: the run is safe-stopped, nothing is retried.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, kind="autonomy_violation", retryable=False, details=details)


class PathViolation(AutonomyViolation, ValueError):
    """A path or glob is unsafe (traversal, absolute, drive letter, reserved name, ...).

    Subclass of ``AutonomyViolation`` => CRITICAL => safe-stop, and of ``ValueError``
    so validators can be used like ordinary parsers.
    """


class ToolDenied(AgentError):
    """A toolbox call was refused (e.g. a write outside ``write_globs``).

    Recoverable: the attempt fails as a policy violation and is retried with
    feedback. Repeated denials escalate to ``AutonomyViolation``.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, kind="policy_violation", retryable=True, details=details)


class LockError(OrchestratorError):
    """The run lock is held by another live process."""


class PolicyDenied(AgentError):
    """A policy gate denied an output. Carries the findings for agent feedback."""

    def __init__(self, message: str, *, findings: tuple[Any, ...] = ()) -> None:
        super().__init__(message, kind="policy_violation", retryable=True)
        self.findings = findings


class SafeStopRequested(OrchestratorError):
    """Raised inside the engine to unwind into the safe-stop procedure."""

    def __init__(self, reason: str, *, trigger: str = "policy") -> None:
        super().__init__(reason)
        self.reason = reason
        self.trigger = trigger
