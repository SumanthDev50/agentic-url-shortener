"""Name -> implementation registries (everything a persisted ``NodeSpec`` refers to by name).

Specs only carry names (``agent``, ``fallback_agent``, ``skip_if``, ``compensator``, gate types), so a
run can be resumed by a new process that rebuilds the same registries. Lookups of unknown names raise
``ConfigError`` listing the known names; re-registering a name is refused (no silent override of a
governance component); names must match a strict grammar so they can safely appear in logs and paths.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from orchestrator.core.errors import ConfigError, OrchestratorError
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.types import Agent, Compensator, Gate, GateSpec, SkipPredicate

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}\Z")
_MAX_LISTED = 40


def _q(value: object) -> str:
    return repr(sanitize_for_terminal(value, 60))


class _Registry:
    """Shared name table: validation, duplicate refusal, thread-safe lookup."""

    kind = "entry"

    def __init__(self, name_re: re.Pattern[str]) -> None:
        self._name_re = name_re
        self._items: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _add(self, name: str, item: Any) -> None:
        if not isinstance(name, str) or not self._name_re.fullmatch(name):
            raise ConfigError(f"invalid {self.kind} name {_q(name)}")
        with self._lock:
            if name in self._items:
                raise ConfigError(f"{self.kind} {_q(name)} is already registered")
            self._items[name] = item

    def _lookup(self, name: str) -> Any:
        with self._lock:
            try:
                return self._items[name]
            except (KeyError, TypeError):
                known = sorted(self._items)
        listed = ", ".join(known[:_MAX_LISTED]) + (", ..." if len(known) > _MAX_LISTED else "")
        raise ConfigError(f"unknown {self.kind} {_q(name)}; known: [{listed}]") from None

    def has(self, name: str) -> bool:
        with self._lock:
            return isinstance(name, str) and name in self._items

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._items)


class AgentRegistry(_Registry):
    """``agent_id -> factory``. ``create`` returns a **fresh** instance every call (agents are stateless)."""

    kind = "agent"

    def __init__(self) -> None:
        super().__init__(NODE_ID_RE)

    def register(self, agent_id: str, factory: Callable[[], Agent]) -> None:
        if not callable(factory):
            raise ConfigError(f"factory of agent {_q(agent_id)} is not callable")
        self._add(agent_id, factory)

    def create(self, agent_id: str) -> Agent:
        factory: Callable[[], Agent] = self._lookup(agent_id)
        try:
            agent = factory()
        except OrchestratorError:
            raise
        except Exception as exc:
            raise ConfigError(f"factory of agent {_q(agent_id)} failed: {type(exc).__name__}") from exc
        if not isinstance(agent, Agent):
            raise ConfigError(f"factory of agent {_q(agent_id)} did not return an Agent")
        return agent

    def ids(self) -> list[str]:
        return self.names()


class GateRegistry(_Registry):
    """``gate type -> factory(params)``; ``build`` turns a serialisable ``GateSpec`` into a ``Gate``."""

    kind = "gate type"

    def __init__(self) -> None:
        super().__init__(_NAME_RE)

    def register(self, type: str, factory: Callable[[Mapping[str, Any]], Gate]) -> None:
        if not callable(factory):
            raise ConfigError(f"factory of gate {_q(type)} is not callable")
        self._add(type, factory)

    def build(self, spec: GateSpec) -> Gate:
        factory: Callable[[Mapping[str, Any]], Gate] = self._lookup(spec.type)
        try:
            gate = factory(spec.params)
        except OrchestratorError:
            raise
        except Exception as exc:
            raise ConfigError(f"gate {_q(spec.type)} could not be built: {type(exc).__name__}") from exc
        if not isinstance(gate, Gate):
            raise ConfigError(f"factory of gate {_q(spec.type)} did not return a Gate")
        return gate

    def types(self) -> list[str]:
        return self.names()


class SkipRegistry(_Registry):
    """``name -> SkipPredicate`` (``NodeSpec.skip_if``)."""

    kind = "skip predicate"

    def __init__(self) -> None:
        super().__init__(_NAME_RE)

    def register(self, name: str, predicate: SkipPredicate) -> None:
        if not callable(predicate):
            raise ConfigError(f"skip predicate {_q(name)} is not callable")
        self._add(name, predicate)

    def get(self, name: str) -> SkipPredicate:
        predicate: SkipPredicate = self._lookup(name)
        return predicate


class CompensatorRegistry(_Registry):
    """``name -> Compensator`` (``NodeSpec.compensator``)."""

    kind = "compensator"

    def __init__(self) -> None:
        super().__init__(_NAME_RE)

    def register(self, name: str, compensator: Compensator) -> None:
        if not isinstance(compensator, Compensator):
            raise ConfigError(f"compensator {_q(name)} has no compensate() method")
        self._add(name, compensator)

    def get(self, name: str) -> Compensator:
        compensator: Compensator = self._lookup(name)
        return compensator


@dataclass
class Registries:
    """Everything the engine needs to resolve names. Gates start empty: L5 fills them
    (``gates.default_gate_registry()`` plus the workflow's own gates)."""

    agents: AgentRegistry = field(default_factory=AgentRegistry)
    gates: GateRegistry = field(default_factory=GateRegistry)
    skips: SkipRegistry = field(default_factory=SkipRegistry)
    compensators: CompensatorRegistry = field(default_factory=CompensatorRegistry)
