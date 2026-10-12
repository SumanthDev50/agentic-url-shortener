"""Wiring: agents, gates, skip predicates, compensators and fault injection (L5).

Reference agents are discovered from a fixed list of modules in this repository (no directory
scanning, no plugin loading: agents are trusted code). Fault injection wraps the *primary* agent of
the targeted nodes in a ``ChaosAgent``; the fallback agent is never wrapped, so fallback and
remediation stay demonstrable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from orchestrator.core.errors import ConfigError
from orchestrator.core.types import Agent
from orchestrator.engine.gates import default_gate_registry
from orchestrator.engine.registry import (
    AgentRegistry,
    CompensatorRegistry,
    Registries,
    SkipRegistry,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.engine import PolicyEngine
from orchestrator.workflows.gates_sdlc import register_sdlc_gates, register_sdlc_skips

AGENT_MODULES: tuple[str, ...] = (
    "orchestrator.agents.intake",
    "orchestrator.agents.requirements",
    "orchestrator.agents.analyst",
    "orchestrator.agents.threat_model",
    "orchestrator.agents.architect",
    "orchestrator.agents.planner",
    "orchestrator.agents.implementer",
    "orchestrator.agents.test_author",
    "orchestrator.agents.integrator",
    "orchestrator.agents.verifier",
    "orchestrator.agents.security_scanner",
    "orchestrator.agents.documenter",
    "orchestrator.agents.release_manager",
    "orchestrator.agents.promoter",
    "orchestrator.agents.release_verifier",
)


@dataclass(frozen=True)
class FaultSpec:
    """Inject ``fault`` into the primary agent of the nodes matching ``node`` (exact id or ``prefix.*``)."""

    node: str
    fault: str
    attempts: int = 99


@dataclass(frozen=True)
class RegistryOptions:
    faults: tuple[FaultSpec, ...] = ()
    node_agents: Mapping[str, str] = field(
        default_factory=dict
    )  # node id -> agent id (from the graph)


def _discover_agent_classes() -> dict[str, Callable[[], Agent]]:
    found: dict[str, Callable[[], Agent]] = {}
    for module_name in AGENT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise ConfigError(f"reference agent module {module_name} is missing") from exc
            raise
        for obj in vars(module).values():
            agent_id = getattr(obj, "agent_id", None)
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and isinstance(agent_id, str)
                and agent_id not in ("", "base")
                and callable(getattr(obj, "run", None))
            ):
                found[agent_id] = obj
    return found


def _target_agent_ids(options: RegistryOptions, spec: FaultSpec) -> set[str]:
    """Agent ids that run the nodes a fault targets (primary agents only)."""
    if spec.node.endswith(".*"):
        prefix = spec.node[:-1]
        targets = {a for n, a in options.node_agents.items() if n.startswith(prefix)}
        if not targets and prefix == "implement.":
            targets = {"implementer"}  # dynamic nodes are not in the static graph
        return targets
    agent = options.node_agents.get(spec.node)
    return {agent} if agent else set()


def build_registries(options: RegistryOptions | None = None) -> Registries:
    opts = options or RegistryOptions()
    classes = _discover_agent_classes()
    for spec in opts.faults:
        for agent_id in _target_agent_ids(opts, spec):
            if agent_id not in classes:
                raise ConfigError(f"fault targets unknown agent {agent_id!r}")
            from orchestrator.agents.chaos import (
                FAULTS,
                ChaosAgent,
            )  # local: only needed with faults

            if spec.fault not in FAULTS:
                raise ConfigError(f"unknown fault {spec.fault!r}; known: {sorted(FAULTS)}")
            inner_factory = classes[agent_id]

            def factory(
                inner: Callable[[], Agent] = inner_factory,
                fault: str = spec.fault,
                attempts: int = spec.attempts,
            ) -> Agent:
                return ChaosAgent(inner(), fault, attempts)  # type: ignore[no-any-return]

            classes[agent_id] = factory
    agents = AgentRegistry()
    for agent_id, factory_fn in sorted(classes.items()):
        agents.register(agent_id, factory_fn)
    gates = default_gate_registry()
    register_sdlc_gates(gates)
    skips = SkipRegistry()
    register_sdlc_skips(skips)
    compensators = CompensatorRegistry()
    from orchestrator.agents.promoter import ApplyBundleCompensator

    compensators.register("apply_bundle", ApplyBundleCompensator())
    return Registries(agents, gates, skips, compensators)


def fault_specs(raw: Sequence[Mapping[str, Any]]) -> tuple[FaultSpec, ...]:
    out: list[FaultSpec] = []
    for item in raw:
        node, fault = item.get("node"), item.get("fault")
        if not isinstance(node, str) or not isinstance(fault, str):
            raise ConfigError("a fault needs string 'node' and 'fault'")
        attempts = item.get("attempts", 99)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 99:
            raise ConfigError("fault 'attempts' must be an integer in [1, 99]")
        out.append(FaultSpec(node, fault, attempts))
    return tuple(out)


# Generated modules that legitimately reach the network (SEC-008 allow-list). Declared in the
# orchestrator's own configuration, never in agent-writable files.
EGRESS_MODULES: tuple[str, ...] = (
    "linkcheck/fetcher.py",
    "linkcheck/guard.py",
    "tests/linkcheck/**",
)


def default_policy() -> PolicyEngine:
    """The policy engine used by the shipped scenarios and the CLI."""
    return PolicyEngine(config=PolicyConfig(egress_modules=EGRESS_MODULES))
