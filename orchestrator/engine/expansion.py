"""Governed normalisation of agent-authored plan expansions (engine-semantics 12.3, E0-E2).

A ``PlanExpansion`` is **untrusted**: a planning node (the *owner*) proposes new nodes and extra
edges. ``normalize_expansion`` turns the proposal into the expansion the engine is willing to apply
and reports everything that falls outside what the human-authored owner allows.

Authority comes only from the owner's ``params``:

* ``expansion_template`` - a ``NodeSpec`` dict. Every new node is *built from the template*: agent,
  fallback agent, stage, consumed / produced kinds, gates, approval, remediation, retry and
  timeout all come from it (retry clamped to ``RetryPolicy.HARD_MAX_ATTEMPTS`` and its delay bounds,
  timeout to ``EngineConfig.max_attempt_s``); ``mandatory``, ``side_effecting``, ``compensator``,
  ``skip_if`` and ``compensates`` are forced off; ``risk`` is the maximum of template and proposal;
  the exit gates get the mandatory defaults (``artifacts_present``; the policy gate is added per
  attempt by the scheduler because it needs the narrowed autonomy profile); ``autonomy`` is the
  template's narrowed by the node's own. The agent contributes only ``id``, ``title``,
  ``depends_on`` (the owner is always forced in) and ``params``.
* ``expansion_policy`` - the allow-lists: ``id_prefix``, ``allowed_agents``, ``allowed_stages``,
  ``allowed_produces``, ``rewire_targets`` (ids or ``prefix.*``) and ``max_nodes``. A missing list
  means "nothing allowed". The *proposal* is checked against them, so a node that asks for a
  privileged agent, a foreign stage or foreign artifact kinds is rejected, not silently rewritten.

Findings are HIGH ``CHG-006`` (a mandatory node would be replaced or rewired) or ``CHG-007``
(anything else outside policy: owner mismatch, ids, agents, stages, produces, privileged flags,
dependencies, rewire targets, size, an unusable template / policy, or a plan that cannot be applied
to the graph). Overwritten attributes that merely differ (approval ``None``, a huge retry, a long
timeout) are not findings: the template simply wins.

Contract: the function never raises on agent-controlled content (any unexpected failure becomes a
finding); it is deterministic and idempotent (normalising its own output changes nothing); the
returned expansion contains only conformant nodes and rewires and must be applied **only when the
findings are empty** - in that case it has also been trial-applied to a copy of ``graph``.

Not handled here (they need node states, so they belong to the scheduler): the diff against the
owner's previous expansion (E3), refusing the removal of live or uncompensated side-effecting nodes
(E4) and tombstones (E5). The ``params`` of new nodes are passed through unchanged (apart from the
two expansion keys); their size is bounded by the ``plan_expansion`` artifact cap.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

from orchestrator.core.errors import ConfigError, OrchestratorError
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.types import (
    CHANGESET_KINDS,
    AutonomyProfile,
    Finding,
    GateSpec,
    NodeSpec,
    PlanExpansion,
    PolicyCategory,
    RetryPolicy,
    Rewire,
    Severity,
)
from orchestrator.engine.config import EngineConfig

RULE_MANDATORY = "CHG-006"
RULE_CONFORMANCE = "CHG-007"
TEMPLATE_KEY = "expansion_template"
POLICY_KEY = "expansion_policy"
MAX_EXPANSION_NODES = 200
MAX_REWIRES = 200
MAX_REWIRE_EDGES = 1000
MAX_FINDINGS = 50
MANDATORY_EXIT_GATES: tuple[GateSpec, ...] = (GateSpec("artifacts_present"),)

_RIGHTS_KEYS = frozenset({TEMPLATE_KEY, POLICY_KEY})
_PREFIX_PATTERN_RE = re.compile(r"^[a-z][a-z0-9_-]{0,40}\.\*\Z")
_RETRY_BOUNDS = {"base_delay_s": (0.0, 60.0), "max_delay_s": (0.0, 60.0), "factor": (1.0, 4.0)}
_DEFAULT_READ_GLOBS = ("**",)

_HINT_OWNER = "Only the planning node that owns an expansion may propose it."
_HINT_ID = "Name new nodes with the id prefix of the owner's expansion policy."
_HINT_AGENT = "Use only agents the owner's expansion policy allows."
_HINT_STAGE = "Use only stages the owner's expansion policy allows."
_HINT_PRODUCES = "Produce only artifact kinds the owner's expansion policy allows."
_HINT_FLAGS = "Governance attributes come from the owner's template; do not set them."
_HINT_DEPS = "Depend only on existing nodes or on nodes of this expansion."
_HINT_REWIRE = "Rewire only targets the owner's expansion policy allows, and only add edges."
_HINT_SIZE = "Propose fewer nodes."
_HINT_OWNER_CONFIG = (
    "The workflow author must give the owner a valid expansion_template and policy."
)


def _s(value: object, limit: int = 60) -> str:
    return sanitize_for_terminal(value, limit)


def _why(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_s(exc, 120)}"


def _finding(
    rule_id: str,
    message: str,
    *,
    location: object = None,
    evidence: str | None = None,
    remediation: str | None = None,
) -> Finding:
    return Finding(
        rule_id,
        PolicyCategory.CHANGE_CONTROL,
        Severity.HIGH,
        _s(message, 240),
        location=_s(location, 80) if location is not None else None,
        evidence=_s(evidence, 80) if evidence is not None else None,
        remediation=remediation,
    )


# --------------------------------------------------------------------------- owner policy


@dataclass(frozen=True)
class _Policy:
    id_prefix: str
    allowed_agents: frozenset[str]
    allowed_stages: frozenset[str]
    allowed_produces: frozenset[str]
    rewire_targets: tuple[str, ...]
    max_nodes: int


def _strings(raw: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = raw.get(name, ())
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{POLICY_KEY}.{name} must be a list of strings")
    return tuple(value)


def _parse_policy(raw: object) -> _Policy:
    """Validate the owner's ``expansion_policy``; a missing list means "nothing allowed"."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{POLICY_KEY} must be a mapping")
    prefix, max_nodes = raw.get("id_prefix", ""), raw.get("max_nodes", 0)
    if not isinstance(prefix, str):
        raise ValueError(f"{POLICY_KEY}.id_prefix must be a string")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 0:
        raise ValueError(f"{POLICY_KEY}.max_nodes must be a non-negative integer")
    targets = _strings(raw, "rewire_targets")
    for pattern in targets:
        if not (NODE_ID_RE.fullmatch(pattern) or _PREFIX_PATTERN_RE.fullmatch(pattern)):
            raise ValueError(f"{POLICY_KEY}.rewire_targets has an invalid pattern")
    return _Policy(
        prefix,
        frozenset(_strings(raw, "allowed_agents")),
        frozenset(_strings(raw, "allowed_stages")),
        frozenset(_strings(raw, "allowed_produces")),
        targets,
        min(max_nodes, MAX_EXPANSION_NODES),
    )


def _bounded(value: object, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{TEMPLATE_KEY}.{name} must be a finite number")
    return float(min(max(value, low), high))


def _clamped_retry(raw: object, name: str) -> dict[str, Any]:
    """A retry dict with every bound clamped (``RetryPolicy`` itself would reject out-of-range)."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{TEMPLATE_KEY}.{name} must be a mapping")
    out = dict(raw)
    if "max_attempts" in out:
        attempts = _bounded(out["max_attempts"], 1, RetryPolicy.HARD_MAX_ATTEMPTS, name)
        out["max_attempts"] = int(attempts)
    for field_name, (low, high) in _RETRY_BOUNDS.items():
        if field_name in out:
            out[field_name] = _bounded(out[field_name], low, high, f"{name}.{field_name}")
    return out


def _parse_template(raw: object, config: EngineConfig) -> NodeSpec:
    """The owner's ``expansion_template`` as a ``NodeSpec`` with retry and timeout clamped."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{TEMPLATE_KEY} must be a mapping (a NodeSpec dict)")
    data = dict(raw)
    data["retry"] = _clamped_retry(data.get("retry", {}), "retry")
    if "fallback_retry" in data:
        data["fallback_retry"] = _clamped_retry(data["fallback_retry"], "fallback_retry")
    timeout = _bounded(data.get("timeout_s", 300.0), 0.0, config.max_attempt_s, "timeout_s")
    if timeout <= 0:
        raise ValueError(f"{TEMPLATE_KEY}.timeout_s must be positive")
    data["timeout_s"] = timeout
    return NodeSpec.from_dict(data)


def _matches(target: str, pattern: str) -> bool:
    """``pattern`` is an exact node id or ``prefix.*`` (validated by ``_parse_policy``)."""
    if pattern.endswith(".*"):
        prefix = pattern[:-1]
        return target.startswith(prefix) and len(target) > len(prefix)
    return target == pattern


# --------------------------------------------------------------------------- nodes


def _narrowed(
    template: AutonomyProfile | None, proposed: AutonomyProfile | None
) -> AutonomyProfile | None:
    return proposed if template is None else template.narrow(proposed)


def _autonomy_problems(
    template: AutonomyProfile | None,
    proposed: AutonomyProfile | None,
    result: AutonomyProfile | None,
) -> list[str]:
    """A scope request that narrows to *nothing* tries to widen beyond the template (and would
    re-widen if normalised twice, because an empty ``write_globs`` means "no opinion")."""
    if template is None or proposed is None or result is None:
        return []
    problems = []
    if proposed.write_globs and not result.write_globs:
        problems.append("requested write scope is outside the owner's template")
    if tuple(proposed.read_globs) != _DEFAULT_READ_GLOBS and not result.read_globs:
        problems.append("requested read scope is outside the owner's template")
    return problems


def _with_mandatory_gates(gates: tuple[GateSpec, ...]) -> tuple[GateSpec, ...]:
    present = {gate.type for gate in gates}
    return (*gates, *(gate for gate in MANDATORY_EXIT_GATES if gate.type not in present))


def _build(template: NodeSpec, spec: NodeSpec, owner: NodeSpec) -> NodeSpec:
    params = {k: v for k, v in spec.params.items() if k not in _RIGHTS_KEYS}
    params.update(template.params)
    return dataclasses.replace(
        template,
        id=spec.id,
        title=_s(spec.title, 120).strip() or spec.id,
        depends_on=tuple(dict.fromkeys((*template.depends_on, owner.id, *spec.depends_on))),
        params=params,
        risk=max(template.risk, spec.risk),
        autonomy=_narrowed(template.autonomy, spec.autonomy),
        exit_gates=_with_mandatory_gates(template.exit_gates),
        mandatory=False,
        side_effecting=False,
        compensator=None,
        idempotent=False,
        compensates=(),
        skip_if=None,
    )


def _normalise_node(
    spec: object,
    owner: NodeSpec,
    template: NodeSpec,
    policy: _Policy,
    graph: WorkflowGraph,
    new_ids: set[str],
    seen: set[str],
    findings: list[Finding],
) -> NodeSpec | None:
    """The conformant node built from ``spec``, or ``None`` (the reasons go into ``findings``)."""
    if not isinstance(spec, NodeSpec):
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                "expansion contains an entry that is not a node",
                location=owner.id,
            )
        )
        return None
    node_id = spec.id
    if not isinstance(node_id, str) or not NODE_ID_RE.fullmatch(node_id):
        findings.append(_finding(RULE_CONFORMANCE, "node id is not valid", location=owner.id))
        return None
    before = len(findings)

    def problem(message: str, *, evidence: str, hint: str, rule: str = RULE_CONFORMANCE) -> None:
        findings.append(
            _finding(rule, message, location=node_id, evidence=evidence, remediation=hint)
        )

    prefix = policy.id_prefix
    if not prefix or not node_id.startswith(prefix) or len(node_id) <= len(prefix):
        problem(
            "node id is outside the id prefix of the owner's expansion policy",
            evidence="id",
            hint=_HINT_ID,
        )
    if node_id in seen:
        problem("node id is used twice in the expansion", evidence="duplicate-id", hint=_HINT_ID)
        return None
    seen.add(node_id)
    if graph.has(node_id) and graph.owner_of(node_id) != owner.id:
        if graph.node(node_id).mandatory:
            problem(
                "expansion would replace a mandatory node",
                evidence="mandatory-id",
                hint=_HINT_ID,
                rule=RULE_MANDATORY,
            )
        else:
            problem("node id clashes with an existing node", evidence="id-clash", hint=_HINT_ID)
    for agent in dict.fromkeys((spec.agent, spec.fallback_agent)):
        if agent and agent not in policy.allowed_agents:
            problem(
                f"agent {_s(agent)!r} is not allowed by the owner's expansion policy",
                evidence=f"agent:{_s(agent)}",
                hint=_HINT_AGENT,
            )
    stage = getattr(spec.stage, "value", spec.stage)
    if stage not in policy.allowed_stages:
        problem(
            f"stage {_s(stage)!r} is not allowed by the owner's expansion policy",
            evidence=f"stage:{_s(stage)}",
            hint=_HINT_STAGE,
        )
    for kind in dict.fromkeys(spec.produces):
        if kind not in policy.allowed_produces:
            problem(
                f"produces {_s(kind)!r}, not allowed by the owner's expansion policy",
                evidence=f"produces:{_s(kind)}",
                hint=_HINT_PRODUCES,
            )
    privileges = (
        ("mandatory", spec.mandatory),
        ("side_effecting", spec.side_effecting),
        ("compensator", spec.compensator),
        ("skip_if", spec.skip_if),
        ("compensates", spec.compensates),
    )
    for name, value in privileges:
        if value:
            problem(
                f"proposed node sets governance attribute {name!r}",
                evidence=f"attribute:{name}",
                hint=_HINT_FLAGS,
            )
    for dep in dict.fromkeys(spec.depends_on):
        if not isinstance(dep, str) or dep == node_id or not (dep in new_ids or graph.has(dep)):
            problem(
                f"node has an invalid dependency {_s(dep)!r}",
                evidence=f"dependency:{_s(dep)}",
                hint=_HINT_DEPS,
            )
    for key in sorted(_RIGHTS_KEYS & set(spec.params) - set(template.params)):
        problem(
            f"node params carry {key!r}, which only the owner's author may set",
            evidence=f"params:{key}",
            hint=_HINT_FLAGS,
        )
    try:
        node = _build(template, spec, owner)
    except (ConfigError, ValueError, TypeError, AttributeError) as exc:
        problem(
            f"node cannot be built from the owner's template ({_why(exc)})",
            evidence="template",
            hint=_HINT_OWNER_CONFIG,
        )
        return None
    for message in _autonomy_problems(template.autonomy, spec.autonomy, node.autonomy):
        problem(message, evidence="autonomy", hint=_HINT_FLAGS)
    return node if len(findings) == before else None


# --------------------------------------------------------------------------- rewires


def _is_names(value: object) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) <= MAX_REWIRE_EDGES
        and all(isinstance(item, str) for item in value)
    )


def _rewire_problems(
    target: object,
    rewire: object,
    owner: NodeSpec,
    policy: _Policy,
    graph: WorkflowGraph,
    new_ids: set[str],
) -> list[tuple[str, str]]:
    """``(rule id, message)`` per violation of one rewire entry."""
    if not isinstance(target, str) or not NODE_ID_RE.fullmatch(target):
        return [(RULE_CONFORMANCE, "rewire target is not a valid node id")]
    if not isinstance(rewire, Rewire):
        return [(RULE_CONFORMANCE, "rewire entry is not a Rewire")]
    problems: list[tuple[str, str]] = []
    if not any(_matches(target, pattern) for pattern in policy.rewire_targets):
        problems.append(
            (RULE_CONFORMANCE, "rewire target is outside the allowed targets of the owner's policy")
        )
    if target == owner.id:
        problems.append((RULE_CONFORMANCE, "an expansion may not rewire its own owner"))
    elif target in new_ids or graph.owner_of(target) == owner.id:
        problems.append((RULE_CONFORMANCE, "an expansion may not rewire its own nodes"))
    if not graph.has(target):
        problems.append((RULE_CONFORMANCE, "rewire target does not exist"))
    elif graph.node(target).mandatory:
        problems.append((RULE_MANDATORY, "plan rewires a mandatory node"))
    if not _is_names(rewire.add_depends_on) or not _is_names(rewire.add_overlay_from):
        problems.append((RULE_CONFORMANCE, "rewire edges must be short lists of strings"))
        return problems
    problems.extend(
        (RULE_CONFORMANCE, f"rewire adds an invalid dependency {_s(dep)!r}")
        for dep in rewire.add_depends_on
        if dep == target or not (dep in new_ids or graph.has(dep))
    )
    problems.extend(
        (RULE_CONFORMANCE, f"rewire overlays non-change-set kind {_s(kind)!r}")
        for kind in rewire.add_overlay_from
        if kind not in CHANGESET_KINDS
    )
    return problems


def _normalise_rewires(
    raw: Mapping[str, Rewire],
    owner: NodeSpec,
    policy: _Policy,
    graph: WorkflowGraph,
    new_ids: set[str],
    findings: list[Finding],
) -> dict[str, Rewire]:
    try:
        items = list(islice(raw.items(), MAX_REWIRES + 1))
    except AttributeError:
        findings.append(
            _finding(RULE_CONFORMANCE, "expansion rewire is not a mapping", location=owner.id)
        )
        return {}
    if len(items) > MAX_REWIRES:
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                f"expansion rewires more than {MAX_REWIRES} nodes",
                location=owner.id,
                remediation=_HINT_REWIRE,
            )
        )
        return {}
    out: dict[str, Rewire] = {}
    for target, rewire in sorted(items, key=lambda item: str(item[0])):
        problems = _rewire_problems(target, rewire, owner, policy, graph, new_ids)
        findings.extend(
            _finding(rule, message, location=target, evidence="rewire", remediation=_HINT_REWIRE)
            for rule, message in problems
        )
        if not problems and isinstance(target, str) and isinstance(rewire, Rewire):
            out[target] = Rewire(
                tuple(dict.fromkeys(rewire.add_depends_on)),
                tuple(dict.fromkeys(rewire.add_overlay_from)),
            )
    return out


# --------------------------------------------------------------------------- entry point


def _normalize(
    owner: NodeSpec,
    expansion: PlanExpansion,
    graph: WorkflowGraph,
    config: EngineConfig,
    findings: list[Finding],
) -> PlanExpansion:
    empty = PlanExpansion(owner.id, (), {})
    if not isinstance(expansion, PlanExpansion):
        findings.append(_finding(RULE_CONFORMANCE, "not a PlanExpansion", location=owner.id))
        return empty
    if expansion.owner != owner.id:
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                f"expansion owner {_s(expansion.owner)!r} is not the producing node {owner.id!r}",
                location=owner.id,
                remediation=_HINT_OWNER,
            )
        )
    if not graph.has(owner.id):
        findings.append(
            _finding(
                RULE_CONFORMANCE, "expansion owner is not a node of the plan", location=owner.id
            )
        )
        return empty
    try:
        template = _parse_template(owner.params.get(TEMPLATE_KEY), config)
        policy = _parse_policy(owner.params.get(POLICY_KEY))
    except (ConfigError, KeyError, ValueError, TypeError, AttributeError) as exc:
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                f"node {owner.id!r} may not expand the plan ({_why(exc)})",
                location=owner.id,
                remediation=_HINT_OWNER_CONFIG,
            )
        )
        return empty
    try:
        proposed = tuple(islice(iter(expansion.nodes), policy.max_nodes + 1))
    except TypeError:
        findings.append(
            _finding(RULE_CONFORMANCE, "expansion nodes are not a collection", location=owner.id)
        )
        return empty
    if len(proposed) > policy.max_nodes:
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                f"expansion adds more than {policy.max_nodes} node(s) (the owner's policy limit)",
                location=owner.id,
                remediation=_HINT_SIZE,
            )
        )
        return empty
    new_ids = {n.id for n in proposed if isinstance(n, NodeSpec) and isinstance(n.id, str)}
    seen: set[str] = set()
    built = (
        _normalise_node(spec, owner, template, policy, graph, new_ids, seen, findings)
        for spec in proposed
    )
    nodes = tuple(node for node in built if node is not None)
    rewire = _normalise_rewires(expansion.rewire, owner, policy, graph, new_ids, findings)
    candidate = PlanExpansion(owner.id, nodes, rewire)
    if findings:
        return candidate
    try:
        graph.copy().apply_expansion(candidate)  # structure: cycles, contracts, ids, capacity
    except OrchestratorError as exc:
        message = _s(exc, 200)
        rule = RULE_MANDATORY if "mandatory" in message else RULE_CONFORMANCE
        findings.append(
            _finding(
                rule,
                f"expansion cannot be applied: {message}",
                location=owner.id,
                remediation="Propose an acyclic expansion with valid nodes and edges.",
            )
        )
    return candidate


def _capped(findings: list[Finding]) -> list[Finding]:
    unique = list(dict.fromkeys(findings))
    if len(unique) <= MAX_FINDINGS:
        return unique
    kept = unique[: MAX_FINDINGS - 1]
    omitted = len(unique) - len(kept)
    kept.append(_finding(RULE_CONFORMANCE, f"{omitted} further problem(s) omitted"))
    return kept


def normalize_expansion(
    owner: NodeSpec,
    expansion: PlanExpansion,
    graph: WorkflowGraph,
    config: EngineConfig,
) -> tuple[PlanExpansion, list[Finding]]:
    """Normalise ``expansion`` proposed by ``owner`` against the live ``graph`` (module docstring).

    Returns ``(normalised expansion, findings)``; apply the expansion only if ``findings`` is empty.
    ``graph`` is only read (the trial application runs on a copy).
    """
    findings: list[Finding] = []
    try:
        result = _normalize(owner, expansion, graph, config, findings)
    except Exception as exc:  # hostile content must not crash a tick: fail closed instead
        findings.append(
            _finding(
                RULE_CONFORMANCE,
                f"expansion could not be processed ({type(exc).__name__})",
                location=owner.id,
                remediation=_HINT_OWNER,
            )
        )
        result = PlanExpansion(owner.id, (), {})
    return result, _capped(findings)


__all__ = [
    "MANDATORY_EXIT_GATES",
    "MAX_EXPANSION_NODES",
    "MAX_FINDINGS",
    "POLICY_KEY",
    "RULE_CONFORMANCE",
    "RULE_MANDATORY",
    "TEMPLATE_KEY",
    "normalize_expansion",
]
