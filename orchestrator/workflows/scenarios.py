"""Scenario loading, scripted "human" behaviour and self-checking expectations (L5).

A scenario (``orchestrator/scenarios/<name>/scenario.json``, format in ``docs/scenario-format.md``)
wires a request, a baseline, fault injection, scripted human actions and ``expect`` assertions.
Scripted actions use exactly the public commands a human would (``approve`` / ``answer`` / ``amend`` /
``kill``); approvals made this way carry channel ``scripted`` and mark the run ``DEMO_SCRIPTED``.
Expectations are evaluated against the audit log, run state and artifacts only.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.clock import SystemClock
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import ApprovalError, ConfigError, OrchestratorError, StateError
from orchestrator.core.fsutil import read_text_lf
from orchestrator.core.lineage import LineageGraph
from orchestrator.core.paths import validate_rel_path
from orchestrator.core.serde import loads_strict
from orchestrator.core.types import AuditEvent, DecisionKind, EventType, RunStatus, SafeStopOverride
from orchestrator.engine.config import AutonomyMode, EngineConfig
from orchestrator.engine.scheduler import Orchestrator, RunResult
from orchestrator.workflows.registry import (
    FaultSpec,
    RegistryOptions,
    build_registries,
    default_policy,
    fault_specs,
)
from orchestrator.workflows.sdlc import SdlcOptions, build_sdlc_graph
from orchestrator.workflows.summary import engineering_summary_finalizer

SCENARIO_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")
SCENARIOS_ROOT = Path(__file__).resolve().parent.parent / "scenarios"
_TRIGGER_RE = re.compile(r"^(run\.waiting|run\.started|node\.succeeded:[a-z][a-z0-9_.*-]{0,60})$")
_ACTIONS = frozenset({"approve", "answer", "amend", "kill"})


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "shortener").is_dir():
            return parent
    raise ConfigError("cannot locate the repository root (pyproject.toml next to shortener/)")


def node_matches(pattern: str, node_id: str) -> bool:
    """Exact node id or ``prefix.*``."""
    if pattern.endswith(".*"):
        return node_id.startswith(pattern[:-1])
    return pattern == node_id


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    kind: str
    directory: Path
    request_text: str
    baseline_empty: bool
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    params: Mapping[str, Any]
    autonomy_mode: str
    max_parallel: int
    promote: str
    faults: tuple[Mapping[str, Any], ...]
    script: tuple[Mapping[str, Any], ...]
    expect: Mapping[str, Any]

    def read_file(self, relative: str) -> str:
        return read_text_lf(self.directory / validate_rel_path(relative))


def list_scenarios(root: Path | None = None) -> list[str]:
    base = root or SCENARIOS_ROOT
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "scenario.json").is_file())


def load_scenario(name: str, root: Path | None = None) -> Scenario:
    if not SCENARIO_NAME_RE.match(name):
        raise ConfigError(f"invalid scenario name: {name!r}")
    directory = (root or SCENARIOS_ROOT) / name
    path = directory / "scenario.json"
    if not path.is_file():
        raise ConfigError(
            f"unknown scenario {name!r}; available: {', '.join(list_scenarios(root)) or 'none'}"
        )
    raw = loads_strict(path.read_bytes())
    if not isinstance(raw, Mapping) or raw.get("schema") != 1:
        raise ConfigError(f"{name}: unsupported scenario schema")
    baseline = raw.get("baseline", {})
    workflow = raw.get("workflow", {})
    script = tuple(raw.get("script", ()))
    for i, entry in enumerate(script):
        if (
            not isinstance(entry, Mapping)
            or entry.get("do") not in _ACTIONS
            or not _TRIGGER_RE.match(str(entry.get("on", "")))
        ):
            raise ConfigError(f"{name}: script entry {i} is invalid")
    scenario = Scenario(
        name=name,
        title=str(raw.get("title", name)),
        kind=str(raw.get("kind", name)),
        directory=directory,
        request_text=read_text_lf(directory / validate_rel_path(str(raw["request_file"]))),
        baseline_empty=bool(baseline.get("empty", False)),
        include=tuple(baseline.get("include", ("**",))),
        exclude=tuple(baseline.get("exclude", ())),
        params=dict(raw.get("params", {})),
        autonomy_mode=str(workflow.get("autonomy_mode", "guided")),
        max_parallel=int(workflow.get("max_parallel", 4)),
        promote=str(workflow.get("promote", "bundle")),
        faults=tuple(raw.get("faults", ())),
        script=script,
        expect=dict(raw.get("expect", {})),
    )
    AutonomyMode(scenario.autonomy_mode)
    return scenario


@dataclass
class ScenarioResult:
    run_id: str
    status: RunStatus
    exit_code: int
    problems: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return not self.problems


class ScenarioRunner:
    def __init__(self, runs_dir: Path | str, *, scenarios_root: Path | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.scenarios_root = scenarios_root

    # ------------------------------------------------------------------ public
    def run(
        self,
        scenario: Scenario,
        *,
        run_id: str | None = None,
        extra_faults: Sequence[FaultSpec] = (),
        faults_off: bool = False,
        apply_root: Path | None = None,
        verify: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> ScenarioResult:
        out = log or (lambda _msg: None)
        transcript: list[str] = []

        def say(message: str) -> None:
            transcript.append(message)
            out(message)

        faults = () if faults_off else fault_specs(scenario.faults)
        faults = (*faults, *extra_faults)
        graph = build_sdlc_graph(SdlcOptions(promote_mode=scenario.promote, params=scenario.params))
        registries = build_registries(
            RegistryOptions(
                faults=faults, node_agents={n.id: n.agent for n in graph.nodes.values()}
            )
        )
        fired_events: set[int] = set()
        lock = threading.Lock()
        holder: dict[str, Any] = {}

        def listener(event: AuditEvent) -> None:
            if (
                event.type != EventType.NODE_SUCCEEDED
                or event.node_id is None
                or "orch" not in holder
            ):
                return
            for idx, entry in enumerate(scenario.script):
                trigger = str(entry["on"])
                if not trigger.startswith("node.succeeded:") or not node_matches(
                    trigger.split(":", 1)[1], event.node_id
                ):
                    continue
                with lock:
                    if idx in fired_events:
                        continue
                    fired_events.add(idx)
                self._act(holder["orch"], holder["run_id"], scenario, entry, say)

        config = EngineConfig(
            allow_scripted=True,
            autonomy_mode=AutonomyMode(scenario.autonomy_mode),
            max_parallel=scenario.max_parallel,
        )
        orch = Orchestrator(
            self.runs_dir,
            registries,
            policy=default_policy(),
            config=config,
            finalizers=(engineering_summary_finalizer,),
            listeners=(listener,),
        )
        baseline_source = None if scenario.baseline_empty else repo_root()
        run = orch.start(
            graph,
            scenario=scenario.name,
            request_text=scenario.request_text,
            baseline_source=baseline_source,
            include=scenario.include,
            exclude=scenario.exclude,
            params={
                **scenario.params,
                "faults": [dict(f) for f in scenario.faults] if not faults_off else [],
            },
            run_id=run_id,
            created_by="demo-operator",
            apply_root=apply_root,
        )
        holder.update(orch=orch, run_id=run)
        say(f"run {run} started ({scenario.name})")
        result: RunResult = orch.run(run)
        answered: set[tuple[int, str]] = set()
        voted: set[tuple[int, str]] = set()
        guard = 0
        while result.status == RunStatus.WAITING_HUMAN and guard < 40:
            guard += 1
            say(
                f"waiting: approvals={list(result.pending_approvals)} questions={list(result.open_questions)}"
            )
            if not self._on_waiting(orch, run, scenario, voted, answered, say):
                say("no scripted action matches what the run is waiting for; stopping here")
                break
            result = orch.resume(run)
        say(f"run {run} finished: {result.status.value} - {result.reason}")
        problems = check_expectations(orch, run, scenario.expect, result.status) if verify else []
        for problem in problems:
            say(f"EXPECTATION FAILED: {problem}")
        return ScenarioResult(run, result.status, result.exit_code, problems, transcript)

    # ------------------------------------------------------------------ scripted actions
    def _on_waiting(
        self,
        orch: Orchestrator,
        run_id: str,
        scenario: Scenario,
        voted: set[tuple[int, str]],
        answered: set[tuple[int, str]],
        say: Callable[[str], None],
    ) -> bool:
        acted = False
        for idx, entry in enumerate(scenario.script):
            if entry["on"] != "run.waiting":
                continue
            summary = orch.status(run_id)
            if entry["do"] == "approve":
                pattern = str(entry.get("request", {}).get("node", "*"))
                for req in summary.pending_approvals:
                    key = (idx, str(req["id"]))
                    if key in voted or not node_matches(pattern, str(req["node"])):
                        continue
                    voted.add(key)
                    self._vote(orch, run_id, req, entry, say)
                    acted = True
            elif entry["do"] == "answer":
                todo = [q for q in summary.open_questions if (idx, str(q["id"])) not in answered]
                if todo:
                    for q in todo:
                        answered.add((idx, str(q["id"])))
                    self._answer(orch, run_id, todo, entry, say)
                    acted = True
            elif entry["do"] == "kill":
                self._act(orch, run_id, scenario, entry, say)
                acted = True
        return acted

    def _vote(
        self,
        orch: Orchestrator,
        run_id: str,
        req: Mapping[str, Any],
        entry: Mapping[str, Any],
        say: Callable[[str], None],
    ) -> None:
        for voter in entry.get("by", ()):
            pending_ids = {str(r["id"]) for r in orch.status(run_id).pending_approvals}
            if str(req["id"]) not in pending_ids:
                break
            try:
                orch.approve(
                    run_id,
                    str(req["id"]),
                    str(voter),
                    str(entry.get("comment", "")),
                    confirm=str(req.get("confirm", "")),
                    channel="scripted",
                )
                say(f"scripted approval of {req['id']} ({req['node']}) by {voter}")
            except (ApprovalError, StateError, OrchestratorError) as exc:
                say(f"vote by {voter!r} on {req['id']} refused as designed: {exc}")

    def _answer(
        self,
        orch: Orchestrator,
        run_id: str,
        questions: Sequence[Mapping[str, Any]],
        entry: Mapping[str, Any],
        say: Callable[[str], None],
    ) -> None:
        rules = [(re.compile(str(r["match"])), str(r["answer"])) for r in entry.get("answers", ())]
        answers: dict[str, str] = {}
        defaults: list[str] = []
        for q in questions:
            for pattern, answer in rules:
                if pattern.search(str(q["text"])):
                    if answer == "@default":
                        if q.get("default"):
                            defaults.append(str(q["id"]))
                    else:
                        answers[str(q["id"])] = answer
                    break
        if not answers and not defaults:
            say("no scripted answer matches the open questions")
            return
        orch.answer(run_id, answers, str(entry.get("by", "demo-human")), accept_default=defaults)
        say(f"scripted answers for {sorted([*answers, *defaults])}")

    def _act(
        self,
        orch: Orchestrator,
        run_id: str,
        scenario: Scenario,
        entry: Mapping[str, Any],
        say: Callable[[str], None],
    ) -> None:
        action = entry["do"]
        try:
            if action == "amend":
                orch.amend(
                    run_id,
                    scenario.read_file(str(entry["text_file"])),
                    str(entry.get("by", "demo-human")),
                    str(entry.get("reason", "amendment")),
                )
                say("scripted amendment of the request")
            elif action == "kill":
                orch.kill(
                    run_id,
                    str(entry.get("by", "operator")),
                    str(entry.get("reason", "scripted kill")),
                )
                say("scripted kill switch")
        except OrchestratorError as exc:
            say(f"scripted {action} failed: {exc}")


# =============================================================================
# expectations
# =============================================================================


def check_expectations(
    orch: Orchestrator, run_id: str, expect: Mapping[str, Any], final: RunStatus
) -> list[str]:
    """Return a list of mismatches between ``expect`` and the finished run (empty = verified)."""
    problems: list[str] = []
    summary = orch.status(run_id)
    events = orch.events(run_id)
    counts = Counter(e.type.value for e in events)
    run_dir = orch.runs_dir / run_id
    store = ArtifactStore(run_dir / "artifacts", clock=SystemClock())
    decisions = DecisionLog(run_dir / "decisions.jsonl", run_id=run_id, clock=SystemClock())

    if "run_status" in expect and final.value != expect["run_status"]:
        problems.append(f"run status is {final.value}, expected {expect['run_status']}")
    for node, status in expect.get("node_statuses", {}).items():
        if summary.nodes.get(node) != status:
            problems.append(f"node {node} is {summary.nodes.get(node)}, expected {status}")
    for pattern, status in expect.get("node_status_patterns", {}).items():
        matched = [n for n in summary.nodes if node_matches(pattern, n)]
        if not matched:
            problems.append(f"no node matches {pattern}")
        problems.extend(
            f"node {n} is {summary.nodes[n]}, expected {status}"
            for n in matched
            if summary.nodes[n] != status
        )
    problems.extend(
        f"event {t} never occurred" for t in expect.get("events_present", ()) if counts[t] == 0
    )
    problems.extend(
        f"event {t} occurred but must not" for t in expect.get("events_absent", ()) if counts[t] > 0
    )
    kinds = {d.kind.value for d in decisions.all()}
    problems.extend(
        f"no {k} decision was recorded" for k in expect.get("decision_kinds", ()) if k not in kinds
    )
    problems.extend(
        f"artifact kind {k} was not produced"
        for k in expect.get("artifacts_present", ())
        if not store.active(k)
    )
    if "min_parallel_width" in expect or "metrics" in expect:
        metrics = orch.metrics(run_id)
        width = getattr(metrics, "max_concurrency_observed", 0)
        if width < int(expect.get("min_parallel_width", 0)):
            problems.append(f"observed parallel width {width} < {expect['min_parallel_width']}")
        limits = expect.get("metrics", {})
        if "retry_frequency_max" in limits and (
            getattr(metrics, "retry_frequency", 0.0) or 0.0
        ) > float(limits["retry_frequency_max"]):
            problems.append("retry frequency above the scenario bound")
        if "rollback_count" in limits and getattr(metrics, "rollbacks", 0) != int(
            limits["rollback_count"]
        ):
            problems.append(
                f"rollbacks {getattr(metrics, 'rollbacks', 0)} != {limits['rollback_count']}"
            )
    replan = expect.get("replan", {})
    if counts[EventType.NODE_CACHE_HIT.value] < int(replan.get("cache_hits_min", 0)):
        problems.append("fewer cache hits than expected after the re-plan")
    for item in expect.get("lineage", ()):
        art = store.latest(str(item["artifact"]))
        if art is None:
            problems.append(f"lineage: artifact {item['artifact']} missing")
        elif not LineageGraph(store, decisions).reaches(
            art.id, DecisionKind(item["reaches_decision_kind"])
        ):
            problems.append(
                f"lineage of {item['artifact']} does not reach a {item['reaches_decision_kind']} decision"
            )
    patch_paths = expect.get("files_in_patch", ())
    if patch_paths:
        patch = store.latest("release_patch")
        text = store.read_text(patch) if patch is not None else ""
        problems.extend(
            f"patch does not touch {p}"
            for p in patch_paths
            if f"diff --git a/{p} b/{p}" not in text
        )
    if "blocking_questions" in expect and len(
        [q for q in summary.open_questions if q.get("blocking")]
    ) != int(expect["blocking_questions"]):
        problems.append("unexpected number of open blocking questions at the end")
    return problems


__all__ = [
    "SCENARIOS_ROOT",
    "Scenario",
    "ScenarioResult",
    "ScenarioRunner",
    "SafeStopOverride",
    "check_expectations",
    "list_scenarios",
    "load_scenario",
    "node_matches",
    "repo_root",
]
