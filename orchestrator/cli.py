"""``agentic`` command line (L5).

Exit codes: 0 succeeded, 10 waiting for a human, 20 failed, 30 safe-stopped, 2 usage / input error.
Everything printed that originates from agents or humans passes ``core.sanitize``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from orchestrator.core.errors import OrchestratorError
from orchestrator.core.fsutil import read_text_lf
from orchestrator.core.metrics import aggregate, load_all, render_table
from orchestrator.core.sanitize import sanitize_for_terminal as clean
from orchestrator.core.state import StateStore
from orchestrator.core.types import RunStatus, SafeStopOverride
from orchestrator.engine.config import AutonomyMode, EngineConfig
from orchestrator.engine.scheduler import Orchestrator, RunResult, RunSummary
from orchestrator.workflows.registry import (
    FaultSpec,
    RegistryOptions,
    build_registries,
    default_policy,
    fault_specs,
)
from orchestrator.workflows.scenarios import (
    ScenarioRunner,
    list_scenarios,
    load_scenario,
    repo_root,
)
from orchestrator.workflows.sdlc import SdlcOptions, build_sdlc_graph
from orchestrator.workflows.summary import engineering_summary_finalizer

DEFAULT_RUNS_DIR = ".agentic/runs"
EXIT_USAGE = 2


def _make_orchestrator(
    runs_dir: Path,
    *,
    faults: Sequence[FaultSpec] = (),
    autonomy: str = "guided",
    max_parallel: int = 4,
    allow_scripted: bool = False,
) -> Orchestrator:
    graph = build_sdlc_graph()
    registries = build_registries(
        RegistryOptions(
            faults=tuple(faults), node_agents={n.id: n.agent for n in graph.nodes.values()}
        )
    )
    return Orchestrator(
        runs_dir,
        registries,
        policy=default_policy(),
        config=EngineConfig(
            autonomy_mode=AutonomyMode(autonomy),
            max_parallel=max_parallel,
            allow_scripted=allow_scripted,
        ),
        finalizers=(engineering_summary_finalizer,),
    )


def _for_run(runs_dir: Path, run_id: str) -> Orchestrator:
    """An orchestrator configured like the run was started (faults are persisted in the run params)."""
    state = StateStore(runs_dir).load(run_id)
    faults = (
        fault_specs(state.params.get("faults", []))
        if isinstance(state.params.get("faults"), list)
        else ()
    )
    return _make_orchestrator(runs_dir, faults=faults, autonomy=state.autonomy_mode)


def _print_summary(summary: RunSummary) -> None:
    print(f"run        {summary.run_id}  ({summary.scenario})")
    print(f"status     {summary.status.value}  {clean(summary.reason)}")
    print(f"assurance  {summary.assurance}")
    print("nodes:")
    for node, status in summary.nodes.items():
        print(f"  {node:<28} {status}")
    if summary.pending_approvals:
        print("pending approvals:")
        for req in summary.pending_approvals:
            votes = ", ".join(req["votes"]) or "none yet"
            print(
                f"  {req['id']}  node={req['node']}  risk={req['risk']}  needs {req['required']} approver(s)  votes: {votes}"
            )
            print(
                f"      confirm code: {req['confirm']}   (agentic approve {summary.run_id} {req['id']} --by NAME --confirm {req['confirm']})"
            )
            for line in str(req["summary"]).splitlines()[:12]:
                print(f"      | {clean(line, 200)}")
    if summary.open_questions:
        print("open questions:")
        for q in summary.open_questions:
            print(
                f"  {q['id']}  [{'blocking' if q['blocking'] else 'non-blocking'}] {clean(q['text'], 240)}"
            )
            if q["options"]:
                print(f"      options: {'; '.join(clean(o, 120) for o in q['options'])}")
            if q["default"]:
                print(f"      default assumption: {clean(q['default'], 160)}")
    print(f"audit head  seq={summary.audit_head[0]} hash={summary.audit_head[1][:16]}")


def _print_result(result: RunResult) -> int:
    print(f"\nrun {result.run_id}: {result.status.value} - {clean(result.reason)}")
    if result.status == RunStatus.WAITING_HUMAN:
        print(f"  approvals pending: {', '.join(result.pending_approvals) or 'none'}")
        print(f"  questions open:    {', '.join(result.open_questions) or 'none'}")
        print(f"  next: agentic status {result.run_id}")
    print(f"  audit head: seq={result.audit_head[0]} hash={result.audit_head[1][:16]}")
    return result.exit_code


def _parse_fault(text: str) -> FaultSpec:
    node, _, rest = text.partition("=")
    fault, _, attempts = rest.partition(":")
    if not node or not fault:
        raise OrchestratorError(
            "--fault must look like NODE=FAULT[:ATTEMPTS] (e.g. 'implement.*=transient:2')"
        )
    return fault_specs(
        [{"node": node, "fault": fault, "attempts": int(attempts) if attempts else 99}]
    )[0]


# --------------------------------------------------------------------------- commands


def cmd_scenarios(_args: argparse.Namespace) -> int:
    for name in list_scenarios():
        sc = load_scenario(name)
        print(f"{name:<12} {sc.title}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    graph = build_sdlc_graph(SdlcOptions(promote_mode="apply" if args.apply else "bundle"))
    if args.format == "mermaid":
        print(graph.to_mermaid())
    elif args.format == "dot":
        print(graph.to_dot())
    else:
        import json

        print(json.dumps(graph.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    faults = () if args.faults_off else fault_specs(scenario.faults)
    faults = (*faults, *(_parse_fault(f) for f in args.fault))
    orch = _make_orchestrator(
        Path(args.runs_dir),
        faults=faults,
        autonomy=scenario.autonomy_mode,
        max_parallel=scenario.max_parallel,
    )
    graph = build_sdlc_graph(SdlcOptions(promote_mode=scenario.promote, params=scenario.params))
    run_id = orch.start(
        graph,
        scenario=scenario.name,
        request_text=scenario.request_text,
        baseline_source=None if scenario.baseline_empty else repo_root(),
        include=scenario.include,
        exclude=scenario.exclude,
        params={
            **scenario.params,
            "faults": [{"node": f.node, "fault": f.fault, "attempts": f.attempts} for f in faults],
        },
        run_id=args.run_id,
        created_by=args.created_by,
        apply_root=Path(args.apply_to) if args.apply_to else None,
    )
    print(f"started run {run_id}")
    return _print_result(orch.run(run_id))


def cmd_demo(args: argparse.Namespace) -> int:
    names = list_scenarios() if not args.names or args.names == ["all"] else args.names
    runner = ScenarioRunner(Path(args.runs_dir))
    failed = 0
    rows: list[tuple[str, str, str]] = []
    for name in names:
        scenario = load_scenario(name)
        extra = tuple(_parse_fault(f) for f in args.fault)
        print(f"\n=== scenario: {scenario.name} - {scenario.title}")
        result = runner.run(
            scenario,
            extra_faults=extra,
            faults_off=args.faults_off,
            verify=args.verify,
            log=lambda m: print(f"  {clean(m, 220)}"),
        )
        verdict = "verified" if result.verified else f"{len(result.problems)} expectation(s) failed"
        rows.append((scenario.name, result.status.value, verdict))
        print(f"  run dir: {Path(args.runs_dir) / result.run_id}")
        if args.verify and not result.verified:
            failed += 1
        elif result.status in (RunStatus.FAILED, RunStatus.SAFE_STOPPED) and args.verify:
            failed += 1
    print("\nsummary")
    for name, status, verdict in rows:
        print(f"  {name:<12} {status:<14} {verdict}")
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    _print_summary(orch.status(args.run_id))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    fn = orch.reject if args.reject else orch.approve
    result = fn(
        args.run_id, args.request_id, args.by, args.comment or "", confirm=args.confirm or ""
    )
    print(f"{args.request_id}: {result['status']} ({result['votes']} vote(s))")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    answers = {}
    for item in args.set or ():
        qid, _, text = item.partition("=")
        answers[qid] = text
    result = orch.answer(args.run_id, answers, args.by, accept_default=args.accept_default or ())
    print(f"answered: {', '.join(result['answered'])}")
    return 0


def cmd_amend(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    result = orch.amend(args.run_id, read_text_lf(args.text_file), args.by, args.reason)
    print(f"request amended ({result['artifact']}, decision {result['decision']})")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    override = None
    if args.override_safe_stop:
        if not args.by or not args.reason:
            raise OrchestratorError("--override-safe-stop needs --by and --reason")
        override = SafeStopOverride(args.by, args.reason, co_approver=args.co_approver)
    return _print_result(orch.resume(args.run_id, override=override))


def cmd_kill(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    result = orch.kill(args.run_id, args.by, args.reason)
    print("kill switch tripped" + (f"; run is now {result.status.value}" if result else ""))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    print(orch.lineage(args.run_id, args.artifact_id))
    return 0


def cmd_audit_verify(args: argparse.Namespace) -> int:
    orch = _for_run(Path(args.runs_dir), args.run_id)
    result = orch.verify_audit(args.run_id, args.anchor)
    print(
        f"audit chain: {'OK' if result.ok else 'BROKEN'}  events={result.count}  {clean(result.reason)}"
    )
    return 0 if result.ok else 1


def cmd_metrics(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    if args.run_id:
        metrics = [_for_run(runs_dir, args.run_id).metrics(args.run_id)]
    else:
        metrics = load_all(runs_dir)
    print(render_table(metrics, aggregate(metrics) if len(metrics) > 1 or args.all else None))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.runs_dir) / args.run_id / "report.md"
    if not path.is_file():
        raise OrchestratorError("no report yet: the run has not reached a terminal state")
    print(read_text_lf(path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic", description="Governed agentic SDLC orchestrator"
    )
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="where runs are stored (default: %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scenarios", help="list the built-in scenarios").set_defaults(func=cmd_scenarios)

    g = sub.add_parser("graph", help="print the SDLC workflow graph")
    g.add_argument("--format", choices=("mermaid", "dot", "json"), default="mermaid")
    g.add_argument("--apply", action="store_true", help="show the apply-mode variant of promote")
    g.set_defaults(func=cmd_graph)

    r = sub.add_parser(
        "run", help="start a scenario run and drive it until it finishes or needs a human"
    )
    r.add_argument("scenario")
    r.add_argument("--run-id")
    r.add_argument("--created-by", default="operator")
    r.add_argument(
        "--fault",
        action="append",
        default=[],
        help="NODE=FAULT[:ATTEMPTS], e.g. 'implement.*=transient:2'",
    )
    r.add_argument(
        "--faults-off", action="store_true", help="ignore the faults declared by the scenario"
    )
    r.add_argument(
        "--apply-to",
        help="directory to really apply the patch to (needs a scenario with promote=apply)",
    )
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("demo", help="run scenarios end to end with scripted human decisions")
    d.add_argument("names", nargs="*", help="scenario names or 'all'")
    d.add_argument("--verify", action="store_true", help="exit non-zero if an expectation fails")
    d.add_argument("--fault", action="append", default=[])
    d.add_argument("--faults-off", action="store_true")
    d.set_defaults(func=cmd_demo)

    for name, fn in (
        ("status", cmd_status),
        ("lineage", cmd_lineage),
        ("audit-verify", cmd_audit_verify),
        ("report", cmd_report),
    ):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        if name == "lineage":
            p.add_argument("artifact_id")
        if name == "audit-verify":
            p.add_argument("--anchor")
        p.set_defaults(func=fn)

    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} a pending request")
        p.add_argument("run_id")
        p.add_argument("request_id")
        p.add_argument("--by", required=True)
        p.add_argument(
            "--confirm", help="first 8 hex characters of the evidence hash shown by 'status'"
        )
        p.add_argument("--comment")
        p.set_defaults(func=cmd_approve, reject=(name == "reject"))

    a = sub.add_parser("answer", help="answer clarification questions")
    a.add_argument("run_id")
    a.add_argument("--by", required=True)
    a.add_argument("--set", action="append", help="QUESTION_ID=ANSWER")
    a.add_argument(
        "--accept-default", action="append", help="QUESTION_ID whose default assumption is accepted"
    )
    a.set_defaults(func=cmd_answer)

    m = sub.add_parser("amend", help="replace the request text (triggers a governed re-plan)")
    m.add_argument("run_id")
    m.add_argument("--by", required=True)
    m.add_argument("--reason", required=True)
    m.add_argument("--text-file", required=True)
    m.set_defaults(func=cmd_amend)

    rs = sub.add_parser("resume", help="continue a paused (or, with an override, safe-stopped) run")
    rs.add_argument("run_id")
    rs.add_argument("--override-safe-stop", action="store_true")
    rs.add_argument("--by")
    rs.add_argument("--reason")
    rs.add_argument("--co-approver")
    rs.set_defaults(func=cmd_resume)

    k = sub.add_parser("kill", help="operator safe-stop of a run")
    k.add_argument("run_id")
    k.add_argument("--by", required=True)
    k.add_argument("--reason", required=True)
    k.set_defaults(func=cmd_kill)

    mt = sub.add_parser("metrics", help="reliability metrics of one run or of all runs")
    mt.add_argument("run_id", nargs="?")
    mt.add_argument("--all", action="store_true")
    mt.set_defaults(func=cmd_metrics)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        code: Any = args.func(args)
    except OrchestratorError as exc:
        print(f"error: {clean(exc, 400)}", file=sys.stderr)
        return EXIT_USAGE
    except (ValueError, OSError) as exc:
        print(f"error: {clean(f'{type(exc).__name__}: {exc}', 400)}", file=sys.stderr)
        return EXIT_USAGE
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
