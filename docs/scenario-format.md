# Scenario file format

A scenario is a directory `orchestrator/scenarios/<name>/` containing `scenario.json` plus the files it references
(`request.md`, optional `answers.json`, `approvals.json`, `amendments.json`). It is the single place that wires a request,
a baseline, scripted human behaviour (for reproducible demos/CI) and **self-checking expectations** together.
`agentic demo <name> --verify` exits non-zero if the run does not match `expect`.

```jsonc
{
  "schema": 1,
  "name": "brownfield",
  "title": "Click limit for short links (enhancement with a schema migration)",
  "kind": "brownfield",                       // greenfield | brownfield | ambiguous (documentation only; classification is derived from the baseline)
  "request_file": "request.md",
  "baseline": {                                // exactly one of:
    "source": "repo",                          //   "repo" = the checkout root (found by walking up to pyproject.toml); or {"empty": true}
    "include": ["shortener/**", "tests/shortener/**"],
    "exclude": []
  },
  "params": { "perf_budget_ms": 250 },        // scenario parameters passed to the workflow builder / agents (JSON, validated)
  "workflow": {
    "autonomy_mode": "guided",                 // supervised | guided | autonomous
    "max_parallel": 4,
    "promote": "bundle"                        // bundle (default) | apply  (apply needs --apply-to DIR at the CLI)
  },
  "faults": [                                  // optional fault injection (ChaosAgent); empty for the happy path
    { "node": "implement.*", "fault": "migration_edit", "attempts": 99 }
  ],
  "script": [                                  // scripted "human" + operator actions, executed in order when their trigger fires
    { "on": "run.waiting", "do": "approve", "request": {"node": "design"}, "by": ["alice", "bob"], "comment": "reviewed migration plan" },
    { "on": "run.waiting", "do": "answer",  "answers_file": "answers.json", "by": "carol" },
    { "on": "node.succeeded:implement.core", "do": "amend", "text_file": "amendment.md", "by": "carol", "reason": "scope change" },
    { "on": "run.waiting", "do": "approve", "request": {"node": "release-readiness"}, "by": ["alice", "bob"] }
  ],
  "expect": {
    "run_status": "succeeded",                 // succeeded | failed | safe_stopped | waiting_human
    "node_statuses": { "codebase-analysis": "succeeded", "verify": "succeeded" },
    "node_status_patterns": { "implement.*": "succeeded" },
    "events_present": ["node.fallback_activated", "approval.granted", "plan.replanned"],
    "events_absent": ["run.safe_stopped"],
    "decision_kinds": ["fallback", "approval", "assumption"],
    "min_parallel_width": 2,                   // max concurrently RUNNING attempts observed (from audit spans)
    "replan": { "cache_hits_min": 1, "added": 1, "removed": 0 },
    "metrics": { "retry_frequency_max": 0.5, "rollback_count": 0 },
    "lineage": [ { "artifact": "release_patch", "reaches_decision_kind": "clarification" } ],
    "artifacts_present": ["release_patch", "readiness_report", "traceability_matrix"],
    "files_in_patch": ["shortener/storage/migrations/003_add_max_clicks.sql"],
    "blocking_questions": 0
  }
}
```

Rules:
* Everything in `script` is executed through the same public commands as a human would use (`approve`, `answer`, `amend`, `kill`) via a `ScenarioControl` facade that only *enqueues*; approvals made this way are `channel="scripted"`, set `RunState.assurance = "DEMO_SCRIPTED"` and are refused unless `AGENTIC_ALLOW_SCRIPTED=1` (the `demo` command sets it for its own process). They deliberately include a *negative control* where a script attempts a duplicate vote and an `agent:`-identity vote, which must appear in the audit log as `approval.refused`.
* `faults` only apply while `ctx.feedback` is empty (so remediation can repair) and never to mandatory governance nodes.
* All paths are relative to the scenario directory and validated (no traversal); all JSON read with `loads_strict`; all strings screened (SEC-001/006) at ingestion exactly like human input.
* `expect` is evaluated against the audit log, run state and artifacts only (never against console output).

## Script action reference (as implemented by `orchestrator/workflows/scenarios.py`)

* **Triggers** (`on`): `run.waiting` (the run paused for humans), `run.started`, `node.succeeded:<node id or prefix.* pattern>`. Each entry fires **once**, in order of first availability.
* **`approve`**: `request: {node: <id or pattern>}`, `by: [names…]` — the listed identities vote *in order* until the request is satisfied (so a longer list than required is harmless). Names listed on purpose that are invalid (a duplicate, `agent:implementer`) are **negative controls**: they must be refused and audited as `approval.refused`.
* **`answer`**: `answers: [{match: <regex over the question text>, answer: <string | "@default">}]` — the first matching rule answers each open blocking question (`@default` = accept the question's `default_assumption`, recorded as an `ASSUMPTION`); `by` is the human name. Question ids are content-derived so scripts match on text, not ids.
* **`amend`**: `text_file` holds the **complete new request text**; `reason` is recorded on the `SCOPE` decision.
* **`kill`**: operator safe-stop (`by`, `reason`).
* `faults`: `{node: <id or pattern>, fault: transient|secret_leak|migration_edit|path_escape|regression, attempts: N}`; the CLI flag `--fault` adds entries; `--faults-off` removes them.
