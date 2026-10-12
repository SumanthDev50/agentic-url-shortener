# Module reference (developer contract)

Public API of every module. Companion documents (all normative):

* [`engine-semantics.md`](engine-semantics.md) — exact behaviour of the engine (state machine, publish, approvals, re-plan, safe-stop, time model)
* [`sdlc-contract.md`](sdlc-contract.md) — the standard workflow, node by node, with gates and approvals
* [`artifact-schemas.md`](artifact-schemas.md) + `orchestrator/core/schemas.py` — artifact kinds and payload shapes
* `orchestrator/core/types.py`, `errors.py`, `serde.py`, `paths.py`, `fsutil.py`, `sanitize.py`, `state.py` — **frozen, already implemented** (read them first)

Conventions: Python ≥ 3.11 (no 3.12+-only APIs: no `Path.walk`, `glob.translate`, `itertools.batched`,
`typing.override`, PEP 695), `from __future__ import annotations`, full type hints (mypy-clean), stdlib only inside
`orchestrator/`, UTF-8 + LF, POSIX relative paths, no global mutable state, thread-safe where stated, **no secrets in
code/logs/tests/docs** (fake secrets in tests are assembled at runtime from fragments, never literals), hostile input is
rejected at the boundary (`validate_rel_path`, `loads_strict`, bounded sizes), **no `fnmatch` / `PurePath.match` /
`glob` for any allow/deny decision** (use `paths.match_glob`), all file I/O via `fsutil` (atomic writes, LF, link-safe).
Tests live under `tests/orchestrator/<layer>/`.

## Layers (enforced by `tests/orchestrator/test_architecture.py`)

```
L0 core      types errors serde paths fsutil sanitize clock schemas graph audit artifacts decisions lineage state metrics
L1 tools     changeset workspace sandbox toolbox effects codebase openapi            (imports L0)
L2 policies  engine config rules_* scanners risk                                     (imports L0, L1)
L3 agents    base catalog reference agents recipes llm chaos                         (imports L0, L1)   L2 ⟂ L3
L4 engine    config registry gates approvals safestop replan expansion scheduler report   (imports L0, L1; policy via protocol only)
L5 workflows sdlc registry gates_sdlc scenarios summary demo ; orchestrator/cli.py   (imports everything)
```
`orchestrator/*/__init__.py` stay empty (no re-exports → no accidental cycles).

---------------------------------------------------------------------------

## L0 — core (to implement: `clock`, `graph`, `audit`, `artifacts`, `decisions`, `lineage`, `metrics`)

### `clock.py`
* `SystemClock` (`now()` UTC-aware, `monotonic()`, `sleep(s)`).
* `FakeClock(start: datetime | None = None)` — deterministic, thread-safe; `sleep(s)` advances virtual time instantly;
  `advance(s)`; `monotonic()` consistent with `now()`.

### `graph.py` — `WorkflowGraph(nodes=(), *, external_kinds=EXTERNAL_KINDS, max_nodes=200)`
* `nodes -> dict[str, NodeSpec]` (insertion-ordered copy), `node(id)`, `has(id)`, `__contains__`, `__len__`, `add_node`, `remove_node`, `replace_node`
* `deps(id)` = `spec.depends_on` ∪ rewired `add_depends_on`; `overlay_of(id)` = `spec.overlay_from` ∪ rewired; `dependents(id)`, `ancestors(id)`, `descendants(id) -> set[str]`
* `topological_order()` (deterministic: Kahn, ties by insertion order), `topo_index(id)`, `layers()`
* `producers_of(kind)`, `consumers_of(kind)` (consumes ∪ optional ∪ overlay ∪ skip_inputs), `critical_path(durations)`
* `resolve_pattern(p) -> list[str]` — exact id or `prefix.*` pattern against the live graph
* `validate()` raises `GraphError` with the **cycle path** in the message; checks every invariant in `sdlc-contract.md §Graph invariants`
* `apply_expansion(PlanExpansion) -> ExpansionDiff(added, removed, retained, changed, rewired)` — structural validation only (the engine has already normalised governance attributes, `engine/expansion.py`): unique ids not clashing with non-owner nodes; **every new node depends on the owner**; mandatory nodes may not be removed/altered; `rewire` only targets existing non-owner nodes and only *adds* edges/overlays; `max_nodes`; result acyclic and contract-valid. **Atomic**: validated on a copy, `PlanPatchError` leaves the graph untouched. Replaces the owner's previous expansion (diff by id + `spec_hash`).
* `remove_expansion(owner)`, `expansion_of(owner)`, `expansions()`
* `to_dict()/from_dict()` (round-trip incl. expansions), `graph_hash()`, `mandatory_snapshot() -> dict[id, spec_hash]`, `to_mermaid(statuses=None)` (generated ids `n0…`, escaped labels via `sanitize.mermaid_label`, never `click`), `to_dot()`

### `audit.py`
* `GENESIS_HASH = "0"*64`; `GOVERNANCE_EVENTS: frozenset[EventType]` (fsynced).
* `Redactor(extra_patterns=())` — `redact_text(str)`, `redact(obj)` (deep). Keys normalised (lowercase, strip `-_ `) and substring-matched against `{password, passwd, pwd, secret, token, apikey, authorization, auth, credential, cookie, session, privatekey, signature, connectionstring, dsn}`; value patterns: AWS key ids, GitHub (`ghp_`, `github_pat_`), Slack, JWT, Google `AIza`, Stripe, `sk-`/`sk-ant-`, URL userinfo, Azure `AccountKey=`/`sig=`, PEM blocks incl. truncated (redact to end), `Bearer …`. **All patterns linear-time** (no nested quantifiers) — fuzz-tested under a time bound; strings capped at 8 KiB then `…[truncated n bytes sha256=…]`.
* `AuditLog(path, *, run_id, trace_id=None, clock, redactor=None, mode="strict")` — thread-safe append-only JSONL, canonical ASCII. Opening an existing file **verifies the chain then continues it** (`mode="recover"` repairs a torn final line, emitting `AUDIT_TAIL_REPAIRED`; a mid-chain mismatch raises `IntegrityError` — `open_new_segment(broken_at, last_good_hash)` starts `audit.1.jsonl`).
  * `emit(type, *, node_id=None, attempt=None, actor=Actor.system(), data=None) -> AuditEvent`; **also callable** with the same keyword signature (`AuditLog` satisfies `types.AuditSink`). Span ids are derived from `(node_id, attempt)` (`"run"`, `"n:<id>"`, `"n:<id>#<n>"`); `ts`/`seq` (starting at 1) assigned inside the lock; `data` redacted before hashing; governance events fsynced.
  * `events() -> list[AuditEvent]`, `head() -> tuple[int, str]`, `verify() -> VerifyResult(ok, count, broken_at, reason)`, `add_listener(cb)` (callbacks are dispatched from a **separate queue after** the append, never under the audit lock), `@staticmethod verify_file(path, anchor=None)`, `normalize_events(events)` (drops ts/hash/span ids; orders `tool.call` per attempt — for golden tests).
* `export_trace(events) -> dict` — run → node → attempt span tree with start/end/duration/status.

### `artifacts.py` — `ArtifactStore(root, *, clock, audit: AuditSink | None = None)`
Thread-safe; index `root/index.json` (atomic); storage path `root/<kind>/<slug(key)>-<sha256(key)[:8]>/v<N>/content.<ext>` with `ext` from `{txt,md,json,py,sql,patch,diff}` — **raw keys and agent filenames never become path components**; every join asserts containment; on index load any absolute / `..` / drive path → `IntegrityError`.
* `put(draft, *, producer, attempt=0, agent_id="", inputs=(), staged=False) -> Artifact` — str/bytes verbatim, Mapping/Sequence → canonical pretty JSON (`.json`); version monotonic per `(kind,key)`; staged ⇒ status `STAGED`; reserved metadata keys (`_*`) stripped; emits `ARTIFACT_CREATED{artifact_id, content_hash, size}`.
* `activate(ids) -> list[Artifact]` — all-or-nothing under one index write, **idempotent**; supersedes the previous ACTIVE of each `(kind,key)`; **dedupe**: if a staged artifact has the same `content_hash` as the current ACTIVE of its `(kind,key)`, mark the staged one SUPERSEDED, emit `ARTIFACT_UNCHANGED`, keep the existing ref; refuses a `(kind,key)` whose ACTIVE artifact has a different producer.
* `quarantine(ids, reason, *, tombstone=False)` (tombstone replaces bytes, keeps hash/size/metadata), `supersede(id, reason)`
* `get(id)`, `latest(kind, key="")` (ACTIVE only), `active(kind)` (one per key, sorted by key), `versions(kind, key="")`, `all(include_inactive=False)`, `staged()`
* `read_bytes/read_text/read_json(artifact)` — verify `content_hash` (→ `IntegrityError`), `loads_strict` for JSON
* `verify_all() -> list[str]`, `verify_against_events(events) -> list[str]` (index ↔ `ARTIFACT_CREATED`)
* `snapshot(kinds: Iterable[str], accept_producer: Callable[[str], bool]) -> PinnedReader` — immutable tuple of refs captured now (ACTIVE only, producer accepted, absent kinds recorded as absent); `PinnedReader` implements `ArtifactReader`, `refs() -> tuple[ArtifactRef, ...]`, raises `UndeclaredInputError` for kinds outside `kinds`; `PinnedReader.with_extra(artifacts)` adds the attempt's STAGED artifacts for exit gates.
* `reader() -> ArtifactReader` (unscoped, live)

### `decisions.py` — `DecisionLog(path, *, run_id, clock, audit: AuditSink | None = None)`
JSONL, thread-safe, ids `D-0001…`. `record(kind, summary, *, actor, node_id=None, rationale="", inputs=(), outputs=(), alternatives=(), outcome="", parents=(), refs=(), data=None) -> Decision` (emits `DECISION_RECORDED{decision_id, kind, decision_hash}`); `all()`, `get(id)`, `for_node(node_id)`, `of_kind(kind)`, `decisions_for(ref)` (decisions whose `outputs` contain `ref`); loading verifies hashes against nothing (the audit event carries the hash).

### `lineage.py` — `LineageGraph(store, decisions)`
`upstream(artifact_id) -> LineageNode` (artifact → its `inputs`, the decisions listing it in `outputs`, recursively through `parents`), `downstream(artifact_id) -> list[Artifact]`, `explain(artifact_id) -> str` (indented, sanitised, human readable), `to_dict(artifact_id)`, `reaches(artifact_id, decision_kind) -> bool`.

### `metrics.py`
Pure functions of audit events with the **definitions of engine-semantics §17.8**: `compute_run_metrics(events) -> RunMetrics` (success, `e2e_latency_s`, `active_latency_s`, `human_wait_s`, attempts, retries, retry_frequency, fallbacks, fallback_rate, rollbacks (compensations), safe_stops, replans, `replan_reuse_ratio`, `first_pass_yield`, `incidents[Incident(node_id, failed_at, recovered_at, mttr_s, status: recovered|unrecovered|abandoned)]`, `mttr_s`, `stage_latency_s`, `node_latency_s`, `critical_path_s`, `max_concurrency_observed`, approvals requested/auto, `policy_violations_by_rule`); `aggregate(runs) -> AggregateMetrics` (success_rate, retry_rate, fallback_rate, rollback_frequency, safe_stop_rate, mttr_mean_s/p95, e2e p50/p95, …); `load_all(runs_dir)`; `render_table(...) -> str`.

---------------------------------------------------------------------------

## L1 — tools

### `changeset.py` and `workspace.py` — **already implemented** (extend + test only)
`ChangeSetConflict`, `DiffStats`, `Conflict`, `MergeResult`, `make_create/make_modify/make_delete`, `compute_ops(view, desired, …)`, `apply_changeset(view, cs, strict=True)`, `merge_changesets(base, [cs…])`, `reverse_changeset(view_before, cs)`, `to_unified_diff(view, cs)` (verified with `git apply`), `diff_stats`, `digest`; `MemoryView`, `DirectoryView`, `StagedView`, `OverlayView`, `BaselineSnapshot` (`create/empty/load/verify`, `manifest`, `manifest_hash`, `is_empty`), `materialize`, `DEFAULT_EXCLUDES`.

### `sandbox.py` — `SandboxRunner(scratch_root)`
`run(argv, *, cwd, timeout_s, cancel: CancelToken | None = None, max_output_bytes=262144) -> CommandResult`. Env is an **allow-list** built from scratch (`PATH` minimal, `SYSTEMROOT`/`WINDIR`/`COMSPEC` on win32, `HOME`/`USERPROFILE`/`TEMP`/`TMP`/`TMPDIR` → scratch, `PYTHONPATH=<cwd>`, `PYTHONUTF8=1`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOD=…` unset-safe `PYTEST_ADDOPTS=""`); no inherited secrets. `Popen` with `start_new_session=True` (POSIX) / `CREATE_NEW_PROCESS_GROUP` (win32); **tree kill** on timeout/cancel (`os.killpg` / `taskkill /PID <pid> /T /F`); output read by capped reader threads (bounded while reading, then redacted + control-stripped, `truncated=True`). `pytest_argv(scratch, targets)` builds the **fixed** argv `[sys.executable,'-m','pytest','-p','no:cacheprovider','-q','--basetemp',<scratch>/.tmp,'--junitxml',<scratch>/junit.xml,'--',*targets]`; each target must match `^[A-Za-z0-9_./-]+$`, not start with `-`/`@`, and resolve inside scratch; `python_argv(args)` accepts only `['-m', <allow-listed module>, …]` or a script path inside scratch, **never `-c`**. Registers `kill_tree` on the cancel token. Documented as process-level isolation, not a security sandbox (no network/memory limits).

### `toolbox.py` — `ScopedToolbox` implementing `types.Toolbox`; `make_toolbox_factory(...) -> ToolboxFactory`
Per-attempt instance built by the engine via `ToolboxFactory` (signature in `types.py`); additionally takes `materialise_dir` / `SandboxRunner`. Reads via `StagedView(view, staged_writes)`; writes staged; **every method**: checks the cancel token, the call budget, the profile (`read_globs`, `write_globs`, `can_execute`, `max_file_bytes`, `max_total_bytes`, `max_changed_files`), validates paths with `validate_rel_path` (→ `PathViolation`, CRITICAL), evaluates an `ActionSubject` through the policy (SEC-005), audits `TOOL_CALL` / `TOOL_DENIED`. A valid path outside `write_globs` raises `ToolDenied`; ≥ 3 denials in one attempt escalate to `AutonomyViolation`. `search` is literal by default (regex: ≤ 200 chars, restricted grammar, bounded work). `run_tests`/`run_python` materialise `view + staged writes` into a fresh scratch dir and go through `SandboxRunner`. After cancel every method raises `AgentCancelled` and drops audit events. `usage()`; `staged_changeset(description)` builds ops with `compute_ops` against the attempt's start view (first-touch `base_hash` rule).

### `effects.py` — `DirectoryEffectPort(target_root, *, protected_globs, audit)` implementing `types.EffectPort`
`apply(cs, dry_run=False) -> ApplyReceipt` (per-op validation + protected-path check + `base_hash` verification against the **target tree**; all-or-nothing with journal and temp+replace; `EFFECT_STARTED/COMPLETED` audited), `revert(receipt)` (only if current hashes equal `post_hashes`, idempotent, else raises). Target root comes from `RunState.apply_root` recorded at start; refuses roots that contain `.git` unless the working tree is clean **and** on a non-default branch (checked with `git` via the sandbox runner, never a network command).

### `codebase.py`
`analyze_codebase(view: FileView) -> CodebaseModel` (AST only, iterative walks, size caps, try/except around `ast.parse` → "unparseable" entry): modules, internal import graph, defs, **FastAPI routes** (method, path, handler, module, line, **auth dependency names**), **SQL tables/columns from migrations** (`CREATE TABLE`, `ALTER TABLE … ADD COLUMN`, in file order), test↔module map, and **data flows** `route → handler → service calls → repository calls → tables/columns` (name-resolution heuristics; unresolved dynamic calls listed in `unresolved` and documented). `impact_analysis(model, keywords, seed_modules=(), depth=2) -> ImpactReport` (dataclass with `to_dict()` matching `Kind.IMPACT_REPORT`): seeds by symbol/keyword match, dependents by reverse-import closure, impacted routes/tables/tests, `coupling_risks` (e.g. caches/async writers around the touched flow), `risk_signals`, `expected_paths`.

### `openapi.py`
`diff_openapi(old, new) -> OpenApiDiff(breaking, additive)` (removed path/method/required request field/response code, type change = breaking; new optional field/path = additive); `export_openapi(runner, cwd, app_import) -> dict` via the sandbox (`python -m` helper script under `orchestrator/tools/_openapi_dump.py`, never `-c`).

---------------------------------------------------------------------------

## L2 — policies (`orchestrator/policies/`)

* `config.py` — `PolicyConfig(severity_overrides, public_routes={GET /{code}, /healthz, /readyz}, auth_dependency_names, migrations_glob, change_budget(max_files=25, max_lines=800), protected_paths, egress_modules, expansion_rules)`; `FLOORS` table (SEC-*, CHG-001/006/007/008, QLT-001 ≥ HIGH; CRITICAL immutable): overrides are **raise-only** (`ConfigError` otherwise); `config_hash()`.
* `engine.py` — `PolicyEngine(rules=None, config=None, waivers=())` implementing `types.PolicyEvaluator`: dispatches only to rules whose `subjects` match; a crashing rule becomes a HIGH `POL-000` finding (fail closed); post-processes **every finding** (evidence masked + capped at ~80 chars); fills `PolicyReport.from_findings(...)` from the floor-protected mapping; `with_waivers(ws)` (MEDIUM-or-lower only, bound to `finding.key()` and the subject's evidence hash); `assess_risk(subject)`; asserts at construction that rule ids == the catalogue manifest and every `SubjectKind` has ≥ 1 rule; `manifest_hash`.
* `scanners.py` — `scan_secrets(text, location)`, `scan_python(text, path)` (iterative `ast.walk`; `ast.parse` guarded for `SyntaxError/RecursionError/MemoryError/ValueError` → "unparseable" finding), `scan_trojan_source(text, path)`; **hard input caps** (1 MiB / 2000-char lines → HIGH "unscannable", never a pass), linear-time regexes, adversarial-input fuzz tests (< 0.5 s).
* `risk.py` — `assess_changeset_risk(subject) -> RiskAssessment` (migration ⇒ HIGH; auth/security module ⇒ HIGH; deletions ⇒ HIGH; new public route / > N files ⇒ MEDIUM; docs/tests only ⇒ LOW).
* `rules_security.py` (SEC-001…008), `rules_compliance.py` (CMP-001…003), `rules_change.py` (CHG-001…008), `rules_quality.py` (QLT-001); `default_rules()`. Catalogue in `architecture.md §1.5`; every rule **fails closed on missing evidence**. Adversarial-corpus tests record known bypasses as `xfail(strict=True)` with a reason — SEC-002/003/004/006 are *heuristic tripwires, not a security boundary* (the boundary is structural: proposals only, autonomy checks, sandboxed execution, human approval).

---------------------------------------------------------------------------

## L3 — agents (`orchestrator/agents/`)

`base.py` (`BaseAgent`, `json_artifact`, `text_artifact`, ids/question-id helpers, `Recipe` protocol with typed `ParamSpec`), `catalog.py` (`Capability`, `CAPABILITIES`, `match_capabilities`; unmatched ⇒ `unsupported`), reference agents (ids in `sdlc-contract.md`), `recipes/` (`linkcheck`, `max_clicks`, `analytics_export`: typed params validated before running — ints within ranges, enums, identifiers `^[a-z_][a-z0-9_]{0,31}$`, **no free-form strings**; templates are static text with typed holes, values rendered via `repr`/`json.dumps`, never f-stringed into SQL/paths; payload files stored as `*.py.tmpl`), `llm.py` (`LLMProvider`, `AnthropicProvider` using the official `anthropic` SDK lazily imported (optional extra `llm`), fixed `base_url`, `ScriptedProvider`, `LLMAgent` with strict allow-listed JSON parsing via `loads_strict`, egress opt-in + denylist + outbound secret scan, token/call budgets, `can_execute=False`), `chaos.py` (`ChaosAgent`, `FAULTS`; faults fire only while `ctx.feedback` is empty; fake secrets assembled at runtime).

## L4 — engine (`orchestrator/engine/`)
`config.py` (`EngineConfig`, `HARD_LIMITS`, `posture()`), `registry.py` (`AgentRegistry.create(agent_id)`, `GateRegistry`, `SkipRegistry`, `CompensatorRegistry`, `Registries`), `gates.py` (generic gates), `approvals.py` (`ApprovalManager`, `normalize_approver`, providers `QueueApprover`/`InteractiveApprover`/`ScriptedApprover`, all non-blocking), `safestop.py` (`KillSwitch`, `CircuitBreaker`, `SafeStopController`), `replan.py` (pure: `invalidate`, `compute_fingerprint`), `expansion.py` (template normalisation, CHG-006/007), `scheduler.py` (`Orchestrator` + `RunSnapshot`), `report.py` (run report, metrics.json, trace.json, seal.json). Signatures: `ApprovalManager.request(...)`, `.vote(request_id, approver, approved, comment, *, channel, os_user)`, `.invalidate(node_id, reason)`, `.is_satisfied(id)`; `replan.invalidate(graph, state, changed) -> list[Transition]`; `compute_fingerprint(node, graph, pinned, agent_version, ...) -> str`.
`Orchestrator(runs_dir, registries, *, config=None, clock=None, approver=None, policy=None, toolbox_factory=None, finalizers=(), listeners=())`: `start(graph, *, scenario, request_text, baseline_source=None, include=(), exclude=(), params=None, run_id=None, created_by, apply_root=None) -> str`, `run(run_id) -> RunResult`, `resume(run_id, *, override=None) -> RunResult`, `approve/reject(run_id, request_id, approver, comment="", *, confirm="")`, `answer(run_id, answers, by, *, accept_default=())`, `amend(run_id, new_request_text, by, reason)`, `kill(run_id, by, reason)`, `status(run_id)`, `lineage(run_id, artifact_id)`, `verify_audit(run_id, anchor=None)`, `events(run_id)`, `metrics(run_id)`. Listeners get each `AuditEvent` **after** the audit append, from a separate queue; scenario scripts act through a `ScenarioControl` facade whose methods only enqueue commands.

## L5 — workflows and CLI
`workflows/sdlc.py` (`build_sdlc_graph(options)`), `registry.py` (`build_registries(options)`), `gates_sdlc.py`, `scenarios.py` (loader/validator for `orchestrator/scenarios/<name>/scenario.json`: `request_file`, `baseline{include,exclude|empty}`, `params`, `faults[]`, `script[{on,do,by,payload}]`, `expect{…}`), `summary.py` (engineering-summary finalizer), `demo.py`; `orchestrator/cli.py` (`agentic …`; exit codes 0 succeeded, 10 waiting for human, 20 failed, 30 safe-stopped, 2 usage error).

## shortener (product)
See `docs/product-spec.md` and `docs/api.md` (generated from the OpenAPI schema). Layering: `api → services → domain/storage`; `api` never touches SQL; `domain` performs no I/O.
