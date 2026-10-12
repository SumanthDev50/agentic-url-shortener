# Engine semantics (normative)

This document is the precise behavioural specification of the orchestration engine
(`orchestrator/engine/`). `architecture.md` explains *what and why*; this explains
*exactly how*. Every rule here is either enforced by code, covered by a test, or listed
as a limitation in `risks-and-tradeoffs.md`. Types referenced live in
`orchestrator/core/types.py`, `state.py`, `schemas.py`.

Vocabulary: **node** (vertex of the DAG), **attempt** (one agent execution; globally
numbered per node), **generation** (per-node counter bumped on every dispatch and cancel),
**epoch_no** (per-node counter bumped on every reset; per-epoch retry budgets restart),
**staged** (produced, not yet visible downstream), **published/ACTIVE** (visible).

---------------------------------------------------------------------------

## 1. Layers, threads, ownership

```
L0 core      types errors serde paths fsutil sanitize clock schemas graph audit artifacts decisions lineage state metrics
L1 tools     changeset workspace(views, snapshot, materialize) sandbox toolbox codebase openapi effects
L2 policies  engine rules scanners risk            (imports L0, L1)
L3 agents    reference agents, recipes, llm, chaos (imports L0, L1)      -- L2 and L3 never import each other
L4 engine    config registry gates approvals safestop replan scheduler report   (imports L0, L1; policy only via the PolicyEvaluator protocol)
L5 workflows sdlc registry scenarios summary demo  + orchestrator/cli.py  (wires everything)
```
An architecture test AST-walks every import against this map.

**Threads.** One *scheduler* thread is the only writer of `RunState`, node statuses, the
publish step and graph mutation. Each attempt runs on a worker thread whose job is, in order:
build the context from an **immutable snapshot** → `agent.run` → validate and **stage**
drafts → run exit gates and policy against the staged artifacts → return one `AttemptOutcome`.
Gates and policy (which scan hostile content) therefore never run on the scheduler thread.
The scheduler only *applies transitions* (publish, approval, retry, fallback, replan).

**Commands** (`approve`, `reject`, `answer`, `amend`, `kill`, `resume --override`) never mutate
state from foreign threads or processes. If a live engine holds `run.lock`, the command is
written to `runs/<id>/inbox/<ts>-<n>-<kind>.json` (`COMMAND_QUEUED`) and drained by the
scheduler at the top of a tick (`COMMAND_APPLIED`); otherwise the caller takes the lock,
applies it through the same code path, and releases. A **terminal run is immutable**:
`approve/answer/amend` raise `StateError`; `SAFE_STOPPED` requires `resume` with an override.

Agent instances are created fresh per attempt (`AgentRegistry.create(agent_id)`); agents must
be stateless. Agent code is trusted code; only model output and requirement text are untrusted.

---------------------------------------------------------------------------

## 2. Scheduler tick

```
loop:
 1 heartbeat (lock mtime; persist last_heartbeat ≥ every 5 s); accrue active runtime (excl. human wait)
 2 safety checks → safe-stop procedure (§14): kill switch, budgets, circuit breaker, zombie cap
 3 drain inbox commands in name order
 4 drain completed AttemptOutcomes (apply each; discard stale generations)
 5 poll approval providers (non-blocking) and apply new votes
 6 deadlines: RUNNING attempts past min(node.timeout_s, profile.max_runtime_s) (measured from
   the worker's "started" message, not from submit) → cancel token, generation++ ,
   outcome FAILED(TIMEOUT); the thread becomes a zombie until its future completes
 7 promote: PENDING → READY | SKIPPED | BLOCKED ; STALE → revalidate (§4.4)
 8 dispatch READY nodes (topological order) while running < max_parallel, zombies < cap,
   not_before <= now: entry gates (fail fast) → BEFORE approval → submit to worker
 9 compute run status (§3.3); terminal → finalize; only humans pending → persist, return WAITING_HUMAN
10 save state (atomic) when dirty
11 wait on the completion queue with timeout=poll_interval_s  (REAL time, never Clock.sleep)
   — except when nothing is running and only back-off timers remain: clock.sleep(min remaining)
```

Readiness is **derived each tick**, never persisted: `READY` is stored only so the audit/status
can show it; dependencies are re-checked at dispatch time.

---------------------------------------------------------------------------

## 3. Node lifecycle

### 3.1 Transition table
`types.ALLOWED_TRANSITIONS` is normative; `Scheduler.transition(node_id, new, reason)` is the
**only** function that changes `NodeState.status` and always calls `assert_transition`.
`replan.invalidate` and `SafeStopController` are *pure*: they return lists of
`Transition(node_id, to, reason)` intents. `NodeStatus.is_settled` ≠ final.

### 3.2 Promotion rules (PENDING)
* all deps `satisfies_dependency` (`SUCCEEDED`/`SKIPPED`) → evaluate `skip_if` (it may read only
  `node.skip_inputs`) → `SKIPPED` (reason recorded) or `READY`;
* any dep in {`FAILED`,`BLOCKED`,`ROLLED_BACK`,`CANCELLED`} → `BLOCKED` (reason names the dep);
* a `SKIPPED` node returns to `PENDING` whenever any of its `skip_inputs` changes (re-evaluate).

### 3.3 Run status
* `RUNNING` if any node is RUNNING, dispatchable/READY, in back-off, or promotable;
* else `WAITING_HUMAN` if any node is AWAITING_APPROVAL / AWAITING_INPUT (opens a human-wait interval);
* else `SUCCEEDED` iff every non-REMOVED node is SUCCEEDED/SKIPPED **and** every id in
  `mandatory_snapshot` is SUCCEEDED (a mandatory node may not end SKIPPED/REMOVED) **and** the
  invariant check (§19) passes;
* else `FAILED` with a reason listing failed/blocked nodes and any PENDING node whose deps can
  never be satisfied (deadlock).
`SAFE_STOPPING`/`SAFE_STOPPED` are entered only by §14.

---------------------------------------------------------------------------

## 4. Dispatch, snapshots, fingerprints, cache hits

### 4.1 Input snapshot (pinned, no torn reads)
At dispatch the engine captures an immutable `PinnedReader`: for every kind in
`node.input_kinds`, the ACTIVE artifacts whose **producer is a strict ancestor or `external`**
(a node never sees its siblings or its own previous generation). Absent optional kinds are
recorded as absent. The snapshot is the single source of truth for: the agent's reader, the
fingerprint, `Artifact.inputs` of every staged draft, `Decision.inputs`, and the approval
evidence. Reading an undeclared kind raises `UndeclaredInputError`.

### 4.2 Fingerprint
```
fingerprint = hash_jsonable({
  spec:     node.spec_hash(),
  agent:    [agent_id_for_epoch, agent.version],
  epoch:    "primary" | "fallback",
  deps:     sorted(graph.deps(node.id)),              # includes rewires
  inputs:   sorted([kind, key, content_hash]) per visible artifact, or [kind, "", null] if absent,
  baseline: state.baseline_manifest_hash,
  policy:   policy_config_hash,
  feedback: hash of remediation-feedback finding keys (empty unless remediated),
  engine:   ENGINE_SCHEMA_VERSION })
```
**Determinism rule:** artifact *content* must not contain times, durations, pids, absolute
paths or random ids (those go in `metadata`, excluded from `content_hash` and the fingerprint).
Test: run every reference agent twice under different `FakeClock` times → identical hashes.

### 4.3 Attempt numbering
`NodeState.attempt_counter` is a per-node monotonic counter, never reset: it is the *attempt
number* in `AuditEvent.attempt`, `Artifact.attempt` and span ids (`n:<node>#<attempt>`).
`generation` increments on every dispatch **and** every cancel/discard. `ctx.attempt` (epoch-local)
is display-only.

### 4.4 Cache hit (revalidation of STALE nodes)
When all deps of a `STALE` node are satisfied the engine recomputes the fingerprint.
If `node.cacheable and not node.side_effecting` and it equals `last_success_fingerprint` and the
published outputs are still ACTIVE and hash-verified, the outputs are **restored — but verdicts
are never restored**: entry gates, exit gates, policy evaluation and approval validity are
**re-run over the restored artifacts** (a changed policy config or waiver set cannot be bypassed;
nodes that read `RunView` — `release-readiness` — are `cacheable=False`). If any re-check fails the
node is treated as a miss (`READY`, `epoch_no+1`). A success emits `NODE_CACHE_HIT` then
`NODE_SUCCEEDED(cache_hit=true)`. A `side_effecting` node is **never** cache-restored: if it had
SUCCEEDED it is first compensated (→ ROLLED_BACK → PENDING) and then re-run; if no compensator
exists the triggering amend is refused with `StateError`.

---------------------------------------------------------------------------

## 5. Attempt execution (worker thread)

1. Build `AgentContext` from the snapshot (`attempt_no`, `generation`, narrowed autonomy
   profile = `agent.autonomy ∩ node.autonomy`, feedback = engine-authored findings:
   `rule_id`, severity, location, remediation hint only — **never** `evidence` or agent prose).
2. `agent.run(ctx)`; exceptions are mapped by `FailureKind.parse(err.kind)` (unknown → FATAL;
   any other exception type → FATAL with a redacted message).
3. Validate drafts: `kind ∈ node.produces`; key forced (`node.id` for `KEYED_KINDS`, else `""`);
   `filename` kept in metadata only; reserved metadata keys stripped; JSON payloads validated with
   `schemas.validate_payload`; sizes ≤ `max_artifact_bytes`; `author` of change sets overwritten with
   the real agent id. Stage via `ArtifactStore.put(staged=True, inputs=snapshot.refs())`.
4. **Change set admission** (kinds in `CHANGESET_KINDS`): `ChangeSet.from_dict` (validates paths);
   `apply_changeset(view_at_attempt_start, cs, strict=True)`; if the node used the Toolbox for writes
   the ops must equal `toolbox.staged_changeset().ops`; `ChangeSetSubject(profile=narrowed profile,
   sources=input change sets)` is evaluated (SEC-005 checks *every* op path against `write_globs` and
   the op count; CHG-007/008 for sources/protected paths). Non-changeset artifacts are evaluated as
   `ArtifactSubject` (never both).
5. `PlanExpansion` (if any): `owner == ctx.node.id`; normalised against the owner's
   `params['expansion_template']` / `params['expansion_policy']` (§12.3); staged as a
   `plan_expansion` artifact; `PlanSubject` evaluated (CHG-006/007).
6. Exit gates (all evaluated; §8) with a `GateContext` whose `workspace_view()` = `view_for(node)` +
   the attempt's staged change sets applied last.
7. `assess_attempt_risk` (§10.2). Return `AttemptOutcome`.

Any exception from a gate or a policy rule is **fail-closed**: it becomes a GATE_REJECTED outcome
with a `GATE-000`/`POL-000` HIGH finding — never a scheduler crash.

---------------------------------------------------------------------------

## 6. Workspace view semantics

`view_for(node) = baseline + [ACTIVE artifacts a : a.kind ∈ node.overlay_from ∧ a.producer ∈
strict_ancestors(node)]`, ordered by `(topo_index(producer), key)`, applied with
`apply_changeset(strict=True)`. A conflict raises `ChangeSetConflict` → GATE_REJECTED.

* Defaults: `implement.*`, `author-tests`, `integrate`: `overlay_from=()` (baseline only; `integrate`
  merges consumed change sets itself, output relative to baseline). `verify`, `security-scan`, `docs`:
  `('integrated_changeset',)`. `promote`/`release-verify`: `('integrated_changeset','docs_changeset')`.
* The Toolbox sees `view_for(node)` + this attempt's staged writes. Siblings never see each other.
* Overlay source artifacts are part of the input snapshot, fingerprint and `Artifact.inputs`.
* `FileOp.base_hash` = `fsutil.sha256_text` of the path's content in the producing attempt's view at the
  **first touch** of that path in that attempt (`None` for CREATE). Create-then-modify stays CREATE;
  create-then-delete is dropped. `docs_changeset` is relative to baseline+integrated.
* **Serialising re-plan**: when `integrate` reports a conflict (two change sets touch one path with
  different results) it owns a `PlanExpansion(nodes=(), rewire={'implement.b': Rewire(
  add_depends_on=('implement.a',), add_overlay_from=('changeset',))})`; b's fingerprint then changes
  (new dependency + overlay input), so b re-runs seeing only a's change set.

---------------------------------------------------------------------------

## 7. Publish protocol (one scheduler tick)

For an outcome that passed all gates and has all required approvals:

* **P1** verify: `generation == node.generation`, node status is RUNNING/AWAITING_APPROVAL, run not
  safe-stopping, approvals satisfied, approval `node_fingerprint` equals the current fingerprint and
  `evidence_hash` equals the hash of the staged content + policy report + risk (TOCTOU guard).
* **P2** audit `NODE_PUBLISHING{staged, expansion_hash}` (fsync) **before any mutation**.
* **P3** `ArtifactStore.activate(ids)` — all-or-nothing under one index write, idempotent. If a staged
  artifact has the same `content_hash` as the current ACTIVE artifact of its `(kind,key)`, the staged
  one is marked SUPERSEDED (`ARTIFACT_UNCHANGED`) and the existing ACTIVE ref is kept (no new version,
  no staleness). Activation refuses a `(kind,key)` whose ACTIVE artifact has a different producer.
* **P4** apply the node's `plan_expansion` artifact (graph mutation, §12.3), supersede the artifacts of
  removed nodes.
* **P5** record the agent's `DecisionDraft`s (§17.3), update `NodeState` (`outputs`, fingerprints,
  `SUCCEEDED`), `state.save` (with `audit_head`).
* **P6** audit `NODE_SUCCEEDED`; then staleness propagation for every `(kind,key)` whose ACTIVE hash
  changed or whose key set changed (a disappearing key counts).

**Crash recovery (`reconcile`)** on `resume`: for every node whose last journal event is
`NODE_PUBLISHING` without `NODE_SUCCEEDED`: if all staged ids are ACTIVE → roll forward (re-apply the
expansion, set SUCCEEDED); else quarantine them and treat the node as RUNNING-at-crash. Every STAGED
artifact not referenced by `NodeState.pending_attempt` is quarantined. For every SUCCEEDED owner assert
`graph.expansion_of(owner)` equals its ACTIVE `plan_expansion` artifact, else re-apply.

---------------------------------------------------------------------------

## 8. Gates

* Entry gates run at dispatch; exit gates in the worker. **All gates are evaluated** (no short-circuit
  except `SAFE_STOP`); precedence `SAFE_STOP > FAIL > AWAIT_INPUT > REQUIRE_APPROVAL > PASS`; findings are unioned.
* An **entry-gate FAIL is not retried** (identical inputs would fail identically): the node goes to
  `FAILED` (kind GATE_REJECTED, phase `entry`), or back to `PENDING` if the cause is "upstream approval
  no longer valid".
* The engine always adds the implicit exit gates `artifacts_present` (all `produces` kinds exist and
  validate) and `policy` (always-on policy evaluation, §5.4) — they cannot be omitted by a node spec.
* `GateResult.questions` and `AgentResult.clarifications` are merged by question id into one open set.
* Positive evidence is audited: `GATE_EVALUATED{gate, phase, decision, rule_ids}` and
  `POLICY_EVALUATED{subject, subject_hash, rules_evaluated, findings}` for every evaluation.

---------------------------------------------------------------------------

## 9. Failure routing and accounting

| Outcome | Node budget | Run budget | Breaker | Next |
|---|---|---|---|---|
| success | – | – | resets agent | publish |
| TRANSIENT / TIMEOUT | counts | counts | counts | retry → fallback → FAILED |
| GATE_REJECTED / POLICY_VIOLATION | counts | counts | **no** | retry with feedback (if node has `remediation`: route back first, §9.2) |
| APPROVAL_REJECTED (human) | counts | counts | no | retry with the human comment as feedback; rejection of a BEFORE approval fails the node |
| FATAL | counts | counts | counts | skip remaining primary attempts → fallback → FAILED |
| AUTONOMY_VIOLATION / CRITICAL finding | – | – | – | **safe-stop** |
| CANCELLED / DISCARDED | no | no | no | – |
| NEEDS_INPUT (clarification) | no | no | no | AWAITING_INPUT (not a failure; interval counts as human wait) |

* Retry iff `err.retryable ∧ kind ∈ retry_on ∧ epoch attempts < max_attempts`; otherwise fallback (if
  `fallback_agent` and the kind is not AUTONOMY_VIOLATION), otherwise `FAILED`. The fallback has its own
  `fallback_retry` budget. Back-off is data: `not_before = clock.now() + delay_for(n)`, node `READY`.
* **Budgets.** `max_total_attempts` counts only *retries and fallbacks* (not first attempts of an
  epoch) and is validated at start to be ≥ 2 × node count. `max_attempts_per_node` is a lifetime cap
  (default 3 × (retry + fallback)) so per-epoch resets cannot amplify. `max_replans` counts
  agent/failure-initiated waves only; human inputs are bounded by `max_human_inputs` and are *refused*
  (not safe-stopped) beyond it.
* **Circuit breaker** is per `agent_id`: counts TRANSIENT/TIMEOUT/FATAL consecutive failures; resets on
  that agent's success or on override; when OPEN the node is routed to its fallback before any run-level
  safe-stop.
* Unknown `AgentError.kind` → FATAL. Any exception escaping a scheduler tick persists state and
  safe-stops with trigger `engine_fault`.

### 9.2 Remediation (the feedback edge)
If a node with `remediation` has an exit gate reject with GATE_REJECTED/POLICY_VIOLATION and
`remediation_loops < max_loops` (and `< EngineConfig.max_remediation_loops`): the engine records a
`REMEDIATION` decision (citing finding keys), emits `REMEDIATION_ROUTED`, resets the nodes in
`route_to` (patterns resolved against the live graph) to `PENDING` with `epoch_no+1` and
`feedback=findings` (engine-authored), and sets the failing node `PENDING` (its staged output is
quarantined; the attempt is not counted as a retry). The route-to nodes re-run (their fingerprint changes
through `feedback`), then the node re-runs. Exhausted loops → normal retry/fallback/FAILED accounting.
Faults (ChaosAgent) fire only while `ctx.feedback` is empty, so the repair loop is demonstrable.

---------------------------------------------------------------------------

## 10. Approvals, risk, waivers, identity

### 10.1 State machine
`PENDING → APPROVED` (≥ `required_approvals` distinct approvers) `| REJECTED` (**any single rejection
vetoes**; further votes refused) `| INVALIDATED | CANCELLED`. An approver cannot change a vote. A vote is
applied only if `node.status == AWAITING_APPROVAL ∧ node.pending_approval_id == request.id ∧
request.node_fingerprint == current fingerprint`. Every refusal is audited as `APPROVAL_REFUSED{reason ∈
self_approval, duplicate, agent_identity, unknown_approver, stale_fingerprint, terminal, insufficient_role}`.

### 10.2 Risk
`assess_attempt_risk(node, outcome, policy) = max(` `node.risk`, every `GateResult.risk` (e.g. the
`design_risk` gate returns HIGH when `data_model_changes` is non-empty), `policy.assess_risk(ChangeSetSubject)`
for change set artifacts, MEDIUM if any REQUIRE_APPROVAL finding `)`. Persisted in `AttemptRecord.risk`,
`ApprovalRequest.risk`, `APPROVAL_REQUESTED.data`.

### 10.3 When approval is needed
`spec.always` **or** `risk ≥ effective_threshold`, where autonomy mode `supervised` treats every
`ApprovalSpec` as `always`, `guided` (default) uses `spec.threshold`, `autonomous` raises non-mandatory
thresholds to HIGH (mandatory/release approvals stay). Otherwise the engine records `APPROVAL_AUTO` and an
`APPROVAL` decision with actor `policy` (never silent). `required_approvals = approvals_by_risk[risk]`
(hard floors LOW 1, MEDIUM 1, **HIGH 2**; the config cannot lower them).

### 10.4 Request content
`ApprovalRequest.summary` is **composed by the engine from structured facts only**: risk reasons, rule
ids/severities/locations, `diff_stats` (files, lines, migrations, routes), artifact refs + content hashes,
fingerprint prefixes, requested_by, assurance. Agent prose appears only in `agent_note`: sanitised,
≤ 500 chars, and rendered in a clearly labelled "agent-authored (untrusted)" block. Human comments are
length-limited and sanitised.

### 10.5 Lazy invalidation
Marking a node `STALE` does not touch its approvals. When its fingerprint is recomputed: equal to the
request's `node_fingerprint` → approval stays valid (cache hit, no re-approval); different →
`INVALIDATED`, a new request is raised on re-run. A pending request of a node that is reset/removed/
cancelled becomes `CANCELLED` and its staged artifacts are quarantined.

### 10.6 Waivers
MEDIUM-or-lower findings (never HIGH/CRITICAL) may be waived by the same human decision that approves
the output (`ApprovalRequest.waives`). Approval creates `Waiver`s bound to `finding.key()` **and** the
`evidence_hash`, records a `WAIVER` decision, and the policy is re-evaluated with
`policy.with_waivers(state.waivers)`. A waiver is never reused across epochs or changed content.

### 10.7 Identity and oversight (honest model)
Identity is **asserted, not authenticated** (limitation). Controls: `normalize_approver` (NFKC, casefold,
ASCII `^[a-z0-9][a-z0-9._@-]{1,63}$`; reserved names — registered agent ids, `system`, `engine`, `policy`,
`scripted`, `agent`, `unknown` — denied), distinctness on the normalised name, optional
`approver_allowlist`, `approver != created_by` for risk ≥ MEDIUM, `os_user` recorded on every vote, and in
`identity_mode=strict` distinct `os_user`s for HIGH. Default mode is `attested`: the report prints
`four-eyes: ATTESTED-ONLY`. CLI `approve` requires `--confirm <first 8 hex of evidence_hash>` (shown by
`status`). Scripted decisions (demos/CI) are allowed only with `AGENTIC_ALLOW_SCRIPTED=1`; they set
`RunState.assurance="DEMO_SCRIPTED"` and every report carries the banner "approvals were scripted, not a
human review". Providers are non-blocking (`poll(request) -> votes`); the interactive provider runs on its
own thread and posts to the inbox.

---------------------------------------------------------------------------

## 11. Clarification flow (AWAITING_INPUT)

* Blocking questions arrive from `AgentResult.clarifications` and/or a gate's `AWAIT_INPUT`, merged by
  question id. The attempt's staged output is **held, not published**; the node goes `AWAITING_INPUT`
  (`ClarificationRound` appended; bounded by `max_clarification_rounds`).
* Question ids are content-derived (`Q-<category>-<sha1(normalised text)[:6]>`): answers survive
  amendments and re-runs. `answer()` rejects unknown ids, enforces `options`, caps values at 2 KiB and
  screens them with SEC-001/SEC-006 *at ingestion* (a MEDIUM hit requires explicit human confirmation).
  `answer(accept_default=[qids])` is allowed only where `default_assumption` is set and records an
  `ASSUMPTION` decision.
* `clarification_answers` is a **cumulative** external artifact (`{"answers": {qid: {answer, by, round}}}`),
  `producer="external"`. Partial answers keep the node waiting. When all blocking questions are answered
  the held output is quarantined, the node becomes `READY` (`generation+1`) and the new attempt is **not
  counted** against any budget. Exhaustion: all remaining questions have defaults → `ASSUMPTION` decisions
  + a mandatory MEDIUM sign-off; otherwise the node **fails** ("never guess").
* Every ambiguity must end in a recorded disposition (`answered | assumed | deferred`); `deferred`
  requires a human `SCOPE` decision.

---------------------------------------------------------------------------

## 12. Re-planning

### 12.1 Triggers
(a) `amend`/`answer` → new version of an external artifact; (b) a node publishes different content;
(c) a planning node returns a new `PlanExpansion`; (d) remediation routing; (e) override-resume.

### 12.2 Invalidation algorithm
`affected = consumers_of(changed kinds) ∪ descendants(changed producers) ∪ changed/rewired expansion nodes`.
Per current status (pure intents → scheduler applies):

| status | action |
|---|---|
| SUCCEEDED | → STALE (outputs stay ACTIVE until republish; consumers are STALE too, so nothing dispatches on them) |
| SKIPPED | → PENDING (re-evaluate `skip_if`) |
| READY | → PENDING |
| RUNNING | trip cancel token, `generation++`, → PENDING, `epoch_no+1`; the late result is discarded (`NODE_ATTEMPT_DISCARDED`), staged artifacts quarantined, not counted |
| AWAITING_APPROVAL | request CANCELLED, staged quarantined, → PENDING, `epoch_no+1` |
| AWAITING_INPUT | → PENDING (unless the change *is* the answers artifact → READY) |
| FAILED / BLOCKED / CANCELLED / ROLLED_BACK | → PENDING, `epoch_no+1` (per-epoch budgets reset; lifetime cap applies) |
| STALE / REMOVED | unchanged |

Order within a tick: trip tokens → bump generations → activate the new artifact version, so a consumer
can never read the new version before its cancel is set. Each reset emits one `NODE_RESET`/`NODE_STALE`.

### 12.3 Expansions
`PlanExpansion` is untrusted. The engine:
* **E0** requires `owner == ctx.node.id`; forces the owner into every new node's `depends_on`;
  applies the expansion only at **publish** (after gates and approvals);
* **E1** overwrites governance attributes of every new node from `params['expansion_template']`
  (a `NodeSpec` dict held by the human-authored owner): approval, risk (`max`), retry (clamped to
  `RetryPolicy.HARD_MAX_ATTEMPTS`), timeout (finite, ≤ `max_attempt_s`), exit gates (+ mandatory
  defaults), `mandatory=False`, `side_effecting=False`, `skip_if=None`, fallback; the agent supplies only
  `id`, `title`, `depends_on`, `params`;
* **E2** rejects agents/stages/produces outside `params['expansion_policy']` and rewire targets outside
  its allow-list (CHG-007); rewire may only add edges; mandatory nodes may not be removed or altered
  (CHG-006, pinned by `mandatory_snapshot`);
* **E3** diff vs the owner's previous expansion: added → PENDING; retained → untouched; **changed** →
  treated as an input change (§12.2); rewire targets whose effective deps changed → §12.2;
* **E4** removed nodes: cancelled, approvals CANCELLED, **all** their ACTIVE artifacts superseded;
  rejected if the removed node is a SUCCEEDED side-effecting node that has not been compensated, or if
  any live node outside the removed set still depends on it;
* **E5** REMOVED nodes stay as tombstones in `RunState.nodes`, are excluded from completion, and their
  ids may be re-added (`epoch_no+1`);
* **E6** a `PlanPatchError` is an attempt failure (GATE_REJECTED with CHG findings), retried with the
  findings as feedback, staged artifacts (incl. `plan_expansion`) quarantined;
* **E7** every re-plan records a `REPLAN` decision with `{trigger, invalidated, cache_hit, re_run,
  added, removed, cancelled_inflight}`; `PLAN_REPLANNED` carries the new `graph_hash`.

---------------------------------------------------------------------------

## 13. Rollback and compensation

Three levels (frequency metric counts levels 2–3 only; attempt quarantines are reported separately):
1. **Attempt** — staged artifacts quarantined (`ARTIFACT_QUARANTINED`); zero side effects. Quarantining a
   SEC-001 artifact replaces its bytes with a tombstone (hash/size kept) so no secret persists on disk.
2. **Node** — a terminally FAILED node: descendants → BLOCKED; if it declares `compensates=(ids…)` the
   listed SUCCEEDED side-effecting ancestors are compensated (saga) → ROLLED_BACK.
3. **Run** — safe-stop compensates every SUCCEEDED side-effecting node in reverse topological order.

Compensators run from the persisted `ApplyReceipt` with a **fresh** cancel token, their own timeout and
bounded retries; best-effort and continue on failure (`ROLLBACK_FAILED` per node →
`safe_stop.compensation_incomplete`). `EffectPort.revert` reverses only files whose current hash equals
`post_hashes` (never clobbers later human edits) and is idempotent. A side-effecting node writes an
`EFFECT_STARTED` journal event **before** each effect and `EFFECT_COMPLETED` after; a non-`idempotent`
side-effecting node is retried only after a successful compensate/verify. `SUCCEEDED → ROLLED_BACK` is a
legal transition.

---------------------------------------------------------------------------

## 14. Safe-stop, kill switch, override, resume

**Triggers**: CRITICAL finding / `AutonomyViolation`; budgets (attempts, replans, active runtime,
zombies → `worker_exhaustion`); circuit breaker with no fallback left; kill switch; audit-integrity
failure; failed compensation; `engine_fault`.

**Journaled procedure** (idempotent; `RunState.safe_stop.step` is persisted after each step; a crash
resumes at the next step; `RunStatus.SAFE_STOPPING` while in progress):
* **S0** persist `SAFE_STOPPING`, record the `SAFE_STOP` decision (trigger, reason, parents) *first*; set
  the stopping flag so the completion handler discards everything (invariant I2);
* **S1** trip all cancel tokens, bump generations; mark PENDING approvals CANCELLED and drop open
  questions (a later `approve`/`answer` raises);
* **S2** quiesce for `safe_stop_quiesce_s`; survivors are zombies; a side-effecting survivor is flagged
  `needs_manual_reconciliation`;
* **S3** quarantine staged artifacts of RUNNING and AWAITING_* nodes;
* **S4** compensate (§13);
* **S5** mark non-settled nodes CANCELLED (non-side-effecting SUCCEEDED nodes keep their status);
* **S6** persist `SAFE_STOPPED`, write the finalizer outputs.

**Kill switch**: `<runs_dir>/KILL` (global) and `<run_dir>/KILL` (per run), default location outside the
sandbox and baseline. Existence alone means *tripped* (contents are never required to be valid); read ≤ 1 KiB,
no symlink following, sanitised; **any OSError other than FileNotFoundError is treated as tripped**
(fail closed). The path comes only from process config (the state snapshot is ignored). Clearing is an
attributed, audited action (`agentic kill --clear`), never plain `rm`. `kill()` with no live engine takes the
lock and performs the safe-stop offline.

**Override** (`resume --override-safe-stop --by --reason [--extend budget=N]`): refused if the trigger
condition still holds after applying `extend` (bounded); at most `max_overrides` (2); recorded as a
`SAFE_STOP` decision with the original as parent. Triggers `kill_switch | budget | circuit_breaker` need one
attested human; `integrity | autonomy_violation | compensation_failed` need the HIGH path (two distinct
approvers) and the check re-run — for `integrity` only a forked, newly anchored run segment is allowed,
never "continue on a broken chain". After an override: CANCELLED/ROLLED_BACK/STALE → PENDING (`epoch_no+1`),
breaker reset, quarantined artifacts stay quarantined, acknowledged kill signals are recorded.

**Resume table**: PENDING/READY recomputed; RUNNING → re-queue with `crash_recovered=true` (staged of that
generation quarantined, one attempt counted); AWAITING_APPROVAL/INPUT stay after re-hashing staged artifacts
and re-validating the fingerprint; back-off stored as absolute `not_before`; open runtime/human-wait
intervals are closed at `last_heartbeat`. Resume first runs `reconcile` (§7) and the audit verification (§17).

---------------------------------------------------------------------------

## 15. Time model

* **T1** the scheduler idles with `completion_queue.get(timeout=poll_interval_s)` in real time, never
  `Clock.sleep`; **T2** back-off is data (`NodeState.not_before`, absolute UTC); **T3** only when nothing is
  running and the completion queue is empty does the loop call `clock.sleep(min(not_before) − now)` —
  instant on `FakeClock`, real on `SystemClock`; **T4** worker threads use `cancel.wait(s)`, never the
  scheduler's `clock.sleep`; **T5** durations are persisted as relative values. Active runtime =
  `clock.monotonic()` accrual minus human-wait intervals. `FakeClock.sleep` is thread-safe and advances one
  global timeline (documented: parallel branches serialise in virtual time; use real time to observe
  concurrency).
* Threads cannot be killed: an attempt that does not honour its token keeps its worker slot as a **zombie**;
  workers are daemon threads; `max_zombie_threads` (default 2) caps them; at the cap the engine stops
  dispatching and, if nothing else can progress, safe-stops (`worker_exhaustion`). Hard guarantees come
  from sandbox subprocess timeouts.

---------------------------------------------------------------------------

## 16. External inputs

`start` stores `request` (producer `external`, `text/markdown`, key `""`); `amend` creates a new version
(`SCOPE` decision, actor human, old/new diff stored); `answer` writes the cumulative answers artifact (§11).
Limits: `max_input_bytes=65536`, valid UTF-8, no NUL, control characters stripped. **Screening at ingestion**:
every human-supplied string (request, answers, amendments, approval comments, scenario scripts) is evaluated
as an `ArtifactSubject(untrusted_input=True)` *before* it becomes ACTIVE and before any staleness
propagation (SEC-001 rejects a secret — only a redacted copy is stored and ever sent to an LLM; SEC-006 flags
prompt-injection indicators → MEDIUM, human confirmation). Baseline snapshots always exclude `runs_dir`, `.git`,
`.venv`, `.env*`, `*.pem`, `*.key`, `*.db`, `__pycache__` and never follow links.

---------------------------------------------------------------------------

## 17. Audit, events, metrics

1. **Chain.** Hash covers the canonical JSON of the raw parsed dict minus `hash` (unknown keys break it).
   `seq` starts at 1, `prev_hash` links, constant `run_id`; lines are read split on `\n` only. Governance
   events (`APPROVAL_*`, `SAFE_STOP_*`, `KILL_*`, `NODE_PUBLISHING`, terminal events) are fsynced before the
   mutation they authorise.
2. **Anchors.** `RunState.audit_head={seq,hash}` is saved with every state save; `resume` asserts the log
   contains that seq/hash (detects tail truncation). The head is appended to `<runs_dir>/anchors.jsonl` after
   every governance event and printed in `report.md` / CLI; `audit-verify --anchor <hash|file>` checks it.
   At a terminal state `seal.json` = {head hash, count, sha256 of `state.json`, artifact index, decisions
   log} (`RUN_SEALED`). *Limitation*: a privileged same-user attacker can rewrite log + anchors; the real fix
   is remote WORM / a transparency log (optional `AGENTIC_AUDIT_HMAC_KEY` keyed mode is out of scope).
3. **Torn tail.** If only the last line is unparseable and everything before verifies → truncate, emit
   `AUDIT_TAIL_REPAIRED`, continue. Any mid-chain mismatch raises `IntegrityError`; the engine then opens a
   **new segment** `audit.1.jsonl` whose first event embeds `broken_at` + last good hash and appends
   `SAFE_STOP_TRIGGERED(integrity)` there.
4. **Binding.** `ARTIFACT_CREATED{artifact_id, content_hash, size}`; `DECISION_RECORDED{decision_hash}`;
   `APPROVAL_VOTE{… evidence_hash}`; `RUN_CREATED{graph_hash, baseline_manifest_hash, policy hash, config hash,
   assurance}`; reads of artifacts cross-check `content_hash` against the chain.
5. **Reconcile (resume).** graph hash; every APPROVED request has ≥ required distinct votes in the chain with
   the current fingerprint; every SUCCEEDED node has a `NODE_SUCCEEDED`; counters ≥ chain counts; artifact
   index ↔ `ARTIFACT_CREATED` events. Mismatch → `IntegrityError` → safe-stop.
6. **Spans** derive inside `AuditLog.append` from `(node_id, attempt)`: run `"run"`, node `"n:<id>"`, attempt
   `"n:<id>#<n>"`. `ts` and `seq` are assigned inside the lock.
7. **Decisions** (§ DecisionDraft): recorded **only at publish** with `inputs` = snapshot refs, `outputs` =
   published refs, `parents` = `draft.parents ∪ decisions whose outputs intersect inputs`; one `RETRY`/
   `REJECTION` decision per failed attempt (finding keys, attempt number). Human decisions take
   `ApprovalRequest.evidence` as `inputs`. `agentic lineage <artifact>` must reach, e.g., the `CLARIFICATION`
   decision from `release_patch` in the ambiguous scenario.
8. **Metrics** (pure functions of events): `attempts=count(NODE_STARTED)`; `retries=count(NODE_RETRY_SCHEDULED)`;
   `retry_frequency=retries/attempts`; `fallback_rate=fallbacks/nodes`; `rollback_frequency=(node+run
   compensations)/runs`; `first_pass_yield`; an **incident** opens at the first `NODE_ATTEMPT_FAILED` (kind ≠
   cancelled) for `(node, epoch)` and closes at `NODE_SUCCEEDED`/`NODE_CACHE_HIT`; `NODE_FAILED` = unrecovered;
   it is **abandoned** (counted separately) on reset/removal/safe-stop. **MTTR excludes human-wait intervals**
   (`RUN_WAITING → RUN_RESUMED`); discarded/cancelled/needs-input attempts are excluded from retry rate;
   `e2e_latency`, `active_latency`, `human_wait`, per-stage latency (from `NODE_STARTED.data.stage`),
   `critical_path_s`, `max_concurrency_observed` (overlapping attempt spans), `replan_reuse_ratio`
   (cache hits / revalidated nodes), policy violations by rule, approvals requested/auto.

---------------------------------------------------------------------------

## 18. Finalizer and reports

The engine runs **finalizers for every terminal state** (success, failure, safe-stop): render `report.md`,
`report.json`, `metrics.json`, `trace.json`, `seal.json` (engine, deterministic, from `RunView` + audit), then
injected finalizers (L5): the **engineering summary** (`engineering_summary` artifact, `actor=system`) with the
five required sections — plan/rationale, artifacts, risks/trade-offs/validation, assumptions, limitations — where
every claim cites an artifact or decision id and *limitations are computed from facts* (unsupported/deferred
requirements, waivers, scripted approvals, injected faults, skipped checks, manual steps, final status,
`audit_head`). All text from untrusted sources passes `sanitize`/`escape_markdown`; diagrams use generated ids.

---------------------------------------------------------------------------

## 19. Invariants (checked by `check_invariants(state)` after every save in tests, and at run end)

* every node status change is an allowed transition;
* a SUCCEEDED node with an approval requirement has an APPROVED request with ≥ required distinct votes whose
  `node_fingerprint` equals the node's `last_success_fingerprint`;
* ACTIVE artifacts have exactly one ACTIVE version per `(kind,key)`; STAGED artifacts are referenced by a
  node's `staged` or are quarantined on resume;
* no ACTIVE artifact belongs to a REMOVED node;
* `mandatory_snapshot` ids exist in the graph with unchanged `spec_hash`;
* counters ≥ audit-derived counts; `audit_head` matches the log;
* replaying the audit log rebuilds node statuses equal to `state.json` (property test).

---------------------------------------------------------------------------

## 20. Hard limits and policy floors

`EngineConfig.__post_init__` enforces `HARD_LIMITS` (e.g. `approvals_by_risk[HIGH] ≥ 2`, `circuit_breaker_failures
∈ [1,20]`, `max_parallel ∈ [1,16]`, budget ceilings); out-of-range values raise `ConfigError`; any value away from the
default is recorded in `RUN_CREATED` and the report prints **Governance posture: DEGRADED**. `PolicyConfig` severity
overrides are **raise-only**: SEC-*, CHG-001, CHG-006/007, QLT-001 floors (`>= HIGH`, DENY) and CRITICAL cannot be
lowered; any downgrade needs an out-of-band flag + SCOPE decision and is refused in `scenario.json`. The
`PolicyEngine` asserts its rule ids equal the catalogue manifest and that every `SubjectKind` has ≥ 1 rule; the
manifest hash goes into `RUN_CREATED`.
