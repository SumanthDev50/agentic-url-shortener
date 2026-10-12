# SDLC workflow contract (normative)

The exact shape of the standard workflow built by `orchestrator/workflows/sdlc.py`
(`build_sdlc_graph`). A test asserts that the built graph equals this table. Node ids never
contain `:`; dynamic nodes are `implement.<group>`. Agent ids are keys of the `AgentRegistry`.

Legend — **C** consumes, **O** optional_consumes, **V** overlay_from, **S** skip_inputs,
**P** produces. `M` = mandatory (cannot be removed/altered by a re-plan, cannot end SKIPPED).

| node | stage | agent (fallback) | deps | C | O | V | S | P | entry gates | exit gates (+ implicit `artifacts_present`, `policy`) | approval | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `intake` | intake | `intake` | – | request | | | | intake_report | | `json_shape` | – | |
| `requirements` | requirements | `requirements` | intake | request, intake_report | clarification_answers | | | requirements | | `json_shape`, `acceptance_criteria_testable`, `ambiguity_dispositions_complete`, `no_blocking_questions` | AFTER, threshold MEDIUM (assumptions ⇒ MEDIUM risk) | clarification loop (§11) |
| `codebase-analysis` | analysis | `analyst` | requirements | requirements | | | intake_report | impact_report | | `json_shape` | – | `skip_if=greenfield` |
| `threat-model` | security | `threat-modeler` | requirements | requirements | | | | threat_model | | `json_shape` | – | parallel with analysis |
| `design` | design | `architect` | requirements, threat-model, codebase-analysis | requirements, threat_model | impact_report | | | design_doc, api_contract | | `json_shape`, `design_risk`, `design_covers_requirements` | AFTER, threshold MEDIUM (`design_risk` ⇒ HIGH if `data_model_changes`) | |
| `decompose` | planning | `planner` | design | requirements, design_doc | impact_report | | | task_plan, plan_expansion | | `json_shape`, `plan_covers_requirements` | – | owns the `implement.*` expansion; `unsupported` ⇒ AWAIT_INPUT |
| `implement.<g>` *(dynamic)* | implementation | `implementer` (`implementer.safe`) | decompose | task_plan, design_doc | impact_report | | | changeset | | `syntax_valid`, `scope_within_plan` | AFTER, risk-based (change-set risk) | from `expansion_template`; narrowed autonomy from `task.write_scope` |
| `author-tests` | testing | `test-author` | decompose | requirements, task_plan, design_doc | | | | changeset (key = node id) | | `syntax_valid`, `tests_cover_acceptance_criteria` | – | parallel with `implement.*` |
| `integrate` | implementation | `integrator` | decompose (+ every `implement.*`, `author-tests` via rewire) | changeset, task_plan | impact_report | | | integrated_changeset, integration_report | | `integration_clean` | – | barrier; conflicts ⇒ serialising re-plan (§6) |
| `verify` **M** | testing | `verifier` | integrate | requirements, integrated_changeset, task_plan | impact_report | integrated_changeset | | test_report | | `json_shape`, `tests_passed`, `fail_to_pass`, `api_compat` | – | `remediation(route_to=implement.*, author-tests; max_loops=2)`; retry only TRANSIENT/TIMEOUT; may execute |
| `security-scan` **M** | security | `security-scanner` | integrate | integrated_changeset | impact_report | integrated_changeset | | security_report | | `json_shape`, `security_clean` | – | parallel with verify, docs |
| `docs` | documentation | `documenter` | integrate | requirements, design_doc, integrated_changeset, task_plan | api_contract | integrated_changeset | | docs_changeset | | `syntax_valid` | – | parallel with verify, security-scan |
| `release-readiness` **M** | release | `release-manager` | verify, security-scan, docs | requirements, design_doc, task_plan, integrated_changeset, integration_report, test_report, security_report, docs_changeset | impact_report, threat_model, intake_report | | | readiness_report, traceability_matrix | | `json_shape`, `traceability`, `release_ready` | **AFTER, always**; ≥ 2 approvers when risk HIGH | `cacheable=False` (reads `RunView`) |
| `promote` **M** | release | `promoter` | release-readiness | integrated_changeset, docs_changeset, readiness_report | | integrated_changeset, docs_changeset | | release_patch, release_bundle | `baseline_unchanged` | `json_shape` | *bundle mode*: none · *apply mode*: **BEFORE, always** | apply mode: `side_effecting`, compensator `apply_bundle`, risk HIGH, uses `EffectPort` |
| `release-verify` **M** | release | `release-verifier` | promote | release_bundle, release_patch | | integrated_changeset, docs_changeset | | release_verification | | `json_shape`, `release_verified` | – | apply mode: `compensates=(promote,)` ⇒ saga rollback on failure |

There is **no `engineering-summary` node**: the summary is an engine *finalizer* that runs for every terminal state
(engine-semantics §18).

## Graph invariants (checked by `WorkflowGraph.validate`)
* acyclic; no dangling deps; node ids match `NODE_ID_RE`;
* every non-optional consumed kind has an **ancestor producer** or is external (`request`, `clarification_answers`);
  optional kinds likewise (so reads never depend on scheduling order);
* each `(kind, "")` has at most one producing node; `KEYED_KINDS` (`changeset`) are produced by many (key = node id);
* `overlay_from ⊆ CHANGESET_KINDS`; mandatory nodes have no `skip_if`; `side_effecting ⇔ compensator`;
* `remediation.route_to` entries resolve to existing nodes or `prefix.*` patterns with ≥ 1 current match.

## Skip predicates and gate catalogue

| name | where | reads | meaning |
|---|---|---|---|
| `greenfield` (skip) | `codebase-analysis` | `intake_report.workspace.is_empty` | no baseline to analyse |
| `artifacts_present`, `json_shape{kind}`, `policy`, `no_blocking_questions` | engine (generic) | | contract / shape / always-on policy / blocking clarifications |
| `acceptance_criteria_testable` | requirements | requirements | every active requirement has ≥ 1 acceptance criterion with measurable text |
| `ambiguity_dispositions_complete` | requirements | requirements | every ambiguity is answered / assumed (with ASSUMPTION decision) / deferred (with human SCOPE decision) |
| `design_risk` | design | design_doc | returns `RiskAssessment` (HIGH if `data_model_changes` or breaking `api_changes`) |
| `design_covers_requirements` | design | requirements, design_doc | every active requirement maps to ≥ 1 component |
| `plan_covers_requirements` | decompose | requirements, task_plan | every active requirement has ≥ 1 task; every task is in a group with an `implement.*` node or declared docs-only; no unsupported requirement left |
| `syntax_valid`, `scope_within_plan` | implement / author-tests / docs | changeset | QLT-001; changed existing files ⊆ impact set ∪ recipe anchors ∪ task `write_scope` |
| `tests_cover_acceptance_criteria` | author-tests | requirements, changeset | each active AC appears in ≥ 1 test docstring `Covers:` line |
| `integration_clean` | integrate | integration_report | no unresolved conflicts |
| `tests_passed`, `fail_to_pass`, `api_compat` | verify | test_report | suite green; authored tests **fail on the baseline and pass on the integrated view**; OpenAPI diff vs declared `api_contract` (CHG-004) |
| `security_clean` | security-scan | security_report | no HIGH+ findings |
| `traceability` | release-readiness | `RunSubject` | CMP-001: each active AC has ≥ 1 *passed* covering test, or `verification=inspection` with evidence, or status deferred/descoped with a human SCOPE decision |
| `release_ready` | release-readiness | `RunSubject` | CMP-003 (approvals/waivers/decisions complete, none stale) and CHG-005 (change reference + rollback plan) |
| `baseline_unchanged` | promote (entry) | target tree | target `manifest_hash == baseline` (else AWAIT_INPUT) |
| `release_verified` | release-verify | release_verification | patch applies cleanly to a fresh baseline copy, suite green, `/healthz` smoke ok |

## Gate/decision wiring that must be demonstrable
Greenfield: skipped analysis, ≥ 3 parallel `implement.*` groups + `author-tests`, join at `integrate`, transient-fault retry.
Brownfield: impact analysis, migration ⇒ HIGH risk ⇒ 2 approvers at `design`, CHG-001 deny → retry → fallback, remediation loop.
Ambiguous: clarification round, deferred requirement with SCOPE decision, mid-run amendment ⇒ re-plan with cache hits, SEC-006 hit.
