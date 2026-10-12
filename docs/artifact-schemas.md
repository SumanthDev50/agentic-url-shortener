# Artifact schemas

The authoritative, executable definition is [`orchestrator/core/schemas.py`](../orchestrator/core/schemas.py)
(`Kind` constants, id grammars, `validate_payload(kind, obj)`); this page records the conventions behind it.

## Conventions
* Every JSON payload carries `"schema": 1`. Text kinds: `request`, `release_patch`, `engineering_summary`.
* **Determinism** — payload bytes never contain wall-clock times, durations, pids, random ids or absolute paths
  (put those in `ArtifactDraft.metadata`: excluded from the content hash and the fingerprint). This is what makes
  fingerprint memoisation and re-plan reuse work. Example: `test_report` records *outcomes only*; the perf p95 is
  metadata.
* Change-set kinds (`changeset`, `integrated_changeset`, `docs_changeset`) hold `ChangeSet.to_dict()`;
  `changeset` is *keyed* (key = producing node id), every other kind has key `""`.
* Artifact ids: `<kind>[.<key>]@v<N>` (`kind` has no `.`; node-id keys such as `implement.core` do).

## Identifier grammar
| id | pattern | example |
|---|---|---|
| requirement | `R-\d{3}` | `R-004` |
| acceptance criterion | `AC-\d{3}-\d{1,2}` | `AC-004-2` |
| assumption | `ASM-\d{3}` | `ASM-001` |
| ambiguity | `AMB-<6 hex>` | `AMB-3fa91c` |
| clarification question | `Q-<category>-<6 hex>` (content-derived) | `Q-vague_quality-9ab04e` |
| task | `T-\d{3}` | `T-007` |
| ADR | `ADR-\d{3}` | `ADR-002` |
| risk | `RISK-\d{3}` | `RISK-005` |
| group | `[a-z][a-z0-9_-]{0,30}` | `core` |

## Traceability convention (CMP-001)
Every generated test carries, in its docstring, a line `Covers: R-001, AC-001-2` (never a pytest marker —
`--strict-markers` is on). The verifier maps pytest outcomes (junit) to acceptance criteria through these lines;
`traceability_matrix` records, per acceptance criterion, the covering tests and their *results*. A criterion is
covered only by a **passed** test (or `verification=inspection` with evidence, or a human-approved deferral).

## Payloads (minimum keys; see `schemas.SHAPES` for the exact grammar)
| kind | producer | key contents |
|---|---|---|
| `request` | external | free text (UTF-8, ≤ 64 KiB) |
| `clarification_answers` | external | `answers: {question_id: {answer, by, round}}` (cumulative) |
| `intake_report` | `intake` | `workspace{file_count, python_files, is_empty, manifest_hash}`, `classification{workspace_kind, change_type}`, `input_findings[]` |
| `requirements` | `requirements` | `problem`, `classification`, `requirements[{id,text,kind,priority,status,status_decision,capabilities,parameters,verification,acceptance_criteria[]}]`, `assumptions[]`, `ambiguities[{id,category,span,question_id,disposition,answer}]`, `open_questions[]`, `unsupported[]`, `capabilities[]` |
| `impact_report` | `analyst` | `modules[]`, `routes[]`, `tables[]`, `data_flows[{route,handler,calls,repositories,tables,unresolved}]`, `tests[]`, `coupling_risks[]`, `risk_signals[]`, `expected_paths[]` |
| `threat_model` | `threat-modeler` | `assets[]`, `trust_boundaries[]`, `risks[{id,category(STRIDE),likelihood,impact,mitigations[{text,requirement_id,test_hint}],status}]` |
| `design_doc` | `architect` | `components[]`, `adrs[{id,title,decision,alternatives[],rationale,consequences}]`, `data_model_changes[]`, `api_changes[]`, `rollback_plan`, `change_reference`, `risk` |
| `api_contract` | `architect` | `paths[{method,path,request_fields[],responses[]}]`, `breaking` (the *declared* delta; `api_compat` compares it with the actual OpenAPI diff) |
| `task_plan` | `planner` | `tasks[{id,group,requirement_ids,depends_on,capability,kind,write_scope,expected_paths,params}]`, `groups[{id,node_id,task_ids,capability,params,write_scope}]`, `critical_path[]`, `unsupported[]` |
| `plan_expansion` | engine | `PlanExpansion.to_dict()` (published atomically with the owner's outputs) |
| `integration_report` | `integrator` | `sources[]`, `merged_paths[]`, `conflicts[]`, `stats{files_*, lines_*}`, `impact_delta{predicted_only, actual_only, precision, recall}` |
| `test_report` | `verifier` | `passed,total,failed,errors,skipped`, `tests[{nodeid,outcome,covers[]}]`, `baseline{ran,passed,failed_tests}`, `fail_to_pass{checked,ok,tests}`, `perf{checked,budget_ms,ok}`, `api_compat{checked,breaking,additive}` |
| `security_report` | `security-scanner` | `rules_evaluated[]`, `findings[]`, `files_scanned`, `passed` |
| `readiness_report` | `release-manager` | `verdict GO/NO_GO`, `checks[{id,title,passed,evidence,detail}]`, `risk`, `rollback_plan`, `change_reference`, `open_items[]` |
| `traceability_matrix` | `release-manager` | `requirements[{id,status,acceptance_criteria[{id,tests,results,covered}],tasks,files,decision_refs}]`, `complete`, `gaps[]` |
| `release_patch` | `promoter` | unified diff text (git style, LF) |
| `release_bundle` | `promoter` | `mode bundle|applied`, `patch_sha256`, `files[]`, `base_manifest_hash`, `receipt` (apply mode) |
| `release_verification` | `release-verifier` | `target`, `applied_cleanly`, `tests_passed`, `healthz_ok`, `checked_files` |
| `engineering_summary` | engine finalizer | markdown; every claim cites artifact/decision ids |
