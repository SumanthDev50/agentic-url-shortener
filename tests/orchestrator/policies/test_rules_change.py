"""CHG-001 ... CHG-008: change-control rules (positive, negative, edge, fail-closed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from orchestrator.core.errors import PathViolation
from orchestrator.core.types import (
    Finding,
    ArtifactStatus,
    ChangeSet,
    ChangeSetSubject,
    FileOp,
    FileOpKind,
    NodeSpec,
    PlanExpansion,
    Rewire,
    Severity,
    Stage,
)
from orchestrator.policies.config import ChangeBudget, ExpansionRules, PolicyConfig
from orchestrator.policies.rules_change import (
    ApiCompatibility,
    ChangeSizeBudget,
    ImmutableMigrations,
    MandatoryNodesPinned,
    PlanConformance,
    ProtectedPaths,
    ReleaseDocumentation,
    ScopeWithinImpact,
    default_rules,
)
from tests.orchestrator.policies.policy_builders import (
    FakeRunView,
    check,
    create,
    cs_subject,
    delete,
    modify,
    node,
    plan_subject,
    release_chain,
    run_subject,
    view,
)

CFG = PolicyConfig()
MIGRATION = "shortener/storage/migrations/0001_init.sql"


def test_default_rules_are_the_catalogue() -> None:
    assert sorted(r.rule_id for r in default_rules()) == [f"CHG-00{i}" for i in range(1, 9)]


# ===============================================================================================
# CHG-001


def test_chg001_modify_or_delete_of_an_applied_migration_is_high() -> None:
    base = view({MIGRATION: "create table t(id integer);\n"})
    for op in (modify(base, MIGRATION, "drop table t;\n"), delete(base, MIGRATION)):
        findings = check(ImmutableMigrations, cs_subject([op], base))
        assert [(f.rule_id, f.severity, f.location) for f in findings] == [
            ("CHG-001", Severity.HIGH, MIGRATION)
        ]


def test_chg001_new_migrations_and_other_sql_are_fine() -> None:
    base = view({MIGRATION: "create table t(id integer);\n"})
    new = create("shortener/storage/migrations/0002_add.sql", "alter table t add column x text;\n")
    other = create("scripts/report.sql", "select 1;\n")
    assert check(ImmutableMigrations, cs_subject([new, other], base)) == []


def test_chg001_a_glob_matching_nothing_passes() -> None:
    cfg = PolicyConfig(migrations_glob="db/changes/*.sql")
    base = view({MIGRATION: "x\n"})
    assert check(ImmutableMigrations, cs_subject([modify(base, MIGRATION, "y\n")], base), cfg) == []
    assert check(ImmutableMigrations, cs_subject([create("a.py", "x=1\n")])) == []


def test_chg001_create_that_overwrites_or_shadows_an_existing_migration() -> None:
    base = view({MIGRATION: "x\n"})
    shadow = create(MIGRATION.replace("0001_init", "0001_INIT"), "y\n")
    assert [f.severity for f in check(ImmutableMigrations, cs_subject([shadow], base))] == [
        Severity.HIGH
    ]
    overwrite = FileOp(MIGRATION, FileOpKind.CREATE, "y\n")
    assert [f.severity for f in check(ImmutableMigrations, cs_subject([overwrite], base))] == [
        Severity.HIGH
    ]


def test_chg001_unverifiable_modify_of_a_missing_migration_fails_closed() -> None:
    ghost = FileOp(MIGRATION, FileOpKind.MODIFY, "y\n", "0" * 64)
    findings = check(ImmutableMigrations, cs_subject([ghost], view()))
    assert [f.severity for f in findings] == [Severity.HIGH]
    assert "cannot be verified" in findings[0].message


def test_chg001_custom_glob_and_nested_migrations() -> None:
    cfg = PolicyConfig(migrations_glob="db/*.sql")
    base = view({"db/001.sql": "x\n", "other/migrations/1.sql": "x\n"})
    assert check(ImmutableMigrations, cs_subject([modify(base, "db/001.sql", "y\n")], base), cfg)
    assert (
        check(
            ImmutableMigrations,
            cs_subject([modify(base, "other/migrations/1.sql", "y\n")], base),
            cfg,
        )
        == []
    )
    deep = view({"a/b/c/migrations/1.sql": "x\n"})
    assert check(
        ImmutableMigrations, cs_subject([modify(deep, "a/b/c/migrations/1.sql", "y\n")], deep)
    )


# ===============================================================================================
# CHG-002


def test_chg002_file_budget() -> None:
    ops = [create(f"f{i}.py", "x = 1\n") for i in range(26)]
    findings = check(ChangeSizeBudget, cs_subject(ops))
    assert [(f.rule_id, f.severity) for f in findings] == [("CHG-002", Severity.MEDIUM)]
    assert "26 files" in findings[0].message
    assert check(ChangeSizeBudget, cs_subject(ops[:25])) == []  # the budget itself is allowed


def test_chg002_line_budget_for_create_modify_and_delete() -> None:
    big = "x = 1\n" * 801
    assert [f.severity for f in check(ChangeSizeBudget, cs_subject([create("a.py", big)]))] == [
        Severity.MEDIUM
    ]
    assert check(ChangeSizeBudget, cs_subject([create("a.py", "x = 1\n" * 800)])) == []
    base = view({"a.py": big})
    assert check(ChangeSizeBudget, cs_subject([delete(base, "a.py")], base))
    changed = "y = 2\n" * 801
    assert check(ChangeSizeBudget, cs_subject([modify(base, "a.py", changed)], base))
    small_edit = big.replace("x = 1", "x = 2", 1)
    assert check(ChangeSizeBudget, cs_subject([modify(base, "a.py", small_edit)], base)) == []


def test_chg002_custom_budget_and_both_dimensions() -> None:
    cfg = PolicyConfig(change_budget=ChangeBudget(max_files=1, max_lines=5))
    subject = cs_subject([create("a.py", "x\n" * 10), create("b.py", "y\n")])
    assert len(check(ChangeSizeBudget, subject, cfg)) == 2


def test_chg002_huge_modifications_are_costed_by_an_upper_bound_quickly() -> None:
    old = "".join(f"line {i}\n" for i in range(5000))
    new = "".join(f"other {i}\n" for i in range(5000))
    base = view({"big.txt": old})
    findings = check(ChangeSizeBudget, cs_subject([modify(base, "big.txt", new)], base))
    assert [f.severity for f in findings] == [Severity.MEDIUM]


# ===============================================================================================
# CHG-003


def _modify_subject(impact: Any, path: str = "shortener/api/routes.py") -> ChangeSetSubject:
    base = view({path: "x = 1\n"})
    return cs_subject([modify(base, path, "x = 2\n")], base, impact=impact)


def test_chg003_missing_impact_report_with_modifications_is_unverifiable() -> None:
    findings = check(ScopeWithinImpact, _modify_subject(None))
    assert [(f.rule_id, f.severity) for f in findings] == [("CHG-003", Severity.MEDIUM)]
    assert "unverifiable" in findings[0].message


@pytest.mark.parametrize("impact", [{}, {"unrelated": 1}, {"modules": "nope"}, "text", 5, []])
def test_chg003_unusable_impact_reports_are_unverifiable(impact: object) -> None:
    assert "unverifiable" in check(ScopeWithinImpact, _modify_subject(impact))[0].message


def test_chg003_only_new_files_need_no_impact_report() -> None:
    base = view({"a.py": "x\n"})
    assert check(ScopeWithinImpact, cs_subject([create("new.py", "x\n")], base)) == []
    assert check(ScopeWithinImpact, cs_subject([create("new.py", "x\n")])) == []  # greenfield


def test_chg003_in_scope_via_modules_expected_paths_tests_and_anchors() -> None:
    path = "shortener/api/routes.py"
    for impact in (
        {"modules": [{"path": path}]},
        {"modules": [], "expected_paths": [path]},
        {"modules": [], "expected_paths": ["shortener/api/*.py"]},
        {"modules": [], "tests": [path]},
        {"modules": [], "recipe_anchors": ["shortener/**"]},
        {
            "modules": [{"path": "other.py"}, "garbage", {"nopath": 1}],
            "expected_paths": [path, 7, None],
        },
    ):
        assert check(ScopeWithinImpact, _modify_subject(impact)) == [], impact


def test_chg003_modifications_outside_the_impact_set_are_flagged_per_file() -> None:
    impact = {"modules": [{"path": "shortener/api/routes.py"}], "expected_paths": []}
    base = view({"shortener/api/routes.py": "a\n", "shortener/core.py": "b\n", "README.md": "c\n"})
    ops = [
        modify(base, "shortener/api/routes.py", "a2\n"),
        modify(base, "shortener/core.py", "b2\n"),
        delete(base, "README.md"),
    ]
    findings = check(ScopeWithinImpact, cs_subject(ops, base, impact=impact))
    assert sorted(f.location or "" for f in findings) == ["README.md", "shortener/core.py"]
    assert all(f.severity == Severity.MEDIUM for f in findings)


def test_chg003_unsafe_impact_entries_are_ignored_not_trusted() -> None:
    impact = {"modules": [], "expected_paths": ["../**", "/etc/passwd", "a\\b", "**"]}
    assert (
        check(ScopeWithinImpact, _modify_subject(impact)) == []
    )  # '**' is a legal (and broad) glob
    impact = {"modules": [], "expected_paths": ["../**", "/etc/passwd"]}
    assert check(ScopeWithinImpact, _modify_subject(impact))  # the unsafe entries grant nothing


# ===============================================================================================
# CHG-004

ROUTE_FILE = (
    "from fastapi import APIRouter\nrouter = APIRouter()\n@router.get('/x')\ndef h():\n    "
    "return 1\n"
)


def test_chg004_no_route_change_needs_no_diff() -> None:
    assert check(ApiCompatibility, cs_subject([create("a.py", "x = 1\n")])) == []
    base = view({"api.py": ROUTE_FILE})
    body_only = ROUTE_FILE.replace("return 1", "return 2")
    assert check(ApiCompatibility, cs_subject([modify(base, "api.py", body_only)], base)) == []


def test_chg004_changed_routes_without_a_diff_fail_closed() -> None:
    added = check(ApiCompatibility, cs_subject([create("api.py", ROUTE_FILE)]))
    assert [(f.rule_id, f.severity) for f in added] == [("CHG-004", Severity.HIGH)]
    assert "GET /x" in (added[0].evidence or "")
    base = view({"api.py": ROUTE_FILE})
    signature = ROUTE_FILE.replace("def h():", "def h(q: int = 0):")
    assert check(ApiCompatibility, cs_subject([modify(base, "api.py", signature)], base))
    path = ROUTE_FILE.replace("'/x'", "'/y'")
    assert check(ApiCompatibility, cs_subject([modify(base, "api.py", path)], base))
    assert check(
        ApiCompatibility, cs_subject([delete(base, "api.py")], base)
    )  # removing a route changes the API


def test_chg004_a_supplied_diff_replaces_the_fail_closed_default() -> None:
    clean = {"openapi_diff": {"breaking": [], "additive": ["POST /x"]}}
    assert check(ApiCompatibility, cs_subject([create("api.py", ROUTE_FILE)], impact=clean)) == []
    breaking = {
        "openapi_diff": {"breaking": ["removed path GET /old", {"path": "/z"}], "additive": []}
    }
    findings = check(ApiCompatibility, cs_subject([create("a.py", "x=1\n")], impact=breaking))
    assert [f.severity for f in findings] == [Severity.HIGH, Severity.HIGH]
    assert "removed path GET /old" in (findings[0].evidence or "")


def test_chg004_accepts_diff_objects_and_rejects_malformed_ones() -> None:
    @dataclass
    class Diff:
        breaking: list[str]

    assert (
        check(
            ApiCompatibility,
            cs_subject([create("a.py", "x=1\n")], impact={"openapi_diff": Diff([])}),
        )
        == []
    )
    assert check(
        ApiCompatibility,
        cs_subject([create("a.py", "x=1\n")], impact={"openapi_diff": Diff(["x"])}),
    )
    malformed_diffs: list[Any] = [{}, {"additive": []}, {"breaking": "none"}, "text", 5]
    for malformed in malformed_diffs:
        findings = check(
            ApiCompatibility,
            cs_subject([create("a.py", "x=1\n")], impact={"openapi_diff": malformed}),
        )
        assert [f.severity for f in findings] == [Severity.HIGH], malformed


def test_chg004_unparseable_new_code_is_left_to_the_syntax_rule() -> None:
    assert check(ApiCompatibility, cs_subject([create("api.py", "def (:\n")])) == []


# ===============================================================================================
# CHG-005


def _release_view(
    payload: object, *, producer: str = "design", status: ArtifactStatus = ArtifactStatus.ACTIVE
) -> FakeRunView:
    rv = FakeRunView(release_chain())
    rv.add_artifact("design_doc", payload, producer=producer, status=status)
    return rv


def test_chg005_requires_change_reference_and_rollback_plan() -> None:
    complete = {"change_reference": "CHG-1234", "rollback_plan": "revert the merge commit"}
    assert check(ReleaseDocumentation, run_subject(_release_view(complete))) == []
    for missing in ("change_reference", "rollback_plan"):
        payload = dict(complete, **{missing: ""})
        findings = check(ReleaseDocumentation, run_subject(_release_view(payload)))
        assert [(f.rule_id, f.severity, f.evidence) for f in findings] == [
            ("CHG-005", Severity.HIGH, missing)
        ]


@pytest.mark.parametrize(
    "placeholder", ["", "  ", "TBD", "n/a", "None", "todo", "-", "?", 7, None, ["x"]]
)
def test_chg005_rejects_placeholders_and_wrong_types(placeholder: object) -> None:
    payload = {"change_reference": placeholder, "rollback_plan": "revert"}
    assert [
        f.evidence for f in check(ReleaseDocumentation, run_subject(_release_view(payload)))
    ] == ["change_reference"]


def test_chg005_missing_or_unreadable_design_document_fails_closed() -> None:
    assert check(ReleaseDocumentation, run_subject(FakeRunView(release_chain())))
    assert check(ReleaseDocumentation, run_subject(_release_view(ValueError("boom"))))
    assert check(ReleaseDocumentation, run_subject(_release_view(["not", "an", "object"])))
    unknown_gate = run_subject(
        _release_view({"change_reference": "X-1", "rollback_plan": "r"}), "no-such-node"
    )
    assert check(
        ReleaseDocumentation, unknown_gate
    )  # unknown position => no ancestors => no evidence


def test_chg005_only_artifacts_of_ancestors_count() -> None:
    payload = {"change_reference": "CHG-1", "rollback_plan": "revert"}
    sibling = FakeRunView([*release_chain(), node("docs", depends_on=("requirements",))])
    sibling.add_artifact("design_doc", payload, producer="docs")
    assert check(
        ReleaseDocumentation, run_subject(sibling)
    )  # 'docs' is not an ancestor of the gate
    descendant = FakeRunView([*release_chain(), node("promote", depends_on=("release-readiness",))])
    descendant.add_artifact("design_doc", payload, producer="promote")
    assert check(ReleaseDocumentation, run_subject(descendant))


def test_chg005_staged_artifact_of_the_gated_attempt_counts() -> None:
    payload = {"change_reference": "CHG-1", "rollback_plan": "revert"}
    staged = _release_view(payload, producer="release-readiness", status=ArtifactStatus.STAGED)
    assert check(ReleaseDocumentation, run_subject(staged)) == []
    stale = _release_view(payload, producer="design", status=ArtifactStatus.SUPERSEDED)
    assert check(ReleaseDocumentation, run_subject(stale))


def test_chg005_empty_gate_position_considers_the_whole_run() -> None:
    payload = {"change_reference": "CHG-1", "rollback_plan": "revert"}
    assert check(ReleaseDocumentation, run_subject(_release_view(payload), "")) == []


# ===============================================================================================
# CHG-006


def _graph() -> dict[str, NodeSpec]:
    return {
        "decompose": node("decompose", stage=Stage.PLANNING),
        "verify": node("verify", depends_on=("decompose",), mandatory=True, stage=Stage.TESTING),
        "integrate": node("integrate", depends_on=("decompose",)),
    }


def test_chg006_unchanged_mandatory_nodes_pass() -> None:
    before = _graph()
    expansion = PlanExpansion("decompose", (node("implement.a", depends_on=("decompose",)),))
    assert check(MandatoryNodesPinned, plan_subject(expansion, before, mandatory=("verify",))) == []


def test_chg006_removed_or_altered_mandatory_node_is_high() -> None:
    before = _graph()
    expansion = PlanExpansion("decompose")
    removed = {k: v for k, v in before.items() if k != "verify"}
    findings = check(MandatoryNodesPinned, plan_subject(expansion, before, removed))
    assert [(f.rule_id, f.severity, f.location) for f in findings] == [
        ("CHG-006", Severity.HIGH, "verify")
    ]
    altered = dict(
        before,
        verify=node(
            "verify", depends_on=("decompose",), mandatory=True, stage=Stage.TESTING, agent="other"
        ),
    )
    assert check(MandatoryNodesPinned, plan_subject(expansion, before, altered))


def test_chg006_mandatory_flag_on_the_spec_is_enough_and_unknown_ids_fail_closed() -> None:
    before = _graph()
    expansion = PlanExpansion("decompose")
    removed = {k: v for k, v in before.items() if k != "verify"}
    assert check(MandatoryNodesPinned, plan_subject(expansion, before, removed, mandatory=()))
    unknown = check(MandatoryNodesPinned, plan_subject(expansion, before, mandatory=("ghost",)))
    assert [f.location for f in unknown] == ["ghost"]


def test_chg006_rewiring_a_mandatory_node_is_flagged() -> None:
    before = _graph()
    expansion = PlanExpansion("decompose", (), {"verify": Rewire(add_depends_on=("integrate",))})
    findings = check(MandatoryNodesPinned, plan_subject(expansion, before, mandatory=("verify",)))
    assert [f.message for f in findings] == ["plan rewires a mandatory node"]


# ===============================================================================================
# CHG-007


def _owner_graph() -> dict[str, NodeSpec]:
    return {
        "decompose": node("decompose", stage=Stage.PLANNING),
        "integrate": node("integrate", depends_on=("decompose",)),
    }


def test_chg007_conforming_expansion_passes() -> None:
    expansion = PlanExpansion(
        "decompose",
        (
            node("implement.a", depends_on=("decompose",)),
            node("implement.b", depends_on=("decompose",)),
        ),
        {"integrate": Rewire(add_depends_on=("implement.a", "implement.b"))},
    )
    assert check(PlanConformance, plan_subject(expansion, _owner_graph())) == []


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        (node("implement.a", depends_on=("decompose",), agent="rogue"), "agent 'rogue'"),
        (
            node("implement.a", depends_on=("decompose",), agent="agent-a", fallback_agent="rogue"),
            "agent 'rogue'",
        ),
        (node("implement.a", depends_on=("decompose",), stage=Stage.RELEASE), "stage 'release'"),
        (
            node("implement.a", depends_on=("decompose",), produces=("release_bundle",)),
            "release_bundle",
        ),
        (node("implement.a", depends_on=()), "does not depend on the expansion owner"),
        (node("implement.a", depends_on=("decompose",), mandatory=True), "'mandatory'"),
        (
            node(
                "implement.a",
                depends_on=("decompose",),
                side_effecting=True,
                compensator="apply_bundle",
            ),
            "'side_effecting'",
        ),
    ],
)
def test_chg007_plan_node_violations(spec: NodeSpec, fragment: str) -> None:
    expansion = PlanExpansion("decompose", (spec,))
    findings = check(PlanConformance, plan_subject(expansion, _owner_graph()))
    assert findings and all(
        f.rule_id == "CHG-007" and f.severity == Severity.HIGH for f in findings
    )
    assert any(fragment in f.message for f in findings)


def test_chg007_empty_allow_lists_allow_nothing() -> None:
    expansion = PlanExpansion("decompose", (node("implement.a", depends_on=("decompose",)),))
    subject = plan_subject(
        expansion,
        _owner_graph(),
        allowed_agents=frozenset(),
        allowed_produces=frozenset(),
        allowed_stages=frozenset(),
    )
    assert len(check(PlanConformance, subject)) == 3


def test_chg007_node_count_and_unknown_owner() -> None:
    cfg = PolicyConfig(expansion_rules=ExpansionRules(max_new_nodes=1))
    nodes = tuple(node(f"implement.{c}", depends_on=("decompose",)) for c in "ab")
    findings = check(
        PlanConformance, plan_subject(PlanExpansion("decompose", nodes), _owner_graph()), cfg
    )
    assert any("limit 1" in f.message for f in findings)
    ghost = check(PlanConformance, plan_subject(PlanExpansion("ghost"), _owner_graph()))
    assert any("owner" in f.message for f in ghost)


@pytest.mark.parametrize(
    ("target", "rewire", "fragment"),
    [
        ("nothing", Rewire(add_depends_on=("decompose",)), "does not exist"),
        ("decompose", Rewire(add_depends_on=("integrate",)), "own owner"),
        ("verify", Rewire(add_depends_on=("decompose",)), "outside the allowed targets"),
        ("integrate", Rewire(add_depends_on=("ghost",)), "unknown node 'ghost'"),
        ("integrate", Rewire(add_overlay_from=("request",)), "non-change-set kind"),
    ],
)
def test_chg007_rewire_violations(target: str, rewire: Rewire, fragment: str) -> None:
    before = {
        **_owner_graph(),
        "verify": node("verify", depends_on=("decompose",), stage=Stage.TESTING),
    }
    expansion = PlanExpansion("decompose", (), {target: rewire})
    findings = check(PlanConformance, plan_subject(expansion, before))
    assert any(fragment in f.message for f in findings), [f.message for f in findings]


def test_chg007_rewire_allow_list_is_configurable() -> None:
    before = {
        **_owner_graph(),
        "verify": node("verify", depends_on=("decompose",), stage=Stage.TESTING),
    }
    expansion = PlanExpansion("decompose", (), {"verify": Rewire(add_depends_on=("integrate",))})
    cfg = PolicyConfig(expansion_rules=ExpansionRules(rewire_targets=("verify",)))
    assert check(PlanConformance, plan_subject(expansion, before), cfg) == []


def test_chg007_integration_must_be_derivable_from_its_sources() -> None:
    src_a = ChangeSet((create("a.py", "a = 1\n"),), "a", "agent-a")
    src_b = ChangeSet((create("b.py", "b = 1\n"),), "b", "agent-b")
    good = cs_subject(
        [create("a.py", "a = 1\n"), create("b.py", "b = 1\n")], sources=[src_a, src_b]
    )
    assert check(PlanConformance, good) == []
    invented = cs_subject(
        [create("a.py", "a = 1\n"), create("c.py", "c = 1\n")], sources=[src_a, src_b]
    )
    assert [f.location for f in check(PlanConformance, invented)] == ["c.py"]
    tampered = cs_subject([create("a.py", "a = 999\n")], sources=[src_a])
    findings = check(PlanConformance, tampered)
    assert [f.severity for f in findings] == [Severity.HIGH]
    assert "not derivable" in findings[0].message


def test_chg007_sources_provenance_is_case_insensitive_and_covers_deletes() -> None:
    base = view({"gone.py": "x\n"})
    src = ChangeSet((delete(base, "gone.py"),), "d", "agent")
    assert check(PlanConformance, cs_subject([delete(base, "gone.py")], base, sources=[src])) == []
    sneaky = ChangeSet((create("Docs/A.md", "x\n"),), "d", "agent")
    assert check(PlanConformance, cs_subject([create("docs/a.md", "x\n")], sources=[sneaky])) == []
    # a source DELETE cannot justify a CREATE with content
    assert check(PlanConformance, cs_subject([create("gone.py", "evil\n")], sources=[src]))


def test_chg007_changeset_without_sources_is_not_checked() -> None:
    assert check(PlanConformance, cs_subject([create("anything.py", "x\n")])) == []


# ===============================================================================================
# CHG-008


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        ".github/CODEOWNERS",
        "Dockerfile",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        ".env",
        ".env.production",
        "certs/server.pem",
        "keys/id.key",
        "id.key",
    ],
)
def test_chg008_protected_paths_are_high(path: str) -> None:
    findings = check(ProtectedPaths, cs_subject([create(path, "x\n")]))
    assert [(f.rule_id, f.severity, f.location) for f in findings] == [
        ("CHG-008", Severity.HIGH, path)
    ]


@pytest.mark.parametrize(
    "path",
    [
        "dockerfile",
        "DOCKERFILE",
        "PyProject.toml",
        ".GITHUB/workflows/x.yml",
        "REQUIREMENTS.TXT",
        "A/B.PEM",
        ".ENV",
    ],
)
def test_chg008_case_variants_cannot_bypass_on_case_insensitive_file_systems(path: str) -> None:
    assert check(ProtectedPaths, cs_subject([create(path, "x\n")]))


@pytest.mark.parametrize(
    "path", ["shortener/api.py", "docs/Dockerfile.md", "src/requirements.py", "env.py", "README.md"]
)
def test_chg008_ordinary_paths_pass(path: str) -> None:
    assert check(ProtectedPaths, cs_subject([create(path, "x\n")])) == []


def test_chg008_modify_and_delete_are_protected_too() -> None:
    base = view({"Dockerfile": "FROM x\n", "pyproject.toml": "[x]\n"})
    ops = [modify(base, "Dockerfile", "FROM y\n"), delete(base, "pyproject.toml")]
    assert len(check(ProtectedPaths, cs_subject(ops, base))) == 2


def test_chg008_config_can_extend_but_the_extension_is_enforced() -> None:
    cfg = PolicyConfig(protected_paths=(*PolicyConfig().protected_paths, "deploy/**"))
    assert check(ProtectedPaths, cs_subject([create("deploy/prod.yml", "x\n")]), cfg)
    assert check(ProtectedPaths, cs_subject([create("deploy/prod.yml", "x\n")])) == []


def test_chg008_git_internals_are_rejected_before_policy_even_runs() -> None:
    with pytest.raises(PathViolation):
        create(".git/config", "x\n")
