"""Tests of the ``max_clicks`` recipe: anchored edits of the REAL shortener, end to end.

The expensive fixtures snapshot the real ``shortener/`` and ``tests/shortener/`` trees, build every
group, the tests and the docs into change sets, merge them (zero conflicts expected) and, in the
``slow`` tests, run the complete product suite - existing tests plus the generated ones - in the
real sandbox.  Deviation from the scenario's include list: ``tests/__init__.py``, ``docs/api.md``
and ``docs/product-notes.md`` are snapshotted too, because the product suite cannot even be
collected without ``tests/__init__.py`` and two of its tests read the two docs files.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents import catalog
from orchestrator.agents.base import split_sentences
from orchestrator.agents.recipes import max_clicks as mc
from orchestrator.agents.recipes.base import (
    BuildContext,
    GroupSpec,
    Recipe,
    RecipeAnchorError,
    RecipeParamError,
    validate_params,
)
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentError, ToolDenied
from orchestrator.core.paths import match_any, match_glob
from orchestrator.core.schemas import GROUP_ID_RE, Kind, validate_payload
from orchestrator.core.types import (
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    ChangeSetSubject,
    Enforcement,
    FileOpKind,
    FileView,
    RiskLevel,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.changeset import diff_stats, digest, merge_changesets
from orchestrator.tools.codebase import CodebaseModel, analyze_codebase, impact_analysis
from orchestrator.tools.openapi import diff_openapi
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.toolbox import ScopedToolbox
from orchestrator.tools.workspace import BaselineSnapshot, MemoryView, OverlayView, materialize

ROOT = Path(__file__).resolve().parents[3]
INCLUDE = [
    "shortener/**",
    "tests/shortener/**",
    "tests/__init__.py",
    "docs/api.md",
    "docs/product-notes.md",
    "pyproject.toml",
]
PARAMS: dict[str, Any] = {"min_value": 1, "max_value": 1_000_000}
REQUEST = (ROOT / "orchestrator" / "scenarios" / "brownfield" / "request.md").read_text(
    encoding="utf-8"
)
MIGRATION = "shortener/storage/migrations/003_add_max_clicks.sql"
REPOSITORY = "shortener/storage/link_repository.py"
STORAGE_TESTS = "tests/shortener/test_shortener_storage.py"
NEW_TESTS = "tests/shortener/test_max_clicks.py"
EXPECTED_PATHS = {
    "storage": {REPOSITORY, MIGRATION, STORAGE_TESTS},
    "domain-api": {
        "shortener/domain/models.py",
        "shortener/domain/errors.py",
        "shortener/domain/validation.py",
        "shortener/api/schemas.py",
        "shortener/api/routes_links.py",
    },
    "service": {"shortener/services/link_service.py", "shortener/services/click_recorder.py"},
}


def requirements(*aspects: tuple[str, str, str], status: str = "active") -> dict[str, Any]:
    """A requirements payload: ``(requirement id, criterion id, aspect)`` per entry."""
    items: dict[str, dict[str, Any]] = {}
    for rid, cid, aspect in aspects:
        entry = items.setdefault(rid, {"id": rid, "status": status, "acceptance_criteria": []})
        entry["acceptance_criteria"].append({"id": cid, "text": "x y z w", "aspect": aspect})
    return {"requirements": list(items.values())}


ALL_ASPECTS = requirements(
    ("R-001", "AC-001-1", "optional-field"),
    ("R-001", "AC-001-2", "range-validation"),
    ("R-002", "AC-002-1", "gone-after-limit"),
    ("R-003", "AC-003-1", "backward-compatible"),
    ("R-004", "AC-004-1", "remaining-visible"),
    ("R-005", "AC-005-1", "concurrency-safe"),
)


def _audit(*args: Any, **kwargs: Any) -> None:
    return None


ENGINE = PolicyEngine()


def toolbox(view: FileView, globs: Sequence[str], node: str = "implement.max_clicks-x") -> Any:
    return ScopedToolbox(
        run_id="run-1",
        node_id=node,
        attempt=1,
        agent_id="implementer",
        profile=AutonomyProfile(write_globs=tuple(globs), max_changed_files=40),
        view=view,
        cancel=CancelToken(),
        clock=FakeClock(),
        policy=ENGINE,
        audit=_audit,
    )


def memory_view(files: Mapping[str, str | None]) -> MemoryView:
    return MemoryView({path: text for path, text in files.items() if text is not None})


@dataclass
class Built:
    base: FileView
    model: CodebaseModel
    groups: tuple[GroupSpec, ...]
    changesets: dict[str, ChangeSet]
    anchors: dict[str, list[str]]
    tests: ChangeSet
    docs: ChangeSet
    merged: ChangeSet
    integrated: OverlayView
    final: OverlayView

    def everything(self) -> list[ChangeSet]:
        return [*self.changesets.values(), self.tests, self.docs]


def build_all(base: FileView, model: CodebaseModel | None, params: Mapping[str, Any]) -> Built:
    recipe = mc.CAPABILITY.recipe
    groups = recipe.groups(params)
    changesets: dict[str, ChangeSet] = {}
    anchors: dict[str, list[str]] = {}
    for group in groups:
        box = toolbox(base, group.write_scope, f"implement.max_clicks-{group.id}")
        ctx = BuildContext(toolbox=box, params=params, model=model)
        recipe.build_group(group.id, ctx)
        changesets[group.id] = box.staged_changeset(group.id)
        anchors[group.id] = list(ctx.anchors.get("recipe_anchors", []))
    box = toolbox(base, ["tests/**"], "author-tests")
    recipe.build_tests(
        BuildContext(toolbox=box, params=params, model=model, requirements=ALL_ASPECTS)
    )
    tests = box.staged_changeset("tests")
    merged = merge_changesets(base, [*changesets.values(), tests])
    assert merged.conflicts == ()
    integrated = OverlayView(base, [merged.changeset])
    box = toolbox(integrated, ["docs/**", "CHANGELOG.md", "README.md"], "docs")
    recipe.build_docs(BuildContext(toolbox=box, params=params, model=model, design=None))
    docs = box.staged_changeset("docs")
    final = OverlayView(base, [merged.changeset, docs])
    return Built(
        base, model or analyze_codebase(base), groups, changesets, anchors, tests, docs,
        merged.changeset, integrated, final,
    )  # fmt: skip


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> BaselineSnapshot:
    return BaselineSnapshot.create(ROOT, tmp_path_factory.mktemp("baseline"), include=INCLUDE)


@pytest.fixture(scope="module")
def base(snapshot: BaselineSnapshot) -> FileView:
    return snapshot.view


@pytest.fixture(scope="module")
def files(base: FileView) -> dict[str, str]:
    return {path: base.read_text(path) for path in base.list_files("**")}


@pytest.fixture(scope="module")
def model(base: FileView) -> CodebaseModel:
    return analyze_codebase(base)


@pytest.fixture(scope="module")
def built(base: FileView, model: CodebaseModel) -> Built:
    return build_all(base, model, PARAMS)


def context(view: FileView, globs: Sequence[str], **extra: Any) -> BuildContext:
    params = extra.pop("params", PARAMS)
    return BuildContext(toolbox=toolbox(view, globs), params=params, **extra)


# ============================================================================ catalog and matching


def test_capability_is_registered_and_implements_the_recipe_protocol() -> None:
    capability = catalog.get_capability("max_clicks")
    assert capability is mc.CAPABILITY
    assert isinstance(capability.recipe, Recipe)
    assert (capability.kind, capability.requires_baseline, capability.risk) == (
        "enhancement",
        True,
        RiskLevel.HIGH,
    )
    assert [a.id for a in capability.aspects] == [
        "optional-field",
        "range-validation",
        "gone-after-limit",
        "backward-compatible",
        "remaining-visible",
        "concurrency-safe",
    ]


def test_brownfield_request_matches_and_yields_the_parameters() -> None:
    matches = catalog.match_capabilities(REQUEST)
    assert matches[0].capability.id == "max_clicks"
    assert matches[0].score >= 3
    extracted = catalog.extract_params(mc.CAPABILITY, REQUEST)
    assert validate_params(mc.PARAMS, extracted) == PARAMS


@pytest.mark.parametrize("scenario", ["greenfield", "ambiguous"])
def test_other_scenarios_never_match_this_capability(scenario: str) -> None:
    folder = ROOT / "orchestrator" / "scenarios" / scenario
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md")))
    assert "max_clicks" not in {m.capability.id for m in catalog.match_capabilities(text)}


def test_every_requirement_sentence_of_the_request_hits_its_aspects() -> None:
    body = "\n".join(line for line in REQUEST.split("\n") if not line.strip().startswith("#"))
    hits = {
        sentence: {
            aspect.id
            for aspect in mc.ASPECTS
            if any(re.search(t, sentence, re.IGNORECASE) for t in aspect.triggers)
        }
        for sentence in split_sentences(body)
    }

    def of(prefix: str) -> set[str]:
        return next(found for sentence, found in hits.items() if sentence.startswith(prefix))

    assert of("When creating") == {"optional-field", "range-validation"}
    assert of("After a link") == {"gone-after-limit"}
    assert of("Existing links") == {"backward-compatible"}
    assert of("The remaining") == {"remaining-visible"}
    assert of("The limit must") == {"concurrency-safe"}
    assert of("Marketing wants") == set()  # intent statement: scope, not a requirement
    assert set().union(*hits.values()) == set(mc.ASPECT_IDS)


def test_criteria_format_with_the_parameters_and_are_measurable() -> None:
    gates = pytest.importorskip("orchestrator.workflows.gates_sdlc")
    for aspect in mc.ASPECTS:
        text = aspect.criterion.format_map(PARAMS)
        assert gates.is_measurable(text), text
    assert "1,000,000" in mc.ASPECTS[1].criterion.format_map(PARAMS)


def test_patterns_stay_linear_on_hostile_input() -> None:
    hostile = [
        "a" * 200_000,
        "max_clicks " * 20_000,
        "1," * 100_000,
        "optional " * 30_000 + "x",
        "between " + "9" * 5_000 + " and ",
        "followed " + "x " * 30_000,
    ]
    patterns = [
        *mc.CAPABILITY.triggers,
        *mc.CAPABILITY.extractors.values(),
        *(t for aspect in mc.ASPECTS for t in aspect.triggers),
    ]
    started = time.perf_counter()
    for pattern in patterns:
        for text in hostile:
            re.search(pattern, text, re.IGNORECASE)
    assert time.perf_counter() - started < 3.0


# ============================================================================ parameters


HOSTILE_PARAMS: list[dict[str, Any]] = [
    {"min_value": True},
    {"min_value": "1"},
    {"min_value": 0},
    {"min_value": -5},
    {"max_value": 2**40},
    {"max_value": 1.5},
    {"min_value": None},
    {"min_value": 5, "max_value": 2},
    {"extra": 1},
    {"max_value": "1; DROP TABLE links"},
]


@pytest.mark.parametrize("params", HOSTILE_PARAMS, ids=lambda p: str(p)[:40])
def test_hostile_parameters_are_rejected_by_every_entry_point(params: dict[str, Any]) -> None:
    recipe = mc.CAPABILITY.recipe
    box = MemoryView({})
    calls: list[Callable[[], object]] = [
        lambda: recipe.groups(params),
        lambda: recipe.design(params, None),
        lambda: recipe.build_group("storage", context(box, ["**"], params=params)),
        lambda: recipe.build_tests(
            context(box, ["tests/**"], params=params, requirements=ALL_ASPECTS)
        ),
        lambda: recipe.build_docs(context(box, ["docs/**"], params=params)),
    ]
    for call in calls:
        with pytest.raises(RecipeParamError):
            call()


def test_custom_range_reaches_every_generated_artifact(
    base: FileView, model: CodebaseModel
) -> None:
    custom = {"min_value": 5, "max_value": 50}
    built = build_all(base, model, custom)
    migration = _content(built.changesets["storage"], MIGRATION)
    assert "max_clicks >= 5" in migration
    validation = _content(built.changesets["domain-api"], "shortener/domain/validation.py")
    assert "MIN_CLICK_LIMIT = 5" in validation and "MAX_CLICK_LIMIT = 50" in validation
    assert "MIN_CLICKS = 5" in _content(built.tests, NEW_TESTS)
    assert "MAX_CLICKS = 50" in _content(built.tests, NEW_TESTS)
    assert "from 5 to" in _content(built.docs, "docs/features/max-clicks.md")


def _content(changeset: ChangeSet, path: str) -> str:
    return next(op.content or "" for op in changeset.ops if op.path == path)


# ============================================================================ groups and design


def test_groups_are_disjoint_valid_and_cover_their_expected_paths() -> None:
    groups = mc.CAPABILITY.recipe.groups(PARAMS)
    assert len(groups) >= 3
    assert len({g.id for g in groups}) == len(groups)
    owners: dict[str, str] = {}
    for group in groups:
        assert GROUP_ID_RE.match(group.id)
        assert group.depends_on == ()
        for expected in group.expected_paths:
            concrete = expected.replace("*_", "003_")
            assert match_any(concrete, group.write_scope), (group.id, expected)
            assert owners.setdefault(concrete, group.id) == group.id
    for group in groups:
        for other in groups:
            if other is not group:
                assert not any(
                    match_any(p, group.write_scope) for p, o in owners.items() if o == other.id
                )


def test_design_is_complete_valid_and_deterministic(model: CodebaseModel) -> None:
    recipe = mc.CAPABILITY.recipe
    design = recipe.design(PARAMS, model)
    assert design == recipe.design(PARAMS, model)
    payload = {
        "schema": 1,
        "summary": "click limits",
        "components": design["components"],
        "adrs": design["adrs"],
        "data_model_changes": design["data_model_changes"],
        "api_changes": design["api_changes"],
        "rollback_plan": design["rollback_plan"],
        "change_reference": "CR-000000",
        "risk": {"level": "HIGH", "reasons": ["data model change"]},
    }
    assert validate_payload(Kind.DESIGN_DOC, payload) == []
    contract = {"schema": 1, "paths": design["api_paths"], "breaking": False}
    assert validate_payload(Kind.API_CONTRACT, contract) == []
    assert design["data_model_changes"][0]["migration"] == MIGRATION
    assert all(change["breaking"] is False for change in design["api_changes"])
    assert {(c["method"], c["path"]) for c in design["api_changes"]} >= {
        ("POST", "/api/v1/links"),
        ("GET", "/{code}"),
    }
    touched = {path for component in design["components"] for path in component["touches"]}
    assert touched >= {REPOSITORY, MIGRATION, "shortener/services/link_service.py"}


def test_adrs_compare_real_alternatives_for_every_enforcement_point() -> None:
    adrs = mc.CAPABILITY.recipe.design(PARAMS, None)["adrs"]
    assert len(adrs) >= 4
    for adr in adrs:
        assert len(adr["alternatives"]) >= 2
        assert all(alt["option"] and alt["rejected_because"] for alt in adr["alternatives"])
    options = " ".join(alt["option"].lower() for adr in adrs for alt in adr["alternatives"])
    for needle in ("in-memory", "eventually consistent", "compare in python", "second counter"):
        assert needle in options
    titles = " ".join(adr["title"].lower() for adr in adrs)
    assert "atomic" in titles and "twice" in titles and "cache" in titles


def test_rollback_plan_is_honest_about_sqlite_and_the_migration_runner() -> None:
    plan = mc.CAPABILITY.recipe.design(PARAMS, None)["rollback_plan"]
    for needle in ("forward", "DROP COLUMN", "3.35", "downgrades", "backup"):
        assert needle in plan


def test_design_without_a_model_uses_the_conventional_layout(model: CodebaseModel) -> None:
    recipe = mc.CAPABILITY.recipe
    assert recipe.design(PARAMS, None) == recipe.design(PARAMS, model)
    assert recipe.design(PARAMS, {"modules": []}) == recipe.design(PARAMS, None)  # foreign model


def test_design_detects_drift_through_the_model(
    files: dict[str, str], built: Built, model: CodebaseModel
) -> None:
    recipe = mc.CAPABILITY.recipe
    applied = analyze_codebase(built.final)
    with pytest.raises(RecipeAnchorError, match="already"):
        recipe.design(PARAMS, applied)
    without = analyze_codebase(memory_view({**files, "shortener/services/click_recorder.py": None}))
    with pytest.raises(RecipeAnchorError, match="click_recorder"):
        recipe.design(PARAMS, without)
    renamed = files["shortener/services/link_service.py"].replace("def _lookup(", "def _find(")
    drifted = analyze_codebase(
        memory_view({**files, "shortener/services/link_service.py": renamed})
    )
    with pytest.raises(RecipeAnchorError, match="_lookup"):
        recipe.design(PARAMS, drifted)


# ============================================================================ building the change


def test_group_change_sets_touch_exactly_the_planned_files(built: Built) -> None:
    for group_id, paths in EXPECTED_PATHS.items():
        assert set(built.changesets[group_id].paths()) == paths
    every = [path for cs in built.changesets.values() for path in cs.paths()]
    assert len(every) == len(set(every))  # one owner per file: change sets can never overlap


def test_every_group_stays_inside_its_write_scope(built: Built) -> None:
    for group in built.groups:
        assert all(match_any(p, group.write_scope) for p in built.changesets[group.id].paths())
        assert set(built.changesets[group.id].paths()) <= {
            p.replace("*_", "003_") for p in group.expected_paths
        }


def test_merge_is_conflict_free_and_complete(built: Built, base: FileView) -> None:
    merged = merge_changesets(base, built.everything())
    assert merged.conflicts == ()
    assert set(merged.changeset.paths()) == set(built.merged.paths()) | set(built.docs.paths())


def test_applied_migrations_stay_untouched_and_the_new_one_is_a_single_nullable_column(
    built: Built,
) -> None:
    from shortener.storage.migrations import split_statements

    ops = {op.path: op for cs in built.everything() for op in cs.ops}
    assert not [p for p in ops if "migrations/001" in p or "migrations/002" in p]
    assert ops[MIGRATION].kind == FileOpKind.CREATE
    statements = split_statements(ops[MIGRATION].content or "")
    assert len(statements) == 1
    assert statements[0].startswith("-- 003_add_max_clicks")
    assert "ALTER TABLE links ADD COLUMN max_clicks INTEGER CHECK" in statements[0]
    assert "NOT NULL" not in statements[0]


def test_existing_files_are_modified_and_new_files_created(built: Built, base: FileView) -> None:
    for cs in built.everything():
        for op in cs.ops:
            assert (op.kind == FileOpKind.MODIFY) == base.exists(op.path), op.path
    created = {
        op.path for cs in built.everything() for op in cs.ops if op.kind == FileOpKind.CREATE
    }
    assert created == {MIGRATION, NEW_TESTS, "CHANGELOG.md", "docs/features/max-clicks.md"}


def test_all_generated_python_compiles(built: Built) -> None:
    checked = 0
    for cs in built.everything():
        for op in cs.ops:
            if op.path.endswith(".py"):
                ast.parse(op.content or "", filename=op.path)
                checked += 1
    assert checked >= 9


def test_recorded_anchors_are_the_modified_existing_files(built: Built, base: FileView) -> None:
    for group_id, anchors in built.anchors.items():
        modified = {
            op.path for op in built.changesets[group_id].ops if op.kind == FileOpKind.MODIFY
        }
        assert set(anchors) == modified


def test_modified_files_lie_inside_the_impact_set(built: Built, model: CodebaseModel) -> None:
    report = impact_analysis(model, mc.CAPABILITY.impact_keywords)
    allowed = {m.path for m in report.modules} | set(report.expected_paths) | set(report.tests)
    for cs in [*built.changesets.values(), built.tests]:
        for op in cs.ops:
            if op.kind == FileOpKind.MODIFY:
                assert any(match_glob(op.path, pattern) for pattern in allowed), op.path


def test_generation_is_deterministic_and_free_of_environment_data(
    built: Built, base: FileView, model: CodebaseModel
) -> None:
    again = build_all(base, model, PARAMS)
    assert [digest(cs) for cs in built.everything()] == [digest(cs) for cs in again.everything()]
    without_model = build_all(base, None, PARAMS)
    assert [digest(cs) for cs in built.everything()] == [
        digest(cs) for cs in without_model.everything()
    ]
    leaked = re.compile(r"[A-Za-z]:\\|(?<![\w.])/(?:home|Users|tmp|var)/|\bpid\b", re.IGNORECASE)
    for cs in built.everything():
        for op in cs.ops:
            assert not leaked.search(op.content or ""), op.path
            assert str(ROOT) not in (op.content or "")


def test_safe_agent_flag_is_accepted_and_changes_nothing(base: FileView, built: Built) -> None:
    group = built.groups[0]
    box = toolbox(base, group.write_scope)
    ctx = BuildContext(toolbox=box, params=PARAMS, anchors={"conservative": True})
    mc.CAPABILITY.recipe.build_group(group.id, ctx)
    assert digest(box.staged_changeset("x")) == digest(built.changesets[group.id])


def test_unknown_group_and_scope_violations_are_refused(base: FileView) -> None:
    recipe = mc.CAPABILITY.recipe
    with pytest.raises(RecipeParamError, match="unknown group"):
        recipe.build_group("../etc", context(base, ["**"]))
    narrow = context(base, ["shortener/storage/**"])  # lacks the storage tests file
    with pytest.raises(ToolDenied):
        recipe.build_group("storage", narrow)


def test_generated_change_sets_pass_the_policy_engine(
    built: Built, model: CodebaseModel, files: dict[str, str]
) -> None:
    impact = impact_analysis(model, mc.CAPABILITY.impact_keywords).to_dict()
    scopes = {g.id: g.write_scope for g in built.groups}
    docs_profile = AutonomyProfile(write_globs=("docs/**", "CHANGELOG.md"))
    # the scenario baseline has no docs/api.md: the docs node then only creates new files
    without_api = OverlayView(
        memory_view({p: t for p, t in files.items() if p != "docs/api.md"}), [built.merged]
    )
    box = toolbox(without_api, ["docs/**", "CHANGELOG.md"], "docs")
    mc.CAPABILITY.recipe.build_docs(BuildContext(toolbox=box, params=PARAMS))
    subjects = [
        (cs, AutonomyProfile(write_globs=scopes[gid]), built.base)
        for gid, cs in built.changesets.items()
    ]
    subjects.append((built.tests, AutonomyProfile(write_globs=("tests/**",)), built.base))
    subjects.append((box.staged_changeset("docs"), docs_profile, without_api))
    for cs, profile, view in subjects:
        subject = ChangeSetSubject(
            changeset=cs, baseline=view, node_id="n", author="implementer",
            impact=impact, profile=profile,
        )  # fmt: skip
        report = ENGINE.evaluate(subject)
        assert report.enforcement in (Enforcement.ALLOW, Enforcement.WARN), [
            (f.rule_id, f.message) for f in report.findings
        ]
    # editing the pre-existing generated docs/api.md is the one thing that needs a human
    with_api = ChangeSetSubject(
        changeset=built.docs, baseline=built.integrated, node_id="n", author="documenter",
        impact=impact, profile=docs_profile,
    )  # fmt: skip
    report = ENGINE.evaluate(with_api)
    assert {(f.rule_id, f.location) for f in report.findings} <= {("CHG-003", "docs/api.md")}
    storage = ChangeSetSubject(
        changeset=built.changesets["storage"], baseline=built.base, node_id="n",
        author="implementer", impact=impact, profile=None,
    )  # fmt: skip
    assert ENGINE.assess_risk(storage).level == RiskLevel.HIGH  # a migration


def test_integrated_change_set_fits_the_change_budget_without_asking_for_approval(
    built: Built, model: CodebaseModel
) -> None:
    budget = PolicyConfig().change_budget
    stats = diff_stats(built.base, built.merged)
    assert stats.files_added + stats.files_modified + stats.files_deleted <= budget.max_files
    assert stats.lines_added + stats.lines_removed <= budget.max_lines - 50  # head-room
    subject = ChangeSetSubject(
        changeset=built.merged,
        baseline=built.base,
        node_id="integrate",
        author="integrator",
        impact=impact_analysis(model, mc.CAPABILITY.impact_keywords).to_dict(),
        profile=AutonomyProfile(write_globs=("**",), max_changed_files=40),
        sources=(*built.changesets.values(), built.tests),
    )
    report = ENGINE.evaluate(subject)
    assert report.enforcement in (Enforcement.ALLOW, Enforcement.WARN), [
        (f.rule_id, f.message) for f in report.findings
    ]


def test_generated_product_files_are_lint_and_format_clean(built: Built, tmp_path: Path) -> None:
    ruff = shutil.which("ruff") or str(Path(sys.executable).with_name("ruff"))
    if not Path(ruff).exists() and not Path(ruff + ".exe").exists():
        pytest.skip("ruff is not installed")
    import subprocess

    materialize(built.final, tmp_path)
    paths = [
        op.path
        for cs in built.everything()
        for op in cs.ops
        if op.path.endswith(".py") and op.path != STORAGE_TESTS
    ]
    for argv in ([ruff, "check", "--no-cache", *paths], [ruff, "format", "--check", *paths]):
        done = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]


# ============================================================================ anchor drift


def _group_of(path: str) -> str:
    return next(group for group, paths in EXPECTED_PATHS.items() if path in paths)


MODELS = "shortener/domain/models.py"
ERRORS = "shortener/domain/errors.py"
VALIDATION = "shortener/domain/validation.py"
SCHEMAS = "shortener/api/schemas.py"
ROUTES = "shortener/api/routes_links.py"
SERVICE = "shortener/services/link_service.py"
RECORDER = "shortener/services/click_recorder.py"

DRIFT = [
    (
        REPOSITORY,
        '"UPDATE links SET click_count = click_count + ? WHERE id = ?"',
        '"UPDATE links SET n = n + ? WHERE id = ?"',
    ),
    (REPOSITORY, "def _parse_optional(text", "def _parse_optional_text(text"),
    (STORAGE_TESTS, "== [1, 2]\n    assert [m.name", "== [1, 2, 3]\n    assert [m.name"),
    (MODELS, "def status_at(self, now: datetime)", "def status_on(self, now: datetime)"),
    (ERRORS, "class InvalidExpiryError(", "class ExpiryError("),
    (VALIDATION, " idempotency\n\n\ndef validate_", " idempotency-keys\n\n\ndef validate_"),
    (SCHEMAS, "examples=[3600],", "examples=[7200],"),
    (
        ROUTES,
        'request_fingerprint(body.model_dump(mode="json"))',
        "request_fingerprint(body.model_dump())",
    ),
    (SERVICE, "def _lookup(self, code: str)", "def _find(self, code: str)"),
    (RECORDER, "the limited-click implications", "something else"),
]


@pytest.mark.parametrize(("path", "old", "new"), DRIFT, ids=[d[0].rsplit("/", 1)[1] for d in DRIFT])
def test_a_drifted_anchor_raises_recipe_anchor_error(
    files: dict[str, str], path: str, old: str, new: str
) -> None:
    assert files[path].count(old) == 1, "the drift case must target a real anchor"
    view = memory_view({**files, path: files[path].replace(old, new)})
    group = _group_of(path)
    scope = next(g.write_scope for g in mc.CAPABILITY.recipe.groups(PARAMS) if g.id == group)
    with pytest.raises(RecipeAnchorError):
        mc.CAPABILITY.recipe.build_group(group, context(view, scope))


def test_duplicated_anchor_is_not_guessed(files: dict[str, str]) -> None:
    path = "shortener/domain/models.py"
    view = memory_view(
        {**files, path: files[path] + '\n\nX = """\n    DISABLED = "disabled"\n"""\n'}
    )
    scope = ("shortener/domain/**", "shortener/api/**")
    with pytest.raises(RecipeAnchorError, match=r"occurs 2 time\(s\)"):
        mc.CAPABILITY.recipe.build_group("domain-api", context(view, scope))


def test_missing_file_is_an_anchor_error(files: dict[str, str]) -> None:
    view = memory_view({**files, "shortener/services/click_recorder.py": None})
    with pytest.raises(RecipeAnchorError, match="does not exist"):
        mc.CAPABILITY.recipe.build_group("service", context(view, ["shortener/services/**"]))


def test_an_already_applied_change_is_refused_everywhere(built: Built) -> None:
    recipe = mc.CAPABILITY.recipe
    for group in built.groups:
        with pytest.raises(RecipeAnchorError):
            recipe.build_group(group.id, context(built.final, group.write_scope))
    with pytest.raises(RecipeAnchorError, match="already exists"):
        recipe.build_tests(context(built.final, ["tests/**"], requirements=ALL_ASPECTS))
    with pytest.raises(RecipeAnchorError):
        recipe.build_docs(context(built.final, ["docs/**", "CHANGELOG.md"]))


@pytest.mark.parametrize(
    "migrations",
    [
        ["001_init.sql", "003_gap.sql"],
        ["001_init.sql", "002_Bad-Name.sql"],
        ["001_init.sql", "002_idempotency.sql", "003_add_max_clicks.sql"],
        [],
    ],
)
def test_unexpected_migration_layouts_are_refused(
    files: dict[str, str], migrations: list[str]
) -> None:
    kept = {p: t for p, t in files.items() if "/migrations/" not in p}
    for name in migrations:
        kept[f"shortener/storage/migrations/{name}"] = "SELECT 1;\n"
    with pytest.raises(RecipeAnchorError):
        mc.CAPABILITY.recipe.build_group(
            "storage", context(memory_view(kept), ["shortener/storage/**", STORAGE_TESTS])
        )


def test_the_next_migration_number_follows_the_baseline(files: dict[str, str]) -> None:
    tests = files[STORAGE_TESTS]
    tests = tests.replace("== [1, 2]", "== [1, 2, 3]").replace(
        '== ["init", "idempotency"]', '== ["init", "idempotency", "other"]'
    )
    view = memory_view(
        {
            **files,
            STORAGE_TESTS: tests,
            "shortener/storage/migrations/003_other.sql": "SELECT 1;\n",
        }
    )
    box = toolbox(view, ["shortener/storage/**", STORAGE_TESTS])
    mc.CAPABILITY.recipe.build_group("storage", BuildContext(toolbox=box, params=PARAMS))
    changeset = box.staged_changeset("x")
    assert "shortener/storage/migrations/004_add_max_clicks.sql" in changeset.paths()
    assert '== ["init", "idempotency", "other", "add_max_clicks"]' in _content(
        changeset, STORAGE_TESTS
    )


# ============================================================================ generated tests


def _tests_of(source: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
    ]


def _covers(function: ast.FunctionDef) -> set[str]:
    found: set[str] = set()
    for line in (ast.get_docstring(function) or "").splitlines():
        if line.strip().startswith("Covers:"):
            found.update(re.findall(r"\b(?:AC-\d{3}-\d{1,2}|R-\d{3})\b", line))
    return found


def test_every_generated_test_names_what_it_covers_and_every_criterion_is_covered(
    built: Built,
) -> None:
    functions = _tests_of(_content(built.tests, NEW_TESTS))
    assert len(functions) >= 18
    assert all(_covers(f) for f in functions), [f.name for f in functions if not _covers(f)]
    wanted = {i for r in ALL_ASPECTS["requirements"] for i in (r["id"], *(
        c["id"] for c in r["acceptance_criteria"]))}  # fmt: skip
    assert set().union(*(_covers(f) for f in functions)) == wanted
    names = [f.name for f in functions]
    assert len(names) == len(set(names))


def test_only_the_aspects_that_are_required_are_generated(base: FileView) -> None:
    req = requirements(("R-007", "AC-007-1", "concurrency-safe"))
    box = toolbox(base, ["tests/**"])
    mc.CAPABILITY.recipe.build_tests(BuildContext(toolbox=box, params=PARAMS, requirements=req))
    source = _content(box.staged_changeset("t"), NEW_TESTS)
    names = {f.name for f in _tests_of(source)}
    assert "test_parallel_redirects_never_exceed_the_limit" in names
    assert "test_link_redirects_until_the_limit_then_answers_410" not in names
    assert all(_covers(f) == {"R-007", "AC-007-1"} for f in _tests_of(source))
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
        if alias.name != "annotations"
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert imported <= used, imported - used


def test_deferred_requirements_and_foreign_aspects_are_ignored(base: FileView) -> None:
    payload = requirements(("R-001", "AC-001-1", "optional-field"))
    payload["requirements"].append(
        {
            "id": "R-002",
            "status": "deferred",
            "acceptance_criteria": [{"id": "AC-002-1", "text": "x", "aspect": "concurrency-safe"}],
        }
    )
    payload["requirements"][0]["acceptance_criteria"].append(
        {"id": "AC-001-2", "text": "x", "aspect": "csv-export"}
    )
    box = toolbox(base, ["tests/**"])
    mc.CAPABILITY.recipe.build_tests(BuildContext(toolbox=box, params=PARAMS, requirements=payload))
    source = _content(box.staged_changeset("t"), NEW_TESTS)
    assert "R-002" not in source and "AC-001-2" not in source


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"requirements": "nope"},
        {"requirements": []},
        {"requirements": [{"id": "R-001", "status": "active", "acceptance_criteria": [
            {"id": "AC-001-1", "aspect": "csv-export"}]}]},
        requirements(("R-1", "AC-001-1", "optional-field")),
        requirements(("R-001", "AC-1", "optional-field")),
        requirements(("R-001\n", "AC-001-1", "optional-field")),
        requirements(('R-001"""; import os #', "AC-001-1", "optional-field")),
    ],
)  # fmt: skip
def test_tests_need_valid_requirements_and_never_embed_unvalidated_text(
    base: FileView, payload: Any
) -> None:
    box = toolbox(base, ["tests/**"])
    ctx = BuildContext(toolbox=box, params=PARAMS, requirements=payload)
    with pytest.raises(RecipeParamError):
        mc.CAPABILITY.recipe.build_tests(ctx)
    assert box.staged_changeset("t").ops == ()


def test_test_file_collision_is_refused(files: dict[str, str]) -> None:
    view = memory_view({**files, NEW_TESTS: "# mine\n"})
    ctx = context(view, ["tests/**"], requirements=ALL_ASPECTS)
    with pytest.raises(RecipeAnchorError, match="already exists"):
        mc.CAPABILITY.recipe.build_tests(ctx)


# ============================================================================ docs


def test_feature_page_changelog_and_api_reference_are_written(built: Built) -> None:
    page = _content(built.docs, "docs/features/max-clicks.md")
    for needle in (
        "ADR-001", "ADR-004", "Alternatives considered", "Roll forward", "DROP COLUMN",
        "1,000,000", "exhausted", "## Limitations", "410",
    ):  # fmt: skip
        assert needle in page
    assert "Change reference: not assigned" in page
    assert "<script" not in page
    assert _content(built.docs, "CHANGELOG.md").startswith("# Changelog\n")
    api = _content(built.docs, "docs/api.md")
    assert "`max_clicks` | `integer or null` | no |" in api
    assert "`remaining_clicks` | `integer or null` | no |" in api
    assert "One of: `active`, `expired`, `disabled`, `exhausted`." in api


def test_docs_without_an_api_reference_still_work_and_say_how_to_regenerate(
    files: dict[str, str], built: Built
) -> None:
    view = memory_view({**{p: t for p, t in files.items() if p != "docs/api.md"}})
    integrated = OverlayView(view, [built.merged])
    box = toolbox(integrated, ["docs/**", "CHANGELOG.md"])
    mc.CAPABILITY.recipe.build_docs(BuildContext(toolbox=box, params=PARAMS))
    changeset = box.staged_changeset("d")
    assert set(changeset.paths()) == {"CHANGELOG.md", "docs/features/max-clicks.md"}
    assert "python -m shortener.apidoc" in _content(changeset, "docs/features/max-clicks.md")


def test_changelog_entry_is_inserted_once_under_unreleased(
    files: dict[str, str], built: Built
) -> None:
    def run(changelog: str | None) -> ChangeSet:
        extra = {"CHANGELOG.md": changelog} if changelog is not None else {}
        view = OverlayView(memory_view({**files, **extra}), [built.merged])
        box = toolbox(view, ["docs/**", "CHANGELOG.md"])
        mc.CAPABILITY.recipe.build_docs(BuildContext(toolbox=box, params=PARAMS))
        return box.staged_changeset("d")

    existing = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- earlier entry\n\n## [0.1.0]\n"
    text = _content(run(existing), "CHANGELOG.md")
    assert text.index("**Click limits.**") < text.index("### Added")
    assert text.count("**Click limits.**") == 1 and "- earlier entry" in text
    assert "CHANGELOG.md" not in run(text).paths()  # the entry is already there: nothing to do
    for broken in ("# Changelog\n", "## [Unreleased]\n\n## [Unreleased]\n"):
        with pytest.raises(RecipeAnchorError, match="Unreleased"):
            run(broken)


def test_documented_facts_follow_the_run_design_and_are_escaped(
    files: dict[str, str], built: Built
) -> None:
    titles = [adr["title"] for adr in mc.CAPABILITY.recipe.design(PARAMS, None)["adrs"]]
    design = {
        "change_reference": "CR-0a1b2c",
        "adrs": [{"id": f"ADR-0{i + 3}<b>|", "title": t} for i, t in enumerate(titles)],
    }
    box = toolbox(built.integrated, ["docs/**", "CHANGELOG.md"])
    mc.CAPABILITY.recipe.build_docs(BuildContext(toolbox=box, params=PARAMS, design=design))
    page = _content(box.staged_changeset("d"), "docs/features/max-clicks.md")
    assert "Change reference: `CR-0a1b2c`." in page
    assert "### ADR-03&lt;b&gt;\\|:" in page and "<b>" not in page
    bad = {"change_reference": "CR-not valid\n# injected"}
    box = toolbox(built.integrated, ["docs/**", "CHANGELOG.md"])
    mc.CAPABILITY.recipe.build_docs(BuildContext(toolbox=box, params=PARAMS, design=bad))
    assert "injected" not in _content(box.staged_changeset("d"), "docs/features/max-clicks.md")


# ============================================================================ payload machinery


def test_every_shipped_edit_script_parses_and_all_holes_are_known() -> None:
    root = Path(mc.__file__).parent / "files" / "max_clicks"
    holes = {**mc._holes(PARAMS), **mc._Migrations("d", ("init", "idempotency")).holes()}
    holes |= dict.fromkeys(("MIGRATION_PATH", "CHANGE_REFERENCE", "ADRS", "ROLLBACK"), "x")
    scripts = sorted(root.glob("*.edits.tmpl"))
    assert {p.name for p in scripts} == set(mc._SCRIPTS.values()) | {"api_doc.edits.tmpl"}
    for script in scripts:
        edits = mc._parse_edits(script.name, mc._fill(script.read_text(encoding="utf-8"), holes))
        assert edits and len({e.ident for e in edits}) == len(edits)
        assert all(e.anchor.endswith("\n") for e in edits)
    for template in root.glob("*.tmpl"):
        assert "\r" not in template.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script",
    [
        "@@@ edit a\n@@@ anchor\nx\n",  # unterminated
        "@@@ edit a\n@@@ anchor\n@@@ replace\ny\n@@@ end\n",  # empty anchor
        "@@@ edit a\n@@@ replace\ny\n@@@ end\n",  # replace before anchor
        "@@@ edit a\n@@@ anchor\nx\n@@@ replace\ny\n@@@ end\nstray text\n",
        "@@@ nonsense\n",
        "@@# only a comment\n",  # no edits at all
        "@@@ edit a b\n",
    ],
)
def test_malformed_edit_scripts_are_rejected_as_internal_defects(script: str) -> None:
    with pytest.raises(AgentError) as error:
        mc._parse_edits("broken.tmpl", script)
    assert error.value.retryable is False


def test_template_holes_are_strict() -> None:
    assert mc._fill("a @@X@@ b @@X@@", {"X": "1"}) == "a 1 b 1"
    with pytest.raises(AgentError):
        mc._fill("@@UNKNOWN@@", {"X": "1"})
    with pytest.raises(AgentError):
        mc._template("../../etc/passwd")
    with pytest.raises(AgentError):
        mc._template("missing_file.txt.tmpl")


# ============================================================================ rollback honesty


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_documented_emergency_down_script_really_removes_the_migration(
    built: Built, tmp_path: Path
) -> None:
    from shortener.storage.db import Database
    from shortener.storage.migrations import Migration, checksum_of, load_migrations, migrate

    old = load_migrations()
    sql = _content(built.changesets["storage"], MIGRATION)
    new = Migration(3, "add_max_clicks", sql, checksum_of(sql))
    database = Database(tmp_path / "rollback.db")
    database.open()
    try:
        assert migrate(database, [*old, new]) == [1, 2, 3]
        connection = database.connection()
        connection.execute(
            "INSERT INTO links (code, original_url, created_at, max_clicks) VALUES (?, ?, ?, ?)",
            ("rolled", "https://example.com/", "2026-01-01T00:00:00.000000Z", 5),
        )
        plan = mc.CAPABILITY.recipe.design(PARAMS, None)["rollback_plan"]
        statements = re.findall(r"`([^`]+;)`", plan)
        assert statements == [
            "ALTER TABLE links DROP COLUMN max_clicks;",
            "DELETE FROM schema_migrations WHERE version = 3;",
        ]
        for statement in statements:
            connection.execute(statement)
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(links)")]
        assert "max_clicks" not in columns
        assert migrate(database, old) == []  # the previous release starts again
    finally:
        database.close()


# ============================================================================ end to end (slow)


def _outcomes(stdout: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"PASSED": [], "FAILED": [], "ERROR": []}
    for line in stdout.splitlines():
        head, _, rest = line.partition(" ")
        if head in found:
            found[head].append(rest.split(" - ", 1)[0])
    return found


def _run_pytest(
    runner: SandboxRunner,
    view: FileView,
    targets: list[str],
    *,
    mutate: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[Any, Path]:
    scratch = runner.new_scratch("mc")
    materialize(view, scratch)
    for rel, (old, new) in (mutate or {}).items():
        target = scratch / rel
        text = target.read_text(encoding="utf-8")
        assert text.count(old) == 1, rel
        target.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
    result = runner.run(runner.pytest_argv(scratch, targets), cwd=scratch, timeout_s=900)
    return result, scratch


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(tmp_path_factory.mktemp("sandbox"))


@pytest.mark.slow
def test_patched_product_suite_passes_in_the_sandbox(built: Built, runner: SandboxRunner) -> None:
    result, scratch = _run_pytest(runner, built.final, ["tests/shortener"])
    outcomes = _outcomes(result.stdout)
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert outcomes["FAILED"] == [] and outcomes["ERROR"] == []
    new = [t for t in outcomes["PASSED"] if t.startswith(NEW_TESTS)]
    assert len(new) >= 28
    assert len(outcomes["PASSED"]) - len(new) >= 700  # the existing suite is still green
    # the sandbox ran the materialised tree, not the editable install of the repository
    probe = scratch / "probe_origin.py"
    probe.write_text("import shortener\nprint(shortener.__file__)\n", encoding="utf-8")
    origin = runner.run(
        runner.python_argv(["probe_origin.py"], scratch=scratch), cwd=scratch, timeout_s=60
    )
    assert Path(origin.stdout.strip()).resolve().is_relative_to(scratch.resolve())
    # the generated API reference equals what the product's own generator prints
    check = runner.run(
        [sys.executable, "-m", "shortener.apidoc", "--check"], cwd=scratch, timeout_s=120
    )
    assert check.returncode == 0, check.stderr[-500:]


def _openapi_of(runner: SandboxRunner, view: FileView) -> dict[str, Any]:
    """The product's own OpenAPI document, dumped by a script inside a scratch copy.

    One JSON value per line: the sandbox redacts and caps every output line at 8 KiB.
    """
    scratch = runner.new_scratch("api")
    materialize(view, scratch)
    (scratch / "dump_openapi.py").write_text(
        "import json\nfrom shortener.apidoc import build_openapi\n"
        "print(json.dumps(build_openapi(), sort_keys=True, indent=1))\n",
        encoding="utf-8",
    )
    result = runner.run(
        runner.python_argv(["dump_openapi.py"], scratch=scratch), cwd=scratch, timeout_s=120
    )
    assert result.returncode == 0, result.stderr[-500:]
    spec: dict[str, Any] = json.loads(result.stdout)
    return spec


@pytest.mark.slow
def test_the_declared_api_change_is_what_the_openapi_diff_finds(
    built: Built, base: FileView, runner: SandboxRunner
) -> None:
    diff = diff_openapi(_openapi_of(runner, base), _openapi_of(runner, built.final))
    assert diff.breaking == []  # the api_compat gate and CHG-004 would pass
    additive = "\n".join(diff.additive)
    for expected in (
        "POST /api/v1/links request body (application/json) $: new optional field 'max_clicks'",
        "GET /api/v1/links/{code} response 200 (application/json) $: "
        "new optional field 'remaining_clicks'",
        'enum widened, added values ["exhausted"]',
    ):
        assert expected in additive
    design = mc.CAPABILITY.recipe.design(PARAMS, None)
    assert all(change["breaking"] is False for change in design["api_changes"])


@pytest.mark.slow
def test_every_generated_test_fails_on_the_unchanged_baseline(
    built: Built, base: FileView, runner: SandboxRunner
) -> None:
    only_tests = OverlayView(base, [ChangeSet(tuple(op for op in built.tests.ops))])
    result, _ = _run_pytest(runner, only_tests, [NEW_TESTS])
    outcomes = _outcomes(result.stdout)
    assert result.returncode != 0
    assert outcomes["PASSED"] == [], outcomes["PASSED"]
    assert len(outcomes["FAILED"]) + len(outcomes["ERROR"]) >= 28


@pytest.mark.slow
@pytest.mark.parametrize(
    ("rel", "old", "new", "must_fail"),
    [
        (
            REPOSITORY,
            "AND (max_clicks IS NULL OR click_count < max_clicks)",
            "AND (max_clicks IS NULL OR click_count < max_clicks + 1000000)",
            {
                "test_link_redirects_until_the_limit_then_answers_410",
                "test_parallel_redirects_never_exceed_the_limit",
                "test_single_use_link_never_redirects_twice_under_a_race",
            },
        ),
        (
            REPOSITORY,
            "WHERE id = ? AND max_clicks IS NULL",
            "WHERE id = ?",
            {"test_click_recorder_does_not_count_limited_clicks_twice"},
        ),
    ],
    ids=["limit-guard-removed", "writer-counts-limited-links"],
)
def test_generated_tests_catch_broken_implementations(
    built: Built, runner: SandboxRunner, rel: str, old: str, new: str, must_fail: set[str]
) -> None:
    result, _ = _run_pytest(runner, built.final, [NEW_TESTS], mutate={rel: (old, new)})
    failed = {t.split("::")[-1] for t in _outcomes(result.stdout)["FAILED"]}
    assert must_fail <= failed, (must_fail - failed, result.stdout[-1500:])
