"""Tests for the ``analytics_export`` recipe (CSV export of daily click counts).

The recipe edits the REAL product (``shortener/``), so most tests build the change sets against a
snapshot of the repository's own tree and check them from several sides: anchors (perturbed
baselines must be refused, never guessed), determinism, the policy rules (SEC-001..008, QLT-001),
ruff and, in the slow tests, the real sandbox running the whole product suite on the integrated
view.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents import catalog
from orchestrator.agents.recipes import analytics_export as ae
from orchestrator.agents.recipes.base import (
    BuildContext,
    Capability,
    Recipe,
    RecipeAnchorError,
    RecipeParamError,
    validate_params,
)
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentError
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.schemas import GROUP_ID_RE, Kind, validate_payload
from orchestrator.core.types import (
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    ChangeSetSubject,
    FileView,
    RiskLevel,
)
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.changeset import compute_ops, digest, merge_changesets
from orchestrator.tools.codebase import CodebaseModel, analyze_codebase
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.toolbox import ScopedToolbox
from orchestrator.tools.workspace import (
    BaselineSnapshot,
    DirectoryView,
    MemoryView,
    OverlayView,
    materialize,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INCLUDE = (
    "shortener/**",
    "tests/__init__.py",
    "tests/shortener/**",
    "pyproject.toml",
    "docs/api.md",
    "docs/product-notes.md",
)
SCENARIOS = REPO_ROOT / "orchestrator" / "scenarios"
GROUP_IDS = ("csv-encoding", "storage", "service", "route")
NEW_TEST_FILES = ("tests/shortener/test_analytics_export.py", "tests/shortener/test_export_perf.py")
APIDOC_TEST = "tests/shortener/test_shortener_apidoc.py"
SECURITY_RULES = {f"SEC-00{n}" for n in (1, 2, 3, 4, 5, 7, 8)} | {"QLT-001"}
HOLE = re.compile(r"@@([A-Z][A-Z0-9_]*)@@")
KNOWN_HOLES = {
    "MAX_WINDOW_DAYS",
    "DEFAULT_WINDOW_DAYS",
    "PERF_BUDGET_MS",
    "IN_PROCESS_BUDGET_MS",
    "AUTH_DEPENDENCY",
    "COVERS",
    "DESIGN_SECTION",
    "TRACEABILITY_SECTION",
    "ENTRY",
}


# ----------------------------------------------------------------------------- helpers


def requirement(
    rid: str,
    aspects: Sequence[str],
    *,
    status: str = "active",
    capabilities: Sequence[str] = (ae.CAPABILITY_ID,),
) -> dict[str, Any]:
    number = int(rid[2:])
    return {
        "id": rid,
        "text": "requirement text",
        "kind": "functional",
        "priority": "must",
        "status": status,
        "status_decision": None if status == "active" else f"SCOPE:{rid}",
        "capabilities": list(capabilities) if status == "active" else [],
        "parameters": {},
        "verification": "test",
        "acceptance_criteria": [
            {"id": f"AC-{number:03d}-{i}", "text": "criterion", "aspect": aspect}
            for i, aspect in enumerate(aspects, 1)
        ]
        if status == "active"
        else [],
    }


def requirements_of(*entries: Any) -> dict[str, Any]:
    return {"schema": 1, "requirements": list(entries)}


FULL = requirements_of(requirement("R-001", ae.ASPECT_IDS))


def noop_audit(type: Any, *, node_id: Any, attempt: Any, actor: Any, data: Any) -> None:
    return None


def toolbox_for(view: FileView, globs: Sequence[str]) -> ScopedToolbox:
    return ScopedToolbox(
        run_id="run-1",
        node_id="implement.analytics_export-x",
        attempt=1,
        agent_id="implementer",
        profile=AutonomyProfile(write_globs=tuple(globs), max_changed_files=20),
        view=view,
        cancel=CancelToken(),
        clock=FakeClock(),
        policy=PolicyEngine(),
        audit=noop_audit,
    )


def build_group(
    view: FileView,
    group_id: str,
    *,
    params: Mapping[str, Any] | None = None,
    model: Any | None = None,
) -> ChangeSet:
    params = params or {}
    group = next(g for g in ae.RECIPE.groups(params) if g.id == group_id)
    toolbox = toolbox_for(view, group.write_scope)
    ae.RECIPE.build_group(group_id, BuildContext(toolbox, params, model=model))
    return toolbox.staged_changeset(group_id)


def build_tests(
    view: FileView, requirements: Mapping[str, Any] | None, params: Mapping[str, Any] | None = None
) -> ChangeSet:
    toolbox = toolbox_for(view, ("tests/shortener/**",))
    ae.RECIPE.build_tests(BuildContext(toolbox, params or {}, requirements=requirements))
    return toolbox.staged_changeset("tests")


def build_docs(
    view: FileView,
    *,
    requirements: Mapping[str, Any] | None = FULL,
    design: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> ChangeSet:
    toolbox = toolbox_for(view, ("docs/**", "CHANGELOG.md"))
    context = BuildContext(toolbox, params or {}, requirements=requirements, design=design)
    ae.RECIPE.build_docs(context)
    return toolbox.staged_changeset("docs")


def build_everything(
    view: FileView,
    *,
    model: Any | None = None,
    params: Mapping[str, Any] | None = None,
    requirements: Mapping[str, Any] | None = FULL,
) -> dict[str, ChangeSet]:
    result = {gid: build_group(view, gid, params=params, model=model) for gid in GROUP_IDS}
    result["tests"] = build_tests(view, requirements, params)
    result["docs"] = build_docs(view, requirements=requirements, params=params)
    return result


def content_of(changeset: ChangeSet, path: str) -> str:
    return next(op.content or "" for op in changeset.ops if op.path == path)


def policy_findings(changeset: ChangeSet, baseline: FileView, scope: Sequence[str]) -> list[str]:
    subject = ChangeSetSubject(
        changeset,
        baseline,
        node_id="implement.analytics_export-x",
        author="implementer",
        profile=AutonomyProfile(write_globs=tuple(scope), max_changed_files=20),
    )
    report = PolicyEngine().evaluate(subject)
    assert set(report.rules_evaluated) >= SECURITY_RULES
    return [f.rule_id for f in report.findings if f.rule_id in SECURITY_RULES]


def ruff(text: str, filename: str, *args: str) -> subprocess.CompletedProcess[str]:
    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff is not installed")
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, "--stdin-filename", filename, "-"],
        input=text,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        check=False,
    )


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def files() -> dict[str, str]:
    view = DirectoryView(REPO_ROOT, include=INCLUDE)
    return {path: view.read_text(path) for path in view.list_files("**")}


@pytest.fixture(scope="module")
def baseline(files: dict[str, str]) -> MemoryView:
    return MemoryView(files)


@pytest.fixture(scope="module")
def model(baseline: MemoryView) -> CodebaseModel:
    return analyze_codebase(baseline)


@pytest.fixture(scope="module")
def built(baseline: MemoryView, model: CodebaseModel) -> dict[str, ChangeSet]:
    return build_everything(baseline, model=model)


# ----------------------------------------------------------------------------- wiring


def test_capability_is_registered_and_typed() -> None:
    cap = ae.CAPABILITY
    assert isinstance(cap, Capability)
    assert (cap.id, cap.kind, cap.risk, cap.requires_baseline) == (
        "analytics_export",
        "enhancement",
        RiskLevel.MEDIUM,
        True,
    )
    assert isinstance(ae.RECIPE, Recipe)
    assert catalog.get_capability("analytics_export") is cap
    assert tuple(a.id for a in cap.aspects) == (
        "csv-export",
        "window-bounds",
        "formula-injection",
        "auth-required",
        "owner-scoped",
        "performance",
    )
    assert all(a.verification == "test" for a in cap.aspects)


@pytest.mark.parametrize(
    ("name", "good", "bad"),
    [
        ("max_window_days", [1, 90, 730], [0, 731, -1, True, "90", 90.5, None, 10**9]),
        ("perf_budget_ms", [50, 250, 5000], [49, 5001, 0, False, "250", 1e3, None]),
    ],
)
def test_parameters_are_typed_and_bounded(name: str, good: list[Any], bad: list[Any]) -> None:
    for value in good:
        assert validate_params(ae.PARAMS, {name: value})[name] == value
    for value in bad:
        with pytest.raises(RecipeParamError):
            validate_params(ae.PARAMS, {name: value})
    defaults = validate_params(ae.PARAMS, {})
    assert defaults == {"max_window_days": 90, "perf_budget_ms": 250}


@pytest.mark.parametrize(
    "params",
    [{"max_window_days": 99999}, {"max_window_days": "90"}, {"perf_budget_ms": 1}, {"surprise": 1}],
)
def test_every_entry_point_rejects_hostile_parameters(
    baseline: MemoryView, params: dict[str, Any]
) -> None:
    toolbox = toolbox_for(baseline, ("**",))
    ctx = BuildContext(toolbox, params, requirements=FULL)
    calls: list[Callable[[], object]] = [
        lambda: ae.RECIPE.groups(params),
        lambda: ae.RECIPE.design(params, None),
        lambda: ae.RECIPE.build_group("service", ctx),
        lambda: ae.RECIPE.build_tests(ctx),
        lambda: ae.RECIPE.build_docs(ctx),
    ]
    for call in calls:
        with pytest.raises(RecipeParamError):
            call()
    assert toolbox.staged_changeset().ops == ()


def test_an_unknown_group_is_refused(baseline: MemoryView) -> None:
    ctx = BuildContext(toolbox_for(baseline, ("**",)), {})
    with pytest.raises(RecipeParamError):
        ae.RECIPE.build_group("../etc", ctx)


def test_criteria_templates_fit_the_parameters() -> None:
    params = validate_params(ae.PARAMS, {"max_window_days": 365, "perf_budget_ms": 400})
    texts = {a.id: a.criterion.format_map(params) for a in ae.ASPECTS}
    assert "365 days" in texts["window-bounds"]
    assert "400 ms" in texts["performance"]
    assert all("{" not in text for text in texts.values())


def test_patterns_are_valid_and_run_in_linear_time() -> None:
    patterns = [*ae.TRIGGERS, *ae.EXTRACTORS.values(), *(p for a in ae.ASPECTS for p in a.triggers)]
    hostile = ["up to " * 20000, "9" * 200000, " " * 200000 + "csv", "a" * 200000, "ms " * 50000]
    started = time.perf_counter()
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for text in hostile:
            compiled.search(text)
    assert time.perf_counter() - started < 3.0


def test_the_catalog_matches_only_the_clarified_ambiguous_request() -> None:
    answers = (
        "CSV export of daily click counts per link for a selectable window of up to 90 days\n"
        "The export must respond within 250 ms (p95) for 90 days of data"
    )
    matches = catalog.match_capabilities(answers)
    assert [m.capability.id for m in matches] == ["analytics_export"]
    vague = (SCENARIOS / "ambiguous" / "request.md").read_text(encoding="utf-8")
    assert "analytics_export" not in [m.capability.id for m in catalog.match_capabilities(vague)]
    for name in ("greenfield", "brownfield"):
        text = (SCENARIOS / name / "request.md").read_text(encoding="utf-8")
        assert "analytics_export" not in [m.capability.id for m in catalog.match_capabilities(text)]


def test_extractors_read_the_window_and_the_budget() -> None:
    answers = (
        "CSV export of daily click counts per link for a selectable window of up to 90 days\n"
        "The export must respond within 250 ms (p95) for 90 days of data"
    )
    assert catalog.extract_params(ae.CAPABILITY, answers) == {
        "max_window_days": 90,
        "perf_budget_ms": 250,
    }
    amended = "the export window must be selectable up to 365 days, not 90.\n" + answers
    assert catalog.extract_params(ae.CAPABILITY, amended)["max_window_days"] == 365
    assert catalog.extract_params(ae.CAPABILITY, "no numbers at all") == {}
    hostile = catalog.extract_params(ae.CAPABILITY, "export up to 99999999 days")
    with pytest.raises(RecipeParamError):
        validate_params(ae.PARAMS, hostile)


def test_aspect_triggers_cover_the_scenario_sentences() -> None:
    def aspects_of(text: str) -> list[str]:
        return [
            a.id for a in ae.ASPECTS if any(re.search(p, text, re.IGNORECASE) for p in a.triggers)
        ]

    reporting = "CSV export of daily click counts per link for a selectable window of up to 90 days"
    speed = "The export must respond within 250 ms (p95) for 90 days of data"
    assert aspects_of(f"We need better reports. {reporting} {speed}") == [
        "csv-export",
        "window-bounds",
        "performance",
    ]
    assert aspects_of("It must be safe to open in Excel and only for the owner") == [
        "formula-injection",
        "auth-required",
        "owner-scoped",
    ]


# ----------------------------------------------------------------------------- groups


def test_groups_are_disjoint_single_owner_and_well_formed() -> None:
    groups = ae.RECIPE.groups({})
    assert len(groups) >= 3
    ids = [g.id for g in groups]
    assert len(set(ids)) == len(ids)
    owner: dict[str, str] = {}
    for g in groups:
        assert GROUP_ID_RE.match(g.id)
        assert NODE_ID_RE.match(f"implement.{ae.CAPABILITY_ID}-{g.id}")
        assert set(g.expected_paths) <= set(g.write_scope)
        assert g.depends_on == ()
        for path in g.write_scope:
            assert path not in owner, f"{path} belongs to two groups"
            owner[path] = g.id
    assert tuple(ids) == GROUP_IDS


def test_each_group_stages_exactly_its_files(
    built: dict[str, ChangeSet], baseline: MemoryView
) -> None:
    for group in ae.RECIPE.groups({}):
        changeset = built[group.id]
        assert set(changeset.paths()) == set(group.write_scope)
        for op in changeset.ops:
            assert op.kind.value == ("create" if op.path == ae.CSV_PATH else "modify")
            if op.path.endswith(".py"):
                compile(op.content or "", op.path, "exec")  # QLT-001 by hand
    assert set(built["tests"].paths()) == set(NEW_TEST_FILES)
    assert set(built["docs"].paths()) == {ae.FEATURE_DOC_PATH, ae.CHANGELOG_PATH}


def test_all_change_sets_merge_without_conflicts(
    built: dict[str, ChangeSet], baseline: MemoryView
) -> None:
    merged = merge_changesets(baseline, list(built.values()))
    assert merged.conflicts == ()
    all_paths = [path for cs in built.values() for path in cs.paths()]
    assert len(all_paths) == len(set(all_paths))  # every file has exactly one owner
    assert set(merged.merged_paths) == set(all_paths)


def test_building_is_deterministic(baseline: MemoryView, built: dict[str, ChangeSet]) -> None:
    again = build_everything(baseline)  # no model: the light analysis must agree
    assert {k: digest(v) for k, v in again.items()} == {k: digest(v) for k, v in built.items()}


@pytest.mark.parametrize("maximum", [1, 7, 90, 365, 730])
def test_the_service_group_renders_the_window_parameters(
    baseline: MemoryView, maximum: int
) -> None:
    text = content_of(
        build_group(baseline, "service", params={"max_window_days": maximum}), ae.SERVICE_PATH
    )
    assert f"EXPORT_MAX_DAYS = {maximum}\n" in text
    assert f"EXPORT_DEFAULT_DAYS = {min(30, maximum)}\n" in text
    route = content_of(
        build_group(baseline, "route", params={"max_window_days": maximum}), ae.ROUTES_PATH
    )
    assert f"N is at most {maximum}." in route
    doc = content_of(
        build_group(baseline, "route", params={"max_window_days": maximum}), ae.API_DOC_PATH
    )
    assert f"N is at most {maximum}." in doc and f"maximum={maximum}`" in doc


def test_the_route_declares_the_auth_dependency_of_its_sibling(
    built: dict[str, ChangeSet],
) -> None:
    text = content_of(built["route"], ae.ROUTES_PATH)
    tree = ast.parse(text)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "export_link_analytics_csv"
    )
    signature = ast.unparse(function.args)
    assert "Depends(require_api_key)" in signature
    assert text.count("Depends(require_api_key)") == 6  # five existing routes + the export
    assert 'analytics/export.csv"' in text


def test_the_route_group_skips_the_api_doc_when_the_view_has_none(
    files: dict[str, str],
) -> None:
    without = MemoryView({k: v for k, v in files.items() if k != ae.API_DOC_PATH})
    assert build_group(without, "route").paths() == (ae.ROUTES_PATH,)


def test_the_api_doc_section_is_inserted_between_the_analytics_and_redirect_sections(
    built: dict[str, ChangeSet], files: dict[str, str]
) -> None:
    text = content_of(built["route"], ae.API_DOC_PATH)
    old = files[ae.API_DOC_PATH]
    export = text.index("### `GET /api/v1/links/{code}/analytics/export.csv`")
    assert text.index("### `GET /api/v1/links/{code}/analytics`") < export
    assert export < text.index("### `GET /{code}`")
    assert text.replace(text[export : text.index("### `GET /{code}`")], "") == old


# ----------------------------------------------------------------------------- anchors / drift


def edit(path: str, old: str, new: str) -> Callable[[dict[str, str]], dict[str, str]]:
    def apply(files: dict[str, str]) -> dict[str, str]:
        assert old in files[path], f"mutation target moved: {old!r}"
        return {**files, path: files[path].replace(old, new)}

    return apply


def drop(path: str) -> Callable[[dict[str, str]], dict[str, str]]:
    return lambda files: {k: v for k, v in files.items() if k != path}


def add(path: str, text: str) -> Callable[[dict[str, str]], dict[str, str]]:
    return lambda files: {**files, path: text}


ANALYTICS_SIGNATURE = (
    "    code: str,\n"
    "    principal: Annotated[Principal, Depends(require_api_key)],\n"
    "    services: Annotated[Services, Depends(get_services)],\n"
    "    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,\n"
)
PERTURBATIONS: list[tuple[str, str, Callable[[dict[str, str]], dict[str, str]]]] = [
    ("csv-encoding", "module-exists", add(ae.CSV_PATH, "x = 1\n")),
    ("csv-encoding", "package-missing", drop("shortener/services/__init__.py")),
    (
        "storage",
        "protocol-anchor-duplicated",
        edit(
            ae.REPOSITORY_PATH, ae._REPOSITORY_PROTOCOL_ANCHOR, ae._REPOSITORY_PROTOCOL_ANCHOR * 2
        ),
    ),
    (
        "storage",
        "impl-method-renamed",
        edit(
            ae.REPOSITORY_PATH,
            "    def top_referrers(self, link_id: int, since: datetime, limit: int) -> "
            "Sequence[NamedCount]:\n        rows = (",
            "    def top_referrers_x(self, link_id: int, since: datetime, limit: int) -> "
            "Sequence[NamedCount]:\n        rows = (",
        ),
    ),
    (
        "storage",
        "format-import-removed",
        edit(
            ae.REPOSITORY_PATH,
            "from shortener.domain.timeutil import format_utc, parse_utc",
            "from shortener.domain.timeutil import parse_utc",
        ),
    ),
    (
        "storage",
        "syntax-error",
        edit(
            ae.REPOSITORY_PATH,
            "class SqliteClickRepository:",
            "def broken(:\nclass SqliteClickRepository:",
        ),
    ),
    ("storage", "file-missing", drop(ae.REPOSITORY_PATH)),
    (
        "service",
        "constant-anchor-changed",
        edit(ae.SERVICE_PATH, "BREAKDOWN_LIMIT = 10", "BREAKDOWN_LIMIT = 11"),
    ),
    ("service", "method-renamed", edit(ae.SERVICE_PATH, "def purge_expired(", "def purge_old(")),
    ("service", "zero-fill-helper-renamed", edit(ae.SERVICE_PATH, "_zero_filled", "_filled")),
    (
        "service",
        "error-import-removed",
        edit(
            ae.SERVICE_PATH,
            "from shortener.domain.errors import InvalidInputError",
            "from shortener.domain.errors import DomainError",
        ),
    ),
    (
        "service",
        "recorder-import-aliased",
        edit(
            ae.SERVICE_PATH,
            "from shortener.services.click_recorder import ClickRecorder\n",
            "from shortener.services.click_recorder import ClickRecorder as Recorder\n",
        ),
    ),
    (
        "service",
        "repository-attribute-renamed",
        edit(ae.SERVICE_PATH, "self._clicks = clicks", "self._store = clicks"),
    ),
    (
        "route",
        "sibling-route-renamed",
        edit(ae.ROUTES_PATH, '"/{code}/analytics",', '"/{code}/stats",'),
    ),
    (
        "route",
        "sibling-route-unauthenticated",
        edit(
            ae.ROUTES_PATH,
            ANALYTICS_SIGNATURE,
            ANALYTICS_SIGNATURE.replace("Depends(require_api_key)", "Depends(get_services)"),
        ),
    ),
    (
        "route",
        "import-line-changed",
        edit(
            ae.ROUTES_PATH,
            "import DEFAULT_DAYS, MAX_DAYS, MIN_DAYS",
            "import DEFAULT_DAYS, MAX_DAYS, MIN_DAYS, TOP_REFERRERS",
        ),
    ),
    (
        "route",
        "end-anchor-changed",
        edit(
            ae.ROUTES_PATH,
            "services.analytics.report(link, days))",
            "services.analytics.report(link, days=days))",
        ),
    ),
    ("route", "handler-renamed", edit(ae.ROUTES_PATH, "def link_analytics(", "def link_stats(")),
    (
        "route",
        "response-import-removed",
        edit(
            ae.ROUTES_PATH,
            "from fastapi import APIRouter, Depends, Header, Query, Response",
            "from fastapi import APIRouter, Depends, Header, Query",
        ),
    ),
    (
        "route",
        "api-doc-heading-missing",
        edit(
            ae.API_DOC_PATH,
            "### `GET /api/v1/links/{code}/analytics`",
            "### `GET /api/v1/links/{code}/stats`",
        ),
    ),
    (
        "route",
        "api-doc-heading-duplicated",
        edit(
            ae.API_DOC_PATH,
            "### `GET /{code}`",
            "### `GET /api/v1/links/{code}/analytics`\n\n### `GET /{code}`",
        ),
    ),
]


@pytest.mark.parametrize(
    ("group", "mutate"),
    [pytest.param(g, m, id=f"{g}-{name}") for g, name, m in PERTURBATIONS],
)
def test_a_perturbed_baseline_is_refused_not_guessed(
    files: dict[str, str], group: str, mutate: Callable[[dict[str, str]], dict[str, str]]
) -> None:
    with pytest.raises(RecipeAnchorError):
        build_group(MemoryView(mutate(files)), group)


@pytest.mark.parametrize("group", GROUP_IDS)
def test_applying_a_group_twice_is_refused(
    baseline: MemoryView, group: str, built: dict[str, ChangeSet]
) -> None:
    applied = OverlayView(baseline, [built[group]])
    with pytest.raises(RecipeAnchorError):
        build_group(applied, group)


def test_a_model_that_disagrees_with_the_layout_is_refused(
    files: dict[str, str], baseline: MemoryView
) -> None:
    moved = {k: v for k, v in files.items() if k != ae.SERVICE_PATH}
    moved["shortener/services/analytics_svc.py"] = files[ae.SERVICE_PATH]
    stale = analyze_codebase(MemoryView(moved))
    with pytest.raises(RecipeAnchorError):
        build_group(baseline, "service", model=stale)
    open_route = edit(
        ae.ROUTES_PATH,
        ANALYTICS_SIGNATURE,
        ANALYTICS_SIGNATURE.replace("Depends(require_api_key)", "Depends(get_services)"),
    )
    unauthenticated = analyze_codebase(MemoryView(open_route(files)))
    with pytest.raises(RecipeAnchorError):
        build_group(baseline, "route", model=unauthenticated)


def test_hostile_baseline_text_never_reaches_the_error_message(files: dict[str, str]) -> None:
    marker = "HOSTILE_MARKER_" + "Zq81"
    mutated = edit(ae.SERVICE_PATH, "BREAKDOWN_LIMIT = 10", f"BREAKDOWN_LIMIT = 99  # {marker}")
    with pytest.raises(RecipeAnchorError) as raised:
        build_group(MemoryView(mutated(files)), "service")
    assert marker not in str(raised.value)


# ----------------------------------------------------------------------------- design


def test_design_matches_the_artifact_schemas(model: CodebaseModel) -> None:
    design = ae.RECIPE.design({}, model)
    doc = {
        "schema": 1,
        "summary": "s",
        "components": design["components"],
        "adrs": design["adrs"],
        "data_model_changes": design["data_model_changes"],
        "api_changes": design["api_changes"],
        "rollback_plan": design["rollback_plan"],
        "change_reference": "CR-000001",
        "risk": {"level": "MEDIUM", "reasons": ["new route"]},
    }
    assert validate_payload(Kind.DESIGN_DOC, doc) == []
    contract = {"schema": 1, "paths": design["api_paths"], "breaking": False}
    assert validate_payload(Kind.API_CONTRACT, contract) == []


def test_design_is_additive_with_alternatives_and_a_rollback(model: CodebaseModel) -> None:
    design = ae.RECIPE.design({"max_window_days": 365}, model)
    assert design["data_model_changes"] == []
    assert [
        (c["method"], c["path"], c["change"], c["breaking"]) for c in design["api_changes"]
    ] == [("GET", ae.EXPORT_ROUTE, "add", False)]
    assert len(design["adrs"]) >= 3
    assert all(len(adr["alternatives"]) >= 2 for adr in design["adrs"])
    assert [adr["id"] for adr in design["adrs"]] == [f"ADR-{n:03d}" for n in range(1, 6)]
    assert "365" in " ".join(adr["decision"] for adr in design["adrs"])
    plan = design["rollback_plan"]
    assert "route" in plan and "csv_export.py" in plan and "no migration" in plan
    touched = {path for c in design["components"] for path in c["touches"]}
    assert {ae.CSV_PATH, ae.REPOSITORY_PATH, ae.SERVICE_PATH, ae.ROUTES_PATH} <= touched
    assert design == ae.RECIPE.design({"max_window_days": 365}, model)


def test_design_reflects_the_model_it_is_given(model: CodebaseModel) -> None:
    named = ae.RECIPE.design({}, model)["components"][3]["responsibility"]
    assert "Depends(require_api_key)" in named
    report = {"routes": [r.to_dict() for r in model.routes]}
    assert (
        "Depends(require_api_key)"
        in ae.RECIPE.design({}, report)["components"][3]["responsibility"]
    )
    unknown = ae.RECIPE.design({}, None)["components"][3]["responsibility"]
    assert "same API-key dependency" in unknown
    assert ae.RECIPE.design({}, {"routes": []})["data_model_changes"] == []


def test_design_refuses_a_model_without_the_authenticated_sibling_route(
    files: dict[str, str],
) -> None:
    gone = edit(ae.ROUTES_PATH, '"/{code}/analytics",', '"/{code}/stats",')
    with pytest.raises(RecipeAnchorError):
        ae.RECIPE.design({}, analyze_codebase(MemoryView(gone(files))))


# ----------------------------------------------------------------------------- tests built


def test_tests_cover_every_criterion_with_covers_lines(baseline: MemoryView) -> None:
    requirements = requirements_of(
        requirement("R-001", ["csv-export", "window-bounds", "performance"]),
        requirement("R-002", ["formula-injection", "auth-required", "owner-scoped"]),
        requirement("R-003", ["csv-export"]),
    )
    changeset = build_tests(baseline, requirements)
    expected = {
        "csv-export": {"R-001", "AC-001-1", "R-003", "AC-003-1"},
        "window-bounds": {"R-001", "AC-001-2"},
        "formula-injection": {"R-002", "AC-002-1"},
        "auth-required": {"R-002", "AC-002-2"},
        "owner-scoped": {"R-002", "AC-002-3"},
        "performance": {"R-001", "AC-001-3"},
    }
    seen: set[str] = set()
    for path in NEW_TEST_FILES:
        tree = ast.parse(content_of(changeset, path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                lines = [ln for ln in doc.splitlines() if ln.startswith("Covers: ")]
                assert len(lines) == 1, node.name
                tokens = set(lines[0].removeprefix("Covers: ").split(", "))
                assert tokens in expected.values(), node.name
                seen |= tokens
    assert seen == set().union(*expected.values())


def test_aspects_without_a_criterion_get_no_tests(baseline: MemoryView) -> None:
    only_export = build_tests(baseline, requirements_of(requirement("R-001", ["csv-export"])))
    assert only_export.paths() == (NEW_TEST_FILES[0],)
    text = content_of(only_export, NEW_TEST_FILES[0])
    for absent in ("EXPORT_MAX_DAYS ==", "neutralize_cell", "X-API-Key", "shared.bob"):
        assert absent not in text
    only_perf = build_tests(baseline, requirements_of(requirement("R-004", ["performance"])))
    assert only_perf.paths() == (NEW_TEST_FILES[1],)
    assert "Covers: R-004, AC-004-1" in content_of(only_perf, NEW_TEST_FILES[1])


def test_deferred_foreign_and_malformed_requirements_are_ignored(baseline: MemoryView) -> None:
    hostile_ac = 'AC-001-1"""\nimport os\nos.system("x")\n"""'
    requirements = requirements_of(
        requirement("R-001", ae.ASPECT_IDS, status="deferred"),
        requirement("R-002", ae.ASPECT_IDS, status="descoped"),
        requirement("R-003", ae.ASPECT_IDS, capabilities=("linkcheck",)),
        {
            "id": 'R-004"""',
            "status": "active",
            "capabilities": [ae.CAPABILITY_ID],
            "acceptance_criteria": [{"id": "AC-004-1", "aspect": "csv-export"}],
        },
        {
            "id": "R-005",
            "status": "active",
            "capabilities": [ae.CAPABILITY_ID],
            "acceptance_criteria": [
                {"id": hostile_ac, "aspect": "csv-export"},
                {"id": "AC-005-2", "aspect": "unknown-aspect"},
                "not a mapping",
            ],
        },
        "not a mapping",
        {"id": "R-006", "status": "active", "capabilities": "analytics_export"},
    )
    changeset = build_tests(baseline, requirements)
    assert changeset.ops == ()  # nothing valid remains: no tests, no files


def test_hostile_ids_are_never_embedded_in_the_generated_source(baseline: MemoryView) -> None:
    requirements = requirements_of(
        requirement("R-001", ["csv-export"]),
        {
            "id": "R-002",
            "status": "active",
            "capabilities": [ae.CAPABILITY_ID],
            "acceptance_criteria": [{"id": 'AC-002-1"""\nimport os', "aspect": "csv-export"}],
        },
    )
    text = content_of(build_tests(baseline, requirements), NEW_TEST_FILES[0])
    assert "import os" not in text
    assert "Covers: R-001, AC-001-1" in text


def test_tests_need_the_requirements_payload(baseline: MemoryView) -> None:
    for payload in (None, {}, {"requirements": "x"}):
        with pytest.raises(AgentError) as raised:
            build_tests(baseline, payload)
        assert raised.value.kind == "fatal" and not raised.value.retryable


def test_existing_test_files_are_never_overwritten(files: dict[str, str]) -> None:
    for path in NEW_TEST_FILES:
        view = MemoryView({**files, path: "def test_x():\n    pass\n"})
        with pytest.raises(RecipeAnchorError):
            build_tests(view, FULL)


@pytest.mark.parametrize(
    ("budget", "expected"), [(50, 500), (250, 1000), (400, 1600), (5000, 20000)]
)
def test_the_in_process_budget_is_generous_but_bounded(
    baseline: MemoryView, budget: int, expected: int
) -> None:
    text = content_of(build_tests(baseline, FULL, {"perf_budget_ms": budget}), NEW_TEST_FILES[1])
    assert f"IN_PROCESS_BUDGET_MS = {expected}\n" in text
    assert f"within {budget} ms (p95)" in text


# ----------------------------------------------------------------------------- docs built


def test_docs_create_the_feature_page_and_the_changelog(
    baseline: MemoryView, model: CodebaseModel
) -> None:
    design = ae.RECIPE.design({"max_window_days": 365}, model)
    changeset = build_docs(baseline, design=design, params={"max_window_days": 365})
    page = content_of(changeset, ae.FEATURE_DOC_PATH)
    assert page.isascii() and "\r" not in page and page.endswith("\n")
    for fragment in (
        "Integer from 1 to 365 (default 30)",
        "`code,date,clicks,top_referrer`",
        "p95 of 250 ms",
        "## Design decisions",
        "ADR-003",
        "## Requirement traceability",
        "`formula-injection` | R-001, AC-001-3",
        "## Rollback",
    ):
        assert fragment in page, fragment
    changelog = content_of(changeset, ae.CHANGELOG_PATH)
    assert changelog.startswith("# Changelog\n")
    assert "## [Unreleased]\n\n### Added\n\n- CSV export of daily click counts" in changelog
    assert "1 to 365 days" in changelog and changelog.endswith(".md).\n")


def test_docs_without_requirements_or_design_still_build(baseline: MemoryView) -> None:
    page = content_of(build_docs(baseline, requirements=None), ae.FEATURE_DOC_PATH)
    assert "## Design decisions" in page  # derived from the recipe's own design
    assert "## Requirement traceability" not in page


CHANGELOGS = {
    "added-heading": (
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- older entry\n\n## [1.0.0]\n\n- first\n",
        "### Added\n\n- CSV export",
    ),
    "no-added-heading": (
        "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- a fix\n\n## [1.0.0]\n\n- first\n",
        "## [Unreleased]\n\n### Added\n\n- CSV export",
    ),
    "empty-unreleased-at-end": ("# Changelog\n\n## [Unreleased]\n", "### Added\n\n- CSV export"),
    "empty-unreleased-before-release": (
        "# Changelog\n\n## [Unreleased]\n## [1.0.0]\n\n- first\n",
        "## [Unreleased]\n\n### Added\n\n- CSV export",
    ),
}


@pytest.mark.parametrize("name", list(CHANGELOGS))
def test_the_changelog_entry_goes_under_unreleased(files: dict[str, str], name: str) -> None:
    original, expected = CHANGELOGS[name]
    view = MemoryView({**files, ae.CHANGELOG_PATH: original})
    updated = content_of(build_docs(view), ae.CHANGELOG_PATH)
    assert expected in updated
    assert updated.count("CSV export of daily click counts") == 1
    released = updated.find("## [1.0.0]")
    assert released < 0 or updated.index("CSV export") < released
    assert (
        updated.replace(
            updated[updated.index("- CSV export") : updated.index(".md).\n") + 6], ""
        ).count("CSV export")
        == 0
    )


@pytest.mark.parametrize(
    "broken",
    [
        "# Changelog\n",
        "# Changelog\n\n## [Unreleased]\n\n## [Unreleased]\n",
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n### Added\n",
    ],
)
def test_an_unexpected_changelog_layout_is_refused(files: dict[str, str], broken: str) -> None:
    with pytest.raises(RecipeAnchorError):
        build_docs(MemoryView({**files, ae.CHANGELOG_PATH: broken}))


def test_the_feature_page_is_never_overwritten(files: dict[str, str]) -> None:
    view = MemoryView({**files, ae.FEATURE_DOC_PATH: "# mine\n"})
    with pytest.raises(RecipeAnchorError):
        build_docs(view)


def test_untrusted_design_text_is_escaped_in_the_docs(baseline: MemoryView) -> None:
    design = {
        "adrs": [
            {
                "id": "ADR-001",
                "title": "<script>alert(1)</script> caf\u00e9 [x](http://evil)",
                "decision": "use `tick` \u202e and | pipes",
            },
            {"id": 7, "title": "ignored"},
            "not a mapping",
        ]
    }
    page = content_of(build_docs(baseline, design=design), ae.FEATURE_DOC_PATH)
    assert page.isascii()
    assert "<script>" not in page and "[x](http://evil)" not in page
    assert "&lt;script&gt;" in page or "\\<script\\>" in page
    assert "ignored" not in page


# ----------------------------------------------------------------------------- policies and style


@pytest.mark.parametrize("key", [*GROUP_IDS, "tests", "docs"])
def test_generated_changes_pass_the_security_and_quality_rules(
    key: str, built: dict[str, ChangeSet], baseline: MemoryView
) -> None:
    scopes = {g.id: g.write_scope for g in ae.RECIPE.groups({})}
    scopes["tests"] = ("tests/shortener/**",)
    scopes["docs"] = ("docs/**", "CHANGELOG.md")
    assert policy_findings(built[key], baseline, scopes[key]) == []


def test_the_route_check_is_live_an_unauthenticated_export_would_be_flagged(
    built: dict[str, ChangeSet], baseline: MemoryView
) -> None:
    text = content_of(built["route"], ae.ROUTES_PATH)
    start = text.index("def export_link_analytics_csv(")
    broken = text[:start] + text[start:].replace(
        "Depends(require_api_key)", "Depends(get_services)", 1
    )
    assert broken != text
    changeset = compute_ops(baseline, {ae.ROUTES_PATH: broken})
    assert "SEC-003" in policy_findings(changeset, baseline, (ae.ROUTES_PATH,))


def test_the_policy_catches_a_write_outside_the_group_scope(
    built: dict[str, ChangeSet], baseline: MemoryView
) -> None:
    assert "SEC-005" in policy_findings(built["service"], baseline, (ae.CSV_PATH,))


@pytest.mark.slow
def test_generated_python_is_ruff_clean_and_formatted(built: dict[str, ChangeSet]) -> None:
    for changeset in (*(built[g] for g in GROUP_IDS), built["tests"]):
        for op in changeset.ops:
            if op.path.endswith(".py"):
                text = op.content or ""
                lint = ruff(text, op.path, "check")
                assert lint.returncode == 0, f"{op.path}: {lint.stdout}{lint.stderr}"
                fmt = ruff(text, op.path, "format", "--check")
                assert fmt.returncode == 0, f"{op.path} is not formatted: {fmt.stdout}{fmt.stderr}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "aspects",
    [
        ["csv-export"],
        ["window-bounds"],
        ["formula-injection"],
        ["auth-required", "owner-scoped"],
        ["window-bounds", "owner-scoped", "performance"],
    ],
)
def test_every_aspect_subset_yields_lint_clean_tests(
    baseline: MemoryView, aspects: list[str]
) -> None:
    changeset = build_tests(baseline, requirements_of(requirement("R-001", aspects)))
    for op in changeset.ops:
        text = op.content or ""
        compile(text, op.path, "exec")
        lint = ruff(text, op.path, "check")
        assert lint.returncode == 0, f"{op.path}: {lint.stdout}{lint.stderr}"
        assert ruff(text, op.path, "format", "--check").returncode == 0


# ----------------------------------------------------------------------------- templates


def test_template_files_are_ascii_lf_and_use_known_holes() -> None:
    templates = sorted(ae._TEMPLATE_DIR.glob("*.tmpl"))
    assert len(templates) >= 15
    for path in templates:
        raw = path.read_bytes()
        assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n"), path.name
        text = raw.decode("ascii")
        assert set(HOLE.findall(text)) <= KNOWN_HOLES, path.name
        if path.name.endswith(".py.tmpl"):
            assert "\t" not in text, path.name
            assert "TODO" not in text and "print(" not in text


def test_a_missing_template_is_a_fatal_installation_error() -> None:
    with pytest.raises(AgentError) as raised:
        ae._render("does_not_exist.py.tmpl")
    assert raised.value.kind == "fatal" and not raised.value.retryable


def test_rendering_rejects_mismatched_and_unsafe_values() -> None:
    with pytest.raises(AgentError):
        ae._render("service_constants.py.tmpl", max_window_days=1)  # a hole without a value
    with pytest.raises(AgentError):
        ae._render("service_constants.py.tmpl", max_window_days=1, default_window_days=1, extra=2)
    for bad in (True, 1.5, None, "1; import os"):
        with pytest.raises(AgentError):
            ae._render("service_constants.py.tmpl", max_window_days=bad, default_window_days=1)  # type: ignore[arg-type]
    for unsafe in ('R-001"""', "R-001\nimport os", "x" * 5000, "@@COVERS@@"):
        with pytest.raises(AgentError):
            ae._render("tests_owner_scoped.py.tmpl", covers=unsafe)
    with pytest.raises(AgentError):
        ae._render(
            "feature_doc.md.tmpl",
            max_window_days=1,
            default_window_days=1,
            perf_budget_ms=50,
            in_process_budget_ms=500,
            design_section="caf\u00e9",
            traceability_section="",
        )


# ----------------------------------------------------------------------------- the real sandbox


def run_pytest_in_sandbox(
    runner: SandboxRunner, view: FileView, targets: Sequence[str]
) -> tuple[int, ET.Element, Path]:
    scratch = runner.new_scratch()
    materialize(view, scratch)
    result = runner.run(runner.pytest_argv(scratch, list(targets)), cwd=scratch, timeout_s=900)
    junit = scratch / "junit.xml"
    root = ET.parse(junit).getroot() if junit.exists() else ET.Element("testsuites")  # noqa: S314 - written by our own pytest run
    return result.returncode, root, scratch


def outcomes(root: ET.Element) -> dict[str, str]:
    found: dict[str, str] = {}
    for case in root.iter("testcase"):
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                status = child.tag
        found[f"{case.get('classname')}::{case.get('name')}"] = status
    return found


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory) -> SandboxRunner:
    return SandboxRunner(tmp_path_factory.mktemp("sandbox"))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> BaselineSnapshot:
    return BaselineSnapshot.create(REPO_ROOT, tmp_path_factory.mktemp("baseline"), include=INCLUDE)


@pytest.mark.slow
def test_the_whole_product_suite_passes_with_the_integrated_feature(
    runner: SandboxRunner, snapshot: BaselineSnapshot
) -> None:
    changesets = build_everything(snapshot.view, model=analyze_codebase(snapshot.view))
    merged = merge_changesets(snapshot.view, list(changesets.values()))
    assert merged.conflicts == ()
    code, root, scratch = run_pytest_in_sandbox(
        runner, OverlayView(snapshot.view, [merged.changeset]), ["tests"]
    )
    results = outcomes(root)
    bad = {k: v for k, v in results.items() if v not in ("passed", "skipped")}
    assert code == 0 and not bad, bad
    assert len(results) >= 714 + 50  # the existing product suite plus the generated tests
    new = {
        k: v for k, v in results.items() if "test_analytics_export" in k or "test_export_perf" in k
    }
    assert len(new) >= 45 and set(new.values()) == {"passed"}
    apidoc = {k: v for k, v in results.items() if "test_shortener_apidoc" in k}
    assert apidoc and set(apidoc.values()) == {"passed"}  # docs/api.md stayed in sync
    assert (scratch / ae.FEATURE_DOC_PATH).exists()


@pytest.mark.slow
def test_the_new_tests_fail_on_the_baseline(
    runner: SandboxRunner, snapshot: BaselineSnapshot
) -> None:
    only_tests = build_tests(snapshot.view, FULL)
    code, root, _ = run_pytest_in_sandbox(
        runner, OverlayView(snapshot.view, [only_tests]), list(NEW_TEST_FILES)
    )
    results = outcomes(root)
    assert code != 0
    assert "passed" not in set(results.values())  # fail-to-pass: nothing passes without the feature


@pytest.mark.slow
@pytest.mark.parametrize(("maximum", "budget"), [(1, 50), (365, 1000)])
def test_other_parameter_values_integrate_and_pass(
    runner: SandboxRunner, snapshot: BaselineSnapshot, maximum: int, budget: int
) -> None:
    params = {"max_window_days": maximum, "perf_budget_ms": budget}
    changesets = build_everything(snapshot.view, params=params)
    merged = merge_changesets(snapshot.view, list(changesets.values()))
    assert merged.conflicts == ()
    code, root, _ = run_pytest_in_sandbox(
        runner,
        OverlayView(snapshot.view, [merged.changeset]),
        [*NEW_TEST_FILES, APIDOC_TEST],
    )
    results = outcomes(root)
    assert code == 0, {k: v for k, v in results.items() if v not in ("passed", "skipped")}
    assert sum(1 for v in results.values() if v == "passed") >= 40
