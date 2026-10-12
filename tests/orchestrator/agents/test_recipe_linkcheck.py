"""Tests for the ``linkcheck`` capability recipe (greenfield: destination health checker).

Covered: recognition of the scenario request (triggers, aspects, extractors), typed and hostile
parameters, the groups (parallel, disjoint write scopes), the design (ADRs with real alternatives),
the generated package / tests / documentation (determinism, traceability lines, no unused imports,
policy rules SEC-001..008 and QLT-001, change-set size and risk) and - marked ``slow`` - a run
of the generated tests in the real sandbox, including mutations that they must catch.
"""

from __future__ import annotations

import ast
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents import catalog
from orchestrator.agents.base import split_sentences
from orchestrator.agents.recipes import linkcheck
from orchestrator.agents.recipes.base import (
    BuildContext,
    Recipe,
    RecipeAnchorError,
    RecipeParamError,
    validate_params,
)
from orchestrator.agents.recipes.linkcheck import (
    ASPECT_IDS,
    CAPABILITY,
    EGRESS_MODULES,
    PARAMS,
)
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentError, ToolDenied
from orchestrator.core.paths import match_glob
from orchestrator.core.schemas import ADR_ID_RE, GROUP_ID_RE, Kind, validate_payload
from orchestrator.core.types import (
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    ChangeSetSubject,
    Enforcement,
    RiskLevel,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.changeset import apply_changeset, merge_changesets
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.toolbox import ScopedToolbox
from orchestrator.tools.workspace import MemoryView, materialize

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = REPO / "orchestrator" / "scenarios"
TEMPLATES = REPO / "orchestrator" / "agents" / "recipes" / "files" / "linkcheck"
RECIPE = CAPABILITY.recipe
DEFAULTS = validate_params(PARAMS, {})
GREENFIELD = (SCENARIOS / "greenfield" / "request.md").read_text(encoding="utf-8")

# the one sentence of the scenario request each aspect must recognise
SENTENCE_OF_ASPECT = {
    "classification": "classify each result",
    "timeout": "time out after 5 seconds",
    "retry": "must be retried up to 3 times",
    "ssrf-block": "never contact",
    "scheme-allowlist": "Only http and https",
    "report": "summary report",
    "offline-testable": "unit-testable",
}
GROUP_SCOPES = {
    "core": {"linkcheck/__init__.py", "linkcheck/model.py", "linkcheck/settings.py"},
    "guard": {"linkcheck/guard.py"},
    "checker": {"linkcheck/checker.py", "linkcheck/fetcher.py"},
    "report": {"linkcheck/report.py"},
}


# ----------------------------------------------------------------------------- rig


def _audit(*args: Any, **kwargs: Any) -> None:
    return None


def box(
    files: Mapping[str, str] | None = None, globs: Iterable[str] = ("**",), node: str = "n"
) -> ScopedToolbox:
    """A real toolbox over ``files`` (the real policy engine checks every action)."""
    return ScopedToolbox(
        run_id="run-1",
        node_id=node,
        attempt=1,
        agent_id="implementer",
        profile=AutonomyProfile(read_globs=("**",), write_globs=tuple(globs)),
        view=MemoryView(dict(files or {})),
        cancel=CancelToken(),
        clock=FakeClock(),
        policy=PolicyEngine(),
        audit=_audit,
    )


def requirements(
    aspects: Sequence[str] = ASPECT_IDS,
    *,
    status: str = "active",
    capabilities: Sequence[str] = ("linkcheck",),
    start: int = 1,
) -> dict[str, Any]:
    """A ``requirements`` payload shaped like the requirements agent's: one criterion per aspect."""
    by_id = {a.id: a for a in CAPABILITY.aspects}
    return {
        "schema": 1,
        "requirements": [
            {
                "id": f"R-{n:03d}",
                "text": f"requirement {n}",
                "status": status,
                "capabilities": list(capabilities),
                "acceptance_criteria": [
                    {
                        "id": f"AC-{n:03d}-1",
                        "text": by_id[aspect].criterion.format_map(DEFAULTS),
                        "aspect": aspect,
                    }
                ],
            }
            for n, aspect in enumerate(aspects, start)
        ],
    }


@dataclass
class Built:
    groups: dict[str, ChangeSet]
    tests: ChangeSet
    docs: ChangeSet
    files: dict[str, str]


def build_everything(
    params: Mapping[str, Any] | None = None,
    reqs: Mapping[str, Any] | None = None,
    base: Mapping[str, str] | None = None,
) -> Built:
    values = validate_params(PARAMS, params or {})
    reqs = requirements() if reqs is None else reqs
    groups: dict[str, ChangeSet] = {}
    for spec in RECIPE.groups(values):
        tb = box(base, spec.write_scope, f"implement.linkcheck-{spec.id}")
        RECIPE.build_group(spec.id, BuildContext(tb, values))
        groups[spec.id] = tb.staged_changeset(f"implement {spec.id}")
    tb = box(base, ("tests/**",), "author-tests")
    RECIPE.build_tests(BuildContext(tb, values, requirements=reqs))
    tests = tb.staged_changeset("tests")
    tb = box(base, ("docs/**", "CHANGELOG.md"), "docs")
    RECIPE.build_docs(BuildContext(tb, values, requirements=reqs))
    docs = tb.staged_changeset("docs")
    view = MemoryView(dict(base or {}))
    merged = merge_changesets(view, [*groups.values(), tests, docs])
    assert merged.conflicts == ()
    return Built(
        groups, tests, docs, {k: v for k, v in apply_changeset(view, merged.changeset).items() if v}
    )


@pytest.fixture(scope="module")
def built() -> Built:
    return build_everything()


def contents(cs: ChangeSet) -> dict[str, str]:
    return {op.path: op.content or "" for op in cs.ops}


def line_count(cs: ChangeSet) -> int:
    return sum(len((op.content or "").splitlines()) for op in cs.ops)


def covers_lines(source: str) -> dict[str, str]:
    """``{test function name: the 'Covers:' docstring line}`` of a generated test module."""
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = ast.get_docstring(node) or ""
            lines = [ln for ln in doc.splitlines() if ln.startswith("Covers:")]
            assert len(lines) == 1, f"{node.name} needs exactly one Covers line"
            found[node.name] = lines[0]
    return found


def unused_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported.update(a.asname or a.name for a in node.names)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return imported - used


# ----------------------------------------------------------------------------- the capability


def test_capability_is_registered_and_well_formed() -> None:
    assert catalog.get_capability("linkcheck") is CAPABILITY
    assert (CAPABILITY.kind, CAPABILITY.requires_baseline) == ("new_module", False)
    assert CAPABILITY.risk == RiskLevel.MEDIUM
    assert isinstance(RECIPE, Recipe)
    assert tuple(a.id for a in CAPABILITY.aspects) == tuple(SENTENCE_OF_ASPECT)
    assert all(a.verification == "test" for a in CAPABILITY.aspects)
    assert {p.name: (p.kind, p.default, p.minimum, p.maximum) for p in PARAMS} == {
        "timeout_seconds": ("int", 5, 1, 60),
        "max_retries": ("int", 3, 0, 5),
    }
    assert set(CAPABILITY.extractors) == {p.name for p in PARAMS}


def test_every_pattern_runs_in_linear_time_on_hostile_text() -> None:
    patterns = [
        *CAPABILITY.triggers,
        *CAPABILITY.extractors.values(),
        *(p for a in CAPABILITY.aspects for p in a.triggers),
    ]
    hostile = [
        "a" * 65536,
        "time out " * 7000,
        "retried " + "up to " * 10000,
        "timeout " + "1" * 60000,
        "http " * 12000,
        "classify" + " healthy" * 8000,
        "private " * 9000 + "address",
        "1 " * 30000 + "second timeout",
    ]
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        for text in hostile:
            started = time.perf_counter()
            compiled.search(text)
            assert time.perf_counter() - started < 1.0, (pattern, text[:20])


def test_each_aspect_recognises_exactly_its_sentence_of_the_scenario_request() -> None:
    body = "\n".join(ln for ln in GREENFIELD.split("\n") if not ln.strip().startswith("#"))
    sentences = split_sentences(body)  # what the requirements agent analyses (headings removed)
    assert len(sentences) == 9
    hits: dict[str, set[str]] = {s: set() for s in sentences}
    for sentence in sentences:
        for aspect in CAPABILITY.aspects:
            if any(re.search(p, sentence, re.IGNORECASE) for p in aspect.triggers):
                hits[sentence].add(aspect.id)
    for aspect_id, fragment in SENTENCE_OF_ASPECT.items():
        owners = [s for s in sentences if fragment in s]
        assert len(owners) == 1, aspect_id
        assert hits[owners[0]] == {aspect_id}
    claimed = {s for s in sentences if hits[s]}
    assert len(claimed) == 7  # the context sentence and the intent sentence carry no aspect


def test_the_scenario_request_matches_and_the_other_scenarios_do_not() -> None:
    matches = {m.capability.id: m for m in catalog.match_capabilities(GREENFIELD)}
    assert "linkcheck" in matches and matches["linkcheck"].score >= 5
    for name in ("brownfield", "ambiguous"):
        text = "\n".join(p.read_text(encoding="utf-8") for p in (SCENARIOS / name).glob("*.md"))
        assert "linkcheck" not in {m.capability.id for m in catalog.match_capabilities(text)}
    answers = (
        "CSV export of daily click counts per link for a selectable window of up to 90 days\n"
        "The export must respond within 250 ms (p95) for 90 days of data\n"
        "Out of scope for this iteration; tracked separately"
    )
    assert "linkcheck" not in {m.capability.id for m in catalog.match_capabilities(answers)}


def test_parameters_are_extracted_from_the_scenario_request() -> None:
    found = catalog.extract_params(CAPABILITY, GREENFIELD)
    assert found == {"timeout_seconds": 5, "max_retries": 3}
    assert validate_params(PARAMS, found) == DEFAULTS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A check must time out after 12 seconds.", {"timeout_seconds": 12}),
        ("The timeout is 7 s per request.", {"timeout_seconds": 7}),
        ("Use a 20-second timeout.", {"timeout_seconds": 20}),
        ("Failures must be retried up to 4 times.", {"max_retries": 4}),
        ("Retry at most 2 times, please.", {"max_retries": 2}),
        ("Allow 5 retries with backoff.", {"max_retries": 5}),
        ("No numbers here, only timeouts and retries.", {}),
    ],
)
def test_extractors_understand_common_phrasings(text: str, expected: dict[str, int]) -> None:
    assert catalog.extract_params(CAPABILITY, text) == expected


def test_decimal_and_absurd_numbers_fail_loudly_instead_of_defaulting() -> None:
    with pytest.raises(
        ValueError
    ):  # int("2.5"): the requirements agent turns it into a fatal error
        catalog.extract_params(CAPABILITY, "A check must time out after 2.5 seconds.")
    for text, name in (
        ("A check must time out after 99999999999999 seconds.", "timeout_seconds"),
        ("Failures are retried up to 999 times.", "max_retries"),
        ("A check must time out after 0 seconds.", "timeout_seconds"),
    ):
        found = catalog.extract_params(CAPABILITY, text)
        assert name in found
        with pytest.raises(RecipeParamError):
            validate_params(PARAMS, found)


def test_criteria_templates_only_need_the_validated_parameters() -> None:
    texts = {
        a.id: a.criterion.format_map({"timeout_seconds": 9, "max_retries": 2})
        for a in CAPABILITY.aspects
    }
    assert "9 seconds" in texts["timeout"] and "2 times" in texts["retry"]
    assert all(len(t) > 40 and "{" not in t for t in texts.values())


# ----------------------------------------------------------------------------- hostile parameters


HOSTILE_PARAMS: list[dict[str, Any]] = [
    {"max_retries": 999},
    {"max_retries": -1},
    {"max_retries": 6},
    {"max_retries": 2.5},
    {"max_retries": True},
    {"max_retries": "3"},
    {"timeout_seconds": "5; rm -rf /"},
    {"timeout_seconds": 0},
    {"timeout_seconds": 61},
    {"timeout_seconds": None},
    {"timeout_seconds": float("nan")},
    {"os.system": 1},
    {"__class__": 1},
    {"timeout_seconds": 5, "extra": 1},
]


@pytest.mark.parametrize("params", HOSTILE_PARAMS, ids=lambda p: ",".join(map(str, p)))
def test_every_entry_point_rejects_hostile_parameters(params: dict[str, Any]) -> None:
    with pytest.raises(RecipeParamError):
        validate_params(PARAMS, params)
    with pytest.raises(RecipeParamError):
        RECIPE.groups(params)
    with pytest.raises(RecipeParamError):
        RECIPE.design(params, None)
    for call in (
        lambda tb: RECIPE.build_group("guard", BuildContext(tb, params)),
        lambda tb: RECIPE.build_tests(BuildContext(tb, params, requirements=requirements())),
        lambda tb: RECIPE.build_docs(BuildContext(tb, params)),
    ):
        tb = box()
        with pytest.raises(RecipeParamError):
            call(tb)
        assert tb.staged_changeset().ops == ()  # nothing was staged before the check


def test_only_validated_numbers_can_be_rendered_into_source() -> None:
    assert (linkcheck._literal(5), linkcheck._literal(0.5)) == ("5", "0.5")
    for hostile in ("5; import os", True, None, [5], "5"):
        with pytest.raises(RecipeParamError):
            linkcheck._literal(hostile)
    with pytest.raises(AgentError):  # a template hole without a value is a template bug
        linkcheck._fill("x = @@NOT_A_HOLE@@", {})


# ----------------------------------------------------------------------------- groups and design


def test_groups_are_parallel_with_disjoint_write_scopes() -> None:
    specs = RECIPE.groups(DEFAULTS)
    assert len(specs) >= 3
    assert {s.id: set(s.write_scope) for s in specs} == GROUP_SCOPES
    assert all(GROUP_ID_RE.match(s.id) and not s.depends_on for s in specs)  # all parallel
    assert all(set(s.expected_paths) == set(s.write_scope) and s.title for s in specs)
    for a in specs:
        for b in specs:
            if a.id != b.id:
                assert not any(match_glob(p, g) for p in a.expected_paths for g in b.write_scope)


def test_design_has_adrs_with_real_alternatives_and_validates_as_a_design_doc() -> None:
    design = RECIPE.design(DEFAULTS, None)
    assert design["data_model_changes"] == [] and design["api_changes"] == []
    assert "no data" in design["rollback_plan"].lower() and "Delete" in design["rollback_plan"]
    adrs = design["adrs"]
    assert len(adrs) >= 2 and all(ADR_ID_RE.match(a["id"]) for a in adrs)
    assert len({a["id"] for a in adrs}) == len(adrs)
    for adr in adrs:
        assert len(adr["alternatives"]) >= 2
        for alt in adr["alternatives"]:
            assert len(alt["option"]) >= 8 and len(alt["rejected_because"]) > 20
        assert len(adr["rationale"]) > 20 and len(adr["consequences"]) > 20
    titles = " ".join(a["title"].lower() for a in adrs)
    assert "fetcher" in titles and "redirect" in titles  # the two decisions the assignment names
    touched = {t for c in design["components"] for t in c["touches"]}
    assert touched == set().union(*GROUP_SCOPES.values())
    payload = {
        "schema": 1,
        **{k: design[k] for k in ("summary", "components", "adrs", "data_model_changes")},
        "api_changes": design["api_changes"],
        "rollback_plan": design["rollback_plan"],
        "change_reference": "CR-abc123",
        "risk": {"level": "LOW", "reasons": ["no data model or API surface change"]},
    }
    assert validate_payload(Kind.DESIGN_DOC, payload) == []


def test_design_is_deterministic_and_reflects_the_parameters() -> None:
    assert RECIPE.design(DEFAULTS, None) == RECIPE.design(DEFAULTS, None)
    custom = RECIPE.design({"timeout_seconds": 12, "max_retries": 1}, None)
    assert "12 s" in custom["summary"] and "1 retries" in custom["summary"]
    assert custom["adrs"] != RECIPE.design(DEFAULTS, None)["adrs"]


# ----------------------------------------------------------------------------- build_group


def test_each_group_writes_exactly_its_scope_as_clean_ascii_lf_source(built: Built) -> None:
    for group_id, scope in GROUP_SCOPES.items():
        files = contents(built.groups[group_id])
        assert set(files) == scope
        for path, text in files.items():
            assert text.isascii() and "\r" not in text and "\t" not in text, path
            assert "@@" not in text and text.endswith("\n") and not text.endswith("\n\n"), path
            compile(text, path, "exec")


def test_the_groups_merge_without_conflict_into_an_importable_package(built: Built) -> None:
    package = {p: t for p, t in built.files.items() if p.startswith("linkcheck/")}
    assert set(package) == set().union(*GROUP_SCOPES.values())
    assert {p for p in built.files if not p.startswith("linkcheck/")} == {
        "tests/linkcheck/test_linkcheck_checker.py",
        "tests/linkcheck/test_linkcheck_guard.py",
        "tests/linkcheck/test_linkcheck_report.py",
        "docs/features/linkcheck.md",
        "CHANGELOG.md",
    }


def test_settings_are_rendered_from_the_validated_parameters() -> None:
    files = build_everything({"timeout_seconds": 12, "max_retries": 1}).files
    settings = files["linkcheck/settings.py"]
    assert "DEFAULT_TIMEOUT_SECONDS = 12\n" in settings
    assert "DEFAULT_MAX_RETRIES = 1\n" in settings
    assert "DEFAULT_BACKOFF_BASE_SECONDS = 0.5\n" in settings
    tests = files["tests/linkcheck/test_linkcheck_checker.py"]
    assert "== 12" in tests and "1 + 1" in tests and "== 5" not in tests


def test_an_existing_file_is_never_overwritten() -> None:
    base = {"linkcheck/guard.py": "# somebody else's module\n"}
    tb = box(base, ("linkcheck/guard.py",))
    with pytest.raises(RecipeAnchorError, match="never overwrites"):
        RECIPE.build_group("guard", BuildContext(tb, DEFAULTS))
    assert tb.staged_changeset().ops == ()
    other = box(base, ("linkcheck/report.py",))
    RECIPE.build_group("report", BuildContext(other, DEFAULTS))  # other groups are unaffected
    assert contents(other.staged_changeset()).keys() == {"linkcheck/report.py"}


def test_unknown_groups_and_out_of_scope_writes_are_refused() -> None:
    for hostile in ("", "../etc", "guard/../../x", "nope", "CORE", "core\n"):
        with pytest.raises(RecipeParamError):
            RECIPE.build_group(hostile, BuildContext(box(), DEFAULTS))
    narrow = box(None, ("linkcheck/guard.py",))
    with pytest.raises(ToolDenied):  # the checker group may not write the guard's neighbour
        RECIPE.build_group("checker", BuildContext(narrow, DEFAULTS))


def test_two_builds_are_byte_identical_and_independent_of_the_conservative_flag() -> None:
    first, second = build_everything(), build_everything()
    assert first.files == second.files
    tb = box(None, ("linkcheck/guard.py",))
    RECIPE.build_group("guard", BuildContext(tb, DEFAULTS, anchors={"conservative": True}))
    assert (
        contents(tb.staged_changeset())["linkcheck/guard.py"] == first.files["linkcheck/guard.py"]
    )


def test_the_package_never_imports_the_network_outside_the_fetcher(built: Built) -> None:
    for path, text in built.files.items():
        if not path.startswith("linkcheck/"):
            continue
        imported = {
            n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(ast.parse(text))
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in getattr(n, "names", ())
        }
        network = {
            m for m in imported if m and m.split(".")[0] in {"socket", "ssl", "http", "urllib"}
        }
        if path == "linkcheck/fetcher.py":
            assert {"socket", "ssl"} <= network
        elif path == "linkcheck/guard.py":
            assert network <= {"urllib.parse"}  # parsing only: no sockets, no name resolution
        else:
            assert not network, (path, network)


# ----------------------------------------------------------------------------- build_tests


def test_every_test_carries_its_covers_line_and_every_criterion_is_covered(built: Built) -> None:
    reqs = requirements()
    wanted = {
        ac["id"]: (r["id"], ac["aspect"])
        for r in reqs["requirements"]
        for ac in r["acceptance_criteria"]
    }
    seen: set[str] = set()
    functions = 0
    for path, text in built.files.items():
        if not path.startswith("tests/"):
            continue
        for name, line in covers_lines(text).items():
            functions += 1
            ids = [i.strip() for i in line.removeprefix("Covers:").split(",")]
            assert ids and all(re.fullmatch(r"R-\d{3}|AC-\d{3}-\d{1,2}", i) for i in ids), name
            seen.update(i for i in ids if i.startswith("AC-"))
    assert functions >= 15
    assert seen == set(wanted)


def test_the_same_aspect_maps_to_all_of_its_criteria() -> None:
    reqs = requirements(["report"]) | {}
    reqs["requirements"] += requirements(["report"], start=8)["requirements"]
    text = build_everything(reqs=reqs).files["tests/linkcheck/test_linkcheck_report.py"]
    lines = set(covers_lines(text).values())
    assert lines == {"Covers: R-001, AC-001-1, R-008, AC-008-1"}


def test_aspects_without_a_criterion_get_no_test() -> None:
    reqs = requirements(["report", "ssrf-block"])
    reqs["requirements"] += [
        *requirements(["classification"], status="deferred", start=3)["requirements"],
        *requirements(["timeout"], capabilities=("max_clicks",), start=4)["requirements"],
        *requirements(["retry"], status="descoped", start=5)["requirements"],
    ]
    files = build_everything(reqs=reqs).files
    tests = {p: t for p, t in files.items() if p.startswith("tests/")}
    # the ssrf-block aspect has tests in the guard module and in the checker module (the fetcher)
    assert set(tests) == {
        "tests/linkcheck/test_linkcheck_checker.py",
        "tests/linkcheck/test_linkcheck_guard.py",
        "tests/linkcheck/test_linkcheck_report.py",
    }
    guard = tests["tests/linkcheck/test_linkcheck_guard.py"]
    assert "test_non_public_destinations_are_blocked_in_every_spelling" in guard
    assert "scheme_not_allowed" not in guard  # the scheme-allowlist aspect has no criterion
    assert "test_only_http_and_https" not in guard
    checker = tests["tests/linkcheck/test_linkcheck_checker.py"]
    assert "test_the_fetcher_connects_only_to_pinned_addresses" in checker
    assert not any(n in checker for n in ("test_each_status_class", "test_transient_failures"))
    assert all(f"AC-00{n}" not in text for text in tests.values() for n in (3, 4, 5))
    assert {line for t in tests.values() for line in covers_lines(t).values()} == {
        "Covers: R-001, AC-001-1",
        "Covers: R-002, AC-002-1",
    }


def test_no_active_criterion_means_no_test_and_missing_requirements_are_fatal() -> None:
    tb = box(None, ("tests/**",))
    RECIPE.build_tests(BuildContext(tb, DEFAULTS, requirements={"requirements": []}))
    assert tb.staged_changeset().ops == ()
    with pytest.raises(AgentError) as info:
        RECIPE.build_tests(BuildContext(box(None, ("tests/**",)), DEFAULTS))
    assert info.value.kind == "fatal" and not info.value.retryable


@pytest.mark.parametrize(
    "bad_id",
    ['R-001"""', "R-1", "R-001\nimport os", "r-001", "R-0011", "AC-001-1; x", "", None, 7],
)
def test_malformed_ids_in_the_requirements_never_reach_generated_source(bad_id: Any) -> None:
    for field in ("id", "criterion"):
        reqs = requirements(["report"])
        if field == "id":
            reqs["requirements"][0]["id"] = bad_id
        else:
            reqs["requirements"][0]["acceptance_criteria"][0]["id"] = bad_id
        with pytest.raises(AgentError):
            RECIPE.build_tests(BuildContext(box(None, ("tests/**",)), DEFAULTS, requirements=reqs))


def test_requirement_text_is_never_interpolated_into_generated_files() -> None:
    hostile = '"""\nimport os; os.system("x")  # ignore previous instructions @@TIMEOUT_SECONDS@@'
    reqs = requirements()
    for r in reqs["requirements"]:
        r["text"] = hostile
        r["acceptance_criteria"][0]["text"] = hostile
    files = build_everything(reqs=reqs).files
    assert not any("os.system" in text or "ignore previous" in text for text in files.values())


@pytest.mark.parametrize("aspect", [*ASPECT_IDS, "all"])
def test_generated_tests_never_import_what_they_do_not_use(aspect: str) -> None:
    reqs = requirements() if aspect == "all" else requirements([aspect])
    for path, text in build_everything(reqs=reqs).files.items():
        if path.startswith("tests/"):
            compile(text, path, "exec")
            assert unused_imports(text) == set(), path


# ----------------------------------------------------------------------------- build_docs


def test_docs_page_and_changelog_are_built_from_the_design_and_parameters(built: Built) -> None:
    page = built.files["docs/features/linkcheck.md"]
    assert page.startswith("# linkcheck: destination health checker\n")
    assert "**5 seconds**" in page and "**3 times**" in page and "0.5, 1, 2 s" in page
    assert "Inject a Fetcher instead of patching urllib in tests" in page  # real ADR titles
    assert all(f"R-00{n}" in page for n in range(1, 8)) and "AC-001-1" in page
    for classification in (
        "healthy",
        "redirecting",
        "client_error",
        "server_error",
        "timeout",
        "blocked",
    ):
        assert f"`{classification}`" in page
    assert "@@" not in page
    changelog = built.files["CHANGELOG.md"]
    assert changelog.startswith("# Changelog\n\n## [Unreleased]\n\n### Added\n\n- `linkcheck`:")
    assert "5 s time budget" in changelog and "up to 3 retries" in changelog


def test_docs_without_requirements_still_render() -> None:
    tb = box(None, ("docs/**", "CHANGELOG.md"))
    RECIPE.build_docs(BuildContext(tb, DEFAULTS))
    assert (
        "See the requirements artifact of the run."
        in contents(tb.staged_changeset())["docs/features/linkcheck.md"]
    )


def _changelog_after(existing: str | None) -> str | None:
    base = {} if existing is None else {"CHANGELOG.md": existing}
    tb = box(base, ("docs/**", "CHANGELOG.md"))
    RECIPE.build_docs(BuildContext(tb, DEFAULTS))
    return contents(tb.staged_changeset()).get("CHANGELOG.md")


def test_an_existing_changelog_is_extended_in_place() -> None:
    entry = "- `linkcheck`: standalone destination health checker"
    with_added = (
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- old thing\n\n## [0.1.0]\n\n- first\n"
    )
    result = _changelog_after(with_added)
    assert result is not None
    assert result.index(entry) < result.index("- old thing") < result.index("## [0.1.0]")
    assert result.count("### Added") == 1 and result.endswith("- first\n")
    no_added = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- a fix\n\n## [0.1.0]\n\n- first\n"
    result = _changelog_after(no_added)
    assert result is not None
    assert (
        result.index("- a fix")
        < result.index("### Added")
        < result.index(entry)
        < result.index("## [0.1.0]")
    )
    releases_only = "# Changelog\n\n## [0.1.0]\n\n- first\n"
    result = _changelog_after(releases_only)
    assert result is not None
    assert result.index("## [Unreleased]") < result.index(entry) < result.index("## [0.1.0]")
    result = _changelog_after("# Changelog\n")
    assert result is not None and result.rstrip("\n").endswith("See docs/features/linkcheck.md.")


def test_the_changelog_entry_is_idempotent() -> None:
    first = _changelog_after(None)
    assert first is not None
    assert _changelog_after(first) is None  # already there: nothing is staged


# ----------------------------------------------------------------------------- policy and size


def test_generated_files_pass_the_security_and_quality_rules() -> None:
    engine = PolicyEngine(config=PolicyConfig(egress_modules=EGRESS_MODULES))
    result = build_everything()
    subjects = {
        **{f"implement.linkcheck-{g}": (cs, GROUP_SCOPES[g]) for g, cs in result.groups.items()},
        "author-tests": (result.tests, {"tests/**"}),
        "docs": (result.docs, {"docs/**", "CHANGELOG.md"}),
    }
    for node, (cs, scope) in subjects.items():
        subject = ChangeSetSubject(
            cs,
            MemoryView({}),
            node_id=node,
            author="implementer",
            profile=AutonomyProfile(write_globs=tuple(scope)),
        )
        report = engine.evaluate(subject)
        assert report.findings == (), (node, [(f.rule_id, f.location) for f in report.findings])
        assert report.enforcement == Enforcement.ALLOW
        assert {"SEC-001", "SEC-002", "SEC-005", "SEC-007", "SEC-008", "QLT-001"} <= set(
            report.rules_evaluated
        )


def test_the_fetcher_is_the_only_implementation_file_that_needs_the_egress_allow_list() -> None:
    result = build_everything()
    engine = PolicyEngine()  # default: no module may import network libraries
    flagged = {}
    for group, cs in result.groups.items():
        subject = ChangeSetSubject(
            cs,
            MemoryView({}),
            node_id=group,
            author="a",
            profile=AutonomyProfile(write_globs=("**",)),
        )
        flagged[group] = {
            (f.rule_id, f.location.split(":")[0])
            for f in engine.evaluate(subject).findings
            if f.location
        }
    assert flagged["checker"] == {("SEC-008", "linkcheck/fetcher.py")}
    assert flagged["core"] == flagged["guard"] == flagged["report"] == set()
    assert EGRESS_MODULES == ("linkcheck/fetcher.py", "tests/linkcheck/**")


def test_every_implementation_group_is_a_low_risk_change_without_waivable_findings() -> None:
    engine = PolicyEngine(config=PolicyConfig(egress_modules=EGRESS_MODULES))
    budget = PolicyConfig().change_budget
    for group, cs in build_everything().groups.items():
        subject = ChangeSetSubject(
            cs,
            MemoryView({}),
            node_id=group,
            author="a",
            profile=AutonomyProfile(write_globs=("**",)),
        )
        risk = engine.assess_risk(subject)
        assert risk.level == RiskLevel.LOW, (group, risk.reasons)
        assert len(cs.ops) <= min(10, budget.max_files) and line_count(cs) <= min(
            300, budget.max_lines
        )
        assert not engine.evaluate(subject).needs_approval


def test_templates_are_ascii_lf_static_text_with_known_holes() -> None:
    allowed = {
        "TIMEOUT_SECONDS", "MAX_RETRIES", "BACKOFF_BASE_SECONDS", "IMPORTS", "COVERS",
        "BACKOFF_SCHEDULE", "DECISIONS", "CRITERIA",
    }  # fmt: skip
    names = sorted(p.name for p in TEMPLATES.iterdir())
    assert all(n.endswith((".py.tmpl", ".md.tmpl")) for n in names)
    for path in TEMPLATES.iterdir():
        raw = path.read_bytes()
        assert raw.isascii() and b"\r" not in raw and b"\t" not in raw, path.name
        assert set(re.findall(r"@@([A-Z_]+)@@", raw.decode("ascii"))) <= allowed, path.name
    markers = set()
    for path in TEMPLATES.glob("tests_*.py.tmpl"):
        markers.update(
            re.findall(r"^# @@ASPECT ([a-z-]+)@@$", path.read_text(encoding="ascii"), re.M)
        )
    assert markers == set(ASPECT_IDS)


def test_the_recipe_module_only_imports_the_layers_it_may() -> None:
    tree = ast.parse(Path(linkcheck.__file__).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    allowed_orchestrator = {
        "orchestrator.agents.recipes.base",
        "orchestrator.core.errors",
        "orchestrator.core.fsutil",
        "orchestrator.core.schemas",
        "orchestrator.core.types",
    }
    assert {m for m in modules if m.startswith("orchestrator")} <= allowed_orchestrator
    assert "anthropic" not in modules and "requests" not in modules


# ------------------------------------------------------------------ the real requirements agent


def test_the_real_requirements_agent_turns_the_scenario_request_into_covered_criteria() -> None:
    rig = pytest.importorskip("tests.orchestrator.agents.analysis_rig")
    from orchestrator.agents.intake import IntakeAgent
    from orchestrator.agents.requirements import RequirementsAgent

    intake = rig.intake_for(None, GREENFIELD, IntakeAgent)
    result = rig.run_agent(
        RequirementsAgent(),
        files=None,
        artifacts={Kind.REQUEST: GREENFIELD, Kind.INTAKE_REPORT: intake},
    )
    payload = rig.payload(result, Kind.REQUIREMENTS)
    assert validate_payload(Kind.REQUIREMENTS, payload) == []
    assert (
        result.clarifications == []
        and payload["unsupported"] == []
        and payload["ambiguities"] == []
    )
    pairs = [
        (r["acceptance_criteria"][0]["aspect"], r["capabilities"], len(r["acceptance_criteria"]))
        for r in payload["requirements"]
    ]
    assert pairs == [(a, ["linkcheck"], 1) for a in SENTENCE_OF_ASPECT]
    assert payload["capabilities"][0]["params"] == {"timeout_seconds": 5, "max_retries": 3}
    # ... and the tests built from that very payload cover every one of its criteria
    built_from_agent = build_everything(reqs=payload)
    covered: set[str] = set()
    for path, text in built_from_agent.files.items():
        if path.startswith("tests/"):
            for line in covers_lines(text).values():
                covered.update(re.findall(r"AC-\d{3}-\d{1,2}", line))
    assert covered == {ac["id"] for r in payload["requirements"] for ac in r["acceptance_criteria"]}


# ----------------------------------------------------------------------------- the sandbox


def run_generated_tests(
    files: Mapping[str, str], tmp_path: Path, name: str
) -> tuple[int, str, set[str]]:
    """Run ``tests/linkcheck`` in the real sandbox: ``(exit code, stdout, names of failed tests)``.

    The failed names come from the junit file: the sandbox redacts (and caps) stdout, which loses
    the ``-rA`` summary of a run with long tracebacks.
    """
    runner = SandboxRunner(tmp_path / f"sandbox-{name}")
    scratch = runner.new_scratch()
    try:
        materialize(MemoryView(dict(files)), scratch)
        result = runner.run(
            runner.pytest_argv(scratch, ["tests/linkcheck"]), cwd=scratch, timeout_s=180
        )
        junit = (scratch / "junit.xml").read_text(encoding="utf-8")
    finally:
        runner.discard(scratch)
    assert not result.timed_out
    failed = re.findall(r'<testcase [^>]*\bname="([^"]+)"[^>]*>\s*<(?:failure|error)', junit)
    return result.returncode, result.stdout, set(failed)


@pytest.mark.slow
def test_the_generated_tests_pass_in_the_real_sandbox_and_catch_mutations(tmp_path: Path) -> None:
    files = build_everything().files
    expected = {
        f"{path}::{name}"
        for path, text in files.items()
        if path.startswith("tests/")
        for name in covers_lines(text)
    }
    code, out, failed = run_generated_tests(files, tmp_path, "green")
    assert code == 0 and not failed, out[-3000:]
    passed = set(re.findall(r"^PASSED (\S+)$", out, re.MULTILINE))  # -rA: one line per test id
    assert passed == expected and len(expected) >= 15

    mutations = {
        # the guard waves every address through
        "linkcheck/guard.py": (
            "def _is_public(ip: IPAddress) -> bool:\n",
            "def _is_public(ip: IPAddress) -> bool:\n    return True\n",
            "test_non_public_destinations_are_blocked_in_every_spelling",
        ),
        # the checker never retries
        "linkcheck/checker.py": (
            "attempt > self.settings.max_retries",
            "True",
            "test_transient_failures_are_retried_with_exponential_backoff",
        ),
        # the fetcher ignores its deadline
        "linkcheck/fetcher.py": (
            "        if left <= 0:\n",
            "        if False:\n",
            "test_default_fetcher_enforces_a_total_deadline",
        ),
    }
    for n, (path, (old, new, failing)) in enumerate(mutations.items()):
        assert files[path].count(old) == 1, path
        mutated = {**files, path: files[path].replace(old, new)}
        code, out, failed = run_generated_tests(mutated, tmp_path, str(n))
        assert code != 0 and failing in failed, (path, sorted(failed))


@pytest.mark.slow
@pytest.mark.parametrize(
    "params",
    [{"timeout_seconds": 1, "max_retries": 0}, {"timeout_seconds": 60, "max_retries": 5}],
    ids=["minimum", "maximum"],
)
def test_a_non_default_parameter_set_builds_a_package_whose_tests_pass(
    params: dict[str, int], tmp_path: Path
) -> None:
    code, out, failed = run_generated_tests(build_everything(params).files, tmp_path, "custom")
    assert code == 0 and not failed, out[-3000:]
