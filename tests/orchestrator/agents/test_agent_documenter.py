"""Tests for the documenter agent (``docs`` node).

Every scenario runs the real agent through the shared ``Rig``: a pinned reader that serves only
the kinds the node declares, a real scoped toolbox over baseline + integrated change set and the
real PolicyEngine.  Capabilities are fakes installed into the catalog, so no real recipe runs.
Non-ASCII and control characters are written as escapes so the file itself stays plain text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pytest

from orchestrator.agents.documenter import (
    BEGIN_MARKER,
    CHANGELOG,
    END_MARKER,
    DocumenterAgent,
)
from orchestrator.agents.recipes.base import RecipeAnchorError, RecipeParamError
from orchestrator.core.errors import AgentError, PathViolation, ToolDenied, UndeclaredInputError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import match_any
from orchestrator.core.sanitize import escape_markdown
from orchestrator.core.schemas import Kind, validate_payload
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    ChangeSet,
    ChangeSetSubject,
    Enforcement,
    EventType,
    FileOpKind,
    PolicyReport,
)
from orchestrator.tools.changeset import ChangeSetConflict, apply_changeset
from tests.orchestrator.agents.delivery_support import (
    NODES,
    FakeRecipe,
    Rig,
    assert_valid,
    changeset_of,
    content_hashes,
    design_doc_payload,
    fake_capability,
    install_capabilities,
    plain_writes,
    requirement,
    requirements_payload,
    task,
    task_plan_payload,
)

DOCS = "docs"
INTEGRATED = Kind.INTEGRATED_CHANGESET
PAGE = "docs/features/widgets.md"
DEFAULT_INTEGRATED: dict[str, str | None] = {"widgets/core.py": "VALUE = 1\n"}
# The changelog entry the default payloads of ``delivery_support`` produce.
ENTRY = (
    "## Release notes: Widgets\n\n"
    "Change reference: CHG-100\n\n"
    "- R-001 (must, active): The widget shall be exported.\n\n"
    "Rollback: Revert the release patch; no data is migrated.\n"
)
E = TypeVar("E", bound=BaseException)


@pytest.fixture(autouse=True)
def _fast_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)


# ----------------------------------------------------------------------------- helpers


def install(
    monkeypatch: pytest.MonkeyPatch,
    recipe: FakeRecipe | None = None,
    *,
    capability_id: str = "widgets",
    title: str | None = None,
) -> FakeRecipe:
    """Make ``capability_id`` resolve to a fake capability; returns its recipe for inspection."""
    chosen = recipe or FakeRecipe()
    install_capabilities(monkeypatch, fake_capability(capability_id, chosen, title=title))
    return chosen


def prepare(
    root: Path,
    *,
    baseline: Mapping[str, str] | None = None,
    integrated: Mapping[str, str | None] | None = DEFAULT_INTEGRATED,
    requirements: Mapping[str, Any] | None = None,
    design: Mapping[str, Any] | None = None,
    plan: Mapping[str, Any] | None = None,
    clock_start: datetime | None = None,
) -> Rig:
    """A rig with the inputs of the ``docs`` node published (``integrated=None``: no change set)."""
    rig = Rig(root, baseline_files=dict(baseline or {}), clock_start=clock_start)
    rig.publish(Kind.REQUIREMENTS, requirements or requirements_payload())
    rig.publish(Kind.DESIGN_DOC, design or design_doc_payload())
    rig.publish(Kind.TASK_PLAN, plan or task_plan_payload())
    if integrated is not None:
        rig.publish_changeset(INTEGRATED, integrated)
    return rig


def tree_after(rig: Rig, changeset: ChangeSet) -> dict[str, str]:
    """The whole file tree after applying ``changeset`` on baseline + integrated change set."""
    view = rig.view_for(NODES[DOCS])
    tree = {path: view.read_text(path) for path in view.list_files("**")}
    for path, content in apply_changeset(view, changeset).items():
        if content is None:
            tree.pop(path, None)
        else:
            tree[path] = content
    return tree


@dataclass
class Docs:
    rig: Rig
    result: AgentResult
    changeset: ChangeSet

    def files(self) -> dict[str, str]:
        """path -> proposed text of every created or modified file."""
        return {
            op.path: op.content or "" for op in self.changeset.ops if op.kind != FileOpKind.DELETE
        }

    def tree(self) -> dict[str, str]:
        return tree_after(self.rig, self.changeset)


def document(root: Path, **kwargs: Any) -> Docs:
    rig = prepare(root, **kwargs)
    result = rig.run(DocumenterAgent(), DOCS)
    assert_valid(result)
    return Docs(rig, result, changeset_of(result, Kind.DOCS_CHANGESET))


def fails(rig: Rig, exc_type: type[E]) -> tuple[AgentContext, E]:
    """Run the agent expecting ``exc_type``; returns the context (to inspect what was staged)."""
    agent = DocumenterAgent()
    ctx = rig.context(agent, DOCS)
    with pytest.raises(exc_type) as info:
        agent.run(ctx)
    return ctx, info.value


def verdict_for(rig: Rig, changeset: ChangeSet) -> PolicyReport:
    """Evaluate like the engine's policy gate does for a ``docs_changeset``.

    ``sources`` stays empty: the gate only hands input change sets to the integrated change set
    itself (CHG-007 would otherwise demand that every documentation path derives from them).
    """
    spec = rig.node(DOCS)
    subject = ChangeSetSubject(
        changeset,
        rig.view_for(spec),
        node_id=spec.id,
        author=changeset.author,
        impact=None,
        profile=DocumenterAgent.autonomy.narrow(spec.autonomy),
        sources=(),
    )
    return rig.policy.evaluate(subject)


def headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


# "...[truncated N chars sha256=...]" plus the few letters of an entity the cut may have split
_TRUNCATION = re.compile(r"&?[a-z]{0,3};?\.\.\.\[truncated \d+ chars sha256=[0-9a-f]{8}\]")


def body(text: str) -> str:
    """``text`` without what the module itself writes raw: the two markers, truncation notes."""
    return _TRUNCATION.sub("", text.replace(BEGIN_MARKER, "").replace(END_MARKER, ""))


def unescaped(text: str, chars: str = "<>[]`|!{}~") -> list[str]:
    """The characters of ``chars`` that no preceding backslash neutralises."""
    found: list[str] = []
    skip = False
    for char in text:
        if skip:
            skip = False
        elif char == "\\":
            skip = True
        elif char in chars:
            found.append(char)
    return found


# ----------------------------------------------------------------------------- the happy path


def test_the_recipe_prose_is_kept_and_the_design_summary_block_is_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = install(monkeypatch, FakeRecipe(doc_files={PAGE: "# Widgets\n\nRecipe prose.\n"}))
    run = document(tmp_path)
    page = run.files()[PAGE]
    head = "# Widgets\n\nRecipe prose.\n\n" + BEGIN_MARKER + "\n## Design summary: Widgets\n"
    assert page.startswith(head)
    assert page.endswith(END_MARKER + "\n")
    lines = page.splitlines()
    for expected in (
        "Add widgets to the service.",
        "- **R-001** (must, active): The widget shall be exported.",
        "  - AC-001-1: Criterion 1 of R-001 holds.",
        "  - AC-001-2: Criterion 2 of R-001 holds.",
        "- **widgets**: Own the widget behaviour. (touches: widgets/core.py)",
        "- ADR-001: Keep widgets in one module. A single module owns the behaviour."
        " Rationale: The surface is tiny.",
        "- Rollback plan: Revert the release patch; no data is migrated.",
        "- Risk: LOW (small, additive change)",
        "- Change reference: CHG-100",
    ):
        assert expected in lines
    assert len(recipe.doc_contexts) == 1  # the recipe ran once, with the validated facts
    ctx = recipe.doc_contexts[0]
    assert ctx.params == {"limit": 5} and ctx.anchors == {}
    assert ctx.requirements == requirements_payload() and ctx.design == design_doc_payload()


def test_the_changelog_is_created_with_an_unreleased_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    run = document(tmp_path)
    kinds = {op.path: op.kind for op in run.changeset.ops}
    assert kinds[CHANGELOG] is FileOpKind.CREATE
    assert run.files()[CHANGELOG] == "# Changelog\n\n" + ENTRY  # no dates: determinism rule


def test_a_recipe_that_stages_nothing_still_gets_its_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)  # build_docs stages nothing
    run = document(tmp_path)
    assert {op.path: op.kind for op in run.changeset.ops} == {
        PAGE: FileOpKind.CREATE,
        CHANGELOG: FileOpKind.CREATE,
    }
    assert run.files()[PAGE].startswith("# Widgets\n\n" + BEGIN_MARKER + "\n## Design summary")


def test_operations_are_relative_to_the_baseline_plus_the_integrated_change_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    base_log = "# Changelog\n\n## 1.0.0\n\n- First.\n"
    integrated_log = base_log + "\n## Implementer note\n\n- Touched widgets.\n"
    integrated_page = "# Widgets\n\nIntegrated prose.\n"
    integrated = {CHANGELOG: integrated_log, PAGE: integrated_page, "widgets/core.py": "X = 1\n"}
    run = document(tmp_path, baseline={CHANGELOG: base_log}, integrated=integrated)
    ops = {op.path: op for op in run.changeset.ops}
    assert {path: op.kind for path, op in ops.items()} == {
        CHANGELOG: FileOpKind.MODIFY,
        PAGE: FileOpKind.MODIFY,  # created by the integrated change set, so a MODIFY here
    }
    assert ops[CHANGELOG].base_hash == sha256_text(integrated_log) != sha256_text(base_log)
    assert ops[PAGE].base_hash == sha256_text(integrated_page)
    files = run.files()
    assert files[CHANGELOG].startswith(integrated_log)  # the implementer's note survives
    assert files[PAGE].startswith(integrated_page.rstrip("\n"))
    view = run.rig.view_for(NODES[DOCS])
    assert set(apply_changeset(view, run.changeset)) == set(ops)  # strict: base hashes verified
    with pytest.raises(ChangeSetConflict):  # the plain baseline is the wrong base
        apply_changeset(run.rig.baseline, run.changeset)


# ----------------------------------------------------------------------------- brownfield texts

OLD_CHANGELOGS = {
    "single_newline": ("# Changelog\n\n## 1.0.0\n\n- First release.\n", True),
    "no_trailing_newline": ("# Changelog\n\n- First release.", True),
    "odd_whitespace": (
        "# Changelog\n\n\n\t- caf\u00e9  \u00a0\n\n## 1.0 \u2013 \u00fc\n\n- A.\n",
        True,
    ),
    "extra_blank_lines": ("# Changelog\n\n- First.\n\n\n", False),  # blank lines collapse to one
}


@pytest.mark.parametrize(
    ("old", "exact_prefix"), list(OLD_CHANGELOGS.values()), ids=list(OLD_CHANGELOGS)
)
def test_brownfield_changelog_text_is_preserved_and_the_entry_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str, exact_prefix: bool
) -> None:
    install(monkeypatch)
    run = document(tmp_path, baseline={CHANGELOG: old, "README.md": "# Project\n"})
    ops = {op.path: op for op in run.changeset.ops}
    assert ops[CHANGELOG].kind is FileOpKind.MODIFY
    assert ops[CHANGELOG].base_hash == sha256_text(old)
    new = run.files()[CHANGELOG]
    assert new == old.rstrip("\n") + "\n\n" + ENTRY  # append-only, after one blank line
    assert new.startswith(old if exact_prefix else old.rstrip("\n"))
    assert "README.md" not in ops  # an untouched README stays untouched


def test_an_identical_changelog_entry_is_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    old = "# Changelog\n\n## Unreleased: Newer\n\n- x\n\n" + ENTRY + "\n## 0.9.0\n\n- older\n"
    same = document(tmp_path / "same", baseline={CHANGELOG: old})
    assert CHANGELOG not in {op.path for op in same.changeset.ops}  # no op at all
    assert PAGE in same.files()  # the rest of the documentation is still produced
    changed = requirements_payload(requirement(text="The widget shall be exported twice."))
    other = document(tmp_path / "changed", baseline={CHANGELOG: old}, requirements=changed)
    text = other.files()[CHANGELOG]  # a different entry is appended, never merged
    assert text.startswith(old.rstrip("\n")) and text.count("## Release notes: Widgets") == 2


def test_an_existing_block_is_replaced_in_place_and_the_second_run_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    stale = f"{BEGIN_MARKER}\nSTALE BLOCK\n{END_MARKER}\n"
    old_page = f"# Widgets\n\nIntro prose.\n\n{stale}\nTrailing prose.\n"
    first = document(tmp_path / "first", baseline={PAGE: old_page, CHANGELOG: "# Changelog\n"})
    page = first.files()[PAGE]
    before, rest = page.split(BEGIN_MARKER, 1)
    block, after = rest.split(END_MARKER, 1)
    assert before == "# Widgets\n\nIntro prose.\n\n"
    assert "STALE BLOCK" not in page and "## Design summary: Widgets" in block
    assert after.strip() == "Trailing prose."
    assert (page.count(BEGIN_MARKER), page.count(END_MARKER)) == (1, 1)
    again = document(tmp_path / "second", baseline=first.tree(), integrated={})
    assert again.changeset.ops == ()


def test_applying_the_documenter_twice_proposes_nothing_the_second_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = FakeRecipe(doc_files={PAGE: "# Widgets\n\nRecipe prose.\n"})
    install(monkeypatch, recipe)  # the recipe regenerates its prose on every run
    first = document(tmp_path / "first")
    assert len(first.changeset.ops) == 2
    again = document(tmp_path / "second", baseline=first.tree(), integrated={})
    assert again.changeset.ops == ()
    assert len(recipe.doc_contexts) == 2


def test_an_orphaned_begin_marker_never_makes_a_later_run_swallow_human_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    human = "Human text after an orphaned marker.\n"
    old_page = f"# Widgets\n\nProse.\n\n{BEGIN_MARKER}\n{human}"  # no end marker
    first = document(tmp_path / "first", baseline={PAGE: old_page})
    page = first.files()[PAGE]
    assert page.startswith(old_page.rstrip("\n"))  # appended, nothing lost
    assert (page.count(BEGIN_MARKER), page.count(END_MARKER)) == (2, 1)
    changed = requirements_payload(requirement(text="The widget shall be exported twice."))
    second = document(
        tmp_path / "second", baseline=first.tree(), integrated={}, requirements=changed
    )
    page = second.files()[PAGE]
    assert human in page  # the second run replaces only its own block
    assert "exported twice." in page and "be exported." not in page
    assert (page.count(BEGIN_MARKER), page.count(END_MARKER)) == (2, 1)


def test_a_changelog_staged_by_the_recipe_is_extended_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prose = "# Changelog\n\n## Unreleased: Recipe entry\n\nRecipe text.\n"
    install(monkeypatch, FakeRecipe(doc_files={CHANGELOG: prose}))
    assert document(tmp_path).files()[CHANGELOG] == prose.rstrip("\n") + "\n\n" + ENTRY


# ----------------------------------------------------------------------------- capabilities


def test_several_capabilities_are_documented_in_id_order_one_page_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    widgets = FakeRecipe()
    gadgets_page = "docs/features/gadgets.md"
    gadgets = FakeRecipe(doc_files={gadgets_page: "# Gadgets\n\nGadget prose.\n"})
    install_capabilities(
        monkeypatch,
        fake_capability("widgets", widgets, title="Alpha widgets"),
        fake_capability("gadgets", gadgets, title="Zeta gadgets"),
    )
    reqs = requirements_payload(
        requirement("R-001", capability="widgets"),
        requirement("R-002", text="Gadgets shall spin.", capability="gadgets"),
    )
    reqs["capabilities"] = [
        {"id": "gadgets", "title": "Gadgets", "requirement_ids": ["R-002"], "params": {}},
        {"id": "widgets", "title": "Widgets", "requirement_ids": ["R-001"], "params": {}},
    ]
    design = design_doc_payload()
    design["components"] = [
        {
            "name": "widgets",
            "responsibility": "Widget part.",
            "touches": [],
            "requirement_ids": ["R-001"],
        },
        {
            "name": "gadgets",
            "responsibility": "Gadget part.",
            "touches": [],
            "requirement_ids": ["R-002"],
        },
    ]
    plan = task_plan_payload(  # group order (widgets first) differs from capability order
        task("T-001", group="a-widgets", capability="widgets", requirement_ids=("R-001",)),
        task("T-002", group="b-gadgets", capability="gadgets", requirement_ids=("R-002",)),
    )
    files = document(tmp_path, requirements=reqs, design=design, plan=plan).files()
    assert sorted(files) == [CHANGELOG, gadgets_page, PAGE]
    gadget, widget = files[gadgets_page], files[PAGE]
    assert gadget.startswith("# Gadgets\n\nGadget prose.\n\n" + BEGIN_MARKER)  # recipe prose kept
    assert widget.startswith("# Alpha widgets\n\n" + BEGIN_MARKER)
    assert "## Design summary: Zeta gadgets" in gadget and "Gadgets shall spin." in gadget
    assert "The widget shall be exported." not in gadget and "**widgets**" not in gadget
    assert "The widget shall be exported." in widget and "Gadgets shall spin." not in widget
    assert "**gadgets**" not in widget and "**widgets**" in widget
    log = files[CHANGELOG]
    assert "## Release notes: Zeta gadgets, Alpha widgets" in log  # id order, not title order
    assert log.index("R-001") < log.index("R-002")
    assert len(widgets.doc_contexts) == len(gadgets.doc_contexts) == 1


def test_groups_of_one_capability_share_a_page_and_use_the_first_groups_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = install(monkeypatch)
    plan = task_plan_payload(
        task("T-001", group="a-first", expected_paths=("widgets/a.py",)),
        task("T-002", group="b-second", expected_paths=("widgets/b.py",)),
    )
    plan["groups"][0]["params"] = {"limit": 7}
    plan["groups"][1]["params"] = {"limit": 9}
    run = document(tmp_path, plan=plan)
    assert len(recipe.doc_contexts) == 1
    assert recipe.doc_contexts[0].params == {"limit": 7}
    assert sorted(run.files()) == [CHANGELOG, PAGE]


BAD_CAPABILITY_IDS = [
    "../etc",
    "Widgets",
    "",
    "a b",
    "x" * 50,
    "widgets\n",  # a trailing line feed must not pass (\Z, not $)
    "wid.gets",
    "-lead",
    "9lives",
    "wi\x1b[31mdgets",
    "widgets/../x",
    "..",
]


@pytest.mark.parametrize("capability_id", BAD_CAPABILITY_IDS)
def test_a_capability_id_outside_the_grammar_is_fatal_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capability_id: str
) -> None:
    recipe = install(monkeypatch)  # 'widgets' exists; the hostile id must never reach a lookup
    plan = task_plan_payload(task(capability=capability_id))
    rig = prepare(tmp_path, plan=plan, integrated=None)
    ctx, error = fails(rig, AgentError)
    assert (error.kind, error.retryable) == ("fatal", False)
    assert str(error) == "the task plan names a capability with an invalid id"  # id not echoed
    assert recipe.doc_contexts == []
    assert ctx.toolbox.staged_changeset().ops == ()


def test_an_unknown_capability_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)  # only 'widgets' is known
    plan = task_plan_payload(task(capability="gadgets"))
    ctx, error = fails(prepare(tmp_path, plan=plan, integrated=None), AgentError)
    assert (error.kind, error.retryable) == ("fatal", False)
    assert str(error) == "unknown capability gadgets"
    assert ctx.toolbox.staged_changeset().ops == ()


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"limit": "5"}, {"limit": True}, {"bogus": 1}],
    ids=["below_minimum", "above_maximum", "wrong_type", "bool_is_not_int", "unknown_key"],
)
def test_invalid_recipe_parameters_are_fatal_and_stop_before_the_recipe_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any]
) -> None:
    recipe = install(monkeypatch)
    rig = prepare(tmp_path, plan=task_plan_payload(params=params), integrated=None)
    ctx, error = fails(rig, RecipeParamError)
    assert (error.kind, error.retryable) == ("fatal", False)
    assert recipe.doc_contexts == []
    assert ctx.toolbox.staged_changeset().ops == ()


def test_a_recipe_anchor_error_propagates_and_nothing_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeRecipe(docs_error=RecipeAnchorError("anchor is missing")))
    ctx, error = fails(prepare(tmp_path, integrated=None), RecipeAnchorError)
    assert (error.kind, error.retryable) == ("fatal", False)
    assert ctx.toolbox.staged_changeset().ops == ()


# ----------------------------------------------------------------------------- the write scope


@pytest.mark.parametrize(
    "path",
    ["app/x.py", ".github/workflows/ci.yml", "docs.md", "CHANGELOG.md.bak", "README.markdown"],
)
def test_a_recipe_write_outside_the_documentation_scope_is_denied_and_not_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    install(monkeypatch, FakeRecipe(doc_files={path: "X = 1\n"}))
    rig = prepare(tmp_path, integrated=None)
    ctx, error = fails(rig, ToolDenied)
    assert error.details == {"denials": 1}  # the first denial; the third would be a violation
    assert "outside write_globs" in str(error)
    assert ctx.toolbox.staged_changeset().ops == ()
    denied = rig.audit.of(EventType.TOOL_DENIED)
    assert [(d["tool"], d["target"]) for d in denied] == [("write_text", path)]


def test_a_traversal_path_from_a_recipe_is_an_autonomy_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeRecipe(doc_files={"docs/../app/x.py": "X = 1\n"}))
    ctx, error = fails(prepare(tmp_path, integrated=None), PathViolation)
    assert (error.kind, error.retryable) == ("autonomy_violation", False)
    with pytest.raises(PathViolation):  # the toolbox stays poisoned: nothing else can be staged
        ctx.toolbox.write_text("docs/ok.md", "fine\n")


def test_every_operation_path_is_inside_the_documentation_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {PAGE: "# Widgets\n\nProse.\n", "docs/guide/intro.md": "# Intro\n"}
    files["README.md"] = "# Project\n\nWidgets.\n"
    install(monkeypatch, FakeRecipe(doc_files=files))
    run = document(tmp_path, baseline={"README.md": "# Project\n"})
    paths = {op.path for op in run.changeset.ops}
    assert paths == {PAGE, "docs/guide/intro.md", "README.md", CHANGELOG}
    assert all(match_any(path, DocumenterAgent.autonomy.write_globs) for path in paths)
    assert len(paths) <= DocumenterAgent.autonomy.max_changed_files


def test_the_documenter_autonomy_is_limited_to_documentation_files(tmp_path: Path) -> None:
    profile = DocumenterAgent.autonomy
    assert profile.write_globs == ("docs/**", "CHANGELOG.md", "README.md")
    assert profile.can_execute is False
    assert profile.narrow(NODES[DOCS].autonomy).write_globs == profile.write_globs
    box = Rig(tmp_path).context(DocumenterAgent(), DOCS).toolbox
    box.write_text("docs/ok.md", "fine\n")  # inside the scope
    with pytest.raises(ToolDenied):
        box.write_text("shortener/app.py", "X = 1\n")
    with pytest.raises(ToolDenied):
        box.run_tests()


# ----------------------------------------------------------------------------- injection hardening

CONTROLS = "\x1b[31mred\x1b[0m\x07\x00\u202e\u200b\r"  # ANSI colour, BEL, NUL, bidi, zero width
SHORT_HOSTILE = "<i>x</i>[a](javascript:alert(1))&lt;"
LONG_HOSTILE = (
    END_MARKER
    + "\n"
    + BEGIN_MARKER
    + "\n<script>alert(1)</script> [a](javascript:alert(1)) ![i](x) &lt;b&gt;"
    + "\n# Injected\n## Heading\n- bullet\n```\n| a | b |\n"
)


def hostile_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    """Requirements and design with markup, forged markers and headings in every text field."""
    req = requirement("R-001", text=LONG_HOSTILE)
    req["acceptance_criteria"][0]["text"] = LONG_HOSTILE
    design = design_doc_payload(
        summary=LONG_HOSTILE,
        rollback_plan=LONG_HOSTILE,
        change_reference=SHORT_HOSTILE,
        api_changes=[
            {"method": SHORT_HOSTILE, "path": SHORT_HOSTILE, "change": "add", "breaking": True}
        ],
        data_model_changes=[
            {
                "table": SHORT_HOSTILE,
                "change": "create",
                "detail": LONG_HOSTILE,
                "migration": "001.sql",
            }
        ],
    )
    design["components"] = [
        {
            "name": SHORT_HOSTILE,
            "responsibility": LONG_HOSTILE,
            "touches": [SHORT_HOSTILE],
            "requirement_ids": ["R-001"],
        }
    ]
    design["adrs"][0].update(title=SHORT_HOSTILE, decision=LONG_HOSTILE, rationale=LONG_HOSTILE)
    design["risk"]["reasons"] = [LONG_HOSTILE]
    return requirements_payload(req), design


def test_untrusted_text_is_escaped_in_every_field_and_the_result_is_policy_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, title=SHORT_HOSTILE)
    reqs, design = hostile_payloads()
    run = document(tmp_path, requirements=reqs, design=design)
    files = run.files()
    assert sorted(files) == [CHANGELOG, PAGE]
    for path, text in files.items():
        visible = body(text)
        assert unescaped(visible) == [], path  # no raw '<', '>', '[', ']', backtick, '|', '!'
        assert "](" not in visible and "alert(1)" not in visible, path  # no link syntax
        assert re.search(r"&(?!amp;)", visible) is None, path  # no raw HTML entity
    page, log = files[PAGE], files[CHANGELOG]
    assert (page.count(BEGIN_MARKER), page.count(END_MARKER)) == (1, 1)  # markers cannot be forged
    assert BEGIN_MARKER not in log and END_MARKER not in log
    assert escape_markdown(END_MARKER) in page  # the forged marker survives only as inert text
    assert len(headings(page)) == 8 and len(headings(log)) == 2  # no injected heading
    assert verdict_for(run.rig, run.changeset).blocking == ()  # the real engine agrees


def test_control_characters_and_ansi_sequences_are_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, title="Wid" + CONTROLS + "gets")
    reqs = requirements_payload(requirement("R-001", text="Req " + CONTROLS + " text"))
    design = design_doc_payload(summary="Sum " + CONTROLS, rollback_plan="Back " + CONTROLS)
    run = document(tmp_path, requirements=reqs, design=design)
    for path, text in run.files().items():
        assert "\x1b" not in text, path
        assert all(unicodedata.category(c) not in {"Cc", "Cf"} or c == "\n" for c in text), path
    assert "Req red text" in run.files()[PAGE]  # the readable part survives
    assert verdict_for(run.rig, run.changeset).blocking == ()  # e.g. no hidden bidi characters


def test_overlong_text_is_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, title="T" * 500)
    reqs = requirements_payload(requirement("R-001", text="A" * 5000))
    design = design_doc_payload(summary="B" * 5000, rollback_plan="C" * 5000)
    design["components"][0]["responsibility"] = "D" * 5000
    files = document(tmp_path, requirements=reqs, design=design).files()
    page, log = files[PAGE], files[CHANGELOG]
    assert "A" * 301 not in page and "B" * 601 not in page and "D" * 301 not in page
    assert "C" * 601 not in page and "T" * 121 not in page
    assert "A" * 201 not in log and "C" * 301 not in log and "T" * 61 not in log
    assert "[truncated 5000 chars sha256=" in page and "[truncated 5000 chars sha256=" in log
    assert len(page) < 8000 and len(log) < 2000


# ----------------------------------------------------------------------------- policy, inputs


def test_the_greenfield_change_set_passes_the_policy_engine_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeRecipe(doc_files={PAGE: "# Widgets\n\nRecipe prose.\n"}))
    run = document(tmp_path)
    verdict = verdict_for(run.rig, run.changeset)
    assert verdict.findings == () and verdict.enforcement is Enforcement.ALLOW


def test_a_brownfield_change_set_has_no_blocking_policy_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    baseline = {CHANGELOG: "# Changelog\n\n- First.\n", "README.md": "# Project\n"}
    run = document(tmp_path, baseline=baseline)
    assert {op.path: op.kind for op in run.changeset.ops}[CHANGELOG] is FileOpKind.MODIFY
    verdict = verdict_for(run.rig, run.changeset)
    assert verdict.blocking == ()
    # the docs node has no impact report, so modifying an existing file is CHG-003 (MEDIUM) only
    assert {f.rule_id for f in verdict.findings} <= {"CHG-003"}


def test_the_documenter_needs_neither_optional_nor_undeclared_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    agent = DocumenterAgent()
    ctx = prepare(tmp_path / "plain").context(agent, DOCS)
    assert ctx.has(Kind.API_CONTRACT) is False  # optional and not published
    for kind in (Kind.TEST_REPORT, Kind.SECURITY_REPORT):
        with pytest.raises(UndeclaredInputError):
            ctx.has(kind)
    plain = content_hashes(agent.run(ctx))
    with_contract = prepare(tmp_path / "contract")
    with_contract.publish(Kind.API_CONTRACT, {"schema": 1, "paths": [], "breaking": False})
    assert content_hashes(with_contract.run(agent, DOCS)) == plain  # a contract changes nothing


def test_documentation_is_deterministic_across_clock_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeRecipe(doc_files={PAGE: "# Widgets\n\nProse.\n"}))
    starts = (datetime(2026, 1, 1, tzinfo=UTC), datetime(2031, 7, 9, 12, 30, tzinfo=UTC))
    hashes = [
        content_hashes(document(tmp_path / f"run{index}", clock_start=start).result)
        for index, start in enumerate(starts)
    ]
    assert set(hashes[0]) == {Kind.DOCS_CHANGESET}
    assert hashes[0] == hashes[1]


def test_the_docs_change_set_is_schema_valid_and_attributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)
    run = document(tmp_path)
    (draft,) = run.result.artifacts
    assert draft.kind == Kind.DOCS_CHANGESET and isinstance(draft.content, dict)
    assert validate_payload(Kind.DOCS_CHANGESET, draft.content) == []
    assert ChangeSet.from_dict(draft.content) == run.changeset
    assert run.changeset.author == "documenter"
    assert run.changeset.description == "documentation for the integrated change"
    assert run.result.rationale == "documented 1 capability(ies) in 2 file(s)."
    assert run.result.usage["tool_calls"] > 0 and run.result.usage["bytes_written"] > 0
