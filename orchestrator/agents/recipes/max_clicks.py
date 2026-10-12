"""Capability ``max_clicks``: an optional per-link click limit for the real URL shortener.

The brownfield scenario ("Click limit for short links") changes the *real* product in
``shortener/``.  This recipe does that through **anchored edits**: every edit is an
``(anchor, replacement)`` pair stored in ``files/max_clicks/*.edits.tmpl``; the anchor must occur
**exactly once** in the file the recipe reads through ``ctx.toolbox`` (and the symbols it relies on
must exist in ``ctx.model`` when a ``CodebaseModel`` is given).  Anything else - a renamed method, a
moved module, an already applied change - raises ``RecipeAnchorError``: the recipe never guesses.

Design in one paragraph (the ADRs of ``design()`` carry the alternatives):

* a new forward-only migration adds a nullable ``links.max_clicks`` (existing rows stay unlimited);
* ``LinkRepository.consume_click`` is one atomic conditional ``UPDATE`` whose ``rowcount`` is the
  whole decision, so ``max_clicks=1`` never redirects twice, however many instances serve it;
* ``LinkService.resolve`` keeps the cached fast path for unlimited links; links with a limit take
  their allowance from the database only (the cache holds just the immutable limit) and the cache
  entry is dropped when the limit is reached;
* the background click writer must not count a limited click a second time: ``add_clicks`` skips
  limited links, so ``click_count`` is exact for them and eventually consistent for the rest;
* the API gains an optional ``max_clicks`` request field and the optional response fields
  ``max_clicks`` / ``remaining_clicks`` (omitted for unlimited links, so their JSON is byte for
  byte what it was), plus the additive link status ``exhausted``.

Groups (disjoint write scopes, no dependencies): ``storage`` (migration, repository, the three
existing tests that pin the migration list), ``domain-api`` (model, validation, errors, schemas,
routes) and ``service`` (``LinkService`` and the click writer docs).

Parameters (typed, range checked before anything runs; free text never reaches a template):
``min_value`` / ``max_value`` - the accepted ``max_clicks`` range (default 1 and 1,000,000),
extracted from phrases such as "between 1 and 1,000,000".

Limitations (honest):

* the limit counts *granted redirects* (every 302, including link-preview bots and repeat visits);
* the limit is immutable after creation (the API has no update endpoint) - that immutability is
  what lets the cache keep it;
* ``docs/api.md`` is generated from the OpenAPI schema and a product test compares it with the
  generator.  When it exists in the docs view the recipe edits it so that it equals the
  regenerated file (editing a pre-existing file is what CHG-003 asks a human to approve); the
  scenario baseline does not contain it, so after promotion regenerate it with
  ``python -m shortener.apidoc`` (the feature page says so);
* route decorators and handler signatures are left untouched on purpose: the change policy
  (CHG-004) treats a changed route signature as an API change that needs an OpenAPI diff;
* three existing product tests pin the list of packaged migrations; the storage group updates
  them (a new migration changes the list), so that change set also touches a test file;
* ``BuildContext.anchors['conservative']`` is accepted and ignored: the change set is already the
  minimal one, so the safe fallback agent produces the identical result.

``ctx.anchors['recipe_anchors']`` lists the existing files a group edited (the "recipe anchors" of
the change-scope rule CHG-003).  ``design(params, model)`` verifies a real ``CodebaseModel``
(modules, symbols, routes, the ``links`` table, the migration list) and falls back to the
conventional layout for ``None`` or any other object; ``build_docs`` documents the integrated
view, where the change already exists, and therefore performs no drift check against ``ctx.model``.
"""

from __future__ import annotations

import ast
import importlib.resources
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.recipes.base import (
    AspectSpec,
    BuildContext,
    Capability,
    GroupSpec,
    ParamSpec,
    RecipeAnchorError,
    RecipeParamError,
    validate_params,
)
from orchestrator.core.errors import AgentError
from orchestrator.core.sanitize import strip_controls
from orchestrator.core.schemas import AC_ID_RE, REQ_ID_RE
from orchestrator.core.types import RiskLevel, Toolbox
from orchestrator.tools.codebase import CodebaseModel

CAPABILITY_ID = "max_clicks"
MIGRATION_NAME = "add_max_clicks"
TESTS_PATH = "tests/shortener/test_max_clicks.py"
FEATURE_DOC_PATH = "docs/features/max-clicks.md"
CHANGELOG_PATH = "CHANGELOG.md"
API_DOC_PATH = "docs/api.md"
DEFAULT_MIGRATIONS_DIR = "shortener/storage/migrations"
PARAM_BOUND = 2_147_483_647  # the largest limit a 32-bit client can represent
MAX_DOC_TEXT_CHARS = 700

GROUP_STORAGE = "storage"
GROUP_DOMAIN_API = "domain-api"
GROUP_SERVICE = "service"

_FEATURE_TERM = "max_clicks"
_HOLE = re.compile(r"@@([A-Z][A-Z0-9_]*)@@")
_MIGRATION_FILE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")
_TEMPLATE_NAME = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$")
_CHANGE_REFERENCE = re.compile(r"^CR-[0-9a-f]{6}$")
_DIRECTIVE = "@@@ "
_COMMENT = "@@# "

# --------------------------------------------------------------------------------- parameters

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "min_value",
        "int",
        default=1,
        minimum=1,
        maximum=PARAM_BOUND,
        description="Smallest accepted max_clicks value.",
    ),
    ParamSpec(
        "max_value",
        "int",
        default=1_000_000,
        minimum=1,
        maximum=PARAM_BOUND,
        description="Largest accepted max_clicks value.",
    ),
)

# --------------------------------------------------------------------------------- aspects

ASPECTS: tuple[AspectSpec, ...] = (
    AspectSpec(
        id="optional-field",
        title="max_clicks is an optional field of link creation",
        triggers=(
            r"\boptional(?:ly)?\b[^.]{0,60}\bmax[_ ]?clicks?\b",
            r"\bmax[_ ]?clicks?\b[^.]{0,60}\boptional(?:ly)?\b",
            r"\b(?:supply|provide|specify)\b[^.]{0,40}\bmax[_ ]?clicks?\b",
        ),
        criterion=(
            "Link creation accepts an optional `max_clicks` integer and stores it; "
            "a request without `max_clicks` still returns 201 for an unlimited link."
        ),
    ),
    AspectSpec(
        id="range-validation",
        title="max_clicks must be an integer inside the allowed range",
        triggers=(
            r"\bbetween\s+\d[\d,_]{0,12}\s+and\s+\d",
            r"\binteger\b",
            r"\b(?:valid range|within the range|range of)\b",
        ),
        criterion=(
            "`max_clicks` values below {min_value} or above {max_value:,} and values that "
            "are not integers are rejected with 422 and nothing is stored."
        ),
    ),
    AspectSpec(
        id="gone-after-limit",
        title="requests after the limit answer 410 Gone",
        triggers=(
            r"\b410\b",
            r"\bgone\b",
            r"\bfollowed\b[^.]{0,40}\btimes?\b",
            r"\bfurther requests?\b",
        ),
        criterion=(
            "After a link has been followed `max_clicks` times, "
            "further requests to its short code return 410 Gone."
        ),
    ),
    AspectSpec(
        id="backward-compatible",
        title="links without a limit and existing clients are unaffected",
        triggers=(
            r"\bbreaking\b",
            r"\bnot be affected\b",
            r"\bkeeps? working\b",
            r"\bbackwards?[- ]compatib\w*",
            r"\bexisting (?:links?|clients?|api clients?)\b",
        ),
        criterion=(
            "Links created without `max_clicks`, existing links and existing API clients "
            "behave exactly as before: same responses and no breaking API change."
        ),
    ),
    AspectSpec(
        id="remaining-visible",
        title="the remaining allowance is visible in the link details",
        triggers=(
            r"\bremaining\b",
            r"\ballowance\b",
            r"\bvisible\b[^.]{0,60}\bdetails?\b",
            r"\blink details?\b",
        ),
        criterion=(
            "The link details response shows `max_clicks` and `remaining_clicks` "
            "(never below 0) for a link with a click limit."
        ),
    ),
    AspectSpec(
        id="concurrency-safe",
        title="the limit holds under concurrent requests",
        triggers=(
            r"\bconcurren\w*",
            r"\b(?:race|parallel|simultaneous)\w*",
            r"\bnever redirects? twice\b",
            r"\bhold under\b",
        ),
        criterion=(
            "Under concurrent requests a link with `max_clicks` of N redirects exactly N "
            "times in total, and a link with `max_clicks` of 1 never redirects twice."
        ),
    ),
)
ASPECT_IDS = tuple(aspect.id for aspect in ASPECTS)

# --------------------------------------------------------------------------------- layout

#: role -> (dotted module, symbols the edits rely on). The symbols are checked against the
#: CodebaseModel when one is given; the anchors are checked against the file text always.
_TARGETS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "repository": (
        "shortener.storage.link_repository",
        (
            "LinkRepository.add_clicks",
            "SqliteLinkRepository.insert",
            "SqliteLinkRepository.get_by_code",
            "SqliteLinkRepository.list_by_owner",
            "SqliteLinkRepository.add_clicks",
            "_to_link",
            "_parse_optional",
        ),
    ),
    "storage_tests": (
        "tests.shortener.test_shortener_storage",
        (
            "test_packaged_migrations_are_ordered_and_checksummed",
            "test_migrate_applies_everything_once",
            "test_concurrent_migrators_apply_each_migration_exactly_once",
        ),
    ),
    "models": (
        "shortener.domain.models",
        ("LinkStatus.DISABLED", "NewLink.created_by", "Link.created_by", "Link.status_at"),
    ),
    "errors": ("shortener.domain.errors", ("InvalidExpiryError",)),
    "validation": (
        "shortener.domain.validation",
        ("MAX_EXPIRY_SPAN", "resolve_expiry", "validate_idempotency_key"),
    ),
    "schemas": (
        "shortener.api.schemas",
        ("CreateLinkRequest.ttl_seconds", "LinkResponse.click_count", "LinkResponse.from_link"),
    ),
    "routes_links": ("shortener.api.routes_links", ("create_link", "get_link")),
    "service": (
        "shortener.services.link_service",
        (
            "CachedTarget.disabled",
            "LinkService.create",
            "LinkService.resolve",
            "LinkService._lookup",
            "LinkService._insert_alias",
            "LinkService._insert_generated",
        ),
    ),
    "recorder": ("shortener.services.click_recorder", ("ClickWriter",)),
}

_SCRIPTS: Mapping[str, str] = {
    "repository": "link_repository.edits.tmpl",
    "storage_tests": "storage_tests.edits.tmpl",
    "models": "models.edits.tmpl",
    "errors": "errors.edits.tmpl",
    "validation": "validation.edits.tmpl",
    "schemas": "schemas.edits.tmpl",
    "routes_links": "routes_links.edits.tmpl",
    "service": "link_service.edits.tmpl",
    "recorder": "click_recorder.edits.tmpl",
}

_GROUP_ROLES: Mapping[str, tuple[str, ...]] = {
    GROUP_STORAGE: ("repository", "storage_tests"),
    GROUP_DOMAIN_API: ("models", "errors", "validation", "schemas", "routes_links"),
    GROUP_SERVICE: ("service", "recorder"),
}

_GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        id=GROUP_STORAGE,
        title="Storage: migration, atomic click consumption, repository",
        write_scope=("shortener/storage/**", "tests/shortener/test_shortener_storage.py"),
        expected_paths=(
            f"{DEFAULT_MIGRATIONS_DIR}/*_{MIGRATION_NAME}.sql",
            "shortener/storage/link_repository.py",
            "tests/shortener/test_shortener_storage.py",
        ),
    ),
    GroupSpec(
        id=GROUP_DOMAIN_API,
        title="Domain and API: model, validation, schemas, routes",
        write_scope=("shortener/domain/**", "shortener/api/**"),
        expected_paths=(
            "shortener/domain/models.py",
            "shortener/domain/errors.py",
            "shortener/domain/validation.py",
            "shortener/api/schemas.py",
            "shortener/api/routes_links.py",
        ),
    ),
    GroupSpec(
        id=GROUP_SERVICE,
        title="Service: create and resolve with the atomic consume",
        write_scope=("shortener/services/**",),
        expected_paths=(
            "shortener/services/link_service.py",
            "shortener/services/click_recorder.py",
        ),
    ),
)

_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/api/v1/links"),
    ("GET", "/api/v1/links"),
    ("GET", "/api/v1/links/{code}"),
    ("GET", "/{code}"),
)

# --------------------------------------------------------------------------------- templates


def _template(name: str) -> str:
    """Read a payload file shipped next to this module (LF-normalised)."""
    if not _TEMPLATE_NAME.match(name):
        raise _internal(f"invalid template name {name!r}")
    root = importlib.resources.files("orchestrator.agents.recipes")
    try:
        text = root.joinpath("files", CAPABILITY_ID, name).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise _internal(f"payload file {name} is missing from the recipe package") from exc
    return text.replace("\r\n", "\n")


def _fill(text: str, values: Mapping[str, str]) -> str:
    """Replace ``@@NAME@@`` holes; unknown holes and leftovers are bugs, never silently kept."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise _internal(f"template hole @@{key}@@ has no value")
        return values[key]

    return _HOLE.sub(replace, text)


@dataclass(frozen=True)
class _Edit:
    ident: str
    anchor: str
    replacement: str


def _parse_edits(name: str, text: str) -> list[_Edit]:
    """Parse an edit script: ``@@@ edit <id>`` / ``@@@ anchor`` / ``@@@ replace`` / ``@@@ end``."""
    edits: list[_Edit] = []
    ident = ""
    mode = ""  # "", "edit", "anchor", "replace"
    anchor_lines: list[str] = []
    buffer: list[str] = []
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if line.startswith(_DIRECTIVE):
            words = line[len(_DIRECTIVE) :].split()
            word = words[0] if words else ""
            if word == "edit" and len(words) == 2 and mode == "":
                ident, mode = words[1], "edit"
            elif word == "anchor" and len(words) == 1 and mode == "edit":
                mode, buffer = "anchor", []
            elif word == "replace" and len(words) == 1 and mode == "anchor" and buffer:
                anchor_lines, mode, buffer = buffer, "replace", []
            elif word == "end" and len(words) == 1 and mode == "replace":
                edits.append(_Edit(ident, "".join(anchor_lines), "".join(buffer)))
                mode, buffer = "", []
            else:
                raise _internal(f"{name}:{number}: unexpected directive {line.strip()!r}")
        elif mode in ("anchor", "replace"):
            buffer.append(line)
        elif line.strip() and not line.startswith(_COMMENT):
            raise _internal(f"{name}:{number}: text outside an edit block")
    if mode != "":
        raise _internal(f"{name}: unterminated edit {ident!r}")
    if not edits:
        raise _internal(f"{name}: no edits")
    return edits


def _internal(message: str) -> AgentError:
    """A defect of the recipe's own payload (never of the baseline): fatal, never retried."""
    return AgentError(f"max_clicks recipe: {message}", kind="fatal", retryable=False)


# --------------------------------------------------------------------------------- parameters


def _checked(params: Mapping[str, Any]) -> dict[str, Any]:
    """Validated parameters; the range must be a real range."""
    values = validate_params(PARAMS, params)
    if values["min_value"] > values["max_value"]:
        raise RecipeParamError("min_value must not exceed max_value")
    return values


def _py_int(value: int) -> str:
    """Digits (with underscores) of a validated integer: the only way numbers enter templates."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeParamError("expected an integer")
    return format(value, "_")


def _holes(values: Mapping[str, Any]) -> dict[str, str]:
    low, high = values["min_value"], values["max_value"]
    return {
        "MIN_VALUE": _py_int(low),
        "MAX_VALUE": _py_int(high),
        "MIN_VALUE_PLAIN": str(int(low)),
        "MAX_VALUE_PLAIN": str(int(high)),
        "MIN_VALUE_COMMAS": format(int(low), ","),
        "MAX_VALUE_COMMAS": format(int(high), ","),
    }


# --------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class _Migrations:
    directory: str
    names: tuple[str, ...]  # existing migration names in order ("init", "idempotency", ...)

    @property
    def new_version(self) -> int:
        return len(self.names) + 1

    @property
    def stem(self) -> str:
        return f"{self.new_version:03d}_{MIGRATION_NAME}"

    @property
    def path(self) -> str:
        return f"{self.directory}/{self.stem}.sql"

    def holes(self) -> dict[str, str]:
        old = [str(v) for v in range(1, len(self.names) + 1)]
        new = [*old, str(self.new_version)]
        return {
            "MIGRATION_STEM": self.stem,
            "OLD_VERSIONS": f"[{', '.join(old)}]",
            "NEW_VERSIONS": f"[{', '.join(new)}]",
            "OLD_NAMES": "[" + ", ".join(f'"{n}"' for n in self.names) + "]",
            "NEW_NAMES": "[" + ", ".join(f'"{n}"' for n in (*self.names, MIGRATION_NAME)) + "]",
        }


def _parse_migrations(paths: Sequence[str]) -> _Migrations:
    """Existing migrations 001..N (the runner's own rule) and where the next one goes."""
    parsed: list[tuple[int, str]] = []
    directories: set[str] = set()
    for path in paths:
        directory, _, base = path.rpartition("/")
        match = _MIGRATION_FILE.match(base)
        if match is None:
            raise RecipeAnchorError(f"unexpected migration file name: {base!r}")
        directories.add(directory)
        parsed.append((int(match.group(1)), match.group(2)))
    if not parsed or len(directories) != 1:
        raise RecipeAnchorError("the baseline has no single migrations directory")
    parsed.sort()
    if [version for version, _ in parsed] != list(range(1, len(parsed) + 1)):
        raise RecipeAnchorError("migration files are not numbered 001..N without gaps")
    names = tuple(name for _, name in parsed)
    if MIGRATION_NAME in names:
        raise RecipeAnchorError(f"migration {MIGRATION_NAME!r} already exists: change applied?")
    if len(names) >= 998:
        raise RecipeAnchorError("no free migration number")
    return _Migrations(next(iter(directories)), names)


def _model_of(model: Any) -> CodebaseModel | None:
    """Only a real ``CodebaseModel`` is trusted for verification; anything else is ignored."""
    return model if isinstance(model, CodebaseModel) else None


def _locate(model: CodebaseModel | None, role: str) -> str:
    """Path of ``role``'s module: from the model (symbols verified) or the conventional layout."""
    dotted, symbols = _TARGETS[role]
    if model is None:
        return dotted.replace(".", "/") + ".py"
    info = model.module(dotted)
    if info is None:
        raise RecipeAnchorError(f"module {dotted} is not part of the analysed codebase")
    known = {definition.name for definition in info.defs}
    missing = [symbol for symbol in symbols if symbol not in known]
    if missing:
        raise RecipeAnchorError(f"{dotted} no longer defines {', '.join(missing)}")
    return info.path


def _applied_or_next_migration(toolbox: Toolbox) -> tuple[str, int]:
    """Path and number of this change's migration: the existing file, or where it would go."""
    existing = toolbox.list_files(f"{DEFAULT_MIGRATIONS_DIR}/*.sql")
    applied = [path for path in existing if path.endswith(f"_{MIGRATION_NAME}.sql")]
    if len(applied) > 1:
        raise RecipeAnchorError(f"more than one {MIGRATION_NAME} migration: {applied}")
    if applied:
        match = _MIGRATION_FILE.match(applied[0].rpartition("/")[2])
        if match is None:
            raise RecipeAnchorError(f"unexpected migration file name: {applied[0]!r}")
        return applied[0], int(match.group(1))
    pending = _parse_migrations(existing)
    return pending.path, pending.new_version


def _migrations_of(model: CodebaseModel | None, toolbox: Toolbox | None) -> _Migrations:
    paths = list(model.migrations) if model is not None else []
    if not paths and toolbox is not None:
        paths = toolbox.list_files(f"{DEFAULT_MIGRATIONS_DIR}/*.sql")
    if model is not None:
        for table in model.tables:
            if table.name == "links" and _FEATURE_TERM in table.columns:
                raise RecipeAnchorError("links.max_clicks already exists: change applied?")
    return _parse_migrations(paths)


# --------------------------------------------------------------------------------- file edits


def _read(toolbox: Toolbox, path: str) -> str:
    try:
        return toolbox.read_text(path)
    except FileNotFoundError:
        raise RecipeAnchorError(f"{path} does not exist in the baseline") from None


def _compiles(path: str, text: str) -> bool:
    if not path.endswith(".py"):
        return True
    try:
        ast.parse(text, filename=path)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return True


def _apply_script(
    ctx: BuildContext, path: str, script: str, holes: Mapping[str, str], *, guard: bool = True
) -> None:
    """Apply every edit of ``script`` to ``path``; each anchor must match exactly once."""
    text = _read(ctx.toolbox, path)
    if guard and _FEATURE_TERM in text:
        raise RecipeAnchorError(f"{path} already mentions {_FEATURE_TERM}: change applied?")
    for edit in _parse_edits(script, _fill(_template(script), holes)):
        found = text.count(edit.anchor)
        if found != 1:
            raise RecipeAnchorError(
                f"{path}: anchor '{edit.ident}' occurs {found} time(s), expected exactly 1; "
                "the baseline has drifted from the recipe"
            )
        text = text.replace(edit.anchor, edit.replacement, 1)
    if not _compiles(path, text):
        raise RecipeAnchorError(f"{path} does not compile after the edits; the baseline drifted")
    ctx.toolbox.write_text(path, text)
    recorded: list[str] = ctx.anchors.setdefault("recipe_anchors", [])
    if path not in recorded:
        recorded.append(path)


def _write_new(toolbox: Toolbox, path: str, text: str) -> None:
    if toolbox.exists(path):
        raise RecipeAnchorError(f"{path} already exists in the baseline; refusing to overwrite it")
    toolbox.write_text(path, text)


# --------------------------------------------------------------------------------- tests module

_TEST_STDLIB_PLAIN = ("sqlite3", "threading")
_TEST_STDLIB_FROM: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("collections.abc", ("Callable",)),
    ("concurrent.futures", ("ThreadPoolExecutor",)),
    ("functools", ("partial",)),
    ("pathlib", ("Path",)),
    ("typing", ("Any",)),
)
_TEST_THIRD_PARTY_PLAIN = ("pytest",)
_TEST_THIRD_PARTY_FROM: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fastapi", ("FastAPI",)),
    ("fastapi.testclient", ("TestClient",)),
)
_TEST_FIRST_PARTY_FROM: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("shortener.config", ("Settings",)),
    ("shortener.container", ("Services",)),
    ("shortener.domain.codegen", ("CodeGenerator",)),
    ("shortener.domain.errors", ("LinkGoneError",)),
    ("shortener.domain.models", ("Link", "LinkStatus", "NewLink")),
    ("shortener.domain.validation", ("UrlPolicy",)),
    ("shortener.observability.metrics", ("Metrics",)),
    ("shortener.services.cache", ("LruTtlCache",)),
    ("shortener.services.idempotency", ("request_fingerprint",)),
    ("shortener.services.link_service", ("LinkService", "RedirectCache")),
    ("shortener.storage.db", ("Database",)),
    ("shortener.storage.link_repository", ("SqliteLinkRepository",)),
    ("shortener.storage.migrations", ("load_migrations", "migrate")),
    (
        "tests.shortener.support",
        ("FakeClock", "api_client", "build_app", "make_settings", "new_secret"),
    ),
)


def _import_block(body: str) -> str:
    """isort-ordered imports of exactly the names ``body`` uses (no unused import, ever)."""

    def used(name: str) -> bool:
        return re.search(rf"(?<![\w.]){re.escape(name)}\b", body) is not None

    def statement(module: str, names: Sequence[str]) -> str | None:
        kept = [name for name in names if used(name)]
        if not kept:
            return None
        line = f"from {module} import {', '.join(kept)}"
        if len(line) <= 100:
            return line
        inner = "".join(f"    {name},\n" for name in kept)
        return f"from {module} import (\n{inner})"

    sections: list[str] = []
    for plain, froms in (
        (_TEST_STDLIB_PLAIN, _TEST_STDLIB_FROM),
        (_TEST_THIRD_PARTY_PLAIN, _TEST_THIRD_PARTY_FROM),
        ((), _TEST_FIRST_PARTY_FROM),
    ):
        lines = [f"import {name}" for name in plain if re.search(rf"\b{name}\.", body)]
        lines += [s for module, names in froms if (s := statement(module, names)) is not None]
        if lines:
            sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _coverage(requirements: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """aspect id -> ``R-xxx`` / ``AC-xxx-y`` ids of the active requirements that carry it."""
    if not isinstance(requirements, Mapping):
        raise RecipeParamError("build_tests needs the requirements artifact (ctx.requirements)")
    items = requirements.get("requirements")
    if not isinstance(items, Sequence) or isinstance(items, str | bytes):
        raise RecipeParamError("requirements.requirements must be a list")
    found: dict[str, list[str]] = {}
    for requirement in items:
        if not isinstance(requirement, Mapping) or requirement.get("status") != "active":
            continue
        rid = requirement.get("id")
        if not isinstance(rid, str) or REQ_ID_RE.match(rid) is None:
            raise RecipeParamError("a requirement has an invalid id")
        criteria = requirement.get("acceptance_criteria")
        for criterion in criteria if isinstance(criteria, Sequence) else ():
            if not isinstance(criterion, Mapping) or criterion.get("aspect") not in ASPECT_IDS:
                continue
            cid = criterion.get("id")
            if not isinstance(cid, str) or AC_ID_RE.match(cid) is None:
                raise RecipeParamError("an acceptance criterion has an invalid id")
            ids = found.setdefault(str(criterion["aspect"]), [])
            ids.extend(token for token in (rid, cid) if token not in ids)
    if not found:
        raise RecipeParamError("the requirements carry no max_clicks acceptance criterion")
    return found


# --------------------------------------------------------------------------------- docs


def _text(value: object) -> str:
    """One safe Markdown line: controls stripped, whitespace collapsed, markup neutralised."""
    line = " ".join(strip_controls(str(value)).split())[:MAX_DOC_TEXT_CHARS]
    return line.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _adr_markdown(adrs: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for adr in adrs:
        lines = [f"### {_text(adr['id'])}: {_text(adr['title'])}", ""]
        lines.append(f"- **Decision.** {_text(adr['decision'])}")
        lines.append("- **Alternatives considered.**")
        lines.extend(
            f"  - *{_text(alt['option'])}*: {_text(alt['rejected_because'])}"
            for alt in adr["alternatives"]
        )
        lines.append(f"- **Rationale.** {_text(adr['rationale'])}")
        lines.append(f"- **Consequences.** {_text(adr['consequences'])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _design_adrs(
    design: Mapping[str, Any] | None, own: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The recipe's ADRs, numbered like the run's design document when it carries them."""
    published: dict[str, str] = {}
    adrs = design.get("adrs") if isinstance(design, Mapping) else None
    for adr in adrs if isinstance(adrs, Sequence) else ():
        if isinstance(adr, Mapping) and isinstance(adr.get("title"), str):
            published[adr["title"]] = str(adr.get("id", ""))
    return [{**adr, "id": published.get(str(adr["title"]), adr["id"])} for adr in own]


# --------------------------------------------------------------------------------- design text

_ADRS: tuple[dict[str, Any], ...] = (
    {
        "id": "ADR-001",
        "title": "Enforce the limit with one atomic conditional UPDATE on the redirect path",
        "decision": (
            "LinkRepository.consume_click runs UPDATE links SET click_count = click_count + 1 "
            "WHERE code = ? AND disabled_at IS NULL AND (max_clicks IS NULL OR click_count < "
            "max_clicks) and LinkService.resolve treats rowcount 0 as 'no click left'. Only "
            "links with a limit take this path; links without one keep the cached fast path."
        ),
        "alternatives": [
            {
                "option": "In-memory counter per process",
                "rejected_because": (
                    "State is lost on restart and not shared between workers or instances, so "
                    "every instance would grant the full allowance and the counter would "
                    "diverge from the persisted click_count."
                ),
            },
            {
                "option": "Reuse the eventually consistent click_count written by the recorder",
                "rejected_because": (
                    "The counter lags redirects by the queue delay and is lossy under overload, "
                    "so concurrent requests overshoot the limit and max_clicks=1 could "
                    "redirect twice."
                ),
            },
            {
                "option": "Read the link, compare in Python, then update (cache bypassed)",
                "rejected_because": (
                    "Check-then-act: two requests can both read click_count < max_clicks and "
                    "both redirect. Check and increment must be one atomic statement."
                ),
            },
            {
                "option": "BEGIN IMMEDIATE, SELECT, then UPDATE in one transaction",
                "rejected_because": (
                    "Correct, but two statements under the write lock for every redirect; the "
                    "single conditional UPDATE is equally atomic and cheaper."
                ),
            },
        ],
        "rationale": (
            "SQLite serialises writers, so the WHERE clause is evaluated and the increment "
            "applied under the same write lock; the outcome does not depend on how many "
            "processes serve redirects, and rowcount is the whole decision."
        ),
        "consequences": (
            "Every redirect of a limited link costs one short write transaction (the redirect "
            "rate limiter still runs first). Links without a limit are unaffected. A disabled "
            "limited link stops at once on every instance because the UPDATE re-checks "
            "disabled_at."
        ),
    },
    {
        "id": "ADR-002",
        "title": "Cache only the immutable facts of limited links; invalidate at the limit",
        "decision": (
            "CachedTarget gains max_clicks (immutable after creation). The allowance is never "
            "read from the cache: when consume_click refuses, the entry is invalidated and the "
            "database row decides between 410 disabled and 410 exhausted."
        ),
        "alternatives": [
            {
                "option": "Cache the remaining count with the positive TTL",
                "rejected_because": (
                    "A stale count lets requests through after the limit, for up to the TTL and "
                    "on every other instance."
                ),
            },
            {
                "option": "Never cache limited links (read the row on every redirect)",
                "rejected_because": (
                    "Adds a read per redirect without improving correctness: the conditional "
                    "UPDATE already carries the state check."
                ),
            },
            {
                "option": "Invalidate the entry on every consumed click",
                "rejected_because": (
                    "Defeats the cache for limited links and adds lock traffic; only the "
                    "transition to exhausted matters."
                ),
            },
        ],
        "rationale": (
            "Everything the cache holds about a link stays true for its lifetime except "
            "disabled, which the UPDATE re-checks for limited links; dropping the entry at "
            "exhaustion keeps no stale positive entry around."
        ),
        "consequences": (
            "Links without a limit keep today's cache behaviour and revocation lag. An "
            "exhausted link costs one UPDATE (no row changes), one lookup and a 410 per "
            "request, bounded by the redirect rate limiter; caching the terminal state is a "
            "possible follow-up."
        ),
    },
    {
        "id": "ADR-003",
        "title": "The background click recorder must not count limited links twice",
        "decision": (
            "LinkRepository.add_clicks gets AND max_clicks IS NULL: the writer still stores the "
            "analytics rows of limited links but leaves click_count to consume_click. "
            "click_count is exact for limited links and eventually consistent for the others."
        ),
        "alternatives": [
            {
                "option": "Keep the writer as the only incrementer and let resolve only check",
                "rejected_because": (
                    "That is the eventually consistent counter rejected in ADR-001."
                ),
            },
            {
                "option": "Increment in resolve for every link and drop the writer's update",
                "rejected_because": (
                    "Gives every redirect a database write and removes the batching that keeps "
                    "analytics off the redirect latency."
                ),
            },
            {
                "option": "A second counter column next to click_count",
                "rejected_because": (
                    "Two overlapping counters to migrate, explain and keep in sync; the "
                    "nullable max_clicks already says which writer owns click_count."
                ),
            },
            {
                "option": "Reconcile double counts afterwards",
                "rejected_because": (
                    "Cannot be made atomic and would show a wrong remaining allowance in between."
                ),
            },
        ],
        "rationale": (
            "In v1 click_count is bumped only by the batching writer after the redirect was "
            "already answered. With the atomic consume on the redirect path the writer would "
            "add every limited click a second time, so the single ownership rule lives in SQL."
        ),
        "consequences": (
            "The analytics total of a limited link (clicks table) may be lower than click_count "
            "when the click queue drops events; remaining_clicks always reflects granted "
            "redirects."
        ),
    },
    {
        "id": "ADR-004",
        "title": "Nullable column in a new forward-only migration; roll back by forward fix",
        "decision": (
            "Migration 003 adds links.max_clicks INTEGER with CHECK (max_clicks IS NULL OR "
            "max_clicks >= 1). NULL means unlimited, so existing rows and clients are "
            "unaffected; applied migrations are never touched (their checksums are enforced)."
        ),
        "alternatives": [
            {
                "option": "Edit 001_init.sql",
                "rejected_because": (
                    "Applied migrations are immutable: the checksum is verified at startup and "
                    "orchestrator rule CHG-001 forbids it."
                ),
            },
            {
                "option": "A separate link_limits table",
                "rejected_because": (
                    "A second lookup on the redirect path and two inserts per creation for one "
                    "nullable attribute."
                ),
            },
            {
                "option": "NOT NULL DEFAULT 0 where 0 means unlimited",
                "rejected_because": (
                    "A sentinel that makes every comparison and the API ambiguous."
                ),
            },
        ],
        "rationale": (
            "A nullable column is an additive, cheap schema change that every existing query "
            "and the previous release's statements (which name their columns) tolerate."
        ),
        "consequences": (
            "The upper bound is validated by the API and the domain only, because SQLite cannot "
            "alter a CHECK constraint; raising it later needs no table rebuild. Undoing the "
            "migration is a forward fix (see the rollback plan)."
        ),
    },
)

_ROLLBACK_PLAN = (
    "Roll forward, do not roll back. Migration {version} is additive and forward-only, and the "
    "migration runner refuses to start a build that does not know an applied migration "
    "(downgrades are not supported), so redeploying the previous release against a migrated "
    "database fails at startup. Take a backup of the SQLite file (with its WAL) before "
    "deploying. If the feature must be withdrawn: (1) preferred, ship a forward fix that keeps "
    "the migration and stops accepting max_clicks; (2) emergency, stop the service and run the "
    "documented down script by hand: `ALTER TABLE links DROP COLUMN max_clicks;` then "
    "`DELETE FROM schema_migrations WHERE version = {version};` (SQLite 3.35 or newer; the "
    "stored limits are lost and the previous release then redirects without limits); (3) or "
    "restore the pre-deployment backup (writes since the deployment are lost)."
)


# --------------------------------------------------------------------------------- the recipe


class MaxClicksRecipe:
    """The vetted, parameterised change that adds click limits to the shortener."""

    id = CAPABILITY_ID
    params = PARAMS

    # -- decomposition ----------------------------------------------------------------------
    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        """Three parallel groups; every edited file belongs to exactly one of them."""
        _checked(params)
        return _GROUPS

    # -- design -----------------------------------------------------------------------------
    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        """Components, ADRs with alternatives, the migration, the additive API and the rollback.

        Components carry ``aspects`` (the ids of the capability aspects they implement) so the
        architect can map requirements to them; ``requirement_ids`` is left empty.  ``api_paths``
        is the declared delta for the ``api_contract``.
        """
        values = _checked(params)
        codebase = _model_of(model)
        migrations = _migrations_of(codebase, None) if codebase is not None else None
        migration_path = (
            migrations.path
            if migrations is not None
            else f"{DEFAULT_MIGRATIONS_DIR}/{3:03d}_{MIGRATION_NAME}.sql"
        )
        version = migrations.new_version if migrations is not None else 3
        paths = {role: _locate(codebase, role) for role in _TARGETS if role != "storage_tests"}
        if codebase is not None:
            known = {(route.method, route.path) for route in codebase.routes}
            missing = [f"{m} {p}" for m, p in _ROUTES if (m, p) not in known]
            if missing:
                raise RecipeAnchorError(f"routes missing from the codebase: {', '.join(missing)}")
        return self._compose(values, paths, migration_path, version)

    @staticmethod
    def _compose(
        values: Mapping[str, Any], paths: Mapping[str, str], migration_path: str, version: int
    ) -> dict[str, Any]:
        low, high = values["min_value"], values["max_value"]
        return {
            "components": [
                {
                    "name": "Migration: links.max_clicks",
                    "responsibility": (
                        "Nullable INTEGER column; NULL means unlimited, existing rows keep "
                        "behaving as before."
                    ),
                    "touches": [migration_path],
                    "requirement_ids": [],
                    "aspects": ["backward-compatible"],
                },
                {
                    "name": "LinkRepository.consume_click",
                    "responsibility": (
                        "Persist max_clicks; atomic conditional UPDATE that takes one click; "
                        "the background writer skips limited links."
                    ),
                    "touches": [paths["repository"]],
                    "requirement_ids": [],
                    "aspects": ["gone-after-limit", "concurrency-safe", "remaining-visible"],
                },
                {
                    "name": "Link domain model and validation",
                    "responsibility": (
                        f"Link.max_clicks / remaining_clicks, status exhausted, range check "
                        f"{low}..{high}."
                    ),
                    "touches": [paths["models"], paths["validation"], paths["errors"]],
                    "requirement_ids": [],
                    "aspects": ["optional-field", "range-validation", "remaining-visible"],
                },
                {
                    "name": "API schemas and routes",
                    "responsibility": (
                        "Optional max_clicks request field; max_clicks and remaining_clicks "
                        "response fields omitted for unlimited links; 410 documentation."
                    ),
                    "touches": [paths["schemas"], paths["routes_links"]],
                    "requirement_ids": [],
                    "aspects": [
                        "optional-field",
                        "range-validation",
                        "backward-compatible",
                        "remaining-visible",
                    ],
                },
                {
                    "name": "LinkService.resolve and create",
                    "responsibility": (
                        "Limited links consume a click atomically (cache bypassed for the "
                        "allowance) and answer 410 when the limit is used up; unlimited links "
                        "keep the cached fast path."
                    ),
                    "touches": [paths["service"], paths["recorder"]],
                    "requirement_ids": [],
                    "aspects": ["gone-after-limit", "concurrency-safe", "backward-compatible"],
                },
            ],
            "adrs": [dict(adr) for adr in _ADRS],
            "data_model_changes": [
                {
                    "table": "links",
                    "change": "add_column",
                    "detail": (
                        "max_clicks INTEGER NULL CHECK (max_clicks IS NULL OR max_clicks >= "
                        f"{low}); NULL means unlimited"
                    ),
                    "migration": migration_path,
                }
            ],
            "api_changes": [
                {"method": m, "path": p, "change": "modify", "breaking": False} for m, p in _ROUTES
            ],
            "api_paths": [
                {
                    "method": "POST",
                    "path": "/api/v1/links",
                    "summary": "Create a short link (optional max_clicks)",
                    "request_fields": [
                        {"name": "max_clicks", "required": False, "type": "integer"}
                    ],
                    "responses": [201, 401, 409, 413, 422, 429, 503],
                },
                {
                    "method": "GET",
                    "path": "/api/v1/links",
                    "summary": "List links (max_clicks and remaining_clicks when limited)",
                    "request_fields": [],
                    "responses": [200, 401, 422, 429],
                },
                {
                    "method": "GET",
                    "path": "/api/v1/links/{code}",
                    "summary": "Link details with the remaining click allowance",
                    "request_fields": [],
                    "responses": [200, 401, 404, 422, 429],
                },
                {
                    "method": "GET",
                    "path": "/{code}",
                    "summary": "Redirect; 410 once the click limit is used up",
                    "request_fields": [],
                    "responses": [302, 404, 410, 422, 429],
                },
            ],
            "rollback_plan": _ROLLBACK_PLAN.format(version=version),
        }

    # -- implementation ---------------------------------------------------------------------
    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        """Stage one group's anchored edits (and the new migration for ``storage``)."""
        values = _checked(ctx.params)
        roles = _GROUP_ROLES.get(group_id)
        if roles is None:
            raise RecipeParamError(f"unknown group {group_id!r}; known: {sorted(_GROUP_ROLES)}")
        codebase = _model_of(ctx.model)
        holes = _holes(values)
        migrations = _migrations_of(codebase, ctx.toolbox)
        holes.update(migrations.holes())
        if group_id == GROUP_STORAGE:
            _write_new(ctx.toolbox, migrations.path, _fill(_template("migration.sql.tmpl"), holes))
        for role in roles:
            _apply_script(ctx, _locate(codebase, role), _SCRIPTS[role], holes)

    def build_tests(self, ctx: BuildContext) -> None:
        """Stage ``tests/shortener/test_max_clicks.py``: one test chunk per required aspect."""
        values = _checked(ctx.params)
        coverage = _coverage(ctx.requirements)
        holes = _holes(values)
        chunks = [
            _fill(
                _template(f"tests_{aspect.replace('-', '_')}.py.tmpl"),
                {**holes, "COVERS": ", ".join(ids)},
            ).strip("\n")
            for aspect in ASPECT_IDS
            if (ids := coverage.get(aspect))
        ]
        head, marker, helpers = _template("tests_header.py.tmpl").partition("@@IMPORTS@@\n")
        if not marker:
            raise _internal("tests_header.py.tmpl has no @@IMPORTS@@ line")
        code = _fill(helpers, holes).strip("\n") + "\n\n\n" + "\n\n\n".join(chunks) + "\n"
        source = _fill(head, holes) + _import_block(code) + "\n\n" + code
        _write_new(ctx.toolbox, TESTS_PATH, source)

    def build_docs(self, ctx: BuildContext) -> None:
        """Stage the feature page, the changelog entry and (when present) the API reference."""
        values = _checked(ctx.params)
        # The docs node reads the *integrated* view, where the migration already exists, so the
        # drift checks of ``design()`` do not apply: the documented facts come from that view.
        migration_path, version = _applied_or_next_migration(ctx.toolbox)
        paths = {role: _locate(None, role) for role in _TARGETS if role != "storage_tests"}
        design = self._compose(values, paths, migration_path, version)
        holes = _holes(values)
        reference = ctx.design.get("change_reference") if isinstance(ctx.design, Mapping) else None
        adrs = _design_adrs(ctx.design, design["adrs"])
        holes.update(
            {
                "CHANGE_REFERENCE": (
                    f"`{reference}`"
                    if isinstance(reference, str) and _CHANGE_REFERENCE.match(reference)
                    else "not assigned"
                ),
                "MIGRATION_PATH": _text(design["data_model_changes"][0]["migration"]),
                "ADRS": _adr_markdown(adrs),
                "ROLLBACK": _text(design["rollback_plan"]),
            }
        )
        _write_new(ctx.toolbox, FEATURE_DOC_PATH, _fill(_template("feature.md.tmpl"), holes))
        self._changelog(ctx, holes)
        if ctx.toolbox.exists(API_DOC_PATH):
            _apply_script(ctx, API_DOC_PATH, "api_doc.edits.tmpl", holes)

    @staticmethod
    def _changelog(ctx: BuildContext, holes: Mapping[str, str]) -> None:
        entry = _fill(_template("changelog_entry.md.tmpl"), holes)
        if not ctx.toolbox.exists(CHANGELOG_PATH):
            header = (
                "# Changelog\n\nAll notable changes to this project are documented here.\n\n"
                "## [Unreleased]\n\n"
            )
            ctx.toolbox.write_text(CHANGELOG_PATH, header + entry)
            return
        text = _read(ctx.toolbox, CHANGELOG_PATH)
        marker = entry.split("\n", 1)[0]
        if marker in text:
            return
        heading = "## [Unreleased]\n"
        if text.count(heading) != 1:
            raise RecipeAnchorError(f"{CHANGELOG_PATH} has no single '## [Unreleased]' heading")
        ctx.toolbox.write_text(CHANGELOG_PATH, text.replace(heading, f"{heading}\n{entry}", 1))


CAPABILITY = Capability(
    id=CAPABILITY_ID,
    title="Per-link click limit (max_clicks)",
    kind="enhancement",
    triggers=(
        r"\bmax[_ -]?clicks?\b",
        r"\bclick (?:limit|allowance|cap|quota)s?\b",
        r"\b(?:single|limited)[- ]use\b",
        r"\b410 gone\b",
        r"\bfollowed\b[^.]{0,40}\btimes\b",
    ),
    aspects=ASPECTS,
    extractors={
        "min_value": r"\b(?:between|from)\s+(\d[\d,_]{0,12})\s+(?:and|to)\s+\d[\d,_]{0,12}",
        "max_value": r"\b(?:between|from)\s+\d[\d,_]{0,12}\s+(?:and|to)\s+(\d[\d,_]{0,12})",
    },
    impact_keywords=("link", "click", "redirect", "resolve"),
    risk=RiskLevel.HIGH,
    recipe=MaxClicksRecipe(),
    requires_baseline=True,
)
