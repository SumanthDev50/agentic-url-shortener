"""Capability ``analytics_export``: CSV export of daily click counts per link (brownfield recipe).

Delivers, through *anchored edits* of the real product, ``GET /api/v1/links/{code}/analytics/
export.csv?days=N``: one zero-filled CSV row per UTC day (``code,date,clicks,top_referrer``) for a
window of up to ``max_window_days`` days. Honesty: this is a deterministic stand-in for a
model-backed engineer. It cannot adapt to a product that drifted from the layout it was written
against; it detects that and raises ``RecipeAnchorError`` instead of guessing.

Groups (each file belongs to exactly one, so the change sets never overlap):

* ``csv-encoding`` - new ``shortener/services/csv_export.py``: RFC 4180 rendering, spreadsheet
  formula-injection neutralisation (cells starting with ``=``, ``+``, ``-``, ``@``, tab or CR get a
  single-quote prefix) and header-safe file names.
* ``storage`` - ``click_repository.py``: ``daily_top_referrers`` (one aggregate query, no
  migration).
* ``service`` - ``analytics_service.py``: window constants and ``export_daily_csv`` (bounds check).
* ``route`` - ``routes_links.py``: the authenticated, owner-scoped route (it declares the same
  auth dependency as its sibling analytics route, read from the code model) and, when the
  baseline view contains it, ``docs/api.md`` (generated from the OpenAPI schema; the freshness
  test of the product would otherwise fail).

Anchors. Every edit locates its insertion point by a text anchor that must occur exactly once and
checks its preconditions with the AST of the file (imports, classes, methods) and, when the caller
supplies the ``CodebaseModel``, against the model (module paths, the sibling route and its auth
dependency). A missing, duplicated or already-applied anchor raises ``RecipeAnchorError``; its
message names the anchor, never file content.

Security. Parameters are typed and range-checked before anything runs (``validate_params``);
templates are static text with typed holes (``@@NAME@@``) that only accept integers, identifiers
and a conservative character set, so a value can never escape a string literal or a docstring.

Limitations (documented, not hidden): the per-day top referrer is one value, not a breakdown; the
export is buffered (bounded by the window), not streamed; the generated tests exist only for
aspects that have an acceptance criterion (the security aspects are implemented regardless, but
tested only when a requirement asks for them); ``docs/api.md`` is edited only if the baseline
view contains it, and the section text mirrors what ``python -m shortener.apidoc`` renders for
FastAPI's current OpenAPI output.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from orchestrator.core.fsutil import read_text_lf
from orchestrator.core.sanitize import escape_markdown
from orchestrator.core.types import RiskLevel, Toolbox
from orchestrator.tools.codebase import CodebaseModel, RouteInfo, analyze_codebase
from orchestrator.tools.workspace import MemoryView

CAPABILITY_ID = "analytics_export"

CSV_PATH = "shortener/services/csv_export.py"
REPOSITORY_PATH = "shortener/storage/click_repository.py"
SERVICE_PATH = "shortener/services/analytics_service.py"
ROUTES_PATH = "shortener/api/routes_links.py"
API_DOC_PATH = "docs/api.md"
FEATURE_DOC_PATH = "docs/features/analytics-export.md"
CHANGELOG_PATH = "CHANGELOG.md"
TESTS_PATH = "tests/shortener/test_analytics_export.py"
PERF_TESTS_PATH = "tests/shortener/test_export_perf.py"

SERVICE_MODULE = "shortener.services.analytics_service"
REPOSITORY_MODULE = "shortener.storage.click_repository"
ROUTES_MODULE = "shortener.api.routes_links"
ANALYTICS_ROUTE = "/api/v1/links/{code}/analytics"
EXPORT_ROUTE = ANALYTICS_ROUTE + "/export.csv"

DEFAULT_WINDOW_DAYS = 30
IN_PROCESS_FACTOR = 4
IN_PROCESS_FLOOR_MS = 500

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "max_window_days",
        "int",
        default=90,
        minimum=1,
        maximum=730,
        description="Largest selectable export window in UTC calendar days.",
    ),
    ParamSpec(
        "perf_budget_ms",
        "int",
        default=250,
        minimum=50,
        maximum=5000,
        description="Target p95 latency of the export in milliseconds.",
    ),
)

_SECURITY_WORDS = (r"\bsecur(?:e|ity)\b", r"\bsafe(?:r|ty)?\b")

ASPECTS: tuple[AspectSpec, ...] = (
    AspectSpec(
        "csv-export",
        "CSV export of daily click counts",
        (r"\bcsv\b", r"\bexports?\b"),
        "The export endpoint returns a CSV attachment with one zero-filled row per UTC day of the "
        "requested window (code, date, clicks, top referrer) and a sanitised file name",
    ),
    AspectSpec(
        "window-bounds",
        "Selectable export window",
        (r"\bwindow\b", r"\bup to \d{1,9} days?\b", r"\bselectable\b", r"\bdate range\b"),
        "The export window is selectable from 1 up to {max_window_days} days; values outside that "
        "range are rejected with 422 and nothing is exported",
    ),
    AspectSpec(
        "formula-injection",
        "Spreadsheet formula injection",
        (r"\bformulas?\b", r"\binjection\b", r"\bspreadsheets?\b", r"\bexcel\b", *_SECURITY_WORDS),
        "CSV cells that start with =, +, -, @, tab or carriage return are prefixed with a single "
        "quote so spreadsheets never evaluate them (verified with hostile referrers)",
    ),
    AspectSpec(
        "auth-required",
        "Authenticated export",
        (r"\bauthenticat\w*", r"\bapi[ -]?keys?\b", r"\blogin\b", *_SECURITY_WORDS),
        "The export requires a valid API key and answers 401 without one",
    ),
    AspectSpec(
        "owner-scoped",
        "Owner-scoped export",
        (
            r"\bowners?\b",
            r"\bown (?:links?|data)\b",
            r"\bother (?:users?|keys?|customers?|tenants?|teams?)\b",
            r"\btenants?\b",
            *_SECURITY_WORDS,
        ),
        "Only the owning API key can export a link: other keys and unknown codes receive 404 and "
        "no data",
    ),
    AspectSpec(
        "performance",
        "Export latency",
        (
            r"\b\d{1,9}\s*(?:ms|milliseconds?)\b",
            r"\bp9[0-9]\b",
            r"\blatency\b",
            r"\bresponse times?\b",
            r"\bfast(?:er)?\b",
            r"\bquick(?:ly|er)?\b",
            r"\bspeed\b",
        ),
        "The export responds within {perf_budget_ms} ms (p95), verified in-process with a generous "
        "budget by a benchmark test",
    ),
)
ASPECT_IDS = tuple(aspect.id for aspect in ASPECTS)

TRIGGERS: tuple[str, ...] = (
    r"\bcsv\b",
    r"\bexports?\b",
    r"\bdaily click",
    r"\bclick counts?\b",
    r"\bclick (?:stats|statistics|data|history)\b",
    r"\bspreadsheets?\b",
    r"\banalytics\b",
)
EXTRACTORS: dict[str, str] = {
    "max_window_days": r"\bup to\s+(\d{1,9})\s*days?\b",
    "perf_budget_ms": r"\b(\d{1,9})\s*(?:ms|milliseconds?)\b",
}
IMPACT_KEYWORDS = ("analytics", "click", "referrer", "export", "csv", "report")

_TEMPLATE_DIR = Path(__file__).resolve().parent / "files" / CAPABILITY_ID
_HOLE = re.compile(r"@@([A-Z][A-Z0-9_]*)@@")
_CODE_VALUE = re.compile(r"[A-Za-z0-9_, .-]{0,4000}\Z")
_MARKDOWN_VALUE = re.compile(r"[\x20-\x7e\n]{0,20000}\Z")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_REQ_ID = re.compile(r"R-\d{3}\Z")
_AC_ID = re.compile(r"AC-\d{3}-\d{1,2}\Z")


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


# ----------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class _Values:
    """The validated parameters plus the numbers derived from them."""

    max_window_days: int
    default_window_days: int
    perf_budget_ms: int
    in_process_budget_ms: int


def _values(params: Mapping[str, Any]) -> _Values:
    checked = validate_params(PARAMS, params)
    maximum, budget = int(checked["max_window_days"]), int(checked["perf_budget_ms"])
    return _Values(
        maximum,
        min(DEFAULT_WINDOW_DAYS, maximum),
        budget,
        max(IN_PROCESS_FLOOR_MS, IN_PROCESS_FACTOR * budget),
    )


# ----------------------------------------------------------------------------- templates


def _render(name: str, **values: int | str) -> str:
    """Fill ``@@NAME@@`` holes. Holes and values must match exactly; values are typed.

    Integers are rendered in decimal; strings must stay inside a conservative alphabet (printable
    ASCII for Markdown templates), so a value cannot close a literal, a docstring or a fence.
    """
    try:
        text = read_text_lf(_TEMPLATE_DIR / name)
    except OSError as exc:
        raise _fatal(f"template {name} is not installed (check the package data)") from exc
    holes = set(_HOLE.findall(text))
    given = {key.upper() for key in values}
    if holes != given:
        raise _fatal(f"template {name}: holes {sorted(holes)} do not match values {sorted(given)}")
    allowed = _MARKDOWN_VALUE if name.endswith(".md.tmpl") else _CODE_VALUE
    by_name = {key.upper(): value for key, value in values.items()}

    def fill(match: re.Match[str]) -> str:
        value = by_name[match.group(1)]
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise _fatal(f"template {name}: hole {match.group(1)} needs an int or a string")
        if isinstance(value, int):
            return str(value)
        if "@@" in value or not allowed.fullmatch(value):
            raise _fatal(f"template {name}: hole {match.group(1)} holds unsafe text")
        return value

    return _HOLE.sub(fill, text)


# ----------------------------------------------------------------------------- anchored edits


@dataclass(frozen=True)
class _Source:
    """A baseline file together with its parsed AST (for preconditions)."""

    path: str
    text: str
    tree: ast.Module


def _read_source(toolbox: Toolbox, path: str) -> _Source:
    if not toolbox.exists(path):
        raise RecipeAnchorError(f"{path}: file is missing from the baseline")
    text = toolbox.read_text(path)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise RecipeAnchorError(f"{path}: file cannot be parsed") from exc
    return _Source(path, text, tree)


def _imported(source: _Source, module: str) -> set[str]:
    names: set[str] = set()
    for node in source.tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module and node.level == 0:
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _require_imports(source: _Source, module: str, *names: str) -> None:
    missing = sorted(set(names) - _imported(source, module))
    if missing:
        raise RecipeAnchorError(
            f"{source.path}: expected 'from {module} import {', '.join(missing)}'"
        )


def _require_function(source: _Source, name: str) -> None:
    found = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in source.tree.body
    )
    if not found:
        raise RecipeAnchorError(f"{source.path}: function '{name}' not found")


def _class(source: _Source, name: str) -> ast.ClassDef:
    for node in source.tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RecipeAnchorError(f"{source.path}: class '{name}' not found")


def _method_names(node: ast.ClassDef) -> set[str]:
    return {
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _require_methods(source: _Source, class_name: str, *names: str) -> None:
    missing = sorted(set(names) - _method_names(_class(source, class_name)))
    if missing:
        raise RecipeAnchorError(f"{source.path}: {class_name} lacks {', '.join(missing)}")


def _forbid_methods(source: _Source, class_name: str, *names: str) -> None:
    present = sorted(set(names) & _method_names(_class(source, class_name)))
    if present:
        raise RecipeAnchorError(f"{source.path}: {class_name}.{present[0]} is already present")


def _require_self_attributes(source: _Source, class_name: str, *names: str) -> None:
    """``__init__`` must assign ``self.<name>`` (the generated code relies on it)."""
    init = next(
        (
            child
            for child in _class(source, class_name).body
            if isinstance(child, ast.FunctionDef) and child.name == "__init__"
        ),
        None,
    )
    if init is None:
        raise RecipeAnchorError(f"{source.path}: {class_name}.__init__ not found")
    assigned = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }
    missing = sorted(set(names) - assigned)
    if missing:
        raise RecipeAnchorError(f"{source.path}: {class_name}.__init__ does not set {missing[0]}")


def _check_module(model: object, module: str, path: str) -> None:
    """With a ``CodebaseModel``: the module must live at the path the group writes to."""
    if isinstance(model, CodebaseModel):
        info = model.module(module)
        if info is None or info.path != path:
            raise RecipeAnchorError(f"module {module} is not at {path} in the analysed baseline")


class _Edit:
    """Sequential anchored edits of one file; every anchor must occur exactly once."""

    def __init__(self, source: _Source) -> None:
        self.path = source.path
        self.text = source.text

    def _locate(self, anchor: str, what: str) -> int:
        count = self.text.count(anchor)
        if count != 1:
            raise RecipeAnchorError(
                f"{self.path}: anchor '{what}' found {count} times (expected exactly once)"
            )
        return self.text.index(anchor)

    def insert_before(self, anchor: str, block: str, what: str) -> None:
        index = self._locate(anchor, what)
        self.text = self.text[:index] + block + self.text[index:]

    def insert_after(self, anchor: str, block: str, what: str) -> None:
        index = self._locate(anchor, what) + len(anchor)
        self.text = self.text[:index] + block + self.text[index:]

    def replace(self, anchor: str, block: str, what: str) -> None:
        index = self._locate(anchor, what)
        self.text = self.text[:index] + block + self.text[index + len(anchor) :]

    def result(self) -> str:
        try:
            ast.parse(self.text)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            raise RecipeAnchorError(f"{self.path}: the edit would produce invalid Python") from exc
        return self.text


def _find_route(routes: Sequence[RouteInfo], path: str) -> RouteInfo | None:
    return next((r for r in routes if r.method == "GET" and r.path == path), None)


# ----------------------------------------------------------------------------- group builders


def _build_csv_encoding(ctx: BuildContext, values: _Values) -> None:
    toolbox = ctx.toolbox
    if not toolbox.exists("shortener/services/__init__.py"):
        raise RecipeAnchorError("shortener/services: package is missing from the baseline")
    if toolbox.exists(CSV_PATH):
        raise RecipeAnchorError(f"{CSV_PATH}: already exists, refusing to overwrite it")
    toolbox.write_text(CSV_PATH, _render("csv_export.py.tmpl"))


_REPOSITORY_PROTOCOL_ANCHOR = (
    "    def top_referrers(self, link_id: int, since: datetime, limit: int) -> "
    "Sequence[NamedCount]: ...\n"
)
_REPOSITORY_IMPL_ANCHOR = (
    "    def top_referrers(self, link_id: int, since: datetime, limit: int) -> "
    "Sequence[NamedCount]:\n        rows = (\n"
)


def _build_storage(ctx: BuildContext, values: _Values) -> None:
    _check_module(ctx.model, REPOSITORY_MODULE, REPOSITORY_PATH)
    source = _read_source(ctx.toolbox, REPOSITORY_PATH)
    _require_imports(source, "datetime", "date", "datetime")
    _require_imports(source, "shortener.domain.timeutil", "format_utc")
    for name in ("ClickRepository", "SqliteClickRepository"):
        _require_methods(source, name, "daily_counts", "top_referrers")
        _forbid_methods(source, name, "daily_top_referrers")
    edit = _Edit(source)
    edit.insert_before(
        _REPOSITORY_PROTOCOL_ANCHOR,
        _render("repo_protocol.py.tmpl"),
        "ClickRepository.top_referrers (protocol)",
    )
    edit.insert_before(
        _REPOSITORY_IMPL_ANCHOR,
        _render("repo_impl.py.tmpl"),
        "SqliteClickRepository.top_referrers",
    )
    ctx.toolbox.write_text(REPOSITORY_PATH, edit.result())


_SERVICE_IMPORT_ANCHOR = "from shortener.services.click_recorder import ClickRecorder\n"
_SERVICE_CONSTANT_ANCHOR = "BREAKDOWN_LIMIT = 10\n"
_SERVICE_METHOD_ANCHOR = "    def purge_expired(self) -> int:\n"


def _build_service(ctx: BuildContext, values: _Values) -> None:
    _check_module(ctx.model, SERVICE_MODULE, SERVICE_PATH)
    source = _read_source(ctx.toolbox, SERVICE_PATH)
    _require_imports(source, "datetime", "UTC", "datetime", "time", "timedelta")
    _require_imports(source, "shortener.domain.errors", "InvalidInputError")
    _require_imports(source, "shortener.domain.models", "Link")
    _require_imports(source, "shortener.services.click_recorder", "ClickRecorder")
    _require_function(source, "_zero_filled")
    _require_methods(source, "AnalyticsService", "report", "purge_expired")
    _forbid_methods(source, "AnalyticsService", "export_daily_csv")
    _require_self_attributes(source, "AnalyticsService", "_clicks", "_clock")
    if "EXPORT_MAX_DAYS" in source.text:
        raise RecipeAnchorError(f"{SERVICE_PATH}: EXPORT_MAX_DAYS is already present")
    edit = _Edit(source)
    edit.insert_after(
        _SERVICE_IMPORT_ANCHOR, _render("service_import.py.tmpl"), "click_recorder import"
    )
    edit.insert_after(
        _SERVICE_CONSTANT_ANCHOR,
        _render(
            "service_constants.py.tmpl",
            max_window_days=values.max_window_days,
            default_window_days=values.default_window_days,
        ),
        "BREAKDOWN_LIMIT constant",
    )
    edit.insert_before(
        _SERVICE_METHOD_ANCHOR, _render("service_method.py.tmpl"), "AnalyticsService.purge_expired"
    )
    ctx.toolbox.write_text(SERVICE_PATH, edit.result())


_ROUTES_IMPORT_ANCHOR = (
    "from shortener.services.analytics_service import DEFAULT_DAYS, MAX_DAYS, MIN_DAYS\n"
)
_ROUTES_END_ANCHOR = (
    "    return AnalyticsResponse.from_report(link.code, services.analytics.report(link, days))\n"
)


def _sibling_route(model: object, source: _Source) -> RouteInfo:
    """The authenticated analytics route this export is modelled on (model or file content)."""
    analysed = (
        model
        if isinstance(model, CodebaseModel)
        else analyze_codebase(MemoryView({source.path: source.text}))
    )
    if _find_route(analysed.routes, EXPORT_ROUTE) is not None:
        raise RecipeAnchorError(f"route GET {EXPORT_ROUTE} is already present")
    sibling = _find_route(analysed.routes, ANALYTICS_ROUTE)
    if sibling is None or sibling.module != ROUTES_MODULE:
        raise RecipeAnchorError(f"sibling route GET {ANALYTICS_ROUTE} not found in {ROUTES_MODULE}")
    if not sibling.auth:
        raise RecipeAnchorError(f"sibling route GET {ANALYTICS_ROUTE} has no auth dependency")
    return sibling


def _api_doc_edit(ctx: BuildContext, values: _Values) -> None:
    """Add the export section to ``docs/api.md`` when the baseline view has the file."""
    toolbox = ctx.toolbox
    if not toolbox.exists(API_DOC_PATH):
        return
    text = toolbox.read_text(API_DOC_PATH)
    heading = f"### `GET {ANALYTICS_ROUTE}`\n"
    if text.count(heading) != 1 or f"### `GET {EXPORT_ROUTE}`" in text:
        raise RecipeAnchorError(f"{API_DOC_PATH}: analytics section not found exactly once")
    start = text.index(heading) + len(heading)
    ends = [i for i in (text.find("\n### ", start), text.find("\n## ", start)) if i >= 0]
    if not ends:
        raise RecipeAnchorError(f"{API_DOC_PATH}: no section follows the analytics section")
    cut = min(ends) + 1
    section = _render("api_doc_section.md.tmpl", max_window_days=values.max_window_days)
    toolbox.write_text(API_DOC_PATH, text[:cut] + section + text[cut:])


def _build_route(ctx: BuildContext, values: _Values) -> None:
    _check_module(ctx.model, ROUTES_MODULE, ROUTES_PATH)
    source = _read_source(ctx.toolbox, ROUTES_PATH)
    sibling = _sibling_route(ctx.model, source)
    auth = sibling.auth[0]
    if not _IDENT.fullmatch(auth):
        raise RecipeAnchorError("sibling auth dependency is not a plain identifier")
    _require_imports(source, "fastapi", "APIRouter", "Depends", "Query", "Response")
    _require_imports(source, "typing", "Annotated")
    _require_imports(source, "shortener.api.deps", "get_services", auth)
    _require_imports(source, "shortener.api.schemas", "AUTHENTICATED_ERRORS", "error_responses")
    _require_imports(source, "shortener.container", "Services")
    _require_imports(source, "shortener.services.auth", "Principal")
    _require_imports(source, "shortener.services.analytics_service", "DEFAULT_DAYS", "MAX_DAYS")
    _require_function(source, "link_analytics")
    if "EXPORT_MAX_DAYS" in source.text:
        raise RecipeAnchorError(f"{ROUTES_PATH}: the export constants are already imported")
    edit = _Edit(source)
    edit.replace(_ROUTES_IMPORT_ANCHOR, _render("route_import.py.tmpl"), "analytics_service import")
    edit.insert_after(
        _ROUTES_END_ANCHOR,
        "\n\n"
        + _render("route.py.tmpl", max_window_days=values.max_window_days, auth_dependency=auth),
        "end of link_analytics",
    )
    ctx.toolbox.write_text(ROUTES_PATH, edit.result())
    _api_doc_edit(ctx, values)


_GROUP_BUILDERS: dict[str, Callable[[BuildContext, _Values], None]] = {
    "csv-encoding": _build_csv_encoding,
    "storage": _build_storage,
    "service": _build_service,
    "route": _build_route,
}


# ----------------------------------------------------------------------------- design


def _sibling_auth(model: object) -> tuple[str, ...] | None:
    """Auth dependency names of the sibling route; ``None`` when no model says."""
    if isinstance(model, CodebaseModel):
        sibling = _find_route(model.routes, ANALYTICS_ROUTE)
        if sibling is None or not sibling.auth:
            raise RecipeAnchorError(f"authenticated sibling route GET {ANALYTICS_ROUTE} not found")
        return sibling.auth
    routes = model.get("routes") if isinstance(model, Mapping) else None
    if isinstance(routes, Sequence) and not isinstance(routes, (str, bytes)):
        for entry in routes:
            if (
                isinstance(entry, Mapping)
                and entry.get("method") == "GET"
                and entry.get("path") == ANALYTICS_ROUTE
            ):
                auth = entry.get("auth")
                if isinstance(auth, Sequence) and not isinstance(auth, (str, bytes)):
                    return tuple(str(name) for name in auth)
    return None


def _adrs(values: _Values) -> list[dict[str, Any]]:
    def adr(
        number: int,
        title: str,
        decision: str,
        alternatives: list[tuple[str, str]],
        rationale: str,
        consequences: str,
    ) -> dict[str, Any]:
        return {
            "id": f"ADR-{number:03d}",
            "title": title,
            "decision": decision,
            "alternatives": [{"option": o, "rejected_because": r} for o, r in alternatives],
            "rationale": rationale,
            "consequences": consequences,
        }

    return [
        adr(
            1,
            "Encode the CSV in a dedicated services module",
            "A pure module (shortener/services/csv_export.py) renders the rows and owns the "
            "formula neutralisation and file-name sanitising; AnalyticsService.export_daily_csv "
            "gathers the data and calls it; the route only wraps the result in a Response.",
            [
                (
                    "Encode in the route handler",
                    "security-critical encoding would sit next to HTTP concerns, be harder to "
                    "unit-test in isolation and could not be reused by a job or a CLI",
                ),
                (
                    "Encode in the repository",
                    "storage must not know presentation formats; neutralisation belongs at the "
                    "output boundary",
                ),
                (
                    "Add a third-party CSV or spreadsheet library",
                    "the standard csv writer already handles quoting; a new dependency adds "
                    "supply-chain surface for no gain",
                ),
            ],
            "One small, side-effect free module is the single place where untrusted text becomes "
            "spreadsheet input, so it can be reviewed and tested on its own.",
            "The service layer gains a module; every text cell passes through it by construction.",
        ),
        adr(
            2,
            "Buffer the response instead of streaming it",
            f"Build the body in memory (at most {values.max_window_days} rows) and return a plain "
            "Response with Content-Disposition.",
            [
                (
                    "StreamingResponse",
                    "headers are committed before an error can surface, the body is small and "
                    "bounded, and streaming complicates the error model and the tests",
                ),
                (
                    "Write a file to object storage and return a signed URL",
                    "new infrastructure, retention and privacy questions for a few hundred rows",
                ),
            ],
            "The size is bounded by the validated window, so buffering is simpler and keeps "
            "errors as ordinary 4xx responses before any byte is sent.",
            "A much larger window or many more columns would need a streaming design.",
        ),
        adr(
            3,
            "Neutralise formula cells with a single-quote prefix",
            "Every text cell that starts (after spaces or invisible characters) with =, +, -, @, "
            "tab or carriage return, or with a full-width look-alike, gets a leading single "
            "quote; cells are also length-capped.",
            [
                (
                    "Strip or reject dangerous characters",
                    "it silently changes or loses data and is easy to bypass with encodings",
                ),
                (
                    "Rely on CSV quoting only",
                    "quoting protects the file structure, not the evaluation of a cell",
                ),
                (
                    "Export a typed .xlsx workbook",
                    "needs a new dependency and a much larger format surface",
                ),
            ],
            "Prefixing a quote is the widely used OWASP mitigation: spreadsheets show the value "
            "as text and never evaluate it, and the data stays readable.",
            "The quote is visible in spreadsheets and must be stripped by machine consumers.",
        ),
        adr(
            4,
            "Per-day top referrer through one aggregate query, without a migration",
            "Add ClickRepository.daily_top_referrers: one parameterised GROUP BY over the "
            "existing clicks columns, reduced to the first row per day in Python.",
            [
                (
                    "Reuse the window-level top_referrers",
                    "it is not per day, so the export could not show the referrer of a day",
                ),
                (
                    "Load every click and aggregate in the service",
                    "memory and time grow with traffic; the database already aggregates",
                ),
                (
                    "Add a summary table or index",
                    "needs a migration (HIGH risk) and the (link_id, clicked_at) index suffices",
                ),
            ],
            "The statement reads existing columns only, so the data model is unchanged.",
            "The repository protocol gains one method; there is no schema change.",
        ),
        adr(
            5,
            "Independent window limit, checked twice",
            f"EXPORT_MAX_DAYS ({values.max_window_days}) is its own constant; the route declares "
            "it as a Query bound (422 error model) and the service checks it again.",
            [
                (
                    "Reuse the analytics MAX_DAYS (365)",
                    "it would couple the export to the JSON endpoint and could not honour a "
                    "requirement above 365 days",
                ),
                (
                    "Validate only in the route",
                    "other callers of the use case could bypass the limit",
                ),
            ],
            "The limit is a requirement parameter, not a property of the JSON analytics.",
            "Windows longer than the click retention period show zero for the purged days.",
        ),
    ]


# ----------------------------------------------------------------------------- tests and docs


def _covers_by_aspect(requirements: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """aspect id -> ``R-001, AC-001-1, ...`` tokens of the active requirements' criteria."""
    entries = requirements.get("requirements") if isinstance(requirements, Mapping) else None
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise _fatal("the requirements payload is needed to derive the traceability lines")
    found: dict[str, dict[str, list[str]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("status") != "active":
            continue
        capabilities = entry.get("capabilities")
        if (
            not isinstance(capabilities, Sequence)
            or isinstance(capabilities, (str, bytes))
            or CAPABILITY_ID not in capabilities
        ):
            continue
        rid = entry.get("id")
        criteria = entry.get("acceptance_criteria")
        if not isinstance(rid, str) or not _REQ_ID.fullmatch(rid):
            continue
        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
            continue
        for criterion in criteria:
            if not isinstance(criterion, Mapping):
                continue
            ac, aspect = criterion.get("id"), criterion.get("aspect")
            if isinstance(ac, str) and _AC_ID.fullmatch(ac) and aspect in ASPECT_IDS:
                found.setdefault(str(aspect), {}).setdefault(rid, []).append(ac)
    return {
        aspect: tuple(
            token
            for rid in sorted(by_req)
            for token in (rid, *sorted(set(by_req[rid]), key=lambda a: int(a.rsplit("-", 1)[1])))
        )
        for aspect, by_req in found.items()
    }


_TEST_FRAGMENTS: dict[str, str] = {
    "csv-export": "tests_csv_export.py.tmpl",
    "window-bounds": "tests_window_bounds.py.tmpl",
    "formula-injection": "tests_formula_injection.py.tmpl",
    "auth-required": "tests_auth_required.py.tmpl",
    "owner-scoped": "tests_owner_scoped.py.tmpl",
}


def _md(text: object, limit: int) -> str:
    """Untrusted text as literal, ASCII-only Markdown."""
    return escape_markdown(text, limit).encode("ascii", "replace").decode("ascii")


def _design_section(design: Mapping[str, Any]) -> str:
    adrs = design.get("adrs")
    if not isinstance(adrs, Sequence) or isinstance(adrs, (str, bytes)):
        return ""
    lines: list[str] = []
    for adr in adrs:
        if isinstance(adr, Mapping) and isinstance(adr.get("id"), str):
            title, decision = _md(adr.get("title", ""), 160), _md(adr.get("decision", ""), 400)
            lines.append(f"* {_md(adr['id'], 16)} {title}: {decision}")
    return "\n## Design decisions\n\n" + "\n".join(lines) + "\n" if lines else ""


def _traceability_section(covers: Mapping[str, tuple[str, ...]]) -> str:
    if not covers:
        return ""
    rows = [
        f"| `{aspect}` | {', '.join(covers[aspect])} |" for aspect in ASPECT_IDS if aspect in covers
    ]
    return (
        "\n## Requirement traceability\n\n| Aspect | Requirement and criteria |\n|---|---|\n"
        + "\n".join(rows)
        + "\n"
    )


_UNRELEASED = re.compile(r"^## \[Unreleased\][ \t]*\n", re.MULTILINE)
_ADDED = re.compile(r"^### Added[ \t]*\n", re.MULTILINE)


def _changelog(ctx: BuildContext, values: _Values) -> None:
    toolbox = ctx.toolbox
    entry = _render("changelog_entry.md.tmpl", max_window_days=values.max_window_days)
    if not toolbox.exists(CHANGELOG_PATH):
        toolbox.write_text(
            CHANGELOG_PATH, _render("changelog_new.md.tmpl", entry=entry.rstrip("\n"))
        )
        return
    text = toolbox.read_text(CHANGELOG_PATH)
    headings = list(_UNRELEASED.finditer(text))
    if len(headings) != 1:
        raise RecipeAnchorError(f"{CHANGELOG_PATH}: '## [Unreleased]' not found exactly once")
    start = headings[0].end()
    following = text.find("\n## ", start)
    end = len(text) if following < 0 else following + 1
    added = list(_ADDED.finditer(text, start, end))
    if len(added) > 1:
        raise RecipeAnchorError(f"{CHANGELOG_PATH}: several '### Added' headings under Unreleased")
    if added:
        at = added[0].end()
        if text.startswith("\n", at):
            at += 1
        updated = text[:at] + entry + text[at:]
    else:
        rest = text[start:]
        separator = "" if not rest or rest.startswith("\n") else "\n"
        updated = text[:start] + f"\n### Added\n\n{entry}" + separator + rest
    toolbox.write_text(CHANGELOG_PATH, updated)


# ----------------------------------------------------------------------------- the recipe


class AnalyticsExportRecipe:
    """Vetted, parameterised recipe (see the module docstring)."""

    id = CAPABILITY_ID
    params = PARAMS

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        _values(params)
        return (
            GroupSpec(
                "csv-encoding",
                "CSV encoding with formula-injection neutralisation (new module)",
                (CSV_PATH,),
                (CSV_PATH,),
            ),
            GroupSpec(
                "storage",
                "Per-day top referrer query in the click repository",
                (REPOSITORY_PATH,),
                (REPOSITORY_PATH,),
            ),
            GroupSpec(
                "service",
                "Export use case and window limit in the analytics service",
                (SERVICE_PATH,),
                (SERVICE_PATH,),
            ),
            GroupSpec(
                "route",
                "Authenticated, owner-scoped export route and its API documentation",
                (ROUTES_PATH, API_DOC_PATH),
                (ROUTES_PATH, API_DOC_PATH),
            ),
        )

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        values = _values(params)
        auth = _sibling_auth(model)
        guard = (
            f"declares Depends({', '.join(auth)}) exactly like GET {ANALYTICS_ROUTE}"
            if auth
            else f"declares the same API-key dependency as GET {ANALYTICS_ROUTE}"
        )
        return {
            "components": [
                {
                    "name": "CSV encoding module",
                    "responsibility": "Render RFC 4180 rows, neutralise spreadsheet formulas, "
                    "sanitise the download file name.",
                    "touches": [CSV_PATH],
                    "requirement_ids": [],
                },
                {
                    "name": "Click repository: per-day top referrer",
                    "responsibility": "One parameterised aggregate query over the existing "
                    "clicks table (no migration).",
                    "touches": [REPOSITORY_PATH],
                    "requirement_ids": [],
                },
                {
                    "name": "Analytics service: export use case",
                    "responsibility": "Validate the window (1 to "
                    f"{values.max_window_days} days), gather daily counts and referrers, "
                    "build the CSV download.",
                    "touches": [SERVICE_PATH],
                    "requirement_ids": [],
                },
                {
                    "name": "Export route",
                    "responsibility": f"GET {EXPORT_ROUTE}: owner-scoped, {guard}, returns the "
                    "CSV attachment; documented in docs/api.md.",
                    "touches": [ROUTES_PATH, API_DOC_PATH],
                    "requirement_ids": [],
                },
            ],
            "adrs": _adrs(values),
            "data_model_changes": [],
            "api_changes": [
                {"method": "GET", "path": EXPORT_ROUTE, "change": "add", "breaking": False}
            ],
            "api_paths": [
                {
                    "method": "GET",
                    "path": EXPORT_ROUTE,
                    "summary": "Export daily click counts as CSV",
                    "request_fields": [
                        {"name": "code", "required": True, "type": "string"},
                        {"name": "days", "required": False, "type": "integer"},
                    ],
                    "responses": [200, 401, 404, 422, 429],
                }
            ],
            "rollback_plan": "Revert the release patch: remove the route (and its section in "
            "docs/api.md), the export use case and constants, the daily_top_referrers query and "
            "shortener/services/csv_export.py. There is no migration and no data change, so "
            "nothing has to be undone in the database; clients that never call the endpoint "
            "are unaffected.",
        }

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        values = _values(ctx.params)
        builder = _GROUP_BUILDERS.get(group_id)
        if builder is None:
            raise RecipeParamError("unknown group id for capability analytics_export")
        builder(ctx, values)

    def build_tests(self, ctx: BuildContext) -> None:
        values = _values(ctx.params)
        covers = _covers_by_aspect(ctx.requirements)
        toolbox = ctx.toolbox
        fragments: list[str] = []
        for aspect in ASPECT_IDS:
            template = _TEST_FRAGMENTS.get(aspect)
            if aspect not in covers or template is None:
                continue
            extra: dict[str, int | str] = {"covers": ", ".join(covers[aspect])}
            if aspect == "window-bounds":
                extra["max_window_days"] = values.max_window_days
            fragments.append(_render(template, **extra))
        if fragments:
            if toolbox.exists(TESTS_PATH):
                raise RecipeAnchorError(f"{TESTS_PATH}: already exists, refusing to overwrite it")
            toolbox.write_text(
                TESTS_PATH, "\n\n".join([_render("tests_header.py.tmpl"), *fragments])
            )
        if "performance" in covers:
            if toolbox.exists(PERF_TESTS_PATH):
                raise RecipeAnchorError(
                    f"{PERF_TESTS_PATH}: already exists, refusing to overwrite it"
                )
            toolbox.write_text(
                PERF_TESTS_PATH,
                _render(
                    "perf_test.py.tmpl",
                    covers=", ".join(covers["performance"]),
                    perf_budget_ms=values.perf_budget_ms,
                    in_process_budget_ms=values.in_process_budget_ms,
                ),
            )
        if not fragments and "performance" not in covers:
            toolbox.note("no active acceptance criterion maps to analytics_export: no tests built")

    def build_docs(self, ctx: BuildContext) -> None:
        values = _values(ctx.params)
        toolbox = ctx.toolbox
        if toolbox.exists(FEATURE_DOC_PATH):
            raise RecipeAnchorError(f"{FEATURE_DOC_PATH}: already exists, refusing to overwrite it")
        design = (
            ctx.design if isinstance(ctx.design, Mapping) else self.design(ctx.params, ctx.model)
        )
        try:
            covers = _covers_by_aspect(ctx.requirements)
        except AgentError:
            covers = {}
        toolbox.write_text(
            FEATURE_DOC_PATH,
            _render(
                "feature_doc.md.tmpl",
                max_window_days=values.max_window_days,
                default_window_days=values.default_window_days,
                perf_budget_ms=values.perf_budget_ms,
                in_process_budget_ms=values.in_process_budget_ms,
                design_section=_design_section(design),
                traceability_section=_traceability_section(covers),
            ),
        )
        _changelog(ctx, values)


RECIPE = AnalyticsExportRecipe()

CAPABILITY = Capability(
    id=CAPABILITY_ID,
    title="CSV export of daily click counts",
    kind="enhancement",
    triggers=TRIGGERS,
    aspects=ASPECTS,
    extractors=EXTRACTORS,
    impact_keywords=IMPACT_KEYWORDS,
    risk=RiskLevel.MEDIUM,
    recipe=RECIPE,
    requires_baseline=True,
)
