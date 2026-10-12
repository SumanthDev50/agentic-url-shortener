"""Static codebase analysis (AST only, nothing is imported or executed) and change-impact analysis.

``analyze_codebase(view)`` turns a read-only ``FileView`` into a :class:`CodebaseModel`:

* modules (dotted name, path, definitions), the *internal* import graph, and the test <-> module
  map;
* FastAPI / Starlette routes from decorators, ``add_api_route`` and ``include_router`` with full
  paths (``APIRouter(prefix=...)`` + include prefixes, string constants followed across modules)
  and the names of the ``Depends(...)`` dependencies that guard them (``auth``);
* SQL tables and columns replayed from migration files in file order (``CREATE TABLE``,
  ``ALTER TABLE ... ADD/DROP/RENAME``, ``DROP TABLE``; indexes/views/triggers ignored);
* data flows ``route -> handler -> called functions -> repository methods -> tables``.

``impact_analysis(model, keywords, ...)`` turns that model plus requirement keywords into an
:class:`ImpactReport` whose ``to_dict()`` is the ``impact_report`` artifact payload.

Heuristics and documented limitations (they surface as ``unresolved`` entries, never as a guess
presented as a fact)
-------------------------------------------------------------------------------------------
* Calls are resolved *by name*: imports (absolute, relative, re-exports), the enclosing class and
  ``self``, parameter / variable annotations, ``x = Class(...)``, ``Depends(factory)`` return
  annotations, class attribute annotations and ``self.x = ...`` assignments, and finally an
  attribute-name heuristic (``self.service.create`` -> ``LinkService.create``: role-word match
  against classes that define the method, ties broken by overlap with the caller's own name).
  Calls through ``getattr``, subscripts, callbacks stored in containers, queues and threads are not
  followed (a background writer is *not* reachable from the route that enqueues work for it); where
  the receiver of an attribute call cannot be typed the call is listed in ``unresolved``.
* A repository protocol resolves to its implementations (subclasses, and classes whose name ends
  with the protocol's name).
* Tables are attributed from SQL *string literals* in the traced functions (``FROM/JOIN/INTO/UPDATE
  <known table>``). SQL assembled dynamically (f-strings, ``%``, ``+``, ``.format``) is reported in
  ``unresolved`` as dynamic SQL.

Safety (the analysed tree is hostile input): no recursion over the AST (explicit stacks; every
helper that walks an expression is a bounded loop; type resolution has a depth budget);
``ast.parse`` is guarded for ``SyntaxError``, ``RecursionError``, ``MemoryError``, ``ValueError``
and NUL bytes and yields an *unparseable* record; files over the size / line caps, over the
AST-node budget, or beyond the file/total budgets are skipped with a record; SQL is tokenised by a
linear scanner (no backtracking regexes). Output is deterministic (sorted, no timestamps, no
absolute paths).
"""

from __future__ import annotations

import ast
import builtins
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields
from itertools import islice
from typing import Any, NamedTuple

from orchestrator.core.paths import validate_rel_path
from orchestrator.core.sanitize import strip_controls
from orchestrator.core.serde import to_jsonable
from orchestrator.core.types import FileView

SCHEMA_VERSION = 1

MAX_SOURCE_BYTES = 512 * 1024
MAX_LINES = 20_000
MAX_LINE_CHARS = 10_000
MAX_AST_NODES = 400_000
MAX_PY_FILES = 5_000
MAX_SQL_FILES = 500
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_FLOW_DEPTH = 8
MAX_FLOW_FUNCS = 300
MAX_IMPACT_DEPTH = 10
MAX_KEYWORDS = 64
MAX_KEYWORD_CHARS = 100
WIDE_IMPACT_MODULES = 25

_MAX_CHAIN = 24
_MAX_ANNOTATION_STEPS = 32
_MAX_STRING_DEPTH = 4
_MAX_REEXPORT_DEPTH = 6
_MAX_RESOLVE_DEPTH = 24
_MAX_SQL_TEXT = 4000
_MAX_STRING_CONST = 20_000
_MAX_REASONS = 3
_HEURISTIC_BUDGET = 500_000
MAX_ROUTES = 5_000


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_IGNORED_SEGMENTS = _words(
    ".venv venv node_modules __pycache__ site-packages .tox .mypy_cache .pytest_cache"
)
_HTTP_VERBS = _words("get post put delete patch head options trace")
_ROUTER_CLASSES = _words("APIRouter FastAPI Starlette Router")
_SECURITY_CLASSES = _words(
    "HTTPBearer HTTPBasic HTTPDigest APIKeyHeader APIKeyQuery APIKeyCookie "
    "OAuth2PasswordBearer OAuth2AuthorizationCodeBearer OAuth2 OpenIdConnect"
)
_DEPENDS_NAMES = _words("Depends Security")
_DISPATCHERS = _words("add_task run_in_threadpool to_thread submit run_in_executor start_soon")
# Generic wrappers whose first non-None argument is the type that matters for call resolution.
_TRANSPARENT_GENERICS = _words(
    "Optional Union Final ClassVar Required NotRequired ReadOnly "
    "Iterator Generator AsyncIterator AsyncGenerator Awaitable"
)
_BUILTIN_TYPES = _words(
    "str int float bool bytes bytearray complex list dict set frozenset tuple object type range "
    "Any Callable Type Sequence Mapping Iterable Self None"
)
_BUILTIN_NAMES = frozenset(dir(builtins))
_MUTABLE_FACTORIES = _words("dict list set defaultdict OrderedDict deque Counter")
# A dependency name counts as authentication when its word tokens hit one of these.
_AUTH_SINGLE_TOKENS = _words(
    "auth authn authz authenticate authenticated authentication authorize authorized "
    "authorization bearer oauth oauth2 jwt credential permission principal login apikey security"
)
_AUTH_TOKEN_PAIRS = tuple(
    _words(pair)
    for pair in (
        "api key",
        "current user",
        "require user",
        "require role",
        "require scope",
        "verify token",
        "check token",
    )
)
_ABBREVIATIONS = {"repo": "repository", "svc": "service", "mgr": "manager"}
_REPOSITORY_WORDS = _words("repository repo dao store gateway")
_CACHE_WORDS = _words("cache cached memo memoize lru")
_WRITER_WORDS = _words("recorder writer flusher batcher worker spooler")
_LIMITER_WORDS = _words("limiter throttle ratelimit bucket")
_DESTRUCTIVE_WORDS = _words("delete remove purge drop truncate disable revoke")
_STOPWORDS = _words("the a an of to for and or in on by with is are be")
_SQL_NAME_STARTERS = _words("FROM JOIN INTO UPDATE TABLE")
_THREAD_NAMES = _words("threading concurrent.futures multiprocessing")
_THREAD_CALLS = _words("Thread Timer ThreadPoolExecutor ProcessPoolExecutor")
_QUEUE_CALLS = _words("Queue SimpleQueue PriorityQueue LifoQueue")
_TASK_CALLS = _words("add_task create_task ensure_future run_in_executor to_thread")
_CACHE_DECORATORS = _words("lru_cache cache cached cached_property")
_SKIP_NODES = (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)
_STRING_EXPRS = (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Name, ast.Attribute)

FuncKey = tuple[str, str]  # (module, qualified function name)
ClassKey = tuple[str, str]  # (module, qualified class name)


# =============================================================================
# Public model
# =============================================================================


def _plain(obj: Any) -> dict[str, Any]:
    return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}


@dataclass(frozen=True)
class DefInfo:
    """A definition inside a module (``kind``: class, function, method, field, constant)."""

    kind: str
    name: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    path: str
    is_test: bool
    parsed: bool
    defs: tuple[DefInfo, ...] = ()
    imports: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class RouteInfo:
    """One HTTP route; ``auth`` is the auth-looking subset of ``dependencies`` (all Depends)."""

    method: str
    path: str
    handler: str
    module: str
    line: int
    auth: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.method} {self.path}"

    def to_dict(self) -> dict[str, Any]:
        """Exactly the ``impact_report.routes[]`` shape."""
        return {
            "method": self.method,
            "path": self.path,
            "handler": self.handler,
            "module": self.module,
            "line": self.line,
            "auth": list(self.auth),
        }


@dataclass(frozen=True)
class TableInfo:
    """A table after replaying all migrations; ``migration`` is the file that created it."""

    name: str
    columns: tuple[str, ...]
    migration: str
    altered_in: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The ``impact_report.tables[]`` shape (columns sorted so the payload is order-stable)."""
        return {"table": self.name, "columns": sorted(self.columns), "migration": self.migration}


@dataclass(frozen=True)
class FlowCall:
    module: str
    qualname: str

    @property
    def label(self) -> str:
        return f"{self.module}.{self.qualname}"


@dataclass(frozen=True)
class DataFlow:
    """``route -> handler -> calls -> repositories -> tables`` (limits: module docstring)."""

    route: str
    module: str
    handler: str
    calls: tuple[FlowCall, ...] = ()
    repositories: tuple[FlowCall, ...] = ()
    tables: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def touched_modules(self) -> frozenset[str]:
        return frozenset({self.module, *(c.module for c in self.calls)})

    def to_dict(self) -> dict[str, Any]:
        """The ``impact_report.data_flows[]`` shape; every list sorted."""
        return {
            "route": self.route,
            "handler": f"{self.module}.{self.handler}",
            "calls": sorted(c.label for c in self.calls),
            "repositories": sorted(c.label for c in self.repositories),
            "tables": sorted(self.tables),
            "unresolved": sorted(self.unresolved),
        }


@dataclass(frozen=True)
class FileRecord:
    """A file that was not (fully) analysed, and why. The reason never echoes file content."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class CodebaseModel:
    modules: tuple[ModuleInfo, ...]
    import_graph: Mapping[str, tuple[str, ...]]
    routes: tuple[RouteInfo, ...]
    tables: tuple[TableInfo, ...]
    migrations: tuple[str, ...]
    test_map: Mapping[str, tuple[str, ...]]
    table_refs: Mapping[str, tuple[str, ...]]
    data_flows: tuple[DataFlow, ...]
    unresolved: tuple[str, ...]
    unparseable: tuple[FileRecord, ...]
    skipped: tuple[FileRecord, ...]

    def module(self, name: str) -> ModuleInfo | None:
        return next((m for m in self.modules if m.name == name), None)

    def tests_for(self, modules: Iterable[str]) -> tuple[str, ...]:
        """Test files (sorted) that import at least one of ``modules``."""
        wanted = set(modules)
        return tuple(sorted(path for path, mods in self.test_map.items() if wanted & set(mods)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "modules": [m.to_dict() for m in self.modules],
            "import_graph": {k: list(v) for k, v in sorted(self.import_graph.items())},
            "routes": [{**r.to_dict(), "dependencies": list(r.dependencies)} for r in self.routes],
            "tables": [{**t.to_dict(), "altered_in": list(t.altered_in)} for t in self.tables],
            "migrations": list(self.migrations),
            "test_map": {k: list(v) for k, v in sorted(self.test_map.items())},
            "table_refs": {k: list(v) for k, v in sorted(self.table_refs.items())},
            "data_flows": [f.to_dict() for f in self.data_flows],
            "unresolved": list(self.unresolved),
            "unparseable": [r.to_dict() for r in self.unparseable],
            "skipped": [r.to_dict() for r in self.skipped],
        }


# =============================================================================
# Text helpers
# =============================================================================

_WORD_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_NATURAL = re.compile(r"(\d+)")


def _stem(token: str) -> str:
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def _tokens(text: str) -> list[str]:
    """Lower-cased, singularised word tokens of an identifier, path or phrase."""
    out: list[str] = []
    for chunk in _WORD_SPLIT.split(text[:256]):
        out.extend(_stem(part) for part in _CAMEL.findall(chunk))
    return out


def _natural_key(name: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in _NATURAL.split(name))


# =============================================================================
# SQL: a linear tokenizer and a migration replayer
# =============================================================================


class _Tok(NamedTuple):
    kind: str  # word | quoted | string | number | punct
    text: str

    @property
    def upper(self) -> str:
        """Upper-cased text of an *unquoted* word (quoted identifiers never act as keywords)."""
        return self.text.upper() if self.kind == "word" else ""


def _read_quoted(sql: str, start: int, quote: str) -> tuple[str, int]:
    out: list[str] = []
    i, n = start + 1, len(sql)
    while i < n:
        ch = sql[i]
        if ch == quote:
            if i + 1 < n and sql[i + 1] == quote:
                out.append(quote)
                i += 2
                continue
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return "".join(out), n  # unterminated: consume to the end, once


def _sql_tokens(sql: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end < 0 else end + 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif ch in "'\"`":
            text, i = _read_quoted(sql, i, ch)
            tokens.append(_Tok("string" if ch == "'" else "quoted", text))
        elif ch == "[":
            end = sql.find("]", i + 1, min(n, i + 128))  # bounded look-ahead: no quadratic scan
            if end < 0 or "\n" in sql[i:end]:
                tokens.append(_Tok("punct", ch))
                i += 1
            else:
                tokens.append(_Tok("quoted", sql[i + 1 : end]))
                i = end + 1
        elif ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] in "_$"):
                j += 1
            tokens.append(_Tok("word", sql[i:j]))
            i = j
        elif ch.isdigit():
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "."):
                j += 1
            tokens.append(_Tok("number", sql[i:j]))
            i = j
        else:
            tokens.append(_Tok("punct", ch))
            i += 1
    return tokens


def _is_punct(tok: _Tok | None, char: str) -> bool:
    return tok is not None and tok.kind == "punct" and tok.text == char


def _split_statements(tokens: list[_Tok]) -> list[list[_Tok]]:
    """Split on ``;``, keeping ``CREATE TRIGGER ... BEGIN ... END;`` bodies in one statement."""
    statements: list[list[_Tok]] = []
    current: list[_Tok] = []
    in_trigger = begin_seen = False
    for tok in tokens:
        if _is_punct(tok, ";"):
            if in_trigger and begin_seen and (not current or current[-1].upper != "END"):
                current.append(tok)
                continue
            if current:
                statements.append(current)
            current, in_trigger, begin_seen = [], False, False
            continue
        current.append(tok)
        if tok.upper == "TRIGGER" and len(current) <= 3 and current[0].upper == "CREATE":
            in_trigger = True
        elif in_trigger and tok.upper == "BEGIN":
            begin_seen = True
    if current:
        statements.append(current)
    return statements


def _name_at(toks: Sequence[_Tok], i: int) -> tuple[str, int] | None:
    """A possibly schema-qualified identifier at ``i``: ``(lower-cased last part, next index)``."""
    if i >= len(toks) or toks[i].kind not in ("word", "quoted"):
        return None
    name = toks[i].text.lower()
    i += 1
    while i + 1 < len(toks) and _is_punct(toks[i], ".") and toks[i + 1].kind in ("word", "quoted"):
        name = toks[i + 1].text.lower()
        i += 2
    return name, i


def _words_at(toks: Sequence[_Tok], i: int, *words: str) -> bool:
    return all(i + k < len(toks) and toks[i + k].upper == w for k, w in enumerate(words))


def _guard_len(toks: Sequence[_Tok], i: int) -> int:
    """Length of an ``IF NOT EXISTS`` / ``IF EXISTS`` guard at ``i`` (0 when there is none)."""
    if _words_at(toks, i, "IF", "NOT", "EXISTS"):
        return 3
    return 2 if _words_at(toks, i, "IF", "EXISTS") else 0


def _split_top_level(toks: Sequence[_Tok]) -> list[list[_Tok]]:
    """Split on commas that are not inside parentheses."""
    parts: list[list[_Tok]] = [[]]
    depth = 0
    for tok in toks:
        if _is_punct(tok, "("):
            depth += 1
        elif _is_punct(tok, ")"):
            depth = max(0, depth - 1)
        if depth == 0 and _is_punct(tok, ","):
            parts.append([])
        else:
            parts[-1].append(tok)
    return [p for p in parts if p]


def _is_table_constraint(item: Sequence[_Tok]) -> bool:
    """True for ``CONSTRAINT ...``, ``PRIMARY/FOREIGN KEY ...``, ``UNIQUE(...)``, ``CHECK(...)``.

    A column legitimately called ``key`` or ``unique`` is told apart by what follows the word.
    """
    first = item[0].upper
    second = item[1] if len(item) > 1 else None
    if first == "CONSTRAINT":
        return True
    if first in ("PRIMARY", "FOREIGN"):
        return second is not None and second.upper == "KEY"
    return first in ("UNIQUE", "CHECK", "EXCLUDE") and _is_punct(second, "(")


def _column_names(toks: Sequence[_Tok], start: int) -> list[str]:
    """Column names of the parenthesised body at ``start`` (empty when there is none)."""
    if start >= len(toks) or not _is_punct(toks[start], "("):
        return []
    depth, end = 0, len(toks)
    for i in range(start, len(toks)):
        if _is_punct(toks[i], "("):
            depth += 1
        elif _is_punct(toks[i], ")"):
            depth -= 1
            if depth == 0:
                end = i
                break
    names: list[str] = []
    for item in _split_top_level(toks[start + 1 : end]):
        if item[0].kind in ("word", "quoted") and not _is_table_constraint(item):
            names.append(item[0].text.lower())
    return names


@dataclass
class _MutableTable:
    name: str
    columns: list[str]
    created_in: str
    altered_in: list[str] = field(default_factory=list)

    def touch(self, migration: str) -> None:
        if not self.altered_in or self.altered_in[-1] != migration:
            self.altered_in.append(migration)


class _SchemaReplay:
    """Applies migration statements in order and keeps the resulting table set."""

    def __init__(self) -> None:
        self.tables: dict[str, _MutableTable] = {}

    def apply_file(self, sql: str, migration: str) -> None:
        for stmt in _split_statements(_sql_tokens(sql)):
            head = stmt[0].upper
            if head == "CREATE":
                self._create(stmt, migration)
            elif head == "ALTER" and _words_at(stmt, 1, "TABLE"):
                self._alter(stmt, migration)
            elif head == "DROP" and _words_at(stmt, 1, "TABLE"):
                self._drop(stmt)

    def _create(self, stmt: list[_Tok], migration: str) -> None:
        i = 1
        while i < len(stmt) and stmt[i].upper in ("TEMP", "TEMPORARY", "VIRTUAL"):
            i += 1
        if not _words_at(stmt, i, "TABLE"):
            return
        i += 1
        guard = _guard_len(stmt, i)
        i += guard
        named = _name_at(stmt, i)
        if named is None:
            return
        name, i = named
        if guard == 3 and name in self.tables:  # IF NOT EXISTS on an existing table: a no-op
            return
        self.tables[name] = _MutableTable(name, _column_names(stmt, i), migration)

    def _drop(self, stmt: list[_Tok]) -> None:
        i = 2 + _guard_len(stmt, 2)
        while i < len(stmt):
            named = _name_at(stmt, i)
            if named is None:
                return
            self.tables.pop(named[0], None)
            i = named[1] + 1  # skip the comma

    def _alter(self, stmt: list[_Tok], migration: str) -> None:
        i = 2 + _guard_len(stmt, 2)
        if _words_at(stmt, i, "ONLY"):
            i += 1
        named = _name_at(stmt, i)
        if named is None:
            return
        name, i = named
        table = self.tables.get(name)
        if table is None:
            # altered but never created by a migration we can see: assume it pre-exists
            table = self.tables[name] = _MutableTable(name, [], migration)
        for action in _split_top_level(stmt[i:]):
            self._alter_action(table, action, migration)

    def _alter_action(self, table: _MutableTable, action: list[_Tok], migration: str) -> None:
        verb = action[0].upper
        k = 1
        if verb == "ADD":
            k += 1 if _words_at(action, k, "COLUMN") else 0
            k += _guard_len(action, k)
            rest = action[k:]
            if rest and rest[0].kind in ("word", "quoted") and not _is_table_constraint(rest):
                table.columns.append(rest[0].text.lower())
                table.touch(migration)
        elif verb == "DROP":
            k += 1 if _words_at(action, k, "COLUMN") else 0
            k += _guard_len(action, k)
            if k < len(action) and action[k].upper != "CONSTRAINT":
                target = action[k].text.lower()
                if target in table.columns:
                    table.columns.remove(target)
                    table.touch(migration)
        elif verb == "RENAME":
            self._rename(table, action, migration)

    def _rename(self, table: _MutableTable, action: list[_Tok], migration: str) -> None:
        if _words_at(action, 1, "TO"):
            new = _name_at(action, 2)
            if new is not None:
                self.tables.pop(table.name, None)
                table.name = new[0]
                self.tables[table.name] = table
                table.touch(migration)
            return
        k = 2 if _words_at(action, 1, "COLUMN") else 1
        if k + 2 < len(action) and action[k + 1].upper == "TO":
            old, new_name = action[k].text.lower(), action[k + 2].text.lower()
            table.columns = [new_name if c == old else c for c in table.columns]
            table.touch(migration)

    def result(self) -> tuple[TableInfo, ...]:
        return tuple(
            TableInfo(t.name, tuple(t.columns), t.created_in, tuple(t.altered_in))
            for t in sorted(self.tables.values(), key=lambda t: t.name)
        )


_SQL_PAIRS = (
    ("select", "from"),
    ("insert", "into"),
    ("replace", "into"),
    ("update", "set"),
    ("delete", "from"),
    ("create", "table"),
    ("alter", "table"),
    ("drop", "table"),
)
_SQL_WORD = re.compile(r"[a-z_]+")


def _looks_like_sql(text: str) -> bool:
    words = set(_SQL_WORD.findall(text[:_MAX_SQL_TEXT].lower()))
    return any(a in words and b in words for a, b in _SQL_PAIRS)


def _referenced_tables(text: str) -> tuple[str, ...]:
    """Candidate table names after ``FROM/JOIN/INTO/UPDATE/TABLE`` (FROM comma lists included)."""
    toks = _sql_tokens(text[:_MAX_SQL_TEXT])
    found: list[str] = []
    for i, tok in enumerate(toks):
        if tok.upper not in _SQL_NAME_STARTERS:
            continue
        j = i + 1 + _guard_len(toks, i + 1)
        while True:
            named = _name_at(toks, j)
            if named is None:
                break
            found.append(named[0])
            j = named[1]
            if tok.upper == "FROM" and j < len(toks) and _is_punct(toks[j], ","):
                j += 1
                continue
            break
    return tuple(dict.fromkeys(found))


_NUMBERED_MIGRATION = re.compile(r"^(\d+[_-]|v\d+__)")  # 001_init.sql, V2__add.sql


def _is_migration_path(path: str) -> bool:
    segments = path.lower().split("/")
    name = segments[-1]
    return name.endswith(".sql") and (
        any(s in ("migrations", "migration") for s in segments[:-1])
        or _NUMBERED_MIGRATION.match(name) is not None
    )


def _ordered_migrations(paths: Iterable[str]) -> list[str]:
    """File order: by directory, then by name with digit runs compared as numbers (2_ < 10_)."""
    return sorted(paths, key=lambda p: (p.rpartition("/")[0], _natural_key(p.rpartition("/")[2])))


def _replay_migrations(
    view: FileView, paths: Sequence[str], skipped: list[FileRecord]
) -> _SchemaReplay:
    replay = _SchemaReplay()
    ordered = _ordered_migrations(paths)
    for path in ordered[:MAX_SQL_FILES]:
        text, problem = _acquire_source(view, path)
        if text is None:
            skipped.append(FileRecord(path, problem))
            continue
        replay.apply_file(text, path)
    skipped.extend(FileRecord(p, "migration file limit reached") for p in ordered[MAX_SQL_FILES:])
    return replay


# =============================================================================
# AST helpers (all bounded loops: nothing here recurses over user-controlled depth)
# =============================================================================


def _chain_of(expr: ast.expr, *, allow_call: bool = True) -> tuple[str, ...] | None:
    """``a.b.c`` -> ``("a", "b", "c")``; a call-rooted chain keeps ``"f()"`` as first element."""
    parts: list[str] = []
    cur: ast.expr = expr
    for _ in range(_MAX_CHAIN):
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return tuple(parts)
        elif isinstance(cur, ast.Call) and allow_call:
            inner = _chain_of(cur.func, allow_call=False)
            if inner is None:
                return None
            parts.append(".".join(inner) + "()")
            parts.reverse()
            return tuple(parts)
        else:
            return None
    return None


def _is_none(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and expr.value is None


def _parse_annotation(text: str) -> ast.expr | None:
    try:
        return ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _annotation_info(expr: ast.expr | None) -> tuple[str | None, list[ast.Call]]:
    """Type text of an annotation (``Optional``/``Annotated``/``X | None`` unwrapped) + Depends."""
    deps: list[ast.Call] = []
    cur = expr
    for _ in range(_MAX_ANNOTATION_STEPS):
        if cur is None:
            break
        if isinstance(cur, ast.Constant):
            if isinstance(cur.value, str) and len(cur.value) <= 200:
                cur = _parse_annotation(cur.value)
                continue
            break
        if isinstance(cur, ast.Name):
            return cur.id, deps
        if isinstance(cur, ast.Attribute):
            chain = _chain_of(cur, allow_call=False)
            return (".".join(chain) if chain else None), deps
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.BitOr):
            cur = cur.right if _is_none(cur.left) else cur.left
            continue
        if isinstance(cur, ast.Subscript):
            base = _chain_of(cur.value, allow_call=False)
            name = base[-1] if base else None
            inner = cur.slice
            elts = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
            if name == "Annotated" and elts:
                deps.extend(e for e in elts[1:] if isinstance(e, ast.Call))
                cur = elts[0]
            elif name in _TRANSPARENT_GENERICS:
                real = [e for e in elts if not _is_none(e)]
                cur = real[0] if real else None
            else:
                return name, deps
            continue
        break
    return None, deps


def _joined_text(node: ast.JoinedStr) -> tuple[str, bool, list[ast.expr]]:
    """f-string text with ``?`` per interpolation: (text, has_interpolation, inner expressions)."""
    parts: list[str] = []
    inner: list[ast.expr] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append("?")
            inner.append(value.value)
    return "".join(parts), bool(inner), inner


def _flatten_concat(node: ast.BinOp) -> list[ast.expr]:
    """Leaves of an ``a + b + c`` / ``fmt % args`` chain, left to right, without recursion."""
    leaves: list[ast.expr] = []
    todo: list[ast.expr] = [node]
    while todo:
        cur = todo.pop()
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, (ast.Add, ast.Mod)):
            todo.append(cur.right)
            todo.append(cur.left)
        else:
            leaves.append(cur)
    return leaves


def _looks_like_auth(name: str) -> bool:
    toks = set(_tokens(name))
    return bool(toks & _AUTH_SINGLE_TOKENS) or any(pair <= toks for pair in _AUTH_TOKEN_PAIRS)


# =============================================================================
# Facts gathered per module (phase 1) and the extractor
# =============================================================================


class _Binding(NamedTuple):
    module: str
    symbol: str | None  # None: the name is bound to the module itself


class _ImportRef(NamedTuple):
    module: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class _Value:
    kind: str  # call | chain | literal | unknown
    chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Dep:
    """A ``Depends(...)`` / ``Security(...)`` reference."""

    name: str
    chain: tuple[str, ...] | None
    scheme: bool  # the dependency expression itself constructs a security scheme


@dataclass
class _Call:
    chain: tuple[str, ...] | None
    note: str = ""


@dataclass
class _SqlRef:
    tables: tuple[str, ...]
    dynamic: bool


@dataclass
class _Assign:
    line: int
    target: str
    value: _Value
    annotation: str | None


@dataclass
class _Decorator:
    chain: tuple[str, ...] | None
    call: ast.Call | None
    deps: list[_Dep]


@dataclass
class _Func:
    """Facts about one function; ``names`` holds the CONSTANT-style names its body reads."""

    module: str
    qual: str
    line: int
    cls: str | None
    scope: str
    params: dict[str, str | None]
    param_deps: dict[str, _Dep]
    deps: list[_Dep]
    returns: str | None
    decorators: list[_Decorator]
    stub: bool
    calls: list[_Call] = field(default_factory=list)
    assigns: list[_Assign] = field(default_factory=list)
    sql: list[_SqlRef] = field(default_factory=list)
    returned: list[str] = field(default_factory=list)
    names: set[str] = field(default_factory=set)


@dataclass
class _AttrAssign:
    attr: str
    value: _Value
    params: dict[str, str | None]
    annotation: str | None


@dataclass
class _Class:
    qual: str
    bases: list[str]
    attrs: dict[str, str] = field(default_factory=dict)
    attr_assigns: list[_AttrAssign] = field(default_factory=list)
    sql_consts: dict[str, _SqlRef] = field(default_factory=dict)


@dataclass
class _Router:
    scope: str
    var: str
    kind: str  # app | router
    prefix: ast.expr | None
    deps: list[_Dep]
    line: int


@dataclass
class _Include:
    scope: str
    receiver: str
    target: ast.expr | None
    prefix: ast.expr | None
    deps: list[_Dep]
    line: int


@dataclass
class _ExplicitRoute:
    scope: str
    receiver: str
    path: ast.expr | None
    endpoint: ast.expr | None
    methods: list[str]
    deps: list[_Dep]
    line: int


@dataclass
class _Module:
    name: str
    path: str
    is_test: bool
    is_package: bool
    parsed: bool = True
    bindings: dict[str, _Binding] = field(default_factory=dict)
    imports: list[_ImportRef] = field(default_factory=list)
    classes: dict[str, _Class] = field(default_factory=dict)
    funcs: dict[str, _Func] = field(default_factory=dict)
    defs: list[DefInfo] = field(default_factory=list)
    var_values: dict[str, _Value] = field(default_factory=dict)
    consts: dict[str, ast.expr] = field(default_factory=dict)
    routers: list[_Router] = field(default_factory=list)
    includes: list[_Include] = field(default_factory=list)
    explicit_routes: list[_ExplicitRoute] = field(default_factory=list)
    scheme_vars: set[str] = field(default_factory=set)
    sql_consts: dict[str, _SqlRef] = field(default_factory=dict)
    signal_names: set[str] = field(default_factory=set)
    mutable_globals: list[str] = field(default_factory=list)

    @property
    def package(self) -> str:
        return self.name if self.is_package else self.name.rpartition(".")[0]

    @property
    def basename(self) -> str:
        return self.name.rpartition(".")[2]


@dataclass(frozen=True)
class _Ctx:
    func: _Func | None
    cls: _Class | None
    class_body: bool
    prefix: str


_Stack = list[tuple[ast.AST, _Ctx]]


class _TooComplex(Exception):
    """The AST node budget of one file was exhausted."""


def _is_stub(body: Sequence[ast.stmt]) -> bool:
    """Body is only ``...`` / ``pass`` / ``raise NotImplementedError`` (an interface method)."""
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                continue
        return False
    return True


def _value_desc(expr: ast.expr | None) -> _Value:
    """Classify an assigned value just enough to infer a type for it later."""
    cur = expr
    for _ in range(8):
        if isinstance(cur, ast.Await):
            cur = cur.value
        elif isinstance(cur, ast.BoolOp):
            cur = next((v for v in cur.values if not isinstance(v, ast.Constant)), cur.values[0])
        elif isinstance(cur, ast.IfExp):
            cur = cur.orelse if isinstance(cur.body, ast.Constant) else cur.body
        else:
            break
    if isinstance(cur, ast.Call):
        chain = _chain_of(cur.func)
        return _Value("call", chain) if chain else _Value("unknown")
    if isinstance(cur, (ast.Name, ast.Attribute)):
        chain = _chain_of(cur)
        return _Value("chain", chain) if chain else _Value("unknown")
    if isinstance(
        cur,
        (
            ast.Constant,
            ast.List,
            ast.Dict,
            ast.Set,
            ast.Tuple,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.JoinedStr,
        ),
    ):
        return _Value("literal")
    return _Value("unknown")


class _Extractor:
    """Single iterative pass over one module's AST that fills a :class:`_Module`."""

    def __init__(self, module: _Module) -> None:
        self.m = module
        self._nodes = 0
        self._handlers: dict[type[ast.AST], Callable[[Any, _Ctx, _Stack], None]] = {
            ast.ClassDef: self._class_def,
            ast.FunctionDef: self._function_def,
            ast.AsyncFunctionDef: self._function_def,
            ast.Import: self._import,
            ast.ImportFrom: self._import_from,
            ast.Assign: self._assign,
            ast.AnnAssign: self._ann_assign,
            ast.Call: self._call,
            ast.JoinedStr: self._joined,
            ast.BinOp: self._bin_op,
            ast.Constant: self._constant,
            ast.Name: self._name,
            ast.Attribute: self._attribute,
            ast.Return: self._return,
        }

    def run(self, tree: ast.Module) -> None:
        ctx = _Ctx(None, None, False, "")
        stack: _Stack = [(stmt, ctx) for stmt in reversed(tree.body)]
        while stack:
            node, ctx = stack.pop()
            self._nodes += 1
            if self._nodes > MAX_AST_NODES:
                raise _TooComplex
            handler = self._handlers.get(type(node))
            if handler is None:
                self._push(node, ctx, stack)
            else:
                handler(node, ctx, stack)

    def _constant(self, node: ast.Constant, ctx: _Ctx, stack: _Stack) -> None:
        if isinstance(node.value, str) and len(node.value) <= _MAX_STRING_CONST:
            self._sql(ctx, node.value, False)

    @staticmethod
    def _name(node: ast.Name, ctx: _Ctx, stack: _Stack) -> None:
        if ctx.func is not None and _is_constant_name(node.id):
            ctx.func.names.add(node.id)

    def _attribute(self, node: ast.Attribute, ctx: _Ctx, stack: _Stack) -> None:
        if ctx.func is not None and _is_constant_name(node.attr):
            ctx.func.names.add(node.attr)
        self._push(node, ctx, stack)

    def _return(self, node: ast.Return, ctx: _Ctx, stack: _Stack) -> None:
        if ctx.func is not None and isinstance(node.value, ast.Name):
            ctx.func.returned.append(node.value.id)
        self._push(node, ctx, stack)

    @staticmethod
    def _push(node: ast.AST, ctx: _Ctx, stack: _Stack) -> None:
        children = [c for c in ast.iter_child_nodes(node) if not isinstance(c, _SKIP_NODES)]
        stack.extend((c, ctx) for c in reversed(children))

    # -- definitions --------------------------------------------------------------
    def _class_def(self, node: ast.ClassDef, ctx: _Ctx, stack: _Stack) -> None:
        qual = ctx.prefix + node.name
        bases = [text for b in node.bases if (text := _annotation_info(b)[0])]
        cls = _Class(qual, bases)
        self.m.classes[qual] = cls
        self.m.defs.append(DefInfo("class", qual, node.lineno))
        inner = _Ctx(None, cls, True, qual + ".")
        stack.extend((s, inner) for s in reversed(node.body))

    def _function_def(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        ctx: _Ctx,
        stack: _Stack,
    ) -> None:
        qual = ctx.prefix + node.name
        params, param_deps, deps = self._parameters(node.args)
        returns, return_deps = _annotation_info(node.returns)
        deps.extend(d for d in (self._dep(c, None) for c in return_deps) if d is not None)
        decorators = self._decorators(node)
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # the docstring is prose, never SQL
        fn = _Func(
            module=self.m.name,
            qual=qual,
            line=node.lineno,
            cls=ctx.cls.qual if ctx.cls is not None else None,
            scope=ctx.func.qual if ctx.func is not None else "",
            params=params,
            param_deps=param_deps,
            deps=deps,
            returns=returns,
            decorators=decorators,
            stub=_is_stub(body),
        )
        self.m.funcs[qual] = fn
        self.m.defs.append(DefInfo("method" if ctx.class_body else "function", qual, node.lineno))
        inner = _Ctx(fn, ctx.cls, False, qual + ".")
        stack.extend((s, inner) for s in reversed(body))

    def _decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[_Decorator]:
        out: list[_Decorator] = []
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            chain = _chain_of(call.func if call is not None else dec)
            declared = (
                next((k.value for k in call.keywords if k.arg == "dependencies"), None)
                if call
                else None
            )
            out.append(_Decorator(chain, call, self._dep_list(declared)))
            if chain:
                self.m.signal_names.add(chain[-1])
        return out

    def _parameters(
        self, args: ast.arguments
    ) -> tuple[dict[str, str | None], dict[str, _Dep], list[_Dep]]:
        positional = [*args.posonlyargs, *args.args]
        defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(
            args.defaults
        )
        pairs = [
            *zip(positional, defaults, strict=True),
            *zip(args.kwonlyargs, args.kw_defaults, strict=True),
        ]
        params: dict[str, str | None] = {}
        param_deps: dict[str, _Dep] = {}
        deps: list[_Dep] = []
        for arg, default in pairs:
            text, ann_calls = _annotation_info(arg.annotation)
            params[arg.arg] = text
            found = [d for d in (self._dep(c, text) for c in ann_calls) if d is not None]
            if isinstance(default, ast.Call):
                from_default = self._dep(default, text)
                if from_default is not None:
                    found.append(from_default)
            if found:
                param_deps[arg.arg] = found[-1]
                deps.extend(found)
        return params, param_deps, deps

    def _is_depends(self, call: ast.Call) -> bool:
        chain = _chain_of(call.func, allow_call=False)
        if not chain:
            return False
        binding = self.m.bindings.get(chain[0])
        symbol = binding.symbol if binding is not None else None
        return chain[-1] in _DEPENDS_NAMES or (len(chain) == 1 and symbol in _DEPENDS_NAMES)

    def _dep(self, call: ast.Call, annotation: str | None) -> _Dep | None:
        """``Depends(f)`` names ``f``; ``Depends(HTTPBearer())`` is a scheme; bare uses the type."""
        if not self._is_depends(call):
            return None
        target: ast.expr | None = call.args[0] if call.args else None
        if target is None:
            target = next((k.value for k in call.keywords if k.arg == "dependency"), None)
        if target is None:
            return _Dep(
                annotation or "", tuple(annotation.split(".")) if annotation else None, False
            )
        if isinstance(target, ast.Call):
            callee = _chain_of(target.func, allow_call=False)
            name = callee[-1] if callee else ""
            return _Dep(name, None, name in _SECURITY_CLASSES)
        chain = _chain_of(target, allow_call=False)
        return _Dep(chain[-1] if chain else "", chain, False)

    def _dep_list(self, node: ast.expr | None) -> list[_Dep]:
        if not isinstance(node, (ast.List, ast.Tuple)):
            return []
        return [
            d
            for e in node.elts
            if isinstance(e, ast.Call) and (d := self._dep(e, None)) is not None
        ]

    # -- imports --------------------------------------------------------------------
    def _import(self, node: ast.Import, ctx: _Ctx, stack: _Stack) -> None:
        for alias in node.names:
            self.m.imports.append(_ImportRef(alias.name, ()))
            self.m.signal_names.add(alias.name)
            if alias.asname:
                self.m.bindings[alias.asname] = _Binding(alias.name, None)
            else:
                top = alias.name.split(".")[0]
                self.m.bindings[top] = _Binding(top, None)

    def _import_from(self, node: ast.ImportFrom, ctx: _Ctx, stack: _Stack) -> None:
        base = self._absolute(node.module, node.level)
        if base is None:
            return
        self.m.imports.append(_ImportRef(base, tuple(a.name for a in node.names)))
        self.m.signal_names.add(base)
        for alias in node.names:
            if alias.name != "*":
                self.m.bindings[alias.asname or alias.name] = _Binding(base, alias.name)

    def _absolute(self, module: str | None, level: int) -> str | None:
        if level == 0:
            return module
        parts = self.m.package.split(".") if self.m.package else []
        if level - 1 > len(parts):
            return None
        parts = parts[: len(parts) - (level - 1)]
        if module:
            parts.append(module)
        return ".".join(parts) or None

    # -- assignments -------------------------------------------------------------------
    def _assign(self, node: ast.Assign, ctx: _Ctx, stack: _Stack) -> None:
        value = _value_desc(node.value)
        for target in node.targets:
            self._bind(target, node.value, value, None, ctx, node.lineno)
        if not ctx.class_body:
            stack.append((node.value, ctx))

    def _ann_assign(self, node: ast.AnnAssign, ctx: _Ctx, stack: _Stack) -> None:
        text, _ = _annotation_info(node.annotation)
        value = _value_desc(node.value) if node.value is not None else _Value("unknown")
        self._bind(node.target, node.value, value, text, ctx, node.lineno)
        if node.value is not None and not ctx.class_body:
            stack.append((node.value, ctx))

    def _bind(
        self,
        target: ast.expr,
        expr: ast.expr | None,
        value: _Value,
        annotation: str | None,
        ctx: _Ctx,
        line: int,
    ) -> None:
        if isinstance(target, ast.Name):
            if ctx.class_body and ctx.cls is not None:
                if annotation:
                    ctx.cls.attrs[target.id] = annotation
                self.m.defs.append(DefInfo("field", ctx.prefix + target.id, line))
                ref = _sql_constant(expr)
                if ref is not None:
                    ctx.cls.sql_consts[target.id] = ref
            elif ctx.func is None:
                self._module_var(target.id, expr, value, line)
            else:
                ctx.func.assigns.append(_Assign(line, target.id, value, annotation))
                self._maybe_router(target.id, expr, ctx.func.qual, line)
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and ctx.cls is not None
            and ctx.func is not None
        ):
            ctx.cls.attr_assigns.append(
                _AttrAssign(target.attr, value, ctx.func.params, annotation)
            )

    def _module_var(self, name: str, expr: ast.expr | None, value: _Value, line: int) -> None:
        self.m.var_values[name] = value
        if expr is None:
            return
        if isinstance(expr, _STRING_EXPRS):
            self.m.consts[name] = expr
        ref = _sql_constant(expr)
        if ref is not None:
            self.m.sql_consts[name] = ref
        self._maybe_router(name, expr, "", line)
        if isinstance(expr, ast.Call):
            chain = _chain_of(expr.func, allow_call=False)
            if chain and chain[-1] in _SECURITY_CLASSES:
                self.m.scheme_vars.add(name)
            if (
                chain
                and chain[-1] in _MUTABLE_FACTORIES
                and len(expr.args) <= 1
                and not expr.keywords
            ):
                self.m.mutable_globals.append(name)
        elif isinstance(expr, (ast.Dict, ast.List, ast.Set)) and not _literal_has_items(expr):
            self.m.mutable_globals.append(name)
        if name.isupper() and len(name) > 1 and not name.startswith("__"):
            self.m.defs.append(DefInfo("constant", name, line))

    def _maybe_router(self, var: str, expr: ast.expr | None, scope: str, line: int) -> None:
        if not isinstance(expr, ast.Call):
            return
        chain = _chain_of(expr.func, allow_call=False)
        if not chain:
            return
        binding = self.m.bindings.get(chain[0])
        symbol = (
            chain[-1]
            if len(chain) > 1 or binding is None or binding.symbol is None
            else binding.symbol
        )
        if symbol not in _ROUTER_CLASSES:
            return
        kwargs = {k.arg: k.value for k in expr.keywords if k.arg}
        self.m.routers.append(
            _Router(
                scope,
                var,
                "router" if symbol in ("APIRouter", "Router") else "app",
                kwargs.get("prefix"),
                self._dep_list(kwargs.get("dependencies")),
                line,
            )
        )

    # -- calls ---------------------------------------------------------------------------
    def _call(self, node: ast.Call, ctx: _Ctx, stack: _Stack) -> None:
        # calling the *result* of a call (``getattr(o, n)()``, ``f()()``) has no static callee
        chain = None if isinstance(node.func, ast.Call) else _chain_of(node.func)
        if chain is not None:
            self._route_registration(node, chain, ctx.func.qual if ctx.func is not None else "")
        fn = ctx.func
        if fn is not None:
            self._record_call(node, chain, fn)
            literal = _format_receiver(node.func)
            if literal is not None:  # ``"SELECT ... {}".format(x)``
                self._sql(ctx, literal, True)
        self._push(node, ctx, stack)

    def _record_call(self, node: ast.Call, chain: tuple[str, ...] | None, fn: _Func) -> None:
        if chain is None:
            note = self._dynamic_note(node.func)
            if note:
                fn.calls.append(_Call(None, note))
        elif chain[-1] not in _DEPENDS_NAMES:
            fn.calls.append(_Call(chain))
            self.m.signal_names.add(chain[-1])
            if chain[-1] in _DISPATCHERS:
                self._callbacks(node, fn)

    @staticmethod
    def _dynamic_note(func: ast.expr) -> str:
        if isinstance(func, ast.Call):
            return (
                "getattr"
                if _chain_of(func.func, allow_call=False) == ("getattr",)
                else "call result"
            )
        return "subscript" if isinstance(func, ast.Subscript) else ""

    @staticmethod
    def _callbacks(node: ast.Call, fn: _Func) -> None:
        """``add_task(f)`` / ``to_thread(f)`` / ``submit(f)``: ``f`` runs in the flow."""
        first: ast.expr | None = node.args[0] if node.args else None
        if first is None:
            first = next((k.value for k in node.keywords if k.arg in ("func", "target")), None)
        if first is not None:
            chain = _chain_of(first, allow_call=False)
            if chain:
                fn.calls.append(_Call(chain))

    def _route_registration(self, node: ast.Call, chain: tuple[str, ...], scope: str) -> None:
        if len(chain) != 2:
            return
        receiver, verb = chain
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if verb == "include_router":
            target = node.args[0] if node.args else kwargs.get("router")
            self.m.includes.append(
                _Include(
                    scope,
                    receiver,
                    target,
                    kwargs.get("prefix"),
                    self._dep_list(kwargs.get("dependencies")),
                    node.lineno,
                )
            )
        elif verb in ("add_api_route", "add_route"):
            path = node.args[0] if node.args else kwargs.get("path")
            endpoint = (
                node.args[1]
                if len(node.args) > 1
                else kwargs.get("endpoint") or kwargs.get("route")
            )
            self.m.explicit_routes.append(
                _ExplicitRoute(
                    scope,
                    receiver,
                    path,
                    endpoint,
                    _methods_of(kwargs.get("methods")) or ["GET"],
                    self._dep_list(kwargs.get("dependencies")),
                    node.lineno,
                )
            )

    # -- string literals: SQL ------------------------------------------------------------
    def _sql(self, ctx: _Ctx, text: str, dynamic: bool) -> None:
        if ctx.func is not None and _looks_like_sql(text):
            ctx.func.sql.append(_SqlRef(_referenced_tables(text), dynamic))

    def _joined(self, node: ast.JoinedStr, ctx: _Ctx, stack: _Stack) -> None:
        text, interpolated, inner = _joined_text(node)
        self._sql(ctx, text, interpolated)
        stack.extend((e, ctx) for e in reversed(inner))

    def _bin_op(self, node: ast.BinOp, ctx: _Ctx, stack: _Stack) -> None:
        if not isinstance(node.op, (ast.Add, ast.Mod)):
            self._push(node, ctx, stack)
            return
        leaves = _flatten_concat(node)
        parts: list[str] = []
        dynamic = has_text = False
        pending: list[ast.expr] = []
        for leaf in leaves:
            if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                parts.append(leaf.value)
                has_text = True
            elif isinstance(leaf, ast.JoinedStr):
                text, interpolated, inner = _joined_text(leaf)
                parts.append(text)
                has_text = True
                dynamic = dynamic or interpolated
                pending.extend(inner)
            else:
                parts.append("?")
                dynamic = True
                pending.append(leaf)
        if has_text:
            self._sql(ctx, "".join(parts), dynamic)
        stack.extend((e, ctx) for e in reversed(pending))


def _format_receiver(func: ast.expr) -> str | None:
    """The literal of ``"...".format`` (the only ``str.format`` form whose template is static)."""
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "format"
        and isinstance(func.value, ast.Constant)
        and isinstance(func.value.value, str)
    ):
        return func.value.value
    return None


def _is_constant_name(name: str) -> bool:
    return name.lstrip("_").isupper()


def _sql_constant(expr: ast.expr | None) -> _SqlRef | None:
    """A module/class-level string constant holding SQL (attributed to the code that reads it)."""
    if not (isinstance(expr, ast.Constant) and isinstance(expr.value, str)):
        return None
    if len(expr.value) > _MAX_STRING_CONST or not _looks_like_sql(expr.value):
        return None
    return _SqlRef(_referenced_tables(expr.value), False)


def _literal_has_items(node: ast.Dict | ast.List | ast.Set) -> bool:
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return bool(node.elts)


def _methods_of(node: ast.expr | None) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    return sorted(
        {
            e.value.upper()
            for e in node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        }
    )


# =============================================================================
# Phase 2: symbol index and call resolution
# =============================================================================


class _Sym(NamedTuple):
    kind: str  # module | class | function | var | external
    module: str
    name: str = ""


class _Type(NamedTuple):
    kind: str  # inst | cls | mod | ext
    module: str = ""
    name: str = ""


_EXT = _Type("ext")


class _Scope(NamedTuple):
    module: str
    func: _Func | None
    cls: ClassKey | None


@dataclass
class _Resolved:
    targets: list[FuncKey] = field(default_factory=list)
    ret: _Type | None = None
    unresolved: str = ""


class _Index:
    """Cross-module symbol tables plus the (cached) type and call resolver."""

    def __init__(
        self,
        modules: Mapping[str, _Module],
        tables: Iterable[str],
        internal_imports: Mapping[str, set[str]],
    ):
        self.modules = modules
        self.table_names = frozenset(tables)
        self.imports = internal_imports
        self.funcs: dict[FuncKey, _Func] = {}
        self.classes: dict[ClassKey, _Class] = {}
        self.simple_classes: dict[str, list[ClassKey]] = {}
        for m in modules.values():
            for qual, fn in m.funcs.items():
                self.funcs[(m.name, qual)] = fn
            for qual, cls in m.classes.items():
                self.classes[(m.name, qual)] = cls
                if "." not in qual:
                    self.simple_classes.setdefault(qual, []).append((m.name, qual))
        self._depth = 0
        self._mro: dict[ClassKey, list[ClassKey]] = {}
        self._attr_types: dict[tuple[ClassKey, str], _Type | None] = {}
        self._var_types: dict[tuple[str, str], _Type | None] = {}
        self._envs: dict[FuncKey, dict[str, _Type | None]] = {}
        self._subclasses: dict[ClassKey, list[ClassKey]] | None = None
        self._impls: dict[ClassKey, list[ClassKey]] = {}
        self._name_tokens: dict[ClassKey, list[str]] = {}
        self._by_token: dict[str, list[ClassKey]] | None = None
        self._guesses: dict[tuple[str, str, str | None, ClassKey | None], _Type | None] = {}
        self._guess_budget = _HEURISTIC_BUDGET

    # -- symbols ----------------------------------------------------------------------
    def lookup(self, module: str, name: str, depth: int = 0) -> _Sym | None:
        m = self.modules.get(module)
        if m is None or depth > _MAX_REEXPORT_DEPTH:
            return None
        if name in m.classes:
            return _Sym("class", module, name)
        if name in m.funcs:
            return _Sym("function", module, name)
        if name in m.var_values:
            return _Sym("var", module, name)
        binding = m.bindings.get(name)
        if binding is None:
            return None
        if binding.symbol is None:
            return _Sym("module" if binding.module in self.modules else "external", binding.module)
        sub = f"{binding.module}.{binding.symbol}"
        if sub in self.modules:
            return _Sym("module", sub)
        if binding.module in self.modules:
            return self.lookup(binding.module, binding.symbol, depth + 1)
        return _Sym("external", binding.module, binding.symbol)

    def ref_target(self, chain: Sequence[str], module: str) -> _Sym | None:
        """Walk ``mod.sub.name`` through modules to the final symbol."""
        sym = self.lookup(module, chain[0])
        for part in chain[1:]:
            if sym is None or sym.kind != "module":
                return None
            sub = f"{sym.module}.{part}"
            sym = _Sym("module", sub) if sub in self.modules else self.lookup(sym.module, part)
        return sym

    def _sym_type(self, sym: _Sym | None) -> _Type | None:
        if sym is None:
            return None
        if sym.kind == "class":
            return _Type("cls", sym.module, sym.name)
        if sym.kind == "module":
            return _Type("mod", sym.module)
        if sym.kind == "external":
            return _EXT
        if sym.kind == "var":
            return self._var_type(sym.module, sym.name)
        return None

    def _var_type(self, module: str, name: str) -> _Type | None:
        key = (module, name)
        if key not in self._var_types:
            self._var_types[key] = None  # recursion guard: ``x = x.make()`` must terminate
            value = self.modules[module].var_values[name]
            self._var_types[key] = self.value_type(value, _Scope(module, None, None))
        return self._var_types[key]

    # -- types --------------------------------------------------------------------------
    def type_from_text(self, text: str | None, module: str) -> _Type | None:
        """Instance type denoted by an annotation text (``None`` when it cannot be placed)."""
        if not text:
            return None
        head, *rest = text.split(".")
        if not rest and head in _BUILTIN_TYPES:
            return _EXT
        current = self._sym_type(self.lookup(module, head))
        if current is None and not rest and len(self.simple_classes.get(head, ())) == 1:
            key = self.simple_classes[head][0]
            current = _Type("cls", *key)
        for part in rest:
            current = self.member_type(current, part) if current is not None else None
        if current is not None and current.kind == "cls":
            return _Type("inst", current.module, current.name)
        return current if current is not None and current.kind == "ext" else None

    def member_type(self, t: _Type | None, name: str) -> _Type | None:
        if t is None:
            return None
        if t.kind == "ext":
            return _EXT
        if t.kind == "mod":
            sub = f"{t.module}.{name}"
            return (
                _Type("mod", sub)
                if sub in self.modules
                else self._sym_type(self.lookup(t.module, name))
            )
        if t.kind == "inst":
            return self.attr_type((t.module, t.name), name)
        nested = (t.module, f"{t.name}.{name}")
        return _Type("cls", *nested) if nested in self.classes else None

    def attr_type(self, ckey: ClassKey, attr: str) -> _Type | None:
        key = (ckey, attr)
        if key in self._attr_types:
            return self._attr_types[key]
        if self._depth >= _MAX_RESOLVE_DEPTH:
            return None
        self._depth += 1
        self._attr_types[key] = None  # recursion guard while resolving
        try:
            result: _Type | None = None
            for ck in self.mro(ckey):
                cls = self.classes[ck]
                result = self.type_from_text(cls.attrs.get(attr), ck[0])
                for assign in cls.attr_assigns:
                    if result is None and assign.attr == attr:
                        result = self._assign_type(assign, ck)
                if result is not None:
                    break
            self._attr_types[key] = result
            return result
        finally:
            self._depth -= 1

    def _assign_type(self, assign: _AttrAssign, ckey: ClassKey) -> _Type | None:
        declared = self.type_from_text(assign.annotation, ckey[0])
        if declared is not None:
            return declared
        value = assign.value
        if value.kind == "chain" and len(value.chain) == 1 and value.chain[0] in assign.params:
            return self.type_from_text(assign.params[value.chain[0]], ckey[0])
        return self.value_type(value, _Scope(ckey[0], None, ckey))

    def value_type(self, value: _Value, scope: _Scope) -> _Type | None:
        if value.kind == "literal":
            return _EXT
        if value.kind not in ("call", "chain") or self._depth >= _MAX_RESOLVE_DEPTH:
            return None
        self._depth += 1  # bounds the mutual recursion between values, attributes and calls
        try:
            if value.kind == "call":
                return self.resolve(value.chain, scope).ret
            return self.type_of_chain(value.chain, scope, None)[0]
        finally:
            self._depth -= 1

    def mro(self, ckey: ClassKey) -> list[ClassKey]:
        cached = self._mro.get(ckey)
        if cached is not None:
            return cached
        order = [ckey]
        # visible (partial) while the bases are resolved: a base may refer back to this class
        self._mro[ckey] = order
        queue = deque([ckey])
        while queue and len(order) < 32:
            current = queue.popleft()
            for base in self.classes[current].bases:
                t = self.type_from_text(base, current[0])
                if t is not None and t.kind == "inst" and (t.module, t.name) not in order:
                    order.append((t.module, t.name))
                    queue.append((t.module, t.name))
        return order

    def find_method(self, ckey: ClassKey, name: str) -> FuncKey | None:
        for module, qual in self.mro(ckey):
            if (module, f"{qual}.{name}") in self.funcs:
                return (module, f"{qual}.{name}")
        return None

    def _class_is_interface(self, ckey: ClassKey) -> bool:
        return any(b.rpartition(".")[2] in ("Protocol", "ABC") for b in self.classes[ckey].bases)

    def implementations(self, ckey: ClassKey) -> list[ClassKey]:
        """Subclasses plus classes whose name ends with this one's (structural implementations)."""
        if ckey in self._impls:
            return self._impls[ckey]
        if self._subclasses is None:
            subclasses: dict[ClassKey, list[ClassKey]] = {}
            for key in self.classes:
                for base in self.mro(key)[1:]:
                    subclasses.setdefault(base, []).append(key)
            self._subclasses = subclasses
        found = list(self._subclasses.get(ckey, []))
        own = self._class_tokens(ckey)
        for other in self.classes:
            tokens = self._class_tokens(other)
            if (
                other != ckey
                and other not in found
                and len(tokens) > len(own)
                and tokens[-len(own) :] == own
            ):
                found.append(other)
        self._impls[ckey] = found
        return found

    def _class_tokens(self, ckey: ClassKey) -> list[str]:
        cached = self._name_tokens.get(ckey)
        if cached is None:
            cached = self._name_tokens[ckey] = _tokens(ckey[1].rpartition(".")[2])
        return cached

    def sql_refs(self, fn: _Func) -> list[_SqlRef]:
        """SQL literals in the function plus those in constants (module or own class) it reads."""
        refs = list(fn.sql)
        cls = self.classes.get((fn.module, fn.cls)) if fn.cls else None
        module = self.modules[fn.module]
        for name in sorted(fn.names):
            own = cls.sql_consts.get(name) if cls is not None else None
            found = own or module.sql_consts.get(name)
            if found is not None:
                refs.append(found)
        return refs

    def return_type(self, fkey: FuncKey) -> _Type | None:
        fn = self.funcs[fkey]
        return self.type_from_text(fn.returns, fn.module)

    # -- local variables --------------------------------------------------------------------
    def scope_of(self, fn: _Func) -> _Scope:
        return _Scope(fn.module, fn, (fn.module, fn.cls) if fn.cls else None)

    def local_env(self, fn: _Func) -> dict[str, _Type | None]:
        key = (fn.module, fn.qual)
        cached = self._envs.get(key)
        if cached is not None:
            return cached
        env: dict[str, _Type | None] = {}
        self._envs[key] = env  # filled in place: re-entrant lookups see the partial environment
        scope = self.scope_of(fn)
        for name, text in fn.params.items():
            found = self.type_from_text(text, fn.module)
            if found is None and name in fn.param_deps and fn.param_deps[name].chain:
                found = self.resolve(fn.param_deps[name].chain or (), scope).ret
            env[name] = found
        for assign in sorted(fn.assigns, key=lambda a: a.line):
            found = self.type_from_text(assign.annotation, fn.module) or self.value_type(
                assign.value, scope
            )
            if found is not None or assign.target not in env:
                env[assign.target] = found
        return env

    def _root_type(self, name: str, scope: _Scope) -> _Type | None:
        if name.endswith("()"):
            return self.resolve(tuple(name[:-2].split(".")), scope).ret
        if name in ("self", "cls") and scope.cls is not None:
            return _Type("inst" if name == "self" else "cls", *scope.cls)
        fn = scope.func
        while fn is not None:
            env = self.local_env(fn)
            if name in env:
                return env[name]
            fn = self.modules[fn.module].funcs.get(fn.scope) if fn.scope else None
        return self._sym_type(self.lookup(scope.module, name))

    def type_of_chain(
        self, chain: Sequence[str], scope: _Scope, method: str | None
    ) -> tuple[_Type | None, bool]:
        """Type of an attribute chain; the flag: a *typed* receiver lacked a typable attribute."""
        current = self._root_type(chain[0], scope)
        if current is None:
            if len(chain) == 1 and scope.func is not None and chain[0] in scope.func.params:
                return self._heuristic_type(chain[0], scope, method), False
            return None, False
        for i, part in enumerate(chain[1:], start=1):
            nxt = self.member_type(current, part)
            last = method if i == len(chain) - 1 else None
            if current.kind == "ext" and chain[i - 1] == "state":
                # ``request.app.state.<service>``: the service-locator idiom; only the name can help
                nxt = self._heuristic_type(part, scope, last) or nxt
            elif nxt is None and current.kind == "inst":
                nxt = self._heuristic_type(part, scope, last)
            if nxt is None:
                return None, True
            current = nxt
        return current, False

    # -- calls ----------------------------------------------------------------------------------
    def resolve(self, chain: Sequence[str], scope: _Scope) -> _Resolved:
        if len(chain) == 1:
            return self._resolve_name(chain[0], scope)
        current, report = self.type_of_chain(chain[:-1], scope, chain[-1])
        if current is None:
            return _Resolved(unresolved=".".join(chain) if report else "")
        return self._call_on(current, chain[-1])

    def _resolve_name(self, name: str, scope: _Scope) -> _Resolved:
        fn = scope.func
        while fn is not None:
            nested = (fn.module, f"{fn.qual}.{name}")
            if nested in self.funcs:
                return _Resolved([nested], self.return_type(nested))
            fn = self.modules[fn.module].funcs.get(fn.scope) if fn.scope else None
        sym = self.lookup(scope.module, name)
        if sym is None and name in _BUILTIN_NAMES:
            return _Resolved(ret=_EXT)
        return self._from_symbol(sym)

    def _from_symbol(self, sym: _Sym | None) -> _Resolved:
        if sym is None:
            return _Resolved()
        if sym.kind == "external":  # a library call: its result is some external object
            return _Resolved(ret=_EXT)
        if sym.kind == "function":
            return _Resolved([(sym.module, sym.name)], self.return_type((sym.module, sym.name)))
        if sym.kind == "class":
            init = self.find_method((sym.module, sym.name), "__init__")
            return _Resolved([init] if init else [], _Type("inst", sym.module, sym.name))
        return _Resolved()

    def _call_on(self, t: _Type, method: str) -> _Resolved:
        if t.kind in ("inst", "cls"):
            ckey = (t.module, t.name)
            found = self.find_method(ckey, method)
            if found is None:
                return _Resolved()
            targets = [found]
            if self.funcs[found].stub or self._class_is_interface(ckey):
                for impl in self.implementations(ckey):
                    own = (impl[0], f"{impl[1]}.{method}")
                    if own in self.funcs:
                        targets.append(own)
            return _Resolved(targets, self.return_type(found))
        if t.kind == "mod":
            sub = f"{t.module}.{method}"
            if sub in self.modules:
                return _Resolved()
            return self._from_symbol(self.lookup(t.module, method))
        return _Resolved(ret=_EXT)

    # -- attribute-name heuristic ------------------------------------------------------------------
    def _heuristic_type(self, name: str, scope: _Scope, method: str | None) -> _Type | None:
        key = (name, scope.module, method, scope.cls)
        if key not in self._guesses:
            self._guesses[key] = self._guess_type(name, scope, method)
        return self._guesses[key]

    def _guess_type(self, name: str, scope: _Scope, method: str | None) -> _Type | None:
        words = [_ABBREVIATIONS.get(w, w) for w in _tokens(name.lstrip("_"))]
        if not words:
            return None
        candidates = self._classes_sharing(words)
        visible = {scope.module, *self.imports.get(scope.module, ())}
        best = self._best_class(words, [k for k in candidates if k[0] in visible], method, scope)
        if best is None:
            best = self._best_class(words, candidates, method, scope)
        return _Type("inst", *best) if best else None

    def _classes_sharing(self, words: Sequence[str]) -> list[ClassKey]:
        """Classes whose name has a word in common with ``words`` (a score of 2 needs one)."""
        if self._by_token is None:
            index: dict[str, list[ClassKey]] = {}
            for key in self.classes:
                for token in set(self._class_tokens(key)):
                    index.setdefault(token, []).append(key)
            self._by_token = index
        found: dict[ClassKey, None] = {}
        for word in words:
            found.update(dict.fromkeys(self._by_token.get(word, ())))
        return list(found)

    def _has_method(self, ckey: ClassKey, method: str) -> bool:
        if self.find_method(ckey, method) is not None:
            return True
        return any((m, f"{q}.{method}") in self.funcs for m, q in self.implementations(ckey))

    def _best_class(
        self, words: list[str], candidates: Iterable[ClassKey], method: str | None, scope: _Scope
    ) -> ClassKey | None:
        """Highest-scoring class: an exact name match, a shared role word, shared words."""
        joined = "".join(words)
        scored: list[tuple[int, ClassKey]] = []
        for ckey in candidates:
            if self._guess_budget <= 0:  # hostile trees: stop guessing instead of going quadratic
                break
            self._guess_budget -= 1
            tokens = self._class_tokens(ckey)
            score = len(set(words) & set(tokens))
            score += 1 if tokens[-1] == words[-1] else 0
            score += 3 if "".join(tokens) == joined else 0
            if score >= 2 and (method is None or self._has_method(ckey, method)):
                scored.append((score, ckey))
        if not scored:
            return None
        top = max(s for s, _ in scored)
        best = [k for s, k in scored if s == top]
        if len(best) > 1:
            best = self._prefer_context(best, scope)
        return best[0] if len(best) == 1 else None

    def _prefer_context(self, keys: list[ClassKey], scope: _Scope) -> list[ClassKey]:
        """Break a tie by overlap with the caller's own class / module name."""
        context = set(_tokens(scope.module.rpartition(".")[2]))
        if scope.cls is not None:
            context |= set(self._class_tokens(scope.cls))

        def overlap(key: ClassKey) -> int:
            words = set(self._class_tokens(key)) | set(_tokens(key[0].rpartition(".")[2]))
            return len(context & words)

        top = max(overlap(k) for k in keys)
        return [k for k in keys if overlap(k) == top] if top > 0 else keys

    # -- strings (route prefixes) -------------------------------------------------------
    def eval_str(self, expr: ast.expr | None, module: str, depth: int = 0) -> str | None:
        if expr is None or depth > _MAX_STRING_DEPTH:
            return None
        parts: list[str] = []
        for leaf in _flatten_concat(expr) if isinstance(expr, ast.BinOp) else [expr]:
            text = self._leaf_str(leaf, module, depth)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)

    def _leaf_str(self, leaf: ast.expr, module: str, depth: int) -> str | None:
        if isinstance(leaf, ast.Constant):
            return leaf.value if isinstance(leaf.value, str) else None
        if isinstance(leaf, ast.JoinedStr):
            out: list[str] = []
            for piece in leaf.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    out.append(piece.value)
                elif isinstance(piece, ast.FormattedValue):
                    inner = self.eval_str(piece.value, module, depth + 1)
                    if inner is None:
                        return None
                    out.append(inner)
            return "".join(out)
        if isinstance(leaf, (ast.Name, ast.Attribute)):
            chain = _chain_of(leaf, allow_call=False)
            sym = self.ref_target(chain, module) if chain else None
            if sym is not None and sym.kind == "var":
                return self.eval_str(
                    self.modules[sym.module].consts.get(sym.name), sym.module, depth + 1
                )
        return None


# =============================================================================
# Routes
# =============================================================================


@dataclass
class _RouteRecord:
    info: RouteInfo
    key: FuncKey | None
    route_deps: list[_Dep]


@dataclass
class _RouteSpec:
    method: str
    path: str
    key: FuncKey | None
    handler: str
    line: int
    deps: list[_Dep]


RouterKey = tuple[str, str, str]  # (module, scope, variable)


class _RouteBuilder:
    def __init__(self, index: _Index, extra_auth: frozenset[str]) -> None:
        self.ix = index
        self.extra_auth = extra_auth
        self.routers: dict[RouterKey, _Router] = {}
        self.unresolved: list[str] = []
        for m in index.modules.values():
            if not m.is_test:
                for router in m.routers:
                    self.routers[(m.name, router.scope, router.var)] = router
        self.specs: dict[RouterKey, list[_RouteSpec]] = {}
        self.children: dict[RouterKey, list[tuple[RouterKey, str, list[_Dep]]]] = {}

    def build(self) -> list[_RouteRecord]:
        for m in self.ix.modules.values():
            if m.is_test:
                continue
            self._decorated(m)
            self._explicit(m)
            self._includes(m)
        return self._expand()

    # -- gathering --------------------------------------------------------------------------------
    def _find_router(self, module: str, scope: str, var: str) -> RouterKey | None:
        current = scope
        while True:
            if (module, current, var) in self.routers:
                return (module, current, var)
            if not current:
                break
            current = current.rpartition(".")[0]
        sym = self.ix.lookup(module, var)
        if sym is not None and sym.kind == "var" and (sym.module, "", sym.name) in self.routers:
            return (sym.module, "", sym.name)
        return None

    def _path_text(self, expr: ast.expr | None, module: str, line: int) -> str:
        if expr is None:
            return ""
        text = self.ix.eval_str(expr, module)
        if text is None:
            self.unresolved.append(f"{module}:{line}: route path is not a static string")
            return "<dynamic>"
        return text

    def _decorated(self, m: _Module) -> None:
        for fn in m.funcs.values():
            for dec in fn.decorators:
                if dec.call is None or dec.chain is None or len(dec.chain) != 2:
                    continue
                receiver, verb = dec.chain
                if verb not in _HTTP_VERBS and verb not in ("api_route", "route"):
                    continue
                router = self._find_router(m.name, fn.scope, receiver)
                if router is None:
                    self.unresolved.append(
                        f"{m.name}:{fn.line}: route decorator on unknown receiver '{receiver}'"
                    )
                    continue
                kwargs = {k.arg: k.value for k in dec.call.keywords if k.arg}
                path_expr = dec.call.args[0] if dec.call.args else kwargs.get("path")
                methods = (
                    [verb.upper()]
                    if verb in _HTTP_VERBS
                    else _methods_of(kwargs.get("methods")) or ["GET"]
                )
                deps = [*dec.deps, *fn.deps]
                path = self._path_text(path_expr, m.name, fn.line)
                for method in methods:
                    spec = _RouteSpec(method, path, (m.name, fn.qual), fn.qual, fn.line, deps)
                    self.specs.setdefault(router, []).append(spec)

    def _explicit(self, m: _Module) -> None:
        for route in m.explicit_routes:
            router = self._find_router(m.name, route.scope, route.receiver)
            if router is None:
                self.unresolved.append(
                    f"{m.name}:{route.line}: add_route on unknown receiver '{route.receiver}'"
                )
                continue
            chain = (
                _chain_of(route.endpoint, allow_call=False) if route.endpoint is not None else None
            )
            sym = self.ix.ref_target(chain, m.name) if chain else None
            key = (sym.module, sym.name) if sym is not None and sym.kind == "function" else None
            if key is None:
                self.unresolved.append(
                    f"{m.name}:{route.line}: route endpoint is not a resolvable function"
                )
            path = self._path_text(route.path, m.name, route.line)
            handler = key[1] if key else "<unresolved>"
            for method in route.methods:
                spec = _RouteSpec(method, path, key, handler, route.line, list(route.deps))
                self.specs.setdefault(router, []).append(spec)

    def _router_of_expr(self, expr: ast.expr | None, module: str, scope: str) -> RouterKey | None:
        if isinstance(expr, ast.Call):
            chain = _chain_of(expr.func, allow_call=False)
            if chain is None:
                return None
            fn_scope = _Scope(module, self.ix.funcs.get((module, scope)), None)
            targets = self.ix.resolve(chain, fn_scope).targets
            if len(targets) != 1:
                return None
            fkey = targets[0]
            candidates = [k for k in self.routers if k[0] == fkey[0] and k[1] == fkey[1]]
            returned = [k for k in candidates if k[2] in self.ix.funcs[fkey].returned]
            pool = returned or candidates
            return pool[0] if len(pool) == 1 else None
        chain = _chain_of(expr, allow_call=False) if expr is not None else None
        if chain is None:
            return None
        if len(chain) == 1:
            return self._find_router(module, scope, chain[0])
        sym = self.ix.ref_target(chain, module)
        if sym is not None and sym.kind == "var" and (sym.module, "", sym.name) in self.routers:
            return (sym.module, "", sym.name)
        return None

    def _includes(self, m: _Module) -> None:
        for inc in m.includes:
            parent = self._find_router(m.name, inc.scope, inc.receiver)
            child = self._router_of_expr(inc.target, m.name, inc.scope)
            if parent is None or child is None:
                self.unresolved.append(
                    f"{m.name}:{inc.line}: include_router target is not statically resolvable"
                )
                continue
            prefix = self.ix.eval_str(inc.prefix, m.name) if inc.prefix is not None else ""
            if prefix is None:
                self.unresolved.append(
                    f"{m.name}:{inc.line}: include_router prefix is not a static string"
                )
                prefix = "<dynamic>"
            self.children.setdefault(parent, []).append((child, prefix, list(inc.deps)))

    # -- expansion --------------------------------------------------------------------------------
    def _expand(self) -> list[_RouteRecord]:
        included = {child for edges in self.children.values() for child, _, _ in edges}
        has_app = any(r.kind == "app" for r in self.routers.values())
        records: list[_RouteRecord] = []
        visited: set[RouterKey] = set()
        for root in sorted(self.routers):
            if root in included:
                continue
            if (
                has_app
                and self.routers[root].kind == "router"
                and (self.specs.get(root) or self.children.get(root))
            ):
                self.unresolved.append(
                    f"{root[0]}: router '{root[2]}' is not included by a resolvable include_router"
                )
            records.extend(self._walk(root, visited, len(records)))
        for key in sorted(self.routers):  # routers that only include each other have no entry point
            if key not in visited:
                records.extend(self._walk(key, visited, len(records)))
        unique: dict[tuple[str, str, str, str], _RouteRecord] = {}
        for rec in records:
            unique.setdefault(
                (rec.info.method, rec.info.path, rec.info.module, rec.info.handler), rec
            )
        return [unique[k] for k in sorted(unique)]

    def _walk(self, root: RouterKey, visited: set[RouterKey], already: int) -> list[_RouteRecord]:
        records: list[_RouteRecord] = []
        stack: list[tuple[RouterKey, str, list[_Dep], tuple[RouterKey, ...]]] = [(root, "", [], ())]
        steps = 0
        while stack:
            steps += 1
            # an include "bomb" (every router included twice, many levels) is cut off here
            if already + len(records) >= MAX_ROUTES or steps > 10 * MAX_ROUTES:
                self.unresolved.append(f"route limit reached ({MAX_ROUTES}); the rest is ignored")
                break
            key, prefix, inherited, trail = stack.pop()
            visited.add(key)
            router = self.routers[key]
            own_prefix = (
                self.ix.eval_str(router.prefix, key[0]) if router.prefix is not None else ""
            )
            if own_prefix is None:
                self.unresolved.append(
                    f"{key[0]}:{router.line}: router prefix is not a static string"
                )
                own_prefix = "<dynamic>"
            full_prefix = prefix + own_prefix
            deps_here = [*inherited, *router.deps]
            for spec in self.specs.get(key, []):
                records.append(self._record(key[0], spec, full_prefix, deps_here))
            for child, inc_prefix, inc_deps in reversed(self.children.get(key, [])):
                if child not in trail and child != key:
                    stack.append(
                        (child, full_prefix + inc_prefix, [*deps_here, *inc_deps], (*trail, key))
                    )
        return records

    def _record(
        self, module: str, spec: _RouteSpec, prefix: str, inherited: list[_Dep]
    ) -> _RouteRecord:
        joined = prefix + spec.path
        path = joined if joined.startswith("/") else "/" + joined
        deps = [*inherited, *spec.deps]
        names = sorted({d.name for d in deps if d.name})
        auth = sorted({d.name for d in deps if d.name and self._is_auth(d, module)})
        info = RouteInfo(
            spec.method,
            path,
            spec.handler,
            spec.key[0] if spec.key else module,
            spec.line,
            tuple(auth),
            tuple(names),
        )
        return _RouteRecord(info, spec.key, deps)

    def _is_auth(self, dep: _Dep, module: str) -> bool:
        if dep.scheme or dep.name in self.extra_auth or _looks_like_auth(dep.name):
            return True
        sym = self.ix.ref_target(dep.chain, module) if dep.chain else None
        return (
            sym is not None
            and sym.kind == "var"
            and sym.name in self.ix.modules[sym.module].scheme_vars
        )


# =============================================================================
# Data flows
# =============================================================================


def _is_repository_method(ix: _Index, key: FuncKey) -> bool:
    fn = ix.funcs[key]
    if any(t in ix.table_names for ref in ix.sql_refs(fn) for t in ref.tables):
        return True
    return fn.cls is not None and bool(
        set(_tokens(fn.cls.rpartition(".")[2])[-1:]) & _REPOSITORY_WORDS
    )


@dataclass(frozen=True)
class _Summary:
    """What one function contributes to every flow that reaches it (independent of the route)."""

    callees: tuple[FuncKey, ...]
    tables: frozenset[str]
    unresolved: tuple[str, ...]


class _FlowTracer:
    """Breadth-first ``route -> calls`` traversal over cached per-function summaries."""

    def __init__(self, ix: _Index) -> None:
        self.ix = ix
        self._summaries: dict[FuncKey, _Summary] = {}

    def _summary(self, key: FuncKey) -> _Summary:
        cached = self._summaries.get(key)
        if cached is None:
            cached = self._summaries[key] = self._summarise(key)
        return cached

    def _summarise(self, key: FuncKey) -> _Summary:
        ix = self.ix
        fn = ix.funcs[key]
        scope = ix.scope_of(fn)
        label = f"{key[0]}.{key[1]}"
        tables: set[str] = set()
        unresolved: set[str] = set()
        for ref in ix.sql_refs(fn):
            tables.update(t for t in ref.tables if t in ix.table_names)
            if ref.dynamic:
                unresolved.add(f"dynamic SQL in {label}")
        callees: dict[FuncKey, None] = {}
        for dep in fn.deps:
            if dep.chain:
                callees.update(dict.fromkeys(ix.resolve(dep.chain, scope).targets))
        for call in fn.calls:
            if call.chain is None:
                unresolved.add(f"dynamic call ({call.note}) in {label}")
                continue
            resolved = ix.resolve(call.chain, scope)
            callees.update(dict.fromkeys(resolved.targets))
            if resolved.unresolved:
                unresolved.add(f"{label}: {resolved.unresolved}")
        return _Summary(tuple(callees), frozenset(tables), tuple(sorted(unresolved)))

    def trace(self, record: _RouteRecord) -> DataFlow:
        ix, info, start = self.ix, record.info, record.key
        if start is None or start not in ix.funcs:
            return DataFlow(
                info.label,
                info.module,
                info.handler,
                unresolved=("handler is not a resolvable function",),
            )
        scope = ix.scope_of(ix.funcs[start])
        route_level = [
            target
            for dep in record.route_deps
            if dep.chain
            for target in ix.resolve(dep.chain, scope).targets
        ]
        order: list[FuncKey] = [start]
        seen = {start}
        queue: deque[tuple[FuncKey, int]] = deque([(start, 0)])
        tables: set[str] = set()
        unresolved: set[str] = set()
        truncated = False
        while queue:
            key, depth = queue.popleft()
            summary = self._summary(key)
            tables |= summary.tables
            unresolved.update(summary.unresolved)
            for target in [*summary.callees, *(route_level if key == start else ())]:
                if target in seen:
                    continue
                if depth + 1 > MAX_FLOW_DEPTH or len(seen) >= MAX_FLOW_FUNCS:
                    truncated = True
                    continue
                seen.add(target)
                order.append(target)
                queue.append((target, depth + 1))
        if truncated:
            unresolved.add("flow truncated at the depth/size limit")
        called = [FlowCall(*k) for k in order[1:]]
        repos = [FlowCall(*k) for k in order[1:] if _is_repository_method(ix, k)]
        return DataFlow(
            info.label,
            info.module,
            info.handler,
            tuple(sorted(called, key=lambda c: c.label)),
            tuple(sorted(repos, key=lambda c: c.label)),
            tuple(sorted(tables)),
            tuple(sorted(unresolved)),
        )


# =============================================================================
# analyze_codebase
# =============================================================================


def _read_text(view: FileView, path: str) -> str | None:
    try:
        return view.read_text(path)
    except (OSError, UnicodeError, ValueError):
        return None


def _size_problem(text: str) -> str:
    size = len(text.encode("utf-8", "replace"))
    if size > MAX_SOURCE_BYTES:
        return f"file too large ({size} bytes > {MAX_SOURCE_BYTES})"
    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        return f"too many lines ({len(lines)} > {MAX_LINES})"
    if max(map(len, lines), default=0) > MAX_LINE_CHARS:
        return f"line too long (> {MAX_LINE_CHARS} chars)"
    return ""


def _module_name(path: str) -> tuple[str, bool]:
    parts = path[:-3].split("/")
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts), is_package


def _is_test_path(path: str) -> bool:
    segments = path.split("/")
    name = segments[-1]
    return (
        any(s in ("tests", "test") for s in segments[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _discover(view: FileView) -> tuple[list[str], list[str], list[FileRecord]]:
    skipped: list[FileRecord] = []
    py: list[str] = []
    sql: list[str] = []
    candidates = [*view.list_files("**/*.py"), *view.list_files("**/*.sql")]
    for path in sorted(set(candidates)):
        try:
            validate_rel_path(path)
        except ValueError:
            skipped.append(FileRecord(path, "invalid path"))
            continue
        if any(s in _IGNORED_SEGMENTS for s in path.split("/")[:-1]):
            continue
        if path.endswith(".py"):
            py.append(path)
        elif _is_migration_path(path):
            sql.append(path)
    return py, sql, skipped


def _parse(text: str) -> tuple[ast.Module | None, str]:
    if "\x00" in text:
        return None, "NUL byte in source"
    try:
        return ast.parse(text), ""
    except SyntaxError as exc:
        return None, f"syntax error at line {exc.lineno or 0}"
    except RecursionError:
        return None, "nesting too deep to parse"
    except MemoryError:
        return None, "out of memory while parsing"
    except (ValueError, OverflowError):
        return None, "source rejected by the parser"


def _acquire_source(view: FileView, path: str, total: int = 0) -> tuple[str | None, str]:
    """Read one source file under the size caps; returns ``(text, "")`` or ``(None, reason)``."""
    text = _read_text(view, path)
    if text is None:
        return None, "unreadable"
    problem = _size_problem(text)
    if problem:
        return None, problem
    if total + len(text) > MAX_TOTAL_BYTES:
        return None, "total size budget exhausted"
    return text, ""


def _load_modules(
    view: FileView, paths: Sequence[str], skipped: list[FileRecord], unparseable: list[FileRecord]
) -> dict[str, _Module]:
    modules: dict[str, _Module] = {}
    total = 0
    for count, path in enumerate(paths):
        name, is_package = _module_name(path)
        if not name or name in modules:
            skipped.append(FileRecord(path, "empty or duplicate module name"))
            continue
        if count >= MAX_PY_FILES:
            skipped.append(FileRecord(path, "file limit reached"))
            continue
        text, problem = _acquire_source(view, path, total)
        if text is None:
            skipped.append(FileRecord(path, problem))
            continue
        total += len(text)
        module = _Module(name, path, _is_test_path(path), is_package)
        tree, reason = _parse(text)
        if tree is not None:
            try:
                _Extractor(module).run(tree)
            except _TooComplex:
                tree, reason = None, f"too many AST nodes (> {MAX_AST_NODES})"
        if tree is None:
            unparseable.append(FileRecord(path, reason))
            module = _Module(name, path, module.is_test, is_package, parsed=False)
        modules[name] = module
    return modules


def _internal_imports(module: _Module, known: Mapping[str, _Module]) -> set[str]:
    targets: set[str] = set()
    for ref in module.imports:
        if not ref.names:
            if ref.module in known:
                targets.add(ref.module)
            continue
        for name in ref.names:
            sub = f"{ref.module}.{name}"
            if name != "*" and sub in known:
                targets.add(sub)
            elif ref.module in known:
                targets.add(ref.module)
    targets.discard(module.name)
    return targets


def _signals(m: _Module) -> tuple[str, ...]:
    names = m.signal_names
    code_tokens = set(_tokens(m.basename))
    for d in m.defs:
        if d.kind in ("class", "function", "method"):
            code_tokens.update(_tokens(d.name))
    found: set[str] = set()
    thread = bool(names & _THREAD_NAMES or names & _THREAD_CALLS)
    queue = bool("queue" in names or names & _QUEUE_CALLS)
    if thread:
        found.add("thread")
    if queue:
        found.add("queue")
    if names & _TASK_CALLS:
        found.add("background-task")
    if names & _CACHE_DECORATORS or code_tokens & _CACHE_WORDS:
        found.add("cache")
    if code_tokens & _LIMITER_WORDS:
        found.add("rate-limiter")
    if code_tokens & _WRITER_WORDS and (
        thread or queue or names & _TASK_CALLS or "recorder" in m.basename
    ):
        found.add("background-writer")
    if m.mutable_globals:
        found.add("mutable-global")
    return tuple(sorted(found))


def _table_refs(ix: _Index) -> dict[str, tuple[str, ...]]:
    refs: dict[str, tuple[str, ...]] = {}
    for m in ix.modules.values():
        names = {
            t
            for fn in m.funcs.values()
            for ref in ix.sql_refs(fn)
            for t in ref.tables
            if t in ix.table_names
        }
        if names:
            refs[m.name] = tuple(sorted(names))
    return refs


def analyze_codebase(view: FileView, *, auth_dependency_names: Iterable[str] = ()) -> CodebaseModel:
    """Analyse ``view`` statically; never raises on hostile content (it is recorded instead).

    ``auth_dependency_names`` (optional extension) lists dependency names that must always count as
    authentication in addition to the built-in name heuristic and security-scheme detection.
    """
    py_paths, sql_paths, skipped = _discover(view)
    unparseable: list[FileRecord] = []
    modules = _load_modules(view, py_paths, skipped, unparseable)
    replay = _replay_migrations(view, sql_paths, skipped)
    tables = replay.result()
    internal = {name: _internal_imports(m, modules) for name, m in modules.items()}
    ix = _Index(modules, (t.name for t in tables), internal)
    builder = _RouteBuilder(ix, frozenset(n for n in auth_dependency_names if isinstance(n, str)))
    records = builder.build()
    tracer = _FlowTracer(ix)
    flows = tuple(
        sorted((tracer.trace(r) for r in records), key=lambda f: (f.route, f.module, f.handler))
    )
    infos = {m.name: m for m in modules.values()}
    test_map = {
        m.path: tuple(sorted(t for t in internal[m.name] if not infos[t].is_test))
        for m in sorted(modules.values(), key=lambda x: x.path)
        if m.is_test and (not m.is_package or internal[m.name])
    }
    module_infos = tuple(
        ModuleInfo(
            m.name,
            m.path,
            m.is_test,
            m.parsed,
            tuple(sorted(m.defs, key=lambda d: (d.line, d.name, d.kind))),
            tuple(sorted(internal[m.name])),
            _signals(m),
        )
        for m in sorted(modules.values(), key=lambda x: x.name)
    )
    migrations = tuple(_ordered_migrations(sql_paths)[:MAX_SQL_FILES])
    return CodebaseModel(
        modules=module_infos,
        import_graph={name: tuple(sorted(targets)) for name, targets in sorted(internal.items())},
        routes=tuple(r.info for r in records),
        tables=tables,
        migrations=migrations,
        test_map=test_map,
        table_refs=_table_refs(ix),
        data_flows=flows,
        unresolved=tuple(sorted(set(builder.unresolved))),
        unparseable=tuple(sorted(unparseable, key=lambda r: r.path)),
        skipped=tuple(sorted(skipped, key=lambda r: (r.path, r.reason))),
    )


# =============================================================================
# Impact analysis
# =============================================================================


@dataclass(frozen=True)
class ImpactModule:
    module: str
    path: str
    reason: str
    role: str  # seed | dependent

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class CouplingRisk:
    component: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ImpactReport:
    """What a change described by ``keywords`` is likely to touch; ``to_dict()`` is the payload."""

    keywords: tuple[str, ...]
    modules: tuple[ImpactModule, ...]
    routes: tuple[RouteInfo, ...]
    tables: tuple[TableInfo, ...]
    data_flows: tuple[DataFlow, ...]
    tests: tuple[str, ...]
    coupling_risks: tuple[CouplingRisk, ...]
    risk_signals: tuple[str, ...]
    expected_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "keywords": list(self.keywords),
            "modules": [m.to_dict() for m in self.modules],
            "routes": [r.to_dict() for r in self.routes],
            "tables": [t.to_dict() for t in self.tables],
            "data_flows": [f.to_dict() for f in self.data_flows],
            "tests": list(self.tests),
            "coupling_risks": [c.to_dict() for c in self.coupling_risks],
            "risk_signals": list(self.risk_signals),
            "expected_paths": list(self.expected_paths),
        }


class _Keyword(NamedTuple):
    text: str
    stems: frozenset[str]

    def matches(self, tokens: AbstractSet[str]) -> bool:
        return self.stems <= tokens


def _clean_keywords(keywords: Iterable[str] | str) -> list[_Keyword]:
    raw_items: Iterable[object] = (
        [keywords] if isinstance(keywords, str) else islice(keywords, MAX_KEYWORDS * 4)
    )
    cleaned: dict[frozenset[str], _Keyword] = {}
    for raw in raw_items:
        if not isinstance(raw, str):
            continue
        text = strip_controls(raw[:MAX_KEYWORD_CHARS]).strip().lower()
        stems = frozenset(t for t in _tokens(text) if t not in _STOPWORDS)
        if stems and stems not in cleaned and len(cleaned) < MAX_KEYWORDS:
            cleaned[stems] = _Keyword(text, stems)
    return sorted(cleaned.values(), key=lambda k: k.text)


class _Impact:
    """One impact computation over a fixed model."""

    def __init__(
        self,
        model: CodebaseModel,
        keywords: list[_Keyword],
        seed_modules: Iterable[str],
        depth: int,
    ) -> None:
        self.model = model
        self.keywords = keywords
        self.depth = depth
        self.by_name = {m.name: m for m in model.modules}
        self.by_path = {m.path: m for m in model.modules}
        self.explicit = (
            [seed_modules]
            if isinstance(seed_modules, str)
            else list(islice(seed_modules, MAX_KEYWORDS))
        )
        self.reasons: dict[str, set[str]] = {}
        self.matched_tables: set[str] = set()
        self.flows = {(f.route, f.module, f.handler): f for f in model.data_flows}

    def _seed(self, module: str, reason: str) -> None:
        self.reasons.setdefault(module, set()).add(reason)

    # -- seeds -------------------------------------------------------------------------
    def find_seeds(self) -> None:
        self._explicit_seeds()
        self._symbol_seeds()
        for route in self.model.routes:
            if self._is_test(route.module):
                continue
            tokens = {*_tokens(route.method), *_tokens(route.path), *_tokens(route.handler)}
            for kw in self.keywords:
                if kw.matches(tokens):
                    self._seed(route.module, f"route {route.label} matches '{kw.text}'")
        self._table_seeds()

    def _is_test(self, module: str) -> bool:
        info = self.by_name.get(module)
        return info is not None and info.is_test

    def _explicit_seeds(self) -> None:
        for raw in self.explicit:
            if not isinstance(raw, str):
                continue
            info = self.by_name.get(raw) or self.by_path.get(raw)
            if info is not None and not info.is_test:
                self._seed(info.name, "explicit seed")

    def _symbol_seeds(self) -> None:
        """Index names and definitions by word once; each keyword probes its rarest word."""
        index: dict[str, list[tuple[str, str, frozenset[str], frozenset[str]]]] = {}
        for m in self.model.modules:
            if m.is_test:
                continue
            base = frozenset(_tokens(m.name.rpartition(".")[2]))
            for kw in self.keywords:
                if kw.matches(base):
                    self._seed(m.name, f"module name matches '{kw.text}'")
            for d in m.defs:
                parent, _, leaf = d.name.rpartition(".")
                own = frozenset(_tokens(leaf))
                context = base | frozenset(_tokens(parent))
                for word in own | context:
                    index.setdefault(word, []).append((m.name, d.name, own, context))
        for kw in self.keywords:
            rarest = min(sorted(kw.stems), key=lambda w: len(index.get(w, ())))
            for module, name, own, context in index.get(rarest, ()):
                # A definition matches on its own words; a multi-word keyword may borrow the
                # context (``link create`` -> ``LinkService.create``) if the definition contributes.
                if kw.matches(own) or (not kw.matches(context) and kw.matches(own | context)):
                    self._seed(module, f"defines {name} (matches '{kw.text}')")

    def _table_seeds(self) -> None:
        for table in self.model.tables:
            words = [set(_tokens(table.name)), *(set(_tokens(c)) for c in table.columns)]
            for kw in self.keywords:
                if not any(kw.matches(w) for w in words):
                    continue
                self.matched_tables.add(table.name)
                for module, refs in self.model.table_refs.items():
                    if table.name in refs and not self._is_test(module):
                        self._seed(module, f"references table '{table.name}' (matches '{kw.text}')")

    # -- dependents --------------------------------------------------------------------
    def dependents(self, seeds: set[str]) -> dict[str, tuple[int, str]]:
        reverse: dict[str, list[str]] = {}
        for module, targets in self.model.import_graph.items():
            if self._is_test(module):
                continue
            for target in targets:
                if not self._is_test(target):
                    reverse.setdefault(target, []).append(module)
        found: dict[str, tuple[int, str]] = {}
        frontier = sorted(seeds)
        for level in range(1, self.depth + 1):
            nxt: list[str] = []
            for module in frontier:
                for dep in sorted(reverse.get(module, ())):
                    if dep not in seeds and dep not in found:
                        found[dep] = (level, module)
                        nxt.append(dep)
            frontier = nxt
        return found

    # -- assembling the report -------------------------------------------------------
    def report(self) -> ImpactReport:
        self.find_seeds()
        seeds = set(self.reasons)
        dependents = self.dependents(seeds)
        impacted = seeds | set(dependents)
        routes, flows = self._impacted_routes(seeds)
        tables = self._impacted_tables(seeds, flows)
        tests = self.model.tests_for(impacted)
        modules = tuple(
            sorted(
                (
                    *(
                        self._module_entry(
                            n, "; ".join(sorted(self.reasons[n])[:_MAX_REASONS]), "seed"
                        )
                        for n in seeds
                    ),
                    *(
                        self._module_entry(n, f"imports {via} (depth {d})", "dependent")
                        for n, (d, via) in dependents.items()
                    ),
                ),
                key=lambda e: e.module,
            )
        )
        return ImpactReport(
            keywords=tuple(k.text for k in self.keywords),
            modules=modules,
            routes=routes,
            tables=tables,
            data_flows=flows,
            tests=tests,
            coupling_risks=self._coupling(seeds, impacted, flows),
            risk_signals=self._risk_signals(seeds, impacted, routes, tables, flows),
            expected_paths=self._expected_paths(modules, tests, tables),
        )

    def _module_entry(self, name: str, reason: str, role: str) -> ImpactModule:
        return ImpactModule(name, self.by_name[name].path, reason, role)

    def _impacted_routes(
        self, seeds: set[str]
    ) -> tuple[tuple[RouteInfo, ...], tuple[DataFlow, ...]]:
        seed_tables = self.matched_tables | {
            t for s in seeds for t in self.model.table_refs.get(s, ())
        }
        routes: list[RouteInfo] = []
        flows: list[DataFlow] = []
        for route in self.model.routes:
            flow = self.flows.get((route.label, route.module, route.handler))
            hit = route.module in seeds or (
                flow is not None
                and (flow.touched_modules & seeds or set(flow.tables) & seed_tables)
            )
            if hit:
                routes.append(route)
                if flow is not None:
                    flows.append(flow)
        return tuple(routes), tuple(flows)

    def _impacted_tables(self, seeds: set[str], flows: Sequence[DataFlow]) -> tuple[TableInfo, ...]:
        names = self.matched_tables | {t for s in seeds for t in self.model.table_refs.get(s, ())}
        names |= {t for f in flows for t in f.tables}
        return tuple(t for t in self.model.tables if t.name in names)

    def _coupling(
        self, seeds: set[str], impacted: set[str], flows: Sequence[DataFlow]
    ) -> tuple[CouplingRisk, ...]:
        flow_modules: dict[str, str] = {}
        for flow in flows:
            for module in sorted(flow.touched_modules):
                flow_modules.setdefault(module, flow.route)
        adjacent: dict[str, str] = {}
        for seed in sorted(seeds):
            for target in self.model.import_graph.get(seed, ()):
                adjacent.setdefault(target, seed)
        risks: list[CouplingRisk] = []
        for info in self.model.modules:
            if info.is_test or not info.signals:
                continue
            if info.name in impacted:
                where = "in the impacted set"
            elif info.name in flow_modules:
                where = f"on the flow of {flow_modules[info.name]}"
            elif info.name in adjacent:
                where = f"imported by {adjacent[info.name]}"
            else:
                continue
            risks.append(CouplingRisk(info.name, f"{', '.join(info.signals)} ({where})"))
        return tuple(sorted(risks, key=lambda r: r.component))

    def _risk_signals(
        self,
        seeds: set[str],
        impacted: set[str],
        routes: Sequence[RouteInfo],
        tables: Sequence[TableInfo],
        flows: Sequence[DataFlow],
    ) -> tuple[str, ...]:
        model = self.model
        out: list[str] = []
        if not seeds:
            out.append(
                "no-seed-matched: keywords matched no module, route or table; empty impact set"
            )
        if tables:
            out.append(f"schema-impacted: tables {', '.join(t.name for t in tables)}")
            if model.migrations:
                out.append(
                    f"migrations-present: {len(model.migrations)} applied migration file(s); "
                    "add a new migration, never edit an applied one"
                )
        auth_routes = [r.label for r in routes if r.auth]
        open_routes = [r.label for r in routes if not r.auth]
        if auth_routes:
            out.append(f"auth-routes-impacted: {', '.join(sorted(auth_routes))}")
        if open_routes:
            out.append(f"unauthenticated-routes-impacted: {', '.join(sorted(open_routes))}")
        auth_names = {a for r in model.routes for a in r.auth}
        auth_code = sorted(
            m.name
            for m in model.modules
            if m.name in impacted and any(d.name in auth_names for d in m.defs)
        )
        if auth_code:
            out.append(f"auth-code-impacted: {', '.join(auth_code)}")
        out.extend(self._destructive(seeds, routes))
        if any(f.unresolved for f in flows):
            out.append(
                f"unresolved-flows: {sum(1 for f in flows if f.unresolved)} impacted flow(s) "
                "contain unresolved calls"
            )
        tested = {m for mods in model.test_map.values() for m in mods}
        untested = sorted(seeds - tested)
        if untested:
            out.append(f"untested-modules: {', '.join(untested[:10])}")
        if model.unparseable:
            out.append(f"unparseable-sources: {', '.join(r.path for r in model.unparseable[:5])}")
        if model.skipped:
            out.append(f"skipped-sources: {len(model.skipped)} file(s) were not analysed")
        if len(impacted) > WIDE_IMPACT_MODULES:
            out.append(f"wide-impact: {len(impacted)} modules impacted")
        return tuple(sorted(out))

    def _destructive(self, seeds: set[str], routes: Sequence[RouteInfo]) -> list[str]:
        names = {r.label for r in routes if r.method == "DELETE"}
        for kw in self.keywords:
            if kw.stems & _DESTRUCTIVE_WORDS:
                names.add(f"keyword '{kw.text}'")
        for m in self.model.modules:
            if m.name in seeds:
                names.update(
                    f"{m.name}.{d.name}"
                    for d in m.defs
                    if set(_tokens(d.name)) & _DESTRUCTIVE_WORDS
                )
        return [f"destructive-operations: {', '.join(sorted(names)[:10])}"] if names else []

    def _expected_paths(
        self, modules: Sequence[ImpactModule], tests: Sequence[str], tables: Sequence[TableInfo]
    ) -> tuple[str, ...]:
        paths = {m.path for m in modules} | set(tests)
        if tables:
            paths |= {f"{p.rpartition('/')[0]}/**" for p in self.model.migrations if "/" in p}
        return tuple(sorted(paths))


def impact_analysis(
    model: CodebaseModel,
    keywords: Iterable[str] | str,
    seed_modules: Iterable[str] = (),
    depth: int = 2,
) -> ImpactReport:
    """Predict what a change described by ``keywords`` touches.

    Seeds are modules whose name or definitions match a keyword (case-insensitive, token-based,
    plural folded: ``links`` matches ``LinkService``; a multi-word keyword needs all its words),
    modules that define a matching route, modules whose SQL references a table or column that
    matches, and ``seed_modules`` (dotted names or paths; unknown ones are ignored). Dependents are
    found by reverse-import closure up to ``depth`` (test modules never propagate impact; they are
    reported in ``tests`` when they import an impacted module). Input is bounded (at most 64
    keywords of at most 100 characters).
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= MAX_IMPACT_DEPTH:
        raise ValueError(f"depth must be an integer in [0, {MAX_IMPACT_DEPTH}]")
    return _Impact(model, _clean_keywords(keywords), seed_modules, depth).report()
