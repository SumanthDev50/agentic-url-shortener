"""AST based detectors for Python source (the engine room of SEC-002 / SEC-004 / SEC-007 / SEC-008).

Design constraints (all enforced here, once):

* **Guarded parsing.**  ``parse_python`` never raises: size cap, ``SyntaxError`` / ``ValueError`` /
  ``RecursionError`` / ``MemoryError`` / ``OverflowError`` become an ``unparseable`` hit.  Callers
  must turn that into a finding - an unanalysable file is never a pass.
* **Iterative traversal only.**  Hostile input can nest expressions thousands of levels deep; every
  walk here uses an explicit stack (``ast.walk`` or ``iter_scoped``), never recursion.
* **Alias aware.**  ``import os as o`` / ``from os import system as s`` resolve to the canonical
  dotted name before matching, so the trivial renaming bypasses do not work.  Anything dynamic
  beyond that (``m = os; getattr(m, name)``) is a documented limitation: these detectors are
  tripwires, not a security boundary (the boundary is structural - see docs/architecture.md §1.5).
* **No source leakage.**  Evidence is a callee name or a short tag (``os.system(...)``), never a
  source line, so a finding can not echo a secret that sits next to the construct.
"""

from __future__ import annotations

import ast
import re
import threading
import warnings
from collections.abc import Callable, Iterator

from orchestrator.policies.hits import UNPARSEABLE, Hit, format_location, limit_hit

# ``warnings.catch_warnings`` mutates process-global state; the lock keeps concurrent policy
# evaluations (engine worker threads) from interleaving their save/restore of the filters.
_PARSE_LOCK = threading.Lock()
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError)


def parse_python(text: str, label: str) -> tuple[ast.Module | None, Hit | None]:
    """Parse ``text``; exactly one element of the returned pair is ``None``."""
    capped = limit_hit(text, label, check_lines=False)
    if capped is not None:
        return None, capped
    try:
        with _PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text, filename="<policy>"), None
    except _PARSE_ERRORS as exc:
        line = getattr(exc, "lineno", None)
        detail = getattr(exc, "msg", None) or type(exc).__name__
        return None, Hit(
            UNPARSEABLE,
            format_location(label, line if isinstance(line, int) else 0),
            f"Python source cannot be parsed and therefore cannot be analysed: {str(detail)[:80]}",
        )


def compile_check(text: str, label: str) -> Hit | None:
    """QLT-001: ``compile()`` (never executes) catches errors ``ast.parse`` accepts, e.g. ``return``
    outside a function.
    """
    capped = limit_hit(text, label, check_lines=False)
    if capped is not None:
        return capped
    try:
        with _PARSE_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(text, "<policy>", "exec", dont_inherit=True)
    except _PARSE_ERRORS as exc:
        line = getattr(exc, "lineno", None)
        detail = getattr(exc, "msg", None) or type(exc).__name__
        return Hit(
            "syntax_error",
            format_location(label, line if isinstance(line, int) else 0),
            f"Python does not compile: {str(detail)[:80]}",
        )
    return None


def analyse(text: str, label: str, analyser: Callable[[ast.AST, str], list[Hit]]) -> list[Hit]:
    """Parse ``text`` and run ``analyser``; a parse failure yields its fail-closed hit."""
    tree, problem = parse_python(text, label)
    if tree is None:
        return [problem] if problem is not None else []
    return analyser(tree, label)


def iter_scoped(tree: ast.AST) -> Iterator[tuple[ast.AST, int]]:
    """Yield ``(node, scope_id)`` parents-first; ``scope_id`` is the enclosing scope (0: module)."""
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    scope_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    while stack:
        node, scope = stack.pop()
        yield node, scope
        child_scope = id(node) if isinstance(node, scope_nodes) else scope
        stack.extend((child, child_scope) for child in ast.iter_child_nodes(node))


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map every locally bound import name to its canonical dotted name."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
                else:
                    top = item.name.split(".", 1)[0]
                    aliases.setdefault(top, top)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def dotted_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Canonical dotted name of a ``Name`` / ``Attribute`` chain (iterative), else ``None``."""
    attrs: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    base = aliases.get(cur.id, cur.id)
    return ".".join([base, *reversed(attrs)])


def _is_str_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))


def _line(node: ast.AST) -> int:
    value = getattr(node, "lineno", 0)
    return value if isinstance(value, int) else 0


# ------------------------------------------------------------------- SEC-002 dangerous constructs

_BUILTIN_EXEC = frozenset({"eval", "exec", "compile"})
_OS_COMMANDS = frozenset(
    {"os.system", "os.popen", "os.startfile", "asyncio.create_subprocess_shell"}
)
_OS_COMMAND_PREFIXES = ("os.spawn", "os.exec", "os.posix_spawn")
_SUBPROCESS_CALLS = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
    }
)
_UNSAFE_DESERIALISERS = frozenset(
    {
        "pickle.loads",
        "pickle.load",
        "pickle.Unpickler",
        "cPickle.loads",
        "cPickle.load",
        "_pickle.loads",
        "_pickle.load",
        "marshal.loads",
        "marshal.load",
        "dill.loads",
        "dill.load",
        "shelve.open",
        "jsonpickle.decode",
    }
)
_YAML_ALWAYS_UNSAFE = frozenset(
    {"yaml.unsafe_load", "yaml.unsafe_load_all", "yaml.full_load", "yaml.full_load_all"}
)
_YAML_LOAD = frozenset({"yaml.load", "yaml.load_all"})
_YAML_SAFE_LOADERS = frozenset({"SafeLoader", "CSafeLoader"})
_DYNAMIC_IMPORTS = frozenset({"__import__", "importlib.import_module", "importlib.__import__"})
_DANGEROUS_MODULES = frozenset(
    {
        "os",
        "subprocess",
        "sys",
        "builtins",
        "__builtins__",
        "importlib",
        "ctypes",
        "socket",
        "shutil",
        "pickle",
        "marshal",
    }
)


def dangerous_calls(tree: ast.AST, label: str) -> list[Hit]:
    """SEC-002: eval/exec/compile of non-literals, shell/OS commands, unsafe (de)serialisation, TLS
    off, dynamic import.
    """
    aliases = import_aliases(tree)
    hits: list[Hit] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            hits.extend(_check_call(node, aliases, label))
        elif isinstance(node, ast.Attribute) and dotted_name(node, aliases) == "ssl.CERT_NONE":
            hits.append(
                _hit(
                    "tls_verify_disabled",
                    label,
                    node,
                    "ssl.CERT_NONE disables certificate verification",
                    "ssl.CERT_NONE",
                )
            )
        elif isinstance(node, ast.Assign) and _disables_hostname_check(node):
            hits.append(
                _hit(
                    "tls_verify_disabled",
                    label,
                    node,
                    "check_hostname disabled",
                    "check_hostname=False",
                )
            )
    return hits


def _hit(code: str, label: str, node: ast.AST, message: str, evidence: str) -> Hit:
    return Hit(code, format_location(label, _line(node)), message, evidence)


def _disables_hostname_check(node: ast.Assign) -> bool:
    return (
        isinstance(node.value, ast.Constant)
        and node.value.value is False
        and any(isinstance(t, ast.Attribute) and t.attr == "check_hostname" for t in node.targets)
    )


def _check_call(node: ast.Call, aliases: dict[str, str], label: str) -> list[Hit]:
    callee = dotted_name(node.func, aliases)
    if callee is None:
        return _check_keywords_only(node, label)
    short = callee.removeprefix("builtins.").removeprefix("__builtins__.")
    hits = _check_keywords_only(node, label)
    first = node.args[0] if node.args else None
    if short in _BUILTIN_EXEC and not _is_str_literal(first):
        hits.append(
            _hit(
                "eval_exec",
                label,
                node,
                f"{short}() of a non-literal value",
                f"{short}(<non-literal>)",
            )
        )
    elif short in _OS_COMMANDS or short.startswith(_OS_COMMAND_PREFIXES):
        hits.append(
            _hit("os_command", label, node, f"{short}() runs an OS command", f"{short}(...)")
        )
    elif short in _SUBPROCESS_CALLS:
        hits.extend(_check_shell_flag(node, short, label))
    elif short in _UNSAFE_DESERIALISERS:
        hits.append(
            _hit(
                "unsafe_deserialisation",
                label,
                node,
                f"{short}() can execute code while loading",
                f"{short}(...)",
            )
        )
    elif short in _YAML_ALWAYS_UNSAFE or (
        short in _YAML_LOAD and not _yaml_loader_is_safe(node, aliases)
    ):
        hits.append(
            _hit("unsafe_yaml", label, node, f"{short}() without a safe loader", f"{short}(...)")
        )
    elif short in _DYNAMIC_IMPORTS and not _is_str_literal(first):
        hits.append(
            _hit(
                "dynamic_import",
                label,
                node,
                f"{short}() with a non-literal module name",
                f"{short}(<non-literal>)",
            )
        )
    elif short == "ssl._create_unverified_context":
        hits.append(
            _hit(
                "tls_verify_disabled",
                label,
                node,
                "unverified TLS context",
                "ssl._create_unverified_context()",
            )
        )
    elif short == "getattr" and _is_dynamic_getattr_on_dangerous(node, aliases):
        hits.append(
            _hit(
                "dynamic_getattr",
                label,
                node,
                "getattr() with a computed name on a dangerous module",
                "getattr(<module>, <non-literal>)",
            )
        )
    return hits


def _check_keywords_only(node: ast.Call, label: str) -> list[Hit]:
    """``verify=False`` is dangerous on *any* callee (requests, httpx, aiohttp, ...)."""
    for kw in node.keywords:
        if (
            kw.arg == "verify"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value in (False, 0, None)
        ):
            return [
                _hit(
                    "tls_verify_disabled",
                    label,
                    node,
                    "TLS certificate verification disabled (verify=False)",
                    "verify=False",
                )
            ]
    return []


def _check_shell_flag(node: ast.Call, short: str, label: str) -> list[Hit]:
    for kw in node.keywords:
        if kw.arg != "shell":
            continue
        value = kw.value
        if isinstance(value, ast.Constant) and not value.value:
            return []
        shown = "shell=True" if isinstance(value, ast.Constant) else "shell=<non-literal>"
        return [_hit("shell_true", label, node, f"{short}() with {shown}", f"{short}({shown})")]
    return []


def _yaml_loader_is_safe(node: ast.Call, aliases: dict[str, str]) -> bool:
    candidates = [kw.value for kw in node.keywords if kw.arg == "Loader"]
    if len(node.args) >= 2:
        candidates.append(node.args[1])
    for cand in candidates:
        name = dotted_name(cand, aliases)
        if name is not None and name.rsplit(".", 1)[-1] in _YAML_SAFE_LOADERS:
            return True
    return False


def _is_dynamic_getattr_on_dangerous(node: ast.Call, aliases: dict[str, str]) -> bool:
    if len(node.args) < 2 or _is_str_literal(node.args[1]):
        return False
    target = dotted_name(node.args[0], aliases)
    return target is not None and target.split(".", 1)[0] in _DANGEROUS_MODULES


# ----------------------------------------------------------------------- SEC-004 SQL construction

_SQL_SINKS = frozenset({"execute", "executemany", "executescript"})
_SQL_KEYWORDS = frozenset({"sql", "query", "statement"})
_SQL_SHAPE = re.compile(
    r"^\s*(?:select\b.{1,1500}?\bfrom\b|insert\s+(?:or\s+\w+\s+)?into\b|update\b.{1,1500}?\bset\b"
    r"|delete\s+from\b|drop\s+(?:table|index|view)\b|alter\s+table\b|create\s+(?:table|index|view)\b"
    r"|replace\s+into\b|with\b.{1,1500}?\bselect\b)",
    re.IGNORECASE | re.DOTALL,
)


def sql_construction(tree: ast.AST, label: str) -> list[Hit]:
    """SEC-004: SQL assembled by f-string / ``%`` / ``.format`` / ``+``, passed to ``execute*`` or
    SQL-shaped anywhere.
    """
    nodes = list(iter_scoped(tree))
    tainted = _tainted_names(nodes)
    by_line: dict[int, Hit] = {}
    inner_binops: set[int] = set()
    sink_args: set[int] = set()
    for node, scope in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SQL_SINKS
        ):
            arg = _record_sink(node, scope, tainted, label, by_line)
            if arg is not None:
                sink_args.add(id(arg))
        elif (
            isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Call))
            and id(node) not in inner_binops | sink_args
        ):
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.BinOp):
                inner_binops.add(id(node.left))
            kind, constants = _dynamic_string(node)
            if kind and _SQL_SHAPE.match(constants):
                by_line.setdefault(
                    _line(node),
                    _hit(
                        "sql_string_built",
                        label,
                        node,
                        f"SQL statement assembled by {kind}",
                        f"{kind} SQL string",
                    ),
                )
    return [by_line[line] for line in sorted(by_line)]


def _record_sink(
    node: ast.Call, scope: int, tainted: set[tuple[int, str]], label: str, out: dict[int, Hit]
) -> ast.AST | None:
    """Record a hit if the SQL argument of an ``execute*`` call is dynamic; return it."""
    assert isinstance(node.func, ast.Attribute)
    arg = (
        node.args[0]
        if node.args
        else next((kw.value for kw in node.keywords if kw.arg in _SQL_KEYWORDS), None)
    )
    if arg is None:
        return None
    kind, _ = _dynamic_string(arg)
    if (
        kind is None
        and isinstance(arg, ast.Name)
        and ((scope, arg.id) in tainted or (0, arg.id) in tainted)
    ):
        kind = "a dynamically built variable"
    if kind:
        out[_line(node)] = _hit(
            "sql_injection",
            label,
            node,
            f"SQL passed to .{node.func.attr}() is assembled by {kind}",
            f".{node.func.attr}(<{kind}>)",
        )
    return arg


def _tainted_names(nodes: list[tuple[ast.AST, int]]) -> set[tuple[int, str]]:
    """Names (per scope) assigned a dynamically built string - one flow-insensitive pass."""
    tainted: set[tuple[int, str]] = set()
    for node, scope in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        kind, _ = _dynamic_string(value)
        augmented_dynamic = (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and not _is_str_literal(value)
        )
        if kind or augmented_dynamic:
            tainted.update((scope, t.id) for t in targets if isinstance(t, ast.Name))
    return tainted


def _dynamic_string(node: ast.AST) -> tuple[str | None, str]:
    """Return ``(kind, constant_text)``; ``kind`` is ``None`` for a static string or a non-string
    expression.
    """
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        dynamic = any(isinstance(v, ast.FormattedValue) for v in node.values)
        return ("an f-string" if dynamic else None), "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        left = node.left
        if (
            isinstance(left, ast.Constant)
            and isinstance(left.value, str)
            and not _all_constants(node.right)
        ):
            return "%-formatting", left.value
        return None, ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concatenation(node)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        base = node.func.value
        if (
            isinstance(base, ast.Constant)
            and isinstance(base.value, str)
            and (node.args or node.keywords)
        ):
            return ".format()", base.value
    return None, ""


def _all_constants(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.Tuple) and all(isinstance(e, ast.Constant) for e in node.elts)


def _concatenation(node: ast.BinOp) -> tuple[str | None, str]:
    """Flatten an ``a + b + c`` chain iteratively; dynamic iff it mixes string literals with
    non-constants.
    """
    leaves: list[ast.AST] = []
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
            stack.extend((cur.right, cur.left))
        else:
            leaves.append(cur)
    texts = [
        leaf.value
        for leaf in leaves
        if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
    ]
    has_dynamic = any(not isinstance(leaf, ast.Constant) for leaf in leaves)
    stringish = bool(texts) or any(isinstance(leaf, ast.JoinedStr) for leaf in leaves)
    return ("concatenation" if has_dynamic and stringish else None), "".join(texts)


# ----------------------------------------------------------- SEC-008 egress / execution libraries

EGRESS_MODULES = (
    "subprocess",
    "socket",
    "ctypes",
    "importlib",
    "urllib.request",
    "http.client",
    "httpx",
    "requests",
    "aiohttp",
)
_EGRESS_CALLS = frozenset({"asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell"})


def _egress_match(module: str) -> str | None:
    for banned in EGRESS_MODULES:
        if module == banned or module.startswith(banned + "."):
            return banned
    return None


def egress_uses(tree: ast.AST, label: str) -> list[Hit]:
    """SEC-008: every import / literal dynamic import of an egress or execution library."""
    aliases = import_aliases(tree)
    hits: list[Hit] = []
    for node in ast.walk(tree):
        banned = _egress_target(node, aliases)
        if banned:
            hits.append(
                _hit(
                    "egress_module",
                    label,
                    node,
                    f"use of egress/execution library {banned!r}",
                    banned,
                )
            )
    return hits


def _egress_target(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Import):
        modules = [item.name for item in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        modules = [node.module, *(f"{node.module}.{item.name}" for item in node.names)]
    elif isinstance(node, ast.Call):
        callee = dotted_name(node.func, aliases)
        if callee in _EGRESS_CALLS:
            return callee
        modules = _dynamic_import_target(node, aliases)
    else:
        return None
    return next((m for m in map(_egress_match, modules) if m), None)


def _dynamic_import_target(node: ast.Call, aliases: dict[str, str]) -> list[str]:
    callee = dotted_name(node.func, aliases)
    first = node.args[0] if node.args else None
    if (
        callee in _DYNAMIC_IMPORTS
        and isinstance(first, ast.Constant)
        and isinstance(first.value, str)
    ):
        return [first.value]
    return []


# ------------------------------------------------------------------ SEC-007 non-ASCII identifiers

_IDENTIFIER_ATTRS = ("id", "arg", "name", "asname", "attr", "module", "rest")


def non_ascii_identifiers(tree: ast.AST, label: str) -> list[Hit]:
    """Every identifier containing a non-ASCII character (homoglyph / Trojan-Source vector)."""
    hits: list[Hit] = []
    for node in ast.walk(tree):
        names: list[str] = [
            v for a in _IDENTIFIER_ATTRS if isinstance(v := getattr(node, a, None), str)
        ]
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.extend(node.names)
        for name in names:
            if not name.isascii():
                hits.append(
                    Hit(
                        "non_ascii_identifier",
                        format_location(label, _line(node)),
                        "identifier contains non-ASCII characters (possible homoglyph)",
                        ascii(name)[:60],
                    )
                )
    return hits
