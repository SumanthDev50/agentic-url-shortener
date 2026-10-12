"""FastAPI route extraction (shared by SEC-003, CHG-004 and the change-set risk assessment).

Static and heuristic by design: a route is a function decorated with
``<obj>.get|post|put|delete|patch| head|options|api_route(path, ...)`` or registered with
``<obj>.add_api_route(path, handler, ...)``. We resolve the router ``prefix`` and router-/app-level
``dependencies`` for objects created in the same file (``APIRouter(...)`` / ``FastAPI(...)`` /
``include_router(...)``), and one level of dependency aliasing (``AuthDep =
Depends(require_api_key)``).  Anything more dynamic (routers built in factories, dependencies
appended at run time) is invisible here - documented as an xfail in the adversarial corpus, and
covered structurally by human approval of the release.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from orchestrator.policies.hits import Hit
from orchestrator.policies.pyscan import parse_python

DYNAMIC_PATH = "<dynamic>"
_HTTP_METHODS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})
_DEPENDENCY_CALLS = frozenset({"Depends", "Security"})
_ROUTER_FACTORIES = frozenset({"APIRouter", "FastAPI"})


@dataclass(frozen=True)
class Route:
    """One HTTP route. ``signature`` changes whenever the decorator, parameters or return annotation
    do.
    """

    method: str
    path: str
    handler: str
    line: int
    authenticated: bool
    signature: str

    @property
    def key(self) -> str:
        """``"GET /healthz"`` - the form used by ``PolicyConfig.public_routes``."""
        return f"{self.method} {self.path}"


@dataclass
class _Router:
    prefix: str = ""
    authenticated: bool = False


def routes_of_text(
    text: str, label: str, auth_names: frozenset[str]
) -> tuple[list[Route] | None, Hit | None]:
    """Parse ``text`` and extract its routes; ``(None, hit)`` when the file cannot be analysed."""
    tree, problem = parse_python(text, label)
    if tree is None:
        return None, problem
    return extract_routes(tree, auth_names), None


def extract_routes(tree: ast.AST, auth_names: frozenset[str]) -> list[Route]:
    nodes = list(ast.walk(tree))
    aliases = _dependency_aliases(nodes, auth_names)
    routers = _routers(nodes, auth_names, aliases)
    functions = {n.name: n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    routes: list[Route] = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    routes.extend(_decorated_routes(node, decorator, routers, auth_names, aliases))
        elif isinstance(node, ast.Call) and _call_attr(node) in ("add_api_route", "add_route"):
            routes.extend(_registered_routes(node, functions, routers, auth_names, aliases))
    return sorted(routes, key=lambda r: (r.line, r.method, r.path))


# --------------------------------------------------------------------------- dependency detection


def _call_attr(node: ast.Call) -> str:
    return node.func.attr if isinstance(node.func, ast.Attribute) else ""


def _last_segment(node: ast.AST) -> str:
    """Last identifier of ``name``, ``a.b.name`` or ``name(...)``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _contains_auth(
    expr: ast.AST | None, auth_names: frozenset[str], aliases: frozenset[str]
) -> bool:
    """True if the expression subtree holds ``Depends(<auth>)`` / ``Security(<auth>)`` or an auth
    alias.
    """
    if expr is None:
        return False
    stack: list[ast.AST] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Call) and _last_segment(node) in _DEPENDENCY_CALLS:
            target = (
                node.args[0]
                if node.args
                else next((k.value for k in node.keywords if k.arg == "dependency"), None)
            )
            if target is not None and _last_segment(target) in auth_names:
                return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _dependency_aliases(nodes: list[ast.AST], auth_names: frozenset[str]) -> frozenset[str]:
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Assign) and _contains_auth(node.value, auth_names, frozenset()):
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return frozenset(found)


def _routers(
    nodes: list[ast.AST], auth_names: frozenset[str], aliases: frozenset[str]
) -> dict[str, _Router]:
    routers: dict[str, _Router] = {}
    for node in nodes:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _last_segment(node.value) in _ROUTER_FACTORIES
        ):
            info = _Router(
                _literal_kw(node.value, "prefix"), _kw_auth(node.value, auth_names, aliases)
            )
            for target in node.targets:
                if isinstance(target, ast.Name):
                    routers[target.id] = info
    for node in nodes:
        if isinstance(node, ast.Call) and _call_attr(node) == "include_router" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id in routers:
                inner = routers[first.id]
                routers[first.id] = _Router(
                    _literal_kw(node, "prefix") + inner.prefix,
                    inner.authenticated or _kw_auth(node, auth_names, aliases),
                )
    return routers


def _literal_kw(call: ast.Call, name: str) -> str:
    for kw in call.keywords:
        if (
            kw.arg == name
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return ""


def _kw_auth(call: ast.Call, auth_names: frozenset[str], aliases: frozenset[str]) -> bool:
    return any(
        kw.arg == "dependencies" and _contains_auth(kw.value, auth_names, aliases)
        for kw in call.keywords
    )


# ----------------------------------------------------------------------------- route construction


def _join(prefix: str, path: str) -> str:
    if path == DYNAMIC_PATH:
        return path
    return (prefix.rstrip("/") + path) if path.startswith("/") else f"{prefix.rstrip('/')}/{path}"


def _path_arg(call: ast.Call, index: int = 0) -> str:
    arg = (
        call.args[index]
        if len(call.args) > index
        else next((k.value for k in call.keywords if k.arg == "path"), None)
    )
    return (
        arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else DYNAMIC_PATH
    )


def _methods(call: ast.Call, attr: str) -> list[str]:
    if attr in _HTTP_METHODS:
        return [attr.upper()]
    for kw in call.keywords:
        if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)):
            values = [
                e.value.upper()
                for e in kw.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            if values and len(values) == len(kw.value.elts):
                return sorted(values)
    return ["ANY"]


def _handler_auth(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    auth_names: frozenset[str],
    aliases: frozenset[str],
) -> bool:
    args = func.args
    exprs: list[ast.AST | None] = [*args.defaults, *args.kw_defaults]
    exprs.extend(
        a.annotation
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
        if a
    )
    return any(_contains_auth(e, auth_names, aliases) for e in exprs)


def _signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[ast.AST] = [*func.decorator_list, func.args]
    if func.returns is not None:
        parts.append(func.returns)
    try:
        dump = "|".join(ast.dump(part) for part in parts)
    except RecursionError:
        return "unhashable"
    return hashlib.sha256(dump.encode("utf-8", "replace")).hexdigest()[:16]


def _decorated_routes(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    decorator: ast.Call,
    routers: dict[str, _Router],
    auth_names: frozenset[str],
    aliases: frozenset[str],
) -> list[Route]:
    attr = _call_attr(decorator)
    if attr not in _HTTP_METHODS and attr != "api_route":
        return []
    receiver = decorator.func.value if isinstance(decorator.func, ast.Attribute) else None
    router = routers.get(receiver.id, _Router()) if isinstance(receiver, ast.Name) else _Router()
    authenticated = (
        router.authenticated
        or _kw_auth(decorator, auth_names, aliases)
        or _handler_auth(func, auth_names, aliases)
    )
    path = _join(router.prefix, _path_arg(decorator))
    signature = _signature(func)
    return [
        Route(m, path, func.name, func.lineno, authenticated, signature)
        for m in _methods(decorator, attr)
    ]


def _registered_routes(
    call: ast.Call,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    routers: dict[str, _Router],
    auth_names: frozenset[str],
    aliases: frozenset[str],
) -> list[Route]:
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    router = routers.get(receiver.id, _Router()) if isinstance(receiver, ast.Name) else _Router()
    handler_node = (
        call.args[1]
        if len(call.args) > 1
        else next((k.value for k in call.keywords if k.arg == "endpoint"), None)
    )
    handler = functions.get(handler_node.id) if isinstance(handler_node, ast.Name) else None
    authenticated = router.authenticated or _kw_auth(call, auth_names, aliases)
    if handler is not None:
        authenticated = authenticated or _handler_auth(handler, auth_names, aliases)
    name = (
        handler.name
        if handler is not None
        else (_last_segment(handler_node) if handler_node else "<unknown>")
    )
    signature = _signature(handler) if handler is not None else "unresolved"
    return [
        Route(m, _join(router.prefix, _path_arg(call)), name, call.lineno, authenticated, signature)
        for m in _methods(call, "api_route")
    ]
