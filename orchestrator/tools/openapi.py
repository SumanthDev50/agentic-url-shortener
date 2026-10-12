"""OpenAPI compatibility tooling: diff two specs, export a spec from a sandboxed app.

``diff_openapi(old, new)`` classifies every difference as *breaking* (an existing client can
fail) or *additive* (an existing client is unaffected). It feeds the ``api_compat`` gate, which
compares the actual delta with the one the architect declared in the ``api_contract``
artifact.

Classification (the table the tests pin down)
---------------------------------------------
============================================  ========  ========
change                                        request   response
============================================  ========  ========
path / operation / response code removed      breaking  breaking
path / operation / response code added        additive  additive
parameter or body field removed               breaking  breaking
new *required* parameter / body field         breaking  additive
new optional parameter / field                additive  additive
optional -> required                          breaking  additive
required -> optional                          additive  breaking
type or format change                         breaking  breaking
enum value removed (narrowing)                breaking  breaking
enum value added                              additive  additive
bound tightened (min/max/pattern)             breaking  (ignored)
authentication newly required                 breaking  --
============================================  ========  ========

Enum narrowing is reported as breaking in *both* directions on purpose: the conservative reading is
the safe default for a gate. ``$ref`` is resolved inside each document (``#/components/...``);
``allOf`` is merged; ``anyOf``/``oneOf``/``nullable``/type lists are compared as sets of
alternatives so the 3.0 and 3.1 spellings of "optional" do not differ. Path parameters are
matched by *position* in the template, so renaming ``{code}`` to ``{id}`` is not a change.

Safety: both documents are untrusted (they may come from a generated or agent-edited artifact). The
walk is iterative, bounded in depth and in the number of comparisons, cycle-safe for recursive
models, and **fails closed**: a schema too deep or too large to compare is reported as breaking.
Names copied into messages pass through ``sanitize``.

``export_openapi`` runs ``orchestrator/tools/_openapi_dump.py`` through any object with the
sandbox's ``run`` signature (see :class:`Runner`) and parses its output with ``loads_strict``.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from orchestrator.core.errors import OrchestratorError
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import loads_strict
from orchestrator.core.types import CommandResult
from orchestrator.tools._openapi_dump import APP_IMPORT_RE

HELPER_MODULE = "orchestrator.tools._openapi_dump"
DEFAULT_EXPORT_TIMEOUT_S = 60.0
MAX_EXPORT_TIMEOUT_S = 600.0
MAX_EXPORT_BYTES = 8 * 1024 * 1024

MAX_SCHEMA_DEPTH = 32
MAX_COMPARISONS = 50_000
MAX_ENTRIES = 1000
_MAX_REF_HOPS = 16
_MAX_ALLOF_PARTS = 64
_MAX_VARIANTS = 64
_MAX_NAME_CHARS = 120

_REQUEST = "request"
_RESPONSE = "response"
_OLD = "old"
_NEW = "new"
_HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_PATH_PARAM_RE = re.compile(r"\{([^{}/]*)\}")
_LOWER_BOUNDS = ("minimum", "exclusiveMinimum", "minLength", "minItems", "minProperties")
_UPPER_BOUNDS = ("maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties")


# ----------------------------------------------------------------------------- public types


@dataclass(frozen=True)
class OpenApiDiff:
    """Result of :func:`diff_openapi`: human-readable, sorted, de-duplicated change descriptions."""

    breaking: list[str] = field(default_factory=list)
    additive: list[str] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking)

    def to_dict(self) -> dict[str, list[str]]:
        return {"breaking": list(self.breaking), "additive": list(self.additive)}


class OpenApiExportError(OrchestratorError):
    """The sandboxed export failed (timeout, non-zero exit, unusable output)."""


class Runner(Protocol):
    """Anything shaped like ``SandboxRunner.run`` (kept local so this module does not import it)."""

    def run(self, argv: Sequence[str], *, cwd: Path, timeout_s: float) -> CommandResult: ...


# ----------------------------------------------------------------------------- small helpers


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _name(value: object) -> str:
    return sanitize_for_terminal(value, _MAX_NAME_CHARS)


def _template_key(path: str) -> str:
    """Path with parameter names erased: ``/a/{code}`` and ``/a/{id}`` are the same route."""
    return _PATH_PARAM_RE.sub("{}", path)


def _path_param_names(path: str) -> list[str]:
    return _PATH_PARAM_RE.findall(path)


def _media_key(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _pointer(root: Mapping[str, Any], ref: str) -> Any:
    """Resolve a local JSON pointer (``#/components/schemas/X``); ``None`` when absent."""
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for raw in ref[2:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _signature(schema: Mapping[str, Any]) -> str:
    """Structure-only label of an alternative (never a ``$ref`` name: inlining is no change)."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "any"


def _security_alternatives(security: Any) -> set[frozenset[str]]:
    """Each requirement object is one alternative; no requirement means public."""
    if not isinstance(security, list) or not security:
        return {frozenset()}
    return {
        frozenset(str(k) for k in req) if isinstance(req, Mapping) else frozenset()
        for req in security
    }


def _fmt_set(values: set[str] | frozenset[str]) -> str:
    return "[" + ", ".join(_name(v) for v in sorted(values)) + "]"


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ----------------------------------------------------------------------------- the differ


class _Differ:
    """One comparison run; collects messages in sets so repeats collapse deterministically."""

    def __init__(self, old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
        self._docs = {_OLD: old, _NEW: new}
        self._budget = MAX_COMPARISONS
        self.breaking: set[str] = set()
        self.additive: set[str] = set()

    # -- paths and operations ---------------------------------------------------
    def run(self) -> OpenApiDiff:
        old_paths = self._paths(_OLD)
        new_paths = self._paths(_NEW)
        old_by_tpl = {_template_key(p): p for p in sorted(old_paths, reverse=True)}
        new_by_tpl = {_template_key(p): p for p in sorted(new_paths, reverse=True)}
        for key in sorted(old_by_tpl.keys() | new_by_tpl.keys()):
            old_path, new_path = old_by_tpl.get(key), new_by_tpl.get(key)
            if new_path is None and old_path is not None:
                self.breaking.add(f"removed path {_name(old_path)}")
            elif old_path is None and new_path is not None:
                self.additive.add(f"new path {_name(new_path)}")
            elif old_path is not None and new_path is not None:
                self._compare_path(old_path, old_paths[old_path], new_path, new_paths[new_path])
        return OpenApiDiff(_finalize(self.breaking), _finalize(self.additive))

    def _paths(self, side: str) -> dict[str, Mapping[str, Any]]:
        paths = _mapping(self._docs[side].get("paths"))
        return {p: _mapping(item) for p, item in paths.items() if isinstance(p, str)}

    def _compare_path(
        self, old_path: str, old_item: Mapping[str, Any], new_path: str, new_item: Mapping[str, Any]
    ) -> None:
        renames = dict(zip(_path_param_names(new_path), _path_param_names(old_path), strict=False))
        for method in _HTTP_METHODS:
            old_op, new_op = old_item.get(method), new_item.get(method)
            label = f"{method.upper()} {_name(new_path if new_op is not None else old_path)}"
            if isinstance(old_op, Mapping) and not isinstance(new_op, Mapping):
                self.breaking.add(f"removed operation {label}")
            elif isinstance(new_op, Mapping) and not isinstance(old_op, Mapping):
                self.additive.add(f"new operation {label}")
            elif isinstance(old_op, Mapping) and isinstance(new_op, Mapping):
                self._compare_operation(label, (old_item, old_op), (new_item, new_op), renames)

    def _compare_operation(
        self,
        label: str,
        old: tuple[Mapping[str, Any], Mapping[str, Any]],
        new: tuple[Mapping[str, Any], Mapping[str, Any]],
        renames: Mapping[str, str],
    ) -> None:
        self._compare_parameters(label, old, new, renames)
        self._compare_request_body(label, old[1], new[1])
        self._compare_responses(label, old[1], new[1])
        self._compare_security(label, old[1], new[1])

    # -- parameters -------------------------------------------------------------
    def _collect_parameters(
        self,
        side: str,
        item_and_op: tuple[Mapping[str, Any], Mapping[str, Any]],
        renames: Mapping[str, str],
    ) -> dict[tuple[str, str], Mapping[str, Any]]:
        out: dict[tuple[str, str], Mapping[str, Any]] = {}
        for source in item_and_op:  # path-level first; operation-level overrides
            declared = source.get("parameters")
            for raw in declared if isinstance(declared, list) else []:
                param, _ = self._resolve(raw, side)
                name, where = param.get("name"), param.get("in")
                if not isinstance(name, str) or not isinstance(where, str):
                    continue
                if where == "path" and side == _NEW:
                    name = renames.get(name, name)
                out[(where, name)] = param
        return out

    def _compare_parameters(
        self,
        label: str,
        old: tuple[Mapping[str, Any], Mapping[str, Any]],
        new: tuple[Mapping[str, Any], Mapping[str, Any]],
        renames: Mapping[str, str],
    ) -> None:
        old_params = self._collect_parameters(_OLD, old, {})
        new_params = self._collect_parameters(_NEW, new, renames)
        for key in sorted(old_params.keys() | new_params.keys()):
            where, name = key
            text = f"parameter '{_name(name)}' ({_name(where)})"
            old_param, new_param = old_params.get(key), new_params.get(key)
            if new_param is None:
                self.breaking.add(f"{label}: removed {text}")
            elif old_param is None:
                if new_param.get("required") is True:
                    self.breaking.add(f"{label}: new required {text}")
                else:
                    self.additive.add(f"{label}: new optional {text}")
            else:
                was, now = old_param.get("required") is True, new_param.get("required") is True
                if now and not was:
                    self.breaking.add(f"{label}: {text} became required")
                elif was and not now:
                    self.additive.add(f"{label}: {text} is now optional")
                old_schema, new_schema = old_param.get("schema"), new_param.get("schema")
                if old_schema is not None and new_schema is not None:
                    self._compare_schema(old_schema, new_schema, f"{label} {text}", _REQUEST)

    # -- bodies and responses ---------------------------------------------------
    def _compare_request_body(
        self, label: str, old_op: Mapping[str, Any], new_op: Mapping[str, Any]
    ) -> None:
        old_body = self._optional_object(old_op.get("requestBody"), _OLD)
        new_body = self._optional_object(new_op.get("requestBody"), _NEW)
        if old_body is None and new_body is None:
            return
        if new_body is None:
            self.breaking.add(f"{label}: removed request body")
        elif old_body is None:
            if new_body.get("required") is True:
                self.breaking.add(f"{label}: new required request body")
            else:
                self.additive.add(f"{label}: new optional request body")
        else:
            was, now = old_body.get("required") is True, new_body.get("required") is True
            if now and not was:
                self.breaking.add(f"{label}: request body became required")
            elif was and not now:
                self.additive.add(f"{label}: request body is now optional")
            self._compare_content(f"{label} request body", old_body, new_body, _REQUEST)

    def _compare_responses(
        self, label: str, old_op: Mapping[str, Any], new_op: Mapping[str, Any]
    ) -> None:
        old_responses = {str(c): r for c, r in _mapping(old_op.get("responses")).items()}
        new_responses = {str(c): r for c, r in _mapping(new_op.get("responses")).items()}
        for code in sorted(old_responses.keys() | new_responses.keys()):
            if code not in new_responses:
                self.breaking.add(f"{label}: removed response {_name(code)}")
            elif code not in old_responses:
                self.additive.add(f"{label}: new response {_name(code)}")
            else:
                old_resp, _ = self._resolve(old_responses[code], _OLD)
                new_resp, _ = self._resolve(new_responses[code], _NEW)
                self._compare_content(
                    f"{label} response {_name(code)}", old_resp, new_resp, _RESPONSE
                )

    def _compare_content(
        self,
        label: str,
        old_holder: Mapping[str, Any],
        new_holder: Mapping[str, Any],
        direction: str,
    ) -> None:
        old_content = {
            _media_key(k): v
            for k, v in _mapping(old_holder.get("content")).items()
            if isinstance(k, str)
        }
        new_content = {
            _media_key(k): v
            for k, v in _mapping(new_holder.get("content")).items()
            if isinstance(k, str)
        }
        for media in sorted(old_content.keys() | new_content.keys()):
            if media not in new_content:
                self.breaking.add(f"{label}: removed media type '{_name(media)}'")
            elif media not in old_content:
                self.additive.add(f"{label}: new media type '{_name(media)}'")
            else:
                old_schema = _mapping(old_content[media]).get("schema")
                new_schema = _mapping(new_content[media]).get("schema")
                if old_schema is not None and new_schema is not None:
                    self._compare_schema(
                        old_schema, new_schema, f"{label} ({_name(media)}) $", direction
                    )

    def _optional_object(self, node: Any, side: str) -> Mapping[str, Any] | None:
        resolved, _ = self._resolve(node, side)
        return resolved if resolved else None

    # -- security ---------------------------------------------------------------
    def _effective_security(self, side: str, op: Mapping[str, Any]) -> set[frozenset[str]]:
        security = op["security"] if "security" in op else self._docs[side].get("security")
        return _security_alternatives(security)

    def _compare_security(
        self, label: str, old_op: Mapping[str, Any], new_op: Mapping[str, Any]
    ) -> None:
        old_alts, new_alts = (
            self._effective_security(_OLD, old_op),
            self._effective_security(_NEW, new_op),
        )
        if old_alts == new_alts:
            return
        public = frozenset[str]()
        if public in old_alts and public not in new_alts:
            names = {n for alt in new_alts for n in alt}
            self.breaking.add(f"{label}: authentication is now required {_fmt_set(names)}")
        elif public in new_alts and public not in old_alts:
            self.additive.add(f"{label}: authentication is no longer required")
        else:
            for alt in sorted(old_alts - new_alts, key=sorted):
                self.breaking.add(f"{label}: removed security alternative {_fmt_set(alt)}")
            for alt in sorted(new_alts - old_alts, key=sorted):
                self.additive.add(f"{label}: new security alternative {_fmt_set(alt)}")

    # -- reference resolution and schema normalisation --------------------------
    def _resolve(self, node: Any, side: str) -> tuple[Mapping[str, Any], str | None]:
        """Follow ``$ref`` hops; returns the target and the first reference (``None`` if inline)."""
        first: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_REF_HOPS):
            if not isinstance(node, Mapping):
                return {}, first
            ref = node.get("$ref")
            if not isinstance(ref, str):
                return node, first
            first = first or ref
            if ref in seen:
                return {}, first
            seen.add(ref)
            target = _pointer(self._docs[side], ref)
            if target is None:  # external or dangling: opaque, equal only to the same reference
                return {"x-ref": ref}, first
            node = target
        return {}, first

    def _effective(self, node: Any, side: str) -> tuple[dict[str, Any], str | None]:
        """Resolve ``$ref`` and merge ``allOf`` (members first, the schema's own keys win)."""
        schema, first = self._resolve(node, side)
        parts: list[Mapping[str, Any]] = [schema]
        index = 0
        while index < len(parts) and len(parts) < _MAX_ALLOF_PARTS:
            members = parts[index].get("allOf")
            index += 1
            for member in members if isinstance(members, list) else []:
                parts.append(self._resolve(member, side)[0])
        merged: dict[str, Any] = {}
        for part in reversed(parts):
            for key, value in part.items():
                if key == "allOf":
                    continue
                if key == "properties" and isinstance(value, Mapping):
                    merged.setdefault("properties", {}).update(value)
                elif key == "required" and isinstance(value, list):
                    merged.setdefault("required", []).extend(value)
                else:
                    merged[key] = value
        return merged, first

    def _variants(self, schema: dict[str, Any], side: str) -> list[tuple[str, Any]] | None:
        """Alternatives of a schema, or ``None`` when it is a single, non-nullable shape."""
        raw_members: list[Any] = []
        for key in ("anyOf", "oneOf"):
            declared = schema.get(key)
            raw_members.extend(declared if isinstance(declared, list) else [])
        declared_type = schema.get("type")
        nullable = schema.get("nullable") is True
        if not raw_members and not isinstance(declared_type, list) and not nullable:
            return None
        variants: list[tuple[str, Any]] = []
        if isinstance(declared_type, list):
            rest = {
                k: v for k, v in schema.items() if k not in ("type", "anyOf", "oneOf", "nullable")
            }
            variants.extend((t, {**rest, "type": t}) for t in declared_type if isinstance(t, str))
        elif not raw_members:
            variants.append(
                (_signature(schema), {k: v for k, v in schema.items() if k != "nullable"})
            )
        pending = list(reversed(raw_members))
        while pending and len(variants) < _MAX_VARIANTS:
            member = pending.pop()
            effective, _ = self._effective(member, side)
            nested = [m for key in ("anyOf", "oneOf") for m in _as_list(effective.get(key))]
            if nested:
                pending.extend(reversed(nested))
            else:
                variants.append((_signature(effective), member))
        if nullable and all(sig != "null" for sig, _ in variants):
            variants.append(("null", {"type": "null"}))
        return variants

    # -- schema comparison (iterative, bounded, cycle-safe) ---------------------
    def _compare_schema(self, old: Any, new: Any, where: str, direction: str) -> None:
        stack: list[tuple[Any, Any, str, int]] = [(old, new, where, 0)]
        seen: set[tuple[str, str]] = set()
        while stack:
            old_node, new_node, loc, depth = stack.pop()
            self._budget -= 1
            if self._budget < 0 or depth > MAX_SCHEMA_DEPTH:
                self.breaking.add(
                    f"{loc}: schema too large or deeply nested to compare (treated as breaking)"
                )
                if self._budget < 0:
                    return
                continue
            old_eff, old_ref = self._effective(old_node, _OLD)
            new_eff, new_ref = self._effective(new_node, _NEW)
            if old_ref is not None and new_ref is not None:
                if (
                    old_ref,
                    new_ref,
                ) in seen:  # recursive model: this pair is already being compared
                    continue
                seen.add((old_ref, new_ref))
            old_vars, new_vars = self._variants(old_eff, _OLD), self._variants(new_eff, _NEW)
            if old_vars is not None or new_vars is not None:
                self._compare_variants(
                    old_vars or [(_signature(old_eff), old_node)],
                    new_vars or [(_signature(new_eff), new_node)],
                    loc,
                    depth,
                    stack,
                )
            else:
                self._compare_shape(old_eff, new_eff, loc, direction, depth, stack)

    def _compare_variants(
        self,
        old_vars: list[tuple[str, Any]],
        new_vars: list[tuple[str, Any]],
        loc: str,
        depth: int,
        stack: list[tuple[Any, Any, str, int]],
    ) -> None:
        old_by: dict[str, list[Any]] = {}
        new_by: dict[str, list[Any]] = {}
        for sig, node in old_vars:
            old_by.setdefault(sig, []).append(node)
        for sig, node in new_vars:
            new_by.setdefault(sig, []).append(node)
        for sig in sorted(old_by.keys() | new_by.keys()):
            olds, news = old_by.get(sig, []), new_by.get(sig, [])
            for i in range(max(len(olds), len(news))):
                if i >= len(news):
                    self.breaking.add(f"{loc}: removed alternative '{_name(sig)}'")
                elif i >= len(olds):
                    self.additive.add(f"{loc}: new alternative '{_name(sig)}'")
                else:
                    stack.append((olds[i], news[i], loc, depth + 1))

    def _compare_shape(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        loc: str,
        direction: str,
        depth: int,
        stack: list[tuple[Any, Any, str, int]],
    ) -> None:
        if "x-ref" in old or "x-ref" in new:
            if old.get("x-ref") != new.get("x-ref"):
                self.breaking.add(f"{loc}: schema reference changed")
            return
        old_type, new_type = old.get("type"), new.get("type")
        if isinstance(old_type, str) and isinstance(new_type, str) and old_type != new_type:
            self.breaking.add(
                f"{loc}: type changed from '{_name(old_type)}' to '{_name(new_type)}'"
            )
            return  # the structure below is no longer comparable
        self._compare_enum(old, new, loc)
        self._compare_format(old, new, loc)
        if direction == _REQUEST:
            self._compare_constraints(old, new, loc)
        self._compare_properties(old, new, loc, direction, depth, stack)
        for key, suffix in (("items", "[]"), ("additionalProperties", ".*")):
            old_child, new_child = old.get(key), new.get(key)
            if isinstance(old_child, Mapping) and isinstance(new_child, Mapping):
                stack.append((old_child, new_child, loc + suffix, depth + 1))

    def _compare_enum(self, old: Mapping[str, Any], new: Mapping[str, Any], loc: str) -> None:
        old_enum, new_enum = old.get("enum"), new.get("enum")
        if isinstance(old_enum, list) and isinstance(new_enum, list):
            old_values, new_values = {_canon(v) for v in old_enum}, {_canon(v) for v in new_enum}
            if old_values - new_values:
                self.breaking.add(
                    f"{loc}: enum narrowed, removed values {_fmt_set(old_values - new_values)}"
                )
            if new_values - old_values:
                self.additive.add(
                    f"{loc}: enum widened, added values {_fmt_set(new_values - old_values)}"
                )
        elif isinstance(new_enum, list):
            self.breaking.add(f"{loc}: enum restriction added")
        elif isinstance(old_enum, list):
            self.additive.add(f"{loc}: enum restriction removed")

    def _compare_format(self, old: Mapping[str, Any], new: Mapping[str, Any], loc: str) -> None:
        old_fmt, new_fmt = old.get("format"), new.get("format")
        if isinstance(new_fmt, str) and new_fmt != old_fmt:
            before = _name(old_fmt) if isinstance(old_fmt, str) else "none"
            self.breaking.add(f"{loc}: format changed from '{before}' to '{_name(new_fmt)}'")
        elif isinstance(old_fmt, str) and new_fmt is None:
            self.additive.add(f"{loc}: format '{_name(old_fmt)}' removed")

    def _compare_constraints(
        self, old: Mapping[str, Any], new: Mapping[str, Any], loc: str
    ) -> None:
        for keys, tighter in (
            (_LOWER_BOUNDS, lambda o, n: n > o),
            (_UPPER_BOUNDS, lambda o, n: n < o),
        ):
            for key in keys:
                old_v, new_v = old.get(key), new.get(key)
                if old_v is None and new_v is None:
                    continue
                if _is_number(old_v) and _is_number(new_v):
                    if old_v == new_v:
                        continue
                    target = self.breaking if tighter(old_v, new_v) else self.additive
                    target.add(f"{loc}: {key} changed from {old_v} to {new_v}")
                elif _is_number(new_v):
                    self.breaking.add(f"{loc}: {key} constraint added ({new_v})")
                elif _is_number(old_v):
                    self.additive.add(f"{loc}: {key} constraint removed")
        old_pattern, new_pattern = old.get("pattern"), new.get("pattern")
        if isinstance(new_pattern, str) and new_pattern != old_pattern:
            self.breaking.add(f"{loc}: pattern constraint added or changed")
        elif isinstance(old_pattern, str) and new_pattern is None:
            self.additive.add(f"{loc}: pattern constraint removed")

    def _compare_properties(
        self,
        old: Mapping[str, Any],
        new: Mapping[str, Any],
        loc: str,
        direction: str,
        depth: int,
        stack: list[tuple[Any, Any, str, int]],
    ) -> None:
        old_props, new_props = _mapping(old.get("properties")), _mapping(new.get("properties"))
        old_req = {r for r in _as_list(old.get("required")) if isinstance(r, str)}
        new_req = {r for r in _as_list(new.get("required")) if isinstance(r, str)}
        for prop in sorted(p for p in old_props if isinstance(p, str)):
            text = f"field '{_name(prop)}'"
            if prop not in new_props:
                self.breaking.add(f"{loc}: removed {text}")
                continue
            stack.append((old_props[prop], new_props[prop], f"{loc}.{_name(prop)}", depth + 1))
            was, now = prop in old_req, prop in new_req
            if direction == _REQUEST and now and not was:
                self.breaking.add(f"{loc}: {text} became required")
            elif direction == _REQUEST and was and not now:
                self.additive.add(f"{loc}: {text} is now optional")
            elif direction == _RESPONSE and was and not now:
                self.breaking.add(f"{loc}: {text} is no longer always present")
            elif direction == _RESPONSE and now and not was:
                self.additive.add(f"{loc}: {text} is now always present")
        for prop in sorted(p for p in new_props if isinstance(p, str) and p not in old_props):
            text = f"field '{_name(prop)}'"
            if prop in new_req and direction == _REQUEST:
                self.breaking.add(f"{loc}: new required {text}")
            else:
                self.additive.add(
                    f"{loc}: new {'required ' if prop in new_req else 'optional '}{text}"
                )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finalize(messages: set[str]) -> list[str]:
    ordered = sorted(messages)
    if len(ordered) <= MAX_ENTRIES:
        return ordered
    return [*ordered[: MAX_ENTRIES - 1], f"... and {len(ordered) - MAX_ENTRIES + 1} more"]


def diff_openapi(old: Mapping[str, Any], new: Mapping[str, Any]) -> OpenApiDiff:
    """Classify the differences between two OpenAPI 3.x documents (see the module docstring).

    Raises ``ValueError`` for anything that is not an OpenAPI 3.x style object (a Swagger 2.0
    document has no ``components`` and would silently compare as "no change", so it is refused).
    """
    for label, doc in (("old", old), ("new", new)):
        if not isinstance(doc, Mapping):
            raise ValueError(f"{label} OpenAPI document must be an object")
        if "swagger" in doc and "openapi" not in doc:
            raise ValueError(f"{label} document is Swagger 2.0; only OpenAPI 3.x is supported")
    try:
        if old is new or old == new:
            return (
                OpenApiDiff()
            )  # nothing to classify (and no fail-closed false positive on deep schemas)
    except RecursionError:  # a pathologically deep document: fall through to the bounded walk
        pass
    return _Differ(old, new).run()


# ----------------------------------------------------------------------------- export


def export_openapi(
    runner: Runner,
    cwd: Path | str,
    app_import: str,
    *,
    timeout_s: float = DEFAULT_EXPORT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the dump helper through ``runner`` in ``cwd`` and return the app's OpenAPI document.

    ``app_import`` is ``"package.module:factory"`` (regex-validated here and again by the helper; it
    is never evaluated). The argv is fixed (``python -m orchestrator.tools._openapi_dump spec``),
    so a sandbox allow-list only needs that one module. The output is parsed with ``loads_strict``
    and shape-checked; any failure raises :class:`OpenApiExportError` with a short sanitised reason.
    """
    if not isinstance(app_import, str) or not APP_IMPORT_RE.match(app_import):
        shown = sanitize_for_terminal(app_import, 80)
        raise ValueError(f"invalid app import {shown!r}; expected 'module:factory'")
    if not 0 < timeout_s <= MAX_EXPORT_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_EXPORT_TIMEOUT_S}]")
    argv = [sys.executable, "-m", HELPER_MODULE, app_import]
    result = runner.run(argv, cwd=Path(cwd), timeout_s=timeout_s)
    if result.timed_out:
        raise OpenApiExportError(f"openapi export timed out after {timeout_s:g}s")
    if result.returncode != 0:
        raise OpenApiExportError(
            f"openapi export failed (exit {result.returncode}): {_stderr_tail(result)}"
        )
    if result.truncated:
        raise OpenApiExportError("openapi export output was truncated (schema too large)")
    try:
        spec = loads_strict(result.stdout.strip(), max_bytes=MAX_EXPORT_BYTES)
    except ValueError as exc:
        raise OpenApiExportError(
            f"openapi export produced invalid JSON: {sanitize_for_terminal(exc, 200)}"
        ) from exc
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("openapi"), str)
        or not isinstance(spec.get("paths"), dict)
    ):
        raise OpenApiExportError("openapi export did not produce an OpenAPI document")
    return spec


def _stderr_tail(result: CommandResult) -> str:
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    return sanitize_for_terminal(lines[-1], 300) if lines else "no diagnostic output"
