"""Tests for ``orchestrator.tools.openapi`` and its sandbox helper ``_openapi_dump``.

The diff is exercised through a table of (mutation of a product-shaped spec) -> (exact breaking /
additive messages), then through reference, recursion, composition and hostile-input cases.
Export is
tested with a fake ``Runner`` (argv, failures, malformed output), the helper script in-process, and
once end-to-end with a real FastAPI app in a subprocess.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.types import CommandResult
from orchestrator.tools import _openapi_dump, openapi
from orchestrator.tools.openapi import (
    OpenApiDiff,
    OpenApiExportError,
    diff_openapi,
    export_openapi,
)

# ----------------------------------------------------------------------------- spec builders


def ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def json_content(schema: Any) -> dict[str, Any]:
    return {"application/json": {"schema": schema}}


def base_spec() -> dict[str, Any]:
    """A spec shaped like the shortener's: $ref'd request/response models, auth, error responses."""
    link = {
        "type": "object",
        "required": ["code", "short_url"],
        "properties": {
            "code": {"type": "string"},
            "short_url": {"type": "string", "format": "uri"},
            "expires_at": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
            "status": {"type": "string", "enum": ["active", "expired", "disabled"]},
        },
    }
    create = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string", "maxLength": 2048},
            "alias": {"anyOf": [{"type": "string", "minLength": 3}, {"type": "null"}]},
            "ttl_seconds": {"anyOf": [{"type": "integer", "minimum": 60}, {"type": "null"}]},
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {"title": "shortener", "version": "1"},
        "paths": {
            "/api/v1/links": {
                "post": {
                    "security": [{"ApiKeyAuth": []}],
                    "requestBody": {"required": True, "content": json_content(ref("CreateLink"))},
                    "responses": {
                        "201": {"description": "created", "content": json_content(ref("Link"))},
                        "409": {"description": "alias taken"},
                    },
                },
                "get": {
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "schema": {"type": "integer", "minimum": 1, "maximum": 100},
                        },
                        {"name": "offset", "in": "query", "schema": {"type": "integer"}},
                    ],
                    "responses": {"200": {"description": "page"}},
                },
            },
            "/api/v1/links/{code}": {
                "get": {
                    "security": [{"ApiKeyAuth": []}],
                    "parameters": [
                        {
                            "name": "code",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {"description": "ok", "content": json_content(ref("Link"))}
                    },
                },
                "delete": {
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {"204": {"description": "gone"}, "404": {"description": "nope"}},
                },
            },
            "/{code}": {
                "get": {
                    "responses": {
                        "302": {"description": "redirect"},
                        "404": {"description": "unknown"},
                        "410": {"description": "expired"},
                    }
                }
            },
            "/healthz": {"get": {"responses": {"200": {"description": "ok"}}}},
        },
        "components": {
            "schemas": {"Link": link, "CreateLink": create},
            "securitySchemes": {
                "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
        },
    }


def schemas(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["components"]["schemas"]


def create_op(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["paths"]["/api/v1/links"]["post"]


def list_op(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["paths"]["/api/v1/links"]["get"]


def changed(mutate: Callable[[dict[str, Any]], None]) -> OpenApiDiff:
    old = base_spec()
    new = copy.deepcopy(old)
    mutate(new)
    return diff_openapi(old, new)


REQ = "POST /api/v1/links request body (application/json) $"
RESP = "POST /api/v1/links response 201 (application/json) $"
RESP_GET = "GET /api/v1/links/{code} response 200 (application/json) $"


def link_responses(suffix: str) -> list[str]:
    """The message for a change to the shared ``Link`` model, as reported on both operations."""
    return [f"{RESP_GET}{suffix}", f"{RESP}{suffix}"]


# ------------------------------------------------------------- the classification table


def _drop(path: str, method: str | None = None) -> Callable[[dict[str, Any]], None]:
    def mutate(doc: dict[str, Any]) -> None:
        if method is None:
            del doc["paths"][path]
        else:
            del doc["paths"][path][method]

    return mutate


def _remove_alias(doc: dict[str, Any]) -> None:
    del schemas(doc)["CreateLink"]["properties"]["alias"]


def _add_required_tenant(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["tenant"] = {"type": "string"}
    schemas(doc)["CreateLink"]["required"].append("tenant")


def _add_optional_note(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["note"] = {"type": "string"}


def _url_to_integer(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["url"]["type"] = "integer"


def _alias_required(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["required"].append("alias")


def _url_optional(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["required"].remove("url")


def _narrow_status(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["status"]["enum"].remove("disabled")


def _widen_status(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["status"]["enum"].append("archived")


def _drop_short_url(doc: dict[str, Any]) -> None:
    del schemas(doc)["Link"]["properties"]["short_url"]
    schemas(doc)["Link"]["required"].remove("short_url")


def _short_url_optional(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["required"].remove("short_url")


def _code_always_present(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["required"].append("status")


def _new_response_field(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["clicks"] = {"type": "integer"}


def _new_required_response_field(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["clicks"] = {"type": "integer"}
    schemas(doc)["Link"]["required"].append("clicks")


def _format_change(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["short_url"]["format"] = "uri-reference"


def _tighten_url(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["url"]["maxLength"] = 1024


def _relax_url(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["url"]["maxLength"] = 4096


def _add_pattern(doc: dict[str, Any]) -> None:
    schemas(doc)["CreateLink"]["properties"]["url"]["pattern"] = "^https://"


def _drop_nullable_expiry(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["expires_at"] = {"type": "string", "format": "date-time"}


def _add_alternative(doc: dict[str, Any]) -> None:
    schemas(doc)["Link"]["properties"]["expires_at"]["anyOf"].append({"type": "integer"})


def _secure_redirect(doc: dict[str, Any]) -> None:
    doc["paths"]["/{code}"]["get"]["security"] = [{"ApiKeyAuth": []}]


def _open_up_list(doc: dict[str, Any]) -> None:
    list_op(doc)["security"] = []


def _swap_security(doc: dict[str, Any]) -> None:
    doc["paths"]["/api/v1/links"]["get"]["security"] = [{"BearerAuth": []}]


def _add_security_alternative(doc: dict[str, Any]) -> None:
    list_op(doc)["security"].append({"BearerAuth": []})


def _add_required_tenant_param(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"].append(
        {"name": "tenant", "in": "query", "required": True, "schema": {"type": "string"}}
    )


def _add_optional_param(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"].append({"name": "q", "in": "query", "schema": {"type": "string"}})


def _limit_required(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"][0]["required"] = True


def _noop_edit(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"][0]["required"] = False  # unchanged: it already was optional


def _drop_offset(doc: dict[str, Any]) -> None:
    del list_op(doc)["parameters"][1]


def _limit_to_string(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"][0]["schema"] = {"type": "string"}


def _limit_max_lower(doc: dict[str, Any]) -> None:
    list_op(doc)["parameters"][0]["schema"]["maximum"] = 50


def _drop_response_404(doc: dict[str, Any]) -> None:
    del doc["paths"]["/{code}"]["get"]["responses"]["404"]


def _add_response_429(doc: dict[str, Any]) -> None:
    doc["paths"]["/{code}"]["get"]["responses"]["429"] = {"description": "slow down"}


def _drop_request_body(doc: dict[str, Any]) -> None:
    del create_op(doc)["requestBody"]


def _drop_media_type(doc: dict[str, Any]) -> None:
    create_op(doc)["requestBody"]["content"] = {}


def _add_media_type(doc: dict[str, Any]) -> None:
    create_op(doc)["requestBody"]["content"]["application/x-www-form-urlencoded"] = {
        "schema": {"type": "object"}
    }


def _body_now_optional(doc: dict[str, Any]) -> None:
    create_op(doc)["requestBody"]["required"] = False


def _add_path(doc: dict[str, Any]) -> None:
    doc["paths"]["/api/v1/stats"] = {"get": {"responses": {"200": {"description": "ok"}}}}


def _add_operation(doc: dict[str, Any]) -> None:
    doc["paths"]["/api/v1/links/{code}"]["put"] = {"responses": {"200": {"description": "ok"}}}


def _new_required_body_on_new_op(doc: dict[str, Any]) -> None:
    doc["paths"]["/healthz"]["post"] = {
        "requestBody": {"required": True, "content": json_content({"type": "object"})},
        "responses": {"200": {"description": "ok"}},
    }


TABLE: list[tuple[str, Callable[[dict[str, Any]], None], list[str], list[str]]] = [
    ("removed-path", _drop("/healthz"), ["removed path /healthz"], []),
    (
        "removed-operation",
        _drop("/api/v1/links/{code}", "delete"),
        ["removed operation DELETE /api/v1/links/{code}"],
        [],
    ),
    ("removed-response-code", _drop_response_404, ["GET /{code}: removed response 404"], []),
    (
        "new-required-parameter",
        _add_required_tenant_param,
        ["GET /api/v1/links: new required parameter 'tenant' (query)"],
        [],
    ),
    (
        "optional-parameter-becomes-required",
        _limit_required,
        ["GET /api/v1/links: parameter 'limit' (query) became required"],
        [],
    ),
    (
        "removed-parameter",
        _drop_offset,
        ["GET /api/v1/links: removed parameter 'offset' (query)"],
        [],
    ),
    (
        "parameter-type-change",
        _limit_to_string,
        [
            "GET /api/v1/links parameter 'limit' (query): type changed from 'integer' to 'string'",
        ],
        [],
    ),
    (
        "parameter-bound-tightened",
        _limit_max_lower,
        ["GET /api/v1/links parameter 'limit' (query): maximum changed from 100 to 50"],
        [],
    ),
    (
        "new-required-request-field",
        _add_required_tenant,
        [f"{REQ}: new required field 'tenant'"],
        [],
    ),
    ("removed-request-field", _remove_alias, [f"{REQ}: removed field 'alias'"], []),
    (
        "request-field-type-change",
        _url_to_integer,
        [f"{REQ}.url: type changed from 'string' to 'integer'"],
        [],
    ),
    (
        "request-field-becomes-required",
        _alias_required,
        [f"{REQ}: field 'alias' became required"],
        [],
    ),
    ("request-field-becomes-optional", _url_optional, [], [f"{REQ}: field 'url' is now optional"]),
    (
        "request-bound-tightened",
        _tighten_url,
        [f"{REQ}.url: maxLength changed from 2048 to 1024"],
        [],
    ),
    ("request-bound-relaxed", _relax_url, [], [f"{REQ}.url: maxLength changed from 2048 to 4096"]),
    (
        "request-pattern-added",
        _add_pattern,
        [f"{REQ}.url: pattern constraint added or changed"],
        [],
    ),
    (
        "response-enum-narrowed",
        _narrow_status,
        link_responses('.status: enum narrowed, removed values ["disabled"]'),
        [],
    ),
    (
        "response-enum-widened",
        _widen_status,
        [],
        link_responses('.status: enum widened, added values ["archived"]'),
    ),
    ("removed-response-field", _drop_short_url, link_responses(": removed field 'short_url'"), []),
    (
        "response-field-no-longer-always-present",
        _short_url_optional,
        link_responses(": field 'short_url' is no longer always present"),
        [],
    ),
    (
        "response-field-now-always-present",
        _code_always_present,
        [],
        link_responses(": field 'status' is now always present"),
    ),
    (
        "new-response-field",
        _new_response_field,
        [],
        link_responses(": new optional field 'clicks'"),
    ),
    (
        "new-required-response-field-is-additive",
        _new_required_response_field,
        [],
        link_responses(": new required field 'clicks'"),
    ),
    (
        "format-change",
        _format_change,
        link_responses(".short_url: format changed from 'uri' to 'uri-reference'"),
        [],
    ),
    (
        "nullable-dropped",
        _drop_nullable_expiry,
        link_responses(".expires_at: removed alternative 'null'"),
        [],
    ),
    (
        "alternative-added",
        _add_alternative,
        [],
        link_responses(".expires_at: new alternative 'integer'"),
    ),
    (
        "authentication-newly-required",
        _secure_redirect,
        ["GET /{code}: authentication is now required [ApiKeyAuth]"],
        [],
    ),
    (
        "authentication-dropped",
        _open_up_list,
        [],
        ["GET /api/v1/links: authentication is no longer required"],
    ),
    (
        "security-scheme-swapped",
        _swap_security,
        ["GET /api/v1/links: removed security alternative [ApiKeyAuth]"],
        ["GET /api/v1/links: new security alternative [BearerAuth]"],
    ),
    (
        "security-alternative-added",
        _add_security_alternative,
        [],
        ["GET /api/v1/links: new security alternative [BearerAuth]"],
    ),
    (
        "request-body-removed",
        _drop_request_body,
        ["POST /api/v1/links: removed request body"],
        [],
    ),
    (
        "request-media-type-removed",
        _drop_media_type,
        ["POST /api/v1/links request body: removed media type 'application/json'"],
        [],
    ),
    (
        "request-media-type-added",
        _add_media_type,
        [],
        ["POST /api/v1/links request body: new media type 'application/x-www-form-urlencoded'"],
    ),
    (
        "request-body-now-optional",
        _body_now_optional,
        [],
        ["POST /api/v1/links: request body is now optional"],
    ),
    ("new-path", _add_path, [], ["new path /api/v1/stats"]),
    ("new-operation", _add_operation, [], ["new operation PUT /api/v1/links/{code}"]),
    ("new-response-code", _add_response_429, [], ["GET /{code}: new response 429"]),
    (
        "new-optional-parameter",
        _add_optional_param,
        [],
        ["GET /api/v1/links: new optional parameter 'q' (query)"],
    ),
    ("new-optional-request-field", _add_optional_note, [], [f"{REQ}: new optional field 'note'"]),
    (
        "new-operation-with-required-body-is-only-additive",
        _new_required_body_on_new_op,
        [],
        ["new operation POST /healthz"],
    ),
    ("no-op-edit", _noop_edit, [], []),
]


@pytest.mark.parametrize(
    ("mutate", "breaking", "additive"),
    [pytest.param(m, b, a, id=name) for name, m, b, a in TABLE],
)
def test_classification_table(
    mutate: Callable[[dict[str, Any]], None], breaking: list[str], additive: list[str]
) -> None:
    result = changed(mutate)
    assert (result.breaking, result.additive) == (breaking, additive)
    assert result.is_breaking is bool(breaking)


def test_identical_documents_and_result_helpers() -> None:
    same = diff_openapi(base_spec(), base_spec())
    assert same == OpenApiDiff([], []) and not same.is_breaking
    assert OpenApiDiff(["b"], ["a"]).to_dict() == {"breaking": ["b"], "additive": ["a"]}
    assert OpenApiDiff().breaking == [] and OpenApiDiff().additive == []


def test_results_are_sorted_deduplicated_and_order_independent() -> None:
    def mutate(doc: dict[str, Any]) -> None:
        _drop_short_url(doc)
        _add_required_tenant(doc)
        _add_response_429(doc)
        _add_path(doc)

    result = changed(mutate)
    assert result.breaking == sorted(set(result.breaking))
    assert result.additive == sorted(set(result.additive))
    old = base_spec()
    new = copy.deepcopy(old)
    mutate(new)
    shuffled = json.loads(json.dumps(new), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert diff_openapi(old, shuffled) == result


# ----------------------------------------------------------------------------- paths and parameters


def test_renaming_a_path_parameter_is_not_a_change() -> None:
    old = base_spec()
    new = copy.deepcopy(old)
    paths = new["paths"]
    item = paths.pop("/api/v1/links/{code}")
    item["get"]["parameters"][0]["name"] = "id"
    paths["/api/v1/links/{id}"] = item
    assert diff_openapi(old, new) == OpenApiDiff([], [])


def test_changing_the_shape_of_a_path_is_a_removal_plus_an_addition() -> None:
    old = base_spec()
    new = copy.deepcopy(old)
    new["paths"]["/api/v1/links/{code}/stats"] = new["paths"].pop("/api/v1/links/{code}")
    result = diff_openapi(old, new)
    assert result.breaking == ["removed path /api/v1/links/{code}"]
    assert result.additive == ["new path /api/v1/links/{code}/stats"]


def test_path_level_parameters_are_merged_and_overridden_by_the_operation() -> None:
    old = {
        "paths": {"/x": {"parameters": [{"name": "a", "in": "query", "required": True}], "get": {}}}
    }
    new = {
        "paths": {
            "/x": {
                "parameters": [{"name": "a", "in": "query", "required": True}],
                "get": {"parameters": [{"name": "a", "in": "query", "required": False}]},
            }
        }
    }
    assert diff_openapi(old, new).additive == ["GET /x: parameter 'a' (query) is now optional"]
    assert diff_openapi(old, new).breaking == []


def test_parameters_and_responses_may_be_references() -> None:
    def doc(required: bool) -> dict[str, Any]:
        return {
            "paths": {
                "/x": {
                    "get": {
                        "parameters": [{"$ref": "#/components/parameters/Limit"}],
                        "responses": {"404": {"$ref": "#/components/responses/NotFound"}},
                    }
                }
            },
            "components": {
                "parameters": {"Limit": {"name": "limit", "in": "query", "required": required}},
                "responses": {
                    "NotFound": {"description": "x", "content": json_content({"type": "object"})}
                },
            },
        }

    assert diff_openapi(doc(False), doc(True)).breaking == [
        "GET /x: parameter 'limit' (query) became required"
    ]


# -------------------------------------------------------- schemas: refs and composition


def test_ref_resolution_follows_each_document_independently() -> None:
    old = base_spec()
    new = copy.deepcopy(old)
    # the request model moves to a differently named component: still the same structure
    schemas(new)["CreateLinkV2"] = schemas(new).pop("CreateLink")
    create_op(new)["requestBody"]["content"]["application/json"]["schema"] = ref("CreateLinkV2")
    assert diff_openapi(old, new) == OpenApiDiff([], [])
    # inlining a $ref is not a change either
    inlined = copy.deepcopy(old)
    create_op(inlined)["requestBody"]["content"]["application/json"]["schema"] = copy.deepcopy(
        schemas(old)["CreateLink"]
    )
    assert diff_openapi(old, inlined) == OpenApiDiff([], [])


def test_nullable_spellings_of_3_0_and_3_1_are_equivalent() -> None:
    def doc(prop: dict[str, Any]) -> dict[str, Any]:
        return {
            "paths": {
                "/x": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "x",
                                "content": json_content(
                                    {"type": "object", "properties": {"v": prop}}
                                ),
                            }
                        }
                    }
                }
            }
        }

    three_zero = doc({"type": "string", "nullable": True})
    three_one = doc({"anyOf": [{"type": "string"}, {"type": "null"}]})
    type_list = doc({"type": ["string", "null"]})
    assert diff_openapi(three_zero, three_one) == OpenApiDiff([], [])
    assert diff_openapi(three_one, type_list) == OpenApiDiff([], [])
    dropped = diff_openapi(three_one, doc({"type": "string"}))
    assert dropped.breaking == [
        "GET /x response 200 (application/json) $.v: removed alternative 'null'"
    ]


def test_all_of_members_are_merged_before_comparison() -> None:
    def doc(base_props: dict[str, Any]) -> dict[str, Any]:
        return {
            "paths": {
                "/x": {
                    "post": {
                        "requestBody": {"content": json_content(ref("Child"))},
                        "responses": {},
                    }
                }
            },
            "components": {
                "schemas": {
                    "Base": {"type": "object", "properties": base_props, "required": ["id"]},
                    "Child": {
                        "allOf": [ref("Base"), {"properties": {"name": {"type": "string"}}}],
                    },
                }
            },
        }

    old = doc({"id": {"type": "string"}, "tag": {"type": "string"}})
    new = doc({"id": {"type": "string"}})
    assert diff_openapi(old, new).breaking == [
        "POST /x request body (application/json) $: removed field 'tag'"
    ]
    assert diff_openapi(old, old) == OpenApiDiff([], [])


def test_recursive_models_terminate_and_report_a_change_once() -> None:
    def doc(extra: bool) -> dict[str, Any]:
        node: dict[str, Any] = {
            "type": "object",
            "properties": {
                "children": {"type": "array", "items": ref("Node")},
                "name": {"type": "string"},
            },
        }
        if extra:
            node["properties"]["depth"] = {"type": "integer"}
        return {
            "paths": {
                "/tree": {
                    "get": {
                        "responses": {
                            "200": {"description": "x", "content": json_content(ref("Node"))}
                        }
                    }
                }
            },
            "components": {"schemas": {"Node": node}},
        }

    assert diff_openapi(doc(False), doc(False)) == OpenApiDiff([], [])
    result = diff_openapi(doc(False), doc(True))
    assert result.additive == [
        "GET /tree response 200 (application/json) $: new optional field 'depth'"
    ]


def test_reference_cycles_dangling_and_external_references_do_not_hang_or_crash() -> None:
    def doc(target: str) -> dict[str, Any]:
        return {
            "paths": {"/x": {"post": {"requestBody": {"content": json_content({"$ref": target})}}}},
            "components": {
                "schemas": {
                    "A": {"$ref": "#/components/schemas/B"},
                    "B": {"$ref": "#/components/schemas/A"},
                }
            },
        }

    assert diff_openapi(doc("#/components/schemas/A"), doc("#/components/schemas/A")).breaking == []
    assert (
        diff_openapi(
            doc("#/components/schemas/Missing"), doc("#/components/schemas/Missing")
        ).breaking
        == []
    )
    assert diff_openapi(doc("other.yaml#/X"), doc("other.yaml#/X")) == OpenApiDiff([], [])
    changed_ref = diff_openapi(doc("other.yaml#/X"), doc("other.yaml#/Y"))
    assert changed_ref.breaking == [
        "POST /x request body (application/json) $: schema reference changed"
    ]


def test_all_of_cycles_are_bounded() -> None:
    schema = {
        "allOf": [{"$ref": "#/components/schemas/Loop"}],
        "properties": {"a": {"type": "string"}},
    }
    doc = {
        "paths": {"/x": {"post": {"requestBody": {"content": json_content(ref("Loop"))}}}},
        "components": {"schemas": {"Loop": schema}},
    }
    assert diff_openapi(doc, copy.deepcopy(doc)) == OpenApiDiff([], [])


def test_too_deep_or_too_large_schemas_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def nested(levels: int, leaf: str) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": leaf}
        for _ in range(levels):
            schema = {"type": "object", "properties": {"n": schema}}
        return schema

    def doc(leaf: str) -> dict[str, Any]:
        return {
            "paths": {"/x": {"post": {"requestBody": {"content": json_content(nested(40, leaf))}}}}
        }

    deep = diff_openapi(doc("string"), doc("integer"))
    assert any("too large or deeply nested" in message for message in deep.breaking)

    def wide(kind: str) -> dict[str, Any]:
        schema = {"type": "object", "properties": {f"p{i}": {"type": kind} for i in range(20)}}
        return {"paths": {"/x": {"post": {"requestBody": {"content": json_content(schema)}}}}}

    monkeypatch.setattr(openapi, "MAX_COMPARISONS", 5)
    starved = diff_openapi(wide("string"), wide("integer"))
    assert any("too large or deeply nested" in message for message in starved.breaking)


def test_enum_restrictions_added_or_removed_and_unhashable_values() -> None:
    def doc(enum: Any) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "object", "properties": {"k": {"type": "string"}}}
        if enum is not None:
            schema["properties"]["k"]["enum"] = enum
        return {"paths": {"/x": {"post": {"requestBody": {"content": json_content(schema)}}}}}

    loc = "POST /x request body (application/json) $.k"
    assert diff_openapi(doc(None), doc(["a"])).breaking == [f"{loc}: enum restriction added"]
    assert diff_openapi(doc(["a"]), doc(None)).additive == [f"{loc}: enum restriction removed"]
    odd = diff_openapi(doc([{"x": 1}, ["l"]]), doc([{"x": 1}]))
    assert odd.breaking == [f'{loc}: enum narrowed, removed values [["l"]]']


def test_array_items_and_additional_properties_are_compared() -> None:
    def doc(item_type: str, extra_type: str) -> dict[str, Any]:
        schema = {
            "type": "array",
            "items": {"type": item_type},
        }
        mapping = {"type": "object", "additionalProperties": {"type": extra_type}}
        return {
            "paths": {
                "/x": {
                    "post": {
                        "requestBody": {"content": json_content(schema)},
                        "responses": {
                            "200": {"description": "x", "content": json_content(mapping)}
                        },
                    }
                }
            }
        }

    result = diff_openapi(doc("string", "string"), doc("integer", "integer"))
    assert result.breaking == [
        "POST /x request body (application/json) $[]: type changed from 'string' to 'integer'",
        "POST /x response 200 (application/json) $.*: type changed from 'string' to 'integer'",
    ]


def test_type_change_hides_the_now_incomparable_structure() -> None:
    def doc(schema: dict[str, Any]) -> dict[str, Any]:
        return {"paths": {"/x": {"post": {"requestBody": {"content": json_content(schema)}}}}}

    old = doc({"type": "object", "properties": {"a": {"type": "string"}}})
    new = doc({"type": "array", "items": {"type": "string"}})
    assert diff_openapi(old, new).breaking == [
        "POST /x request body (application/json) $: type changed from 'object' to 'array'"
    ]


# ----------------------------------------------------------------------------- hostile input


@pytest.mark.parametrize("bad", [None, [], "openapi", 7])
def test_non_object_documents_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="must be an object"):
        diff_openapi(bad, {})
    with pytest.raises(ValueError, match="must be an object"):
        diff_openapi({}, bad)


def test_swagger_2_is_refused_instead_of_silently_comparing_equal() -> None:
    with pytest.raises(ValueError, match=r"Swagger 2\.0"):
        diff_openapi({"swagger": "2.0", "paths": {}}, {"openapi": "3.0.0", "paths": {}})


def test_malformed_structures_never_crash_the_diff() -> None:
    junk: list[Any] = [None, 1, "x", [], [1, "a"], {"$ref": 5}, {"$ref": ["#/x"]}, {"": {}}]
    for value in junk:
        doc = {
            "openapi": "3.1.0",
            "paths": {
                "/a": {
                    "get": {
                        "parameters": value,
                        "requestBody": value,
                        "responses": value,
                        "security": value,
                    },
                    "parameters": value,
                },
                "/b": value,
                5: {"get": {}},
            },
            "components": value,
            "security": value,
        }
        for other in (
            doc,
            {"paths": value},
            {"paths": {"/a": {"get": {"responses": {"200": value}}}}},
        ):
            diff_openapi(doc, other)
            diff_openapi(other, doc)


def test_names_in_messages_are_sanitised() -> None:
    old: dict[str, Any] = {"paths": {"/ok": {"get": {}}}}
    new: dict[str, Any] = {"paths": {"/ok": {"get": {}}, "/x\x1b[31m\x07evil\nline": {"get": {}}}}
    assert diff_openapi(old, new).additive == ["new path /xevilline"]
    long_name = "/" + "a" * 5000
    message = diff_openapi({"paths": {}}, {"paths": {long_name: {}}}).additive[0]
    assert len(message) < 300


def test_huge_result_lists_are_capped_with_a_marker() -> None:
    old: dict[str, Any] = {"paths": {f"/p{i:05d}": {"get": {}} for i in range(1500)}}
    result = diff_openapi(old, {"paths": {}})
    assert len(result.breaking) == openapi.MAX_ENTRIES
    assert result.breaking[-1] == "... and 501 more"
    assert result.breaking[0] == "removed path /p00000"


def deep_document(levels: int) -> dict[str, Any]:
    deep: Any = {"type": "string"}
    for _ in range(levels):
        deep = {"type": "object", "properties": {"n": deep}}
    return {"paths": {"/x": {"post": {"requestBody": {"content": json_content(deep)}}}}}


def test_the_same_deep_document_is_no_change_and_distinct_copies_fail_closed() -> None:
    doc = deep_document(3000)
    assert diff_openapi(doc, doc) == OpenApiDiff([], [])  # identity: nothing to classify
    # equal but distinct: comparing 3000 levels exceeds the depth budget -> reported, never a crash
    result = diff_openapi(deep_document(3000), deep_document(3000))
    assert result.additive == []
    assert result.breaking == [
        "POST /x request body (application/json) $.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n.n."
        "n.n.n.n.n.n.n.n.n: schema too large or deeply nested to compare (treated as breaking)"
    ] or all("too large or deeply nested" in m for m in result.breaking)
    assert diff_openapi(deep_document(30), deep_document(30)) == OpenApiDiff(
        [], []
    )  # shallow equal copies


# ----------------------------------------------------------------------------- export (fake runner)


class FakeRunner:
    def __init__(self, outcome: CommandResult) -> None:
        self.outcome = outcome
        self.calls: list[tuple[list[str], Path, float]] = []

    def run(self, argv: Any, *, cwd: Path, timeout_s: float) -> CommandResult:
        self.calls.append((list(argv), cwd, timeout_s))
        return self.outcome


def command_result(
    stdout: str = "", *, code: int = 0, stderr: str = "", **extra: Any
) -> CommandResult:
    return CommandResult(("x",), code, stdout, stderr, 0.0, **extra)


GOOD_SPEC = {"openapi": "3.1.0", "info": {"title": "t", "version": "1"}, "paths": {"/x": {}}}


def test_export_runs_the_helper_module_with_a_fixed_argv_and_parses_the_json(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(command_result(json.dumps(GOOD_SPEC) + "\n"))
    spec = export_openapi(runner, tmp_path, "shortener.main:create_app", timeout_s=12.5)
    assert spec == GOOD_SPEC
    argv, cwd, timeout = runner.calls[0]
    assert argv == [
        sys.executable,
        "-m",
        "orchestrator.tools._openapi_dump",
        "shortener.main:create_app",
    ]
    assert (cwd, timeout) == (tmp_path, 12.5)
    assert "-c" not in argv


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "plain",
        ":factory",
        "module:",
        "a b:c",
        "a;b:c",
        "a:b;c",
        "a:b\n",
        "a:b c",
        "../x:y",
        "a/b:c",
        "a.b.:c",
        "a..b:c",
        "a:b:c",
        "a:b.c",
        "-m:x",
        "ä:b",
        'a:__import__("os")',
        "a:b # c",
        "a" * 65 + ":b",
        "a:" + "b" * 65,
        ".".join(["m"] * 9) + ":f",
    ],
)
def test_app_import_must_be_a_plain_module_colon_name(tmp_path: Path, bad: str) -> None:
    runner = FakeRunner(command_result(json.dumps(GOOD_SPEC)))
    with pytest.raises(ValueError, match="invalid app import"):
        export_openapi(runner, tmp_path, bad)
    assert runner.calls == []  # nothing was run


@pytest.mark.parametrize("bad", [None, 7, b"a:b", ["a:b"]])
def test_non_string_app_import_is_rejected(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(ValueError, match="invalid app import"):
        export_openapi(FakeRunner(command_result("{}")), tmp_path, bad)


@pytest.mark.parametrize("timeout", [0, -1, 601, float("nan"), float("inf")])
def test_timeout_must_be_positive_and_bounded(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        export_openapi(FakeRunner(command_result("{}")), tmp_path, "a:b", timeout_s=timeout)


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (command_result("", timed_out=True, code=-9), "timed out"),
        (
            command_result("", code=3, stderr="noise\nopenapi dump failed: RuntimeError: boom\n"),
            "exit 3): openapi dump failed: RuntimeError: boom",
        ),
        (command_result("", code=1), "no diagnostic output"),
        (command_result(json.dumps(GOOD_SPEC), truncated=True), "truncated"),
        (command_result("not json"), "invalid JSON"),
        (command_result(""), "invalid JSON"),
        (command_result('{"openapi": "3.1.0", "openapi": "3.0.0", "paths": {}}'), "invalid JSON"),
        (command_result('{"openapi": "3.1.0", "paths": {}, "x": NaN}'), "invalid JSON"),
        (command_result("[]"), "did not produce an OpenAPI document"),
        (command_result('{"paths": {}}'), "did not produce an OpenAPI document"),
        (command_result('{"openapi": 3, "paths": {}}'), "did not produce an OpenAPI document"),
        (
            command_result('{"openapi": "3.1.0", "paths": []}'),
            "did not produce an OpenAPI document",
        ),
    ],
)
def test_export_failures_are_reported_as_export_errors(
    tmp_path: Path, outcome: CommandResult, message: str
) -> None:
    with pytest.raises(OpenApiExportError, match=re.escape(message)):
        export_openapi(FakeRunner(outcome), tmp_path, "a:b")


def test_diagnostics_from_the_sandbox_are_sanitised_and_bounded(tmp_path: Path) -> None:
    hostile = "\x1b[2J\x1b[31mFAKE PROMPT>\x07 " + "x" * 5000
    runner = FakeRunner(command_result("", code=3, stderr=hostile))
    with pytest.raises(OpenApiExportError) as info:
        export_openapi(runner, tmp_path, "a:b")
    text = str(info.value)
    assert "\x1b" not in text and "\x07" not in text and len(text) < 450


def test_oversized_output_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openapi, "MAX_EXPORT_BYTES", 50)
    with pytest.raises(OpenApiExportError, match="invalid JSON"):
        export_openapi(FakeRunner(command_result(json.dumps(GOOD_SPEC))), tmp_path, "a:b")


def test_export_accepts_a_str_cwd(tmp_path: Path) -> None:
    runner = FakeRunner(command_result(json.dumps(GOOD_SPEC)))
    export_openapi(runner, str(tmp_path), "a:b")
    assert runner.calls[0][1] == tmp_path


# ------------------------------------------------------------ the helper script (in process)


@pytest.fixture
def dump_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    source = textwrap.dedent(
        """
        import os

        class Api:
            def openapi(self):
                switch = os.environ.get("SHORTENER_OPENAPI_DUMP")
                return {"paths": {}, "openapi": "3.1.0", "z": switch}

        app = Api()

        def factory():
            print("noise printed while importing or building the app")
            return Api()

        def explodes():
            raise RuntimeError("boom \\x1b[31m with escapes " + "x" * 1000)

        def exits():
            raise SystemExit(0)

        def not_an_app():
            return 42

        class BadSchema:
            def openapi(self):
                return {"v": float("nan")}

        def nan_schema():
            return BadSchema()

        class Wrong:
            def openapi(self):
                return ["not", "a", "dict"]

        def wrong_shape():
            return Wrong()
        """
    )
    name = "dumpfixture_mod"
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("SHORTENER_OPENAPI_DUMP", "placeholder")  # records "absent" for teardown...
    monkeypatch.delenv("SHORTENER_OPENAPI_DUMP")  # ...so the switch the helper sets cannot leak
    return name


def test_helper_prints_one_canonical_json_line_and_diverts_noise(
    dump_module: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _openapi_dump.main([f"{dump_module}:factory"]) == 0
    out, err = capsys.readouterr()
    assert out == '{"openapi":"3.1.0","paths":{},"z":"1"}\n'  # sorted keys, compact, dump mode on
    assert "noise printed" in err and "noise printed" not in out


def test_helper_accepts_an_application_instance_as_well_as_a_factory(
    dump_module: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _openapi_dump.main([f"{dump_module}:app"]) == 0
    assert json.loads(capsys.readouterr().out)["openapi"] == "3.1.0"


def test_helper_sets_the_dump_switch_but_never_overrides_an_explicit_value(
    dump_module: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHORTENER_OPENAPI_DUMP", "explicit")
    assert _openapi_dump.main([f"{dump_module}:app"]) == 0
    assert json.loads(capsys.readouterr().out)["z"] == "explicit"


@pytest.mark.parametrize(
    ("attr", "code", "needle"),
    [
        ("explodes", 3, "RuntimeError: boom"),
        ("exits", 3, "SystemExit"),
        ("not_an_app", 3, "neither an application nor a factory"),
        ("missing_name", 3, "AttributeError"),
        ("wrong_shape", 3, "did not return an object"),
        ("nan_schema", 4, "not serialisable"),
    ],
)
def test_helper_failures_have_distinct_exit_codes_and_short_messages(
    dump_module: str, capsys: pytest.CaptureFixture[str], attr: str, code: int, needle: str
) -> None:
    assert _openapi_dump.main([f"{dump_module}:{attr}"]) == code
    out, err = capsys.readouterr()
    assert out == "" and needle in err
    assert "Traceback" not in err and "\x1b" not in err and len(err) < 700


def test_helper_missing_module_is_a_clean_failure(capsys: pytest.CaptureFixture[str]) -> None:
    assert _openapi_dump.main(["no_such_module_for_openapi_dump:factory"]) == 3
    assert "ModuleNotFoundError" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [[], ["a:b", "c:d"], ["a b:c"], ["os:system;x"], ["../x:y"], ["a:b\n"], ["-c"]]
)
def test_helper_rejects_anything_but_one_valid_target_before_importing(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    before = set(sys.modules)
    code = _openapi_dump.main(argv)
    after = set(sys.modules)
    assert code == 2
    assert "usage" in capsys.readouterr().err
    assert after == before


def test_helper_and_export_share_one_validation_regex() -> None:
    assert openapi.APP_IMPORT_RE is _openapi_dump.APP_IMPORT_RE


# ------------------------------------------------------------ end to end (real subprocess)


class LocalRunner:
    """Test-only stand-in for ``SandboxRunner``: a plain subprocess with a minimal environment."""

    def run(self, argv: Any, *, cwd: Path, timeout_s: float) -> CommandResult:
        repo = str(Path(__file__).resolve().parents[3])
        env = {
            "PATH": "",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONPATH": f"{cwd}{os.pathsep}{repo}",
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        done = subprocess.run(
            list(argv),
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandResult(tuple(argv), done.returncode, done.stdout, done.stderr, 0.0)


APP_TEMPLATE = """
import os

from fastapi import FastAPI

def create_app() -> FastAPI:
    assert os.environ.get("SHORTENER_OPENAPI_DUMP") == "1", "dump switch missing"
    app = FastAPI(title="mini", version="1.0", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}
{extra}
    return app
"""


@pytest.mark.slow
def test_real_fastapi_app_is_exported_through_a_subprocess_and_diffed(tmp_path: Path) -> None:
    (tmp_path / "miniapi.py").write_text(APP_TEMPLATE.replace("{extra}", ""), encoding="utf-8")
    spec = export_openapi(LocalRunner(), tmp_path, "miniapi:create_app", timeout_s=60)
    assert spec["info"]["title"] == "mini" and set(spec["paths"]) == {"/ping"}
    assert "get" in spec["paths"]["/ping"]
    grown = copy.deepcopy(spec)
    grown["paths"]["/stats"] = {"get": {"responses": {"200": {"description": "ok"}}}}
    assert diff_openapi(spec, grown) == OpenApiDiff([], ["new path /stats"])
    assert diff_openapi(grown, spec) == OpenApiDiff(["removed path /stats"], [])
