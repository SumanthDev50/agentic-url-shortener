"""Generate ``docs/api.md`` from the application's OpenAPI schema.

``python -m shortener.apidoc`` writes the file; ``python -m shortener.apidoc --check`` exits
non-zero if the committed file is stale (a test runs the same comparison).  The schema is read
from ``create_app(...).openapi()``, which needs no database and no running server; the settings
used for that are throw-away (a key generated at runtime, never printed or written).
"""

from __future__ import annotations

import secrets
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shortener.config import ApiKeyDigest, Settings
from shortener.main import create_app
from shortener.services.auth import digest_api_key

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "api.md"
HTTP_METHODS = ("get", "post", "put", "patch", "delete")
HIDDEN_SCHEMAS = frozenset({"HTTPValidationError", "ValidationError"})
CONSTRAINT_KEYS = ("minLength", "maxLength", "pattern", "minimum", "maximum")

Schema = dict[str, Any]

STATIC_SECTIONS = """\
## Authentication

Every `/api/v1` endpoint and `/metrics` require the header `X-API-Key: <secret>`.
Keys are configured with `SHORTENER_API_KEYS` (`id:secret` pairs). A link belongs to the key
that created it; other keys receive `404` for it. `GET /{code}`, `/healthz` and `/readyz`
are public.

## Error model

All errors use one JSON shape. `code` is stable and machine-readable, `message` is static,
`request_id` matches the `X-Request-ID` response header, and `details` (validation errors
only) lists the offending fields.

```json
{"error": {"code": "not_found", "message": "The link does not exist.",
           "request_id": "6b00369f-e071-4e74-9721-d6956aa29276"}}
```

Rate limited responses (`429`) carry `Retry-After` and `X-RateLimit-Limit`,
`X-RateLimit-Remaining` and `X-RateLimit-Reset` (seconds until the bucket is full again).
"""


def build_openapi() -> Schema:
    """The OpenAPI document of a throw-away application instance."""
    settings = Settings(
        api_keys=(ApiKeyDigest("docs", digest_api_key(secrets.token_urlsafe(32))),),
        database_path=str(Path(tempfile.gettempdir()) / "shortener-apidoc.db"),
    )
    return create_app(settings).openapi()


def render_markdown(spec: Schema) -> str:
    """Deterministic Markdown (LF newlines) for an OpenAPI document."""
    info = spec["info"]
    lines = [
        f"# {info['title']} API",
        "",
        "> Generated from the OpenAPI schema by `python -m shortener.apidoc`. Do not edit by hand.",
        "",
        f"Version: `{info['version']}`",
        "",
        _clean(info.get("description", "")),
        "",
        STATIC_SECTIONS,
        "## Endpoints",
        "",
    ]
    for path, item in spec["paths"].items():
        for method in HTTP_METHODS:
            if method in item:
                lines.extend(_operation(method, path, item[method]))
    lines.extend(["## Schemas", ""])
    components: dict[str, Schema] = spec.get("components", {}).get("schemas", {})
    for name, schema in sorted(components.items()):
        if name not in HIDDEN_SCHEMAS:
            lines.extend(_schema_section(name, schema))
    return "\n".join(lines).rstrip("\n") + "\n"


def _operation(method: str, path: str, operation: Schema) -> list[str]:
    lines = [f"### `{method.upper()} {path}`", ""]
    if operation.get("summary"):
        lines.extend([f"**{operation['summary']}**", ""])
    if operation.get("description"):
        lines.extend([_clean(operation["description"]), ""])
    secured = "`X-API-Key` header" if operation.get("security") else "none (public)"
    lines.extend([f"Authentication: {secured}", ""])
    if operation.get("parameters"):
        lines.extend(_parameters_table(operation["parameters"]))
    if operation.get("requestBody"):
        schema = operation["requestBody"]["content"]["application/json"]["schema"]
        lines.extend([f"Request body (JSON): `{_type_of(schema)}`", ""])
    lines.extend(_responses_table(operation.get("responses", {})))
    return lines


def _parameters_table(parameters: list[Schema]) -> list[str]:
    lines = ["| Parameter | In | Type | Required | Description |", "|---|---|---|---|---|"]
    for parameter in parameters:
        schema = parameter.get("schema", {})
        lines.append(
            f"| `{parameter['name']}` | {parameter['in']} | `{_type_of(schema)}` | "
            f"{_yes_no(parameter.get('required', False))} | "
            f"{_describe(parameter.get('description', ''), schema)} |"
        )
    return [*lines, ""]


def _responses_table(responses: Schema) -> list[str]:
    lines = ["| Status | Description | Body |", "|---|---|---|"]
    for status, response in sorted(responses.items(), key=lambda entry: str(entry[0])):
        lines.append(
            f"| {status} | {_cell(_clean(response.get('description', '')))} | "
            f"{_response_body(response)} |"
        )
    return [*lines, ""]


def _response_body(response: Schema) -> str:
    content = response.get("content")
    if not content:
        return "-"
    media_type, details = next(iter(content.items()))
    schema = details.get("schema", {})
    if "$ref" in schema:
        return f"`{_type_of(schema)}` ({media_type})"
    return f"`{_type_of(schema)}` ({media_type})" if schema else "-"


def _schema_section(name: str, schema: Schema) -> list[str]:
    lines = [f"### {name}", ""]
    if schema.get("description"):
        lines.extend([_clean(schema["description"]), ""])
    if "enum" in schema:
        values = ", ".join(f"`{value}`" for value in schema["enum"])
        return [*lines, f"One of: {values}.", ""]
    required = set(schema.get("required", []))
    lines.extend(["| Field | Type | Required | Description |", "|---|---|---|---|"])
    for field, details in schema.get("properties", {}).items():
        lines.append(
            f"| `{field}` | `{_type_of(details)}` | {_yes_no(field in required)} | "
            f"{_describe(details.get('description', ''), details)} |"
        )
    return [*lines, ""]


def _type_of(schema: Schema) -> str:
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "anyOf" in schema:
        return " or ".join(_type_of(option) for option in schema["anyOf"])
    kind = str(schema.get("type", "any"))
    if kind == "array":
        return f"array of {_type_of(schema.get('items', {}))}"
    fmt = schema.get("format")
    return f"{kind} ({fmt})" if fmt else kind


def _describe(description: str, schema: Schema) -> str:
    """Description cell: text plus the schema's constraints in backticks."""
    text = _clean(description)
    options = [schema, *schema.get("anyOf", [])]
    found = [
        f"{key}={_number(option[key])!r}"
        for option in options
        for key in CONSTRAINT_KEYS
        if key in option
    ]
    if found:
        text = f"{text} Constraints: `{', '.join(found)}`.".strip()
    return _cell(text)


def _number(value: object) -> object:
    """Render integral floats (pydantic emits ``60.0``) as integers."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _clean(text: object) -> str:
    return " ".join(str(text).split())


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    content = render_markdown(build_openapi())
    if "--check" in arguments:
        current = DEFAULT_OUTPUT.read_text(encoding="utf-8") if DEFAULT_OUTPUT.exists() else ""
        if current.replace("\r\n", "\n") != content:
            sys.stderr.write("docs/api.md is out of date; run: python -m shortener.apidoc\n")
            return 1
        return 0
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
