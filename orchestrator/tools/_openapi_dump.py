"""Sandbox-side helper: print the OpenAPI schema of an application as canonical JSON.

Run as ``python -m orchestrator.tools._openapi_dump <module>:<factory>`` (never ``python -c``) from
the scratch directory that holds the code under test; ``openapi.export_openapi`` builds that argv.

Safety properties
-----------------
* The single argument is validated against :data:`APP_IMPORT_RE` (dotted module path, a colon, one
  identifier). It is only ever fed to ``importlib.import_module`` / ``getattr`` -- never ``eval``,
  never a path, so nothing but an importable attribute name can be selected.
* The helper exports ``SHORTENER_OPENAPI_DUMP=1`` (unless the caller already set it) *before* the
  import. Products that refuse to start without secrets, like the shortener, should read it as
  "schema-only mode": no API keys required, in-memory storage, no background threads. The helper
  itself never invents a secret or a settings object.
* Anything the imported code prints is diverted to stderr, so stdout carries exactly one line of
  JSON (sorted keys, ASCII, no NaN) and a noisy import cannot corrupt the document.
* Failures of the (untrusted) factory become one short sanitised stderr line and a distinct exit
  code; no traceback is printed.

Exit codes: 0 ok, 2 usage / invalid target, 3 import or factory failure, 4 schema not serialisable.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import sys
from collections.abc import Sequence
from typing import Any

# ``\Z`` (not ``$``): ``$`` also matches before a trailing newline, which would let "a:b\n" through.
APP_IMPORT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,63}(?:\.[A-Za-z_][A-Za-z0-9_]{0,63}){0,7}:[A-Za-z_][A-Za-z0-9_]{0,63}\Z"
)
DUMP_ENV = "SHORTENER_OPENAPI_DUMP"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_IMPORT = 3
EXIT_SCHEMA = 4
_MAX_MESSAGE_CHARS = 500


def _load_schema(spec: str) -> dict[str, Any]:
    """Import ``module:attr``: an app (has ``openapi``) or a factory that returns one."""
    module_name, _, attr = spec.partition(":")
    target = getattr(importlib.import_module(module_name), attr)
    # A FastAPI instance is itself callable (ASGI), so the ``openapi`` attribute is checked first.
    app = target if callable(getattr(target, "openapi", None)) else target()
    openapi = getattr(app, "openapi", None)
    if not callable(openapi):
        raise TypeError("target is neither an application nor a factory returning one")
    schema = openapi()
    if not isinstance(schema, dict):
        raise TypeError("openapi() did not return an object")
    return schema


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return text[:_MAX_MESSAGE_CHARS]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or not APP_IMPORT_RE.match(args[0]):
        sys.stderr.write("usage: python -m orchestrator.tools._openapi_dump <module>:<factory>\n")
        return EXIT_USAGE
    os.environ.setdefault(DUMP_ENV, "1")
    stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            schema = _load_schema(args[0])
    except (Exception, SystemExit) as exc:  # untrusted factory: report, never propagate
        sys.stderr.write(f"openapi dump failed: {_short(exc)}\n")
        return EXIT_IMPORT
    try:
        text = json.dumps(
            schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"openapi schema is not serialisable: {_short(exc)}\n")
        return EXIT_SCHEMA
    stdout.write(text + "\n")
    stdout.flush()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
