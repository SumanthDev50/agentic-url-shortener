"""docs/api.md is generated from the OpenAPI schema and must stay up to date."""

from __future__ import annotations

from pathlib import Path

import pytest

from shortener import apidoc
from shortener.apidoc import build_openapi, main, render_markdown

API_DOC = Path(__file__).resolve().parents[2] / "docs" / "api.md"


def test_committed_api_documentation_is_up_to_date() -> None:
    expected = render_markdown(build_openapi())
    actual = API_DOC.read_text(encoding="utf-8")  # universal newlines
    assert actual == expected, "docs/api.md is stale: run `python -m shortener.apidoc`"


def test_generated_markdown_is_deterministic_ascii_with_lf_only() -> None:
    first = render_markdown(build_openapi())
    assert first == render_markdown(build_openapi())
    assert first.isascii()
    assert "\r" not in first
    assert first.endswith("\n") and not first.endswith("\n\n")
    raw = API_DOC.read_bytes()
    assert raw.isascii() and b"\r" not in raw


def test_documentation_covers_every_endpoint_schema_and_the_error_model() -> None:
    spec = build_openapi()
    text = render_markdown(spec)
    for path, item in spec["paths"].items():
        for method in item:
            assert f"### `{method.upper()} {path}`" in text
    for name in spec["components"]["schemas"]:
        if name not in apidoc.HIDDEN_SCHEMAS:
            assert f"### {name}" in text
    for fragment in ("X-API-Key", "Error model", "Retry-After", "Idempotency-Key", "ErrorResponse"):
        assert fragment in text
    assert "HTTPValidationError" not in text  # every 422 uses the service's own error model


def test_every_documented_operation_declares_its_error_responses() -> None:
    spec = build_openapi()
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            statuses = set(operation["responses"])
            if "security" in operation:
                assert {"401", "429"} <= statuses, (method, path)
            assert "422" not in statuses or (
                operation["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith(
                    "/ErrorResponse"
                )
            )


def test_check_mode_reports_stale_and_missing_documentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "docs" / "api.md"
    monkeypatch.setattr(apidoc, "DEFAULT_OUTPUT", target)
    assert main(["--check"]) == 1  # missing
    assert "out of date" in capsys.readouterr().err
    target.parent.mkdir()
    target.write_text("stale\n", encoding="utf-8")
    assert main(["--check"]) == 1
    assert main([]) == 0  # regenerate
    assert target.read_text(encoding="utf-8") == render_markdown(build_openapi())
    assert b"\r" not in target.read_bytes()
    assert main(["--check"]) == 0


def test_write_mode_creates_missing_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "a" / "b" / "api.md"
    monkeypatch.setattr(apidoc, "DEFAULT_OUTPUT", target)
    assert main([]) == 0
    assert target.exists()


def test_check_mode_tolerates_windows_checkouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "api.md"
    target.write_bytes(render_markdown(build_openapi()).replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(apidoc, "DEFAULT_OUTPUT", target)
    assert main(["--check"]) == 0


def test_numbers_in_constraints_are_rendered_as_integers() -> None:
    assert apidoc._number(60.0) == 60
    assert apidoc._number(0.5) == 0.5
    assert apidoc._number("x") == "x"
    assert "minimum=60.0" not in render_markdown(build_openapi())
