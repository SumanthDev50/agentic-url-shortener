"""JSON logging (redaction, request id, injection safety) and the Prometheus metrics registry."""

from __future__ import annotations

import io
import json
import logging

import pytest

from shortener.observability.logging import (
    REDACTED,
    JsonFormatter,
    configure_logging,
    is_sensitive_key,
    log_event,
    redact,
    request_id_var,
    scrub_text,
)
from shortener.observability.metrics import CATALOGUE, Metrics
from tests.shortener.support import new_secret


def render(record: logging.LogRecord) -> dict[str, object]:
    line = JsonFormatter().format(record)
    assert "\n" not in line  # exactly one physical line per record
    return json.loads(line)  # type: ignore[no-any-return]


def make_record(message: str, **fields: object) -> logging.LogRecord:
    logger = logging.getLogger("shortener.test")
    return logger.makeRecord(
        "shortener.test", logging.INFO, __file__, 1, message, (), None, extra={"fields": fields}
    )


# ----------------------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    "key",
    ["X-API-Key", "x-api-key", "Authorization", "AUTHORIZATION", "api_key", "Cookie", "token"],
)
def test_sensitive_keys_are_recognised_case_insensitively(key: str) -> None:
    assert is_sensitive_key(key)


def test_non_sensitive_keys_are_not_redacted() -> None:
    assert not is_sensitive_key("route")
    assert not is_sensitive_key("status")


def test_structured_fields_with_sensitive_keys_are_redacted_at_any_depth() -> None:
    secret = new_secret()
    payload = render(
        make_record(
            "request",
            headers={
                "X-API-Key": secret,
                "Accept": "*/*",
                "nested": [{"Authorization": f"Bearer {secret}"}],
            },
            api_key=secret,
        )
    )
    text = json.dumps(payload)
    assert secret not in text
    assert payload["api_key"] == REDACTED
    headers = payload["headers"]
    assert isinstance(headers, dict)
    assert headers["X-API-Key"] == REDACTED
    assert headers["Accept"] == "*/*"
    assert headers["nested"] == [{"Authorization": REDACTED}]


def test_header_fragments_in_free_text_are_scrubbed() -> None:
    secret = new_secret()
    messages = [
        f"failed with X-API-Key: {secret}",
        f"curl -H 'x-api-key={secret}'",
        f"Authorization: Bearer {secret}",
        f"authorization = Basic {secret}",
        f"Cookie: session={secret}",
    ]
    for message in messages:
        scrubbed = scrub_text(message)
        assert secret not in scrubbed
        assert REDACTED in scrubbed
    assert scrub_text("nothing sensitive here") == "nothing sensitive here"


def test_exception_tracebacks_are_scrubbed() -> None:
    secret = new_secret()
    try:
        raise RuntimeError(f"upstream rejected X-API-Key: {secret}")
    except RuntimeError:
        import sys

        record = logging.getLogger("shortener.test").makeRecord(
            "shortener.test", logging.ERROR, __file__, 1, "boom", (), sys.exc_info()
        )
    payload = render(record)
    assert secret not in json.dumps(payload)
    assert "RuntimeError" in str(payload["exception"])


def test_redact_handles_non_container_values() -> None:
    assert redact(5) == 5
    assert redact(None) is None
    assert redact(("a", {"password": "x"})) == ["a", {"password": REDACTED}]


# ----------------------------------------------------------------------------- formatting


def test_records_include_request_id_and_extra_fields() -> None:
    token = request_id_var.set("req-12345678")
    try:
        payload = render(make_record("hello", status=200, route="/x"))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-12345678"
    assert (payload["event"], payload["level"], payload["status"], payload["route"]) == (
        "hello",
        "INFO",
        200,
        "/x",
    )
    assert str(payload["ts"]).endswith("+00:00")


def test_extra_fields_cannot_overwrite_reserved_keys() -> None:
    payload = render(make_record("real", event="forged", level="DEBUG", request_id="forged"))
    assert payload["event"] == "real" and payload["level"] == "INFO"
    assert payload["request_id"] is None


def test_newlines_and_control_characters_cannot_forge_log_lines() -> None:
    record = make_record('user input\n{"event": "forged"}\r\x1b[31m', note="a\nb")
    line = JsonFormatter().format(record)
    assert "\n" not in line and "\r" not in line and "\x1b" not in line
    assert json.loads(line)["event"].startswith("user input")


def test_non_serialisable_values_do_not_break_logging() -> None:
    payload = render(make_record("odd", value=object(), items={1, 2}))
    assert "object" in str(payload["value"])


def test_log_event_helper_attaches_fields() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("shortener.helper-test")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        log_event(logger, logging.INFO, "thing_happened", count=3)
    finally:
        logger.removeHandler(handler)
    assert json.loads(stream.getvalue())["count"] == 3


def test_configure_logging_is_idempotent_and_writes_json() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream)
    configure_logging("WARNING", stream)  # replaces rather than duplicates the handler
    logger = logging.getLogger("shortener")
    managed = [h for h in logger.handlers if getattr(h, "_shortener_json_handler", False)]
    assert len(managed) == 1
    logging.getLogger("shortener.configured").warning("visible")
    logging.getLogger("shortener.configured").info("hidden")
    lines = stream.getvalue().strip().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["visible"]
    logger.removeHandler(managed[0])
    uvicorn_logger = logging.getLogger("uvicorn.error")
    for handler in [
        h for h in uvicorn_logger.handlers if getattr(h, "_shortener_json_handler", False)
    ]:
        uvicorn_logger.removeHandler(handler)
    uvicorn_logger.propagate = True
    configure_logging("INFO")  # restore the default level for other tests
    for name in ("shortener", "uvicorn.error"):
        for handler in [
            h
            for h in logging.getLogger(name).handlers
            if getattr(h, "_shortener_json_handler", False)
        ]:
            logging.getLogger(name).removeHandler(handler)
    uvicorn_logger.propagate = True


# ----------------------------------------------------------------------------- metrics


def test_counters_labels_and_rendering() -> None:
    metrics = Metrics()
    metrics.inc("shortener_http_requests_total", method="GET", route="/{code}", status="302")
    metrics.inc("shortener_http_requests_total", method="GET", route="/{code}", status="302")
    metrics.inc("shortener_http_requests_total", method="POST", route="/api/v1/links", status="201")
    metrics.inc("shortener_links_created_total", 3)
    text = metrics.render()
    assert 'shortener_http_requests_total{method="GET",route="/{code}",status="302"} 2' in text
    assert (
        'shortener_http_requests_total{method="POST",route="/api/v1/links",status="201"} 1' in text
    )
    assert "shortener_links_created_total 3" in text
    assert "# HELP shortener_links_created_total Short links created." in text
    assert "# TYPE shortener_click_queue_depth gauge" in text
    assert text.endswith("\n")


def test_unused_label_free_counters_render_zero_and_labelled_ones_render_no_samples() -> None:
    text = Metrics().render()
    assert "shortener_clicks_written_total 0" in text
    assert "shortener_auth_failures_total 0" in text
    assert not [
        line for line in text.splitlines() if line.startswith("shortener_http_requests_total")
    ]


def test_every_catalogue_entry_is_rendered_with_help_and_type() -> None:
    text = Metrics().render()
    for name, spec in CATALOGUE.items():
        assert f"# HELP {name} " in text
        assert f"# TYPE {name} {spec.kind}" in text


def test_label_values_are_escaped() -> None:
    metrics = Metrics()
    metrics.inc("shortener_redirect_cache_total", result='a"b\\c\nd')
    assert 'result="a\\"b\\\\c\\nd"' in metrics.render()


def test_gauges_are_sampled_at_render_time() -> None:
    metrics = Metrics()
    depth = {"value": 4.0}
    metrics.register_gauge("shortener_click_queue_depth", lambda: depth["value"])
    assert "shortener_click_queue_depth 4" in metrics.render()
    depth["value"] = 2.5
    assert "shortener_click_queue_depth 2.5" in metrics.render()
    assert metrics.value("shortener_click_queue_depth") == 2.5


def test_unknown_metrics_and_non_gauge_registration_are_programming_errors() -> None:
    metrics = Metrics()
    with pytest.raises(ValueError, match="unknown metric"):
        metrics.inc("not_a_metric")
    with pytest.raises(ValueError, match="not a gauge"):
        metrics.register_gauge("shortener_links_created_total", lambda: 1.0)
    with pytest.raises(ValueError, match="not a gauge"):
        metrics.register_gauge("missing", lambda: 1.0)


def test_series_count_is_capped_against_label_explosion() -> None:
    metrics = Metrics(max_series=10)
    for index in range(1000):
        metrics.inc("shortener_http_requests_total", method="GET", route=f"/r{index}", status="200")
    series = [
        line
        for line in metrics.render().splitlines()
        if line.startswith("shortener_http_requests_total{")
    ]
    assert len(series) <= 10
    metrics.inc(
        "shortener_http_requests_total", method="GET", route="/r0", status="200"
    )  # existing still counts
    assert (
        metrics.value("shortener_http_requests_total", method="GET", route="/r0", status="200") == 2
    )
