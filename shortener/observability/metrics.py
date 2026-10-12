"""Thread-safe counters/gauges and a Prometheus text renderer (NFR-05).

Metric names come from a fixed catalogue (typos fail fast) and the number of distinct label
combinations is capped, so attacker-chosen label values can never exhaust memory.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal, NamedTuple

LabelKey = tuple[tuple[str, str], ...]


class MetricSpec(NamedTuple):
    kind: Literal["counter", "gauge"]
    help: str
    labelled: bool = False


#: Order defines the output order.
CATALOGUE: dict[str, MetricSpec] = {
    "shortener_http_requests_total": MetricSpec(
        "counter", "HTTP requests by method, route and status.", labelled=True
    ),
    "shortener_redirect_cache_total": MetricSpec(
        "counter", "Redirect cache lookups by result (hit, miss, negative_hit).", labelled=True
    ),
    "shortener_links_created_total": MetricSpec("counter", "Short links created."),
    "shortener_idempotent_replays_total": MetricSpec(
        "counter", "Creations answered from a stored idempotency key."
    ),
    "shortener_click_queue_depth": MetricSpec(
        "gauge", "Click events waiting for the background writer."
    ),
    "shortener_clicks_enqueued_total": MetricSpec("counter", "Click events accepted."),
    "shortener_clicks_dropped_total": MetricSpec(
        "counter", "Click events dropped, by reason.", labelled=True
    ),
    "shortener_clicks_written_total": MetricSpec("counter", "Click events persisted."),
    "shortener_analytics_errors_total": MetricSpec(
        "counter", "Click recording failures (never fatal to a redirect)."
    ),
    "shortener_rate_limit_rejections_total": MetricSpec(
        "counter", "Requests rejected by rate limiting, by scope.", labelled=True
    ),
    "shortener_auth_failures_total": MetricSpec("counter", "Rejected authentication attempts."),
}

DEFAULT_MAX_SERIES = 2000


class Metrics:
    """Process-local metric registry."""

    def __init__(self, *, max_series: int = DEFAULT_MAX_SERIES) -> None:
        self._lock = threading.Lock()
        self._max_series = max_series
        self._values: dict[str, dict[LabelKey, float]] = {name: {} for name in CATALOGUE}
        self._suppliers: dict[str, Callable[[], float]] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add ``value`` to a counter series (dropped silently once the series cap is hit)."""
        key = self._label_key(name, labels)
        with self._lock:
            series = self._values[name]
            if key in series:
                series[key] += value
            elif self._series_count() < self._max_series:
                series[key] = value

    def register_gauge(self, name: str, supplier: Callable[[], float]) -> None:
        """Sample a label-less gauge from ``supplier`` at render time."""
        spec = CATALOGUE.get(name)
        if spec is None or spec.kind != "gauge":
            raise ValueError(f"{name} is not a gauge in the catalogue")
        self._suppliers[name] = supplier

    def value(self, name: str, **labels: str) -> float:
        """Current value of one series (0 if never touched); for tests and diagnostics."""
        key = self._label_key(name, labels)
        supplier = self._suppliers.get(name)
        if supplier is not None:
            return supplier()
        with self._lock:
            return self._values[name].get(key, 0.0)

    def render(self) -> str:
        """Prometheus text exposition format 0.0.4."""
        with self._lock:
            snapshot = {name: dict(series) for name, series in self._values.items()}
        lines: list[str] = []
        for name, spec in CATALOGUE.items():
            lines.append(f"# HELP {name} {spec.help}")
            lines.append(f"# TYPE {name} {spec.kind}")
            lines.extend(self._samples(name, spec, snapshot[name]))
        return "\n".join(lines) + "\n"

    def _samples(self, name: str, spec: MetricSpec, series: dict[LabelKey, float]) -> list[str]:
        supplier = self._suppliers.get(name)
        if supplier is not None:
            return [f"{name} {_format_number(supplier())}"]
        if not series and not spec.labelled:
            return [f"{name} 0"]
        return [
            f"{name}{_format_labels(key)} {_format_number(value)}"
            for key, value in sorted(series.items())
        ]

    @staticmethod
    def _label_key(name: str, labels: dict[str, str]) -> LabelKey:
        if name not in CATALOGUE:
            raise ValueError(f"unknown metric {name}")
        return tuple(sorted((key, str(value)) for key, value in labels.items()))

    def _series_count(self) -> int:
        return sum(len(series) for series in self._values.values())


def _format_labels(key: LabelKey) -> str:
    if not key:
        return ""
    body = ",".join(f'{name}="{_escape_label(value)}"' for name, value in key)
    return "{" + body + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))
