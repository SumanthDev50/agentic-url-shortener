"""Click analytics: privacy-reducing ingestion and aggregation (FR-08, NFR-01, NFR-08).

``record_click`` runs on the redirect path and therefore must never raise or block: it reduces
the request to a :class:`ClickEvent` (no raw IP, only UA family and referrer scheme+host) and
hands it to the bounded :class:`ClickRecorder`.  Any unexpected failure is counted and logged
(without request data) and swallowed.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta

from shortener.clock import Clock
from shortener.domain.errors import InvalidInputError
from shortener.domain.models import (
    AnalyticsReport,
    ClickEvent,
    DailyClicks,
    Link,
)
from shortener.domain.privacy import referrer_origin, visitor_id
from shortener.domain.useragent import classify_user_agent
from shortener.observability.metrics import Metrics
from shortener.services.click_recorder import ClickRecorder
from shortener.storage.click_repository import ClickRepository

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 30
MIN_DAYS = 1
MAX_DAYS = 365
TOP_REFERRERS = 10
BREAKDOWN_LIMIT = 10


class AnalyticsService:
    def __init__(
        self,
        *,
        clicks: ClickRepository,
        recorder: ClickRecorder,
        clock: Clock,
        server_secret: bytes,
        metrics: Metrics,
        retention_days: int,
    ) -> None:
        self._clicks = clicks
        self._recorder = recorder
        self._clock = clock
        self._server_secret = server_secret
        self._metrics = metrics
        self._retention_days = retention_days

    def record_click(
        self,
        *,
        link_id: int,
        client_ip: str | None,
        user_agent: str | None,
        referer: str | None,
    ) -> None:
        """Queue one click; never raises (a redirect must not fail because analytics did)."""
        try:
            now = self._clock.now()
            browser, device = classify_user_agent(user_agent)
            event = ClickEvent(
                link_id=link_id,
                clicked_at=now,
                referrer_host=referrer_origin(referer),
                browser=browser,
                device=device,
                visitor_hash=visitor_id(self._server_secret, now.date(), client_ip, browser),
            )
            self._recorder.submit(event)
        except Exception:
            self._metrics.inc("shortener_analytics_errors_total")
            logger.exception("click_record_failed")

    def report(self, link: Link, days: int = DEFAULT_DAYS) -> AnalyticsReport:
        """Aggregates for the last ``days`` UTC calendar days (today included)."""
        if not MIN_DAYS <= days <= MAX_DAYS:
            raise InvalidInputError(f"days must be between {MIN_DAYS} and {MAX_DAYS}.")
        first_day = self._clock.now().date() - timedelta(days=days - 1)
        since = datetime.combine(first_day, time.min, tzinfo=UTC)
        per_day = self._clicks.daily_counts(link.id, since)
        return AnalyticsReport(
            days=days,
            summary=self._clicks.summary(link.id, since),
            clicks_by_day=_zero_filled(first_day, days, per_day),
            top_referrers=tuple(self._clicks.top_referrers(link.id, since, TOP_REFERRERS)),
            browsers=tuple(self._clicks.breakdown("browser", link.id, since, BREAKDOWN_LIMIT)),
            devices=tuple(self._clicks.breakdown("device", link.id, since, BREAKDOWN_LIMIT)),
        )

    def purge_expired(self) -> int:
        """Delete clicks older than the retention window; returns the number removed."""
        cutoff = self._clock.now() - timedelta(days=self._retention_days)
        return self._clicks.delete_before(cutoff)


def _zero_filled(first_day: date, days: int, per_day: dict[date, int]) -> tuple[DailyClicks, ...]:
    return tuple(
        DailyClicks(day=day, clicks=per_day.get(day, 0))
        for day in (first_day + timedelta(days=offset) for offset in range(days))
    )
