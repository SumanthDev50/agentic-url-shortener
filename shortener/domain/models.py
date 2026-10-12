"""Domain dataclasses (plain data, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class LinkStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class NewLink:
    """A link about to be inserted (the repository assigns the id)."""

    code: str
    original_url: str
    created_at: datetime
    expires_at: datetime | None
    created_by: str | None


@dataclass(frozen=True, slots=True)
class Link:
    id: int
    code: str
    original_url: str
    created_at: datetime
    expires_at: datetime | None
    disabled_at: datetime | None
    click_count: int
    created_by: str | None

    def status_at(self, now: datetime) -> LinkStatus:
        """Disabled wins over expired; expiry is inclusive of the expiry instant."""
        if self.disabled_at is not None:
            return LinkStatus.DISABLED
        if self.expires_at is not None and now >= self.expires_at:
            return LinkStatus.EXPIRED
        return LinkStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ClickEvent:
    """One privacy-reduced click (no raw IP, no full user agent, no referrer path)."""

    link_id: int
    clicked_at: datetime
    referrer_host: str | None
    browser: str
    device: str
    visitor_hash: str


@dataclass(frozen=True, slots=True)
class ClickSummary:
    total_clicks: int
    unique_visitors: int
    first_click_at: datetime | None
    last_click_at: datetime | None


@dataclass(frozen=True, slots=True)
class DailyClicks:
    day: date
    clicks: int


@dataclass(frozen=True, slots=True)
class NamedCount:
    name: str
    clicks: int


@dataclass(frozen=True, slots=True)
class AnalyticsReport:
    """Aggregates over the last ``days`` UTC calendar days (today included)."""

    days: int
    summary: ClickSummary
    clicks_by_day: tuple[DailyClicks, ...]
    top_referrers: tuple[NamedCount, ...]
    browsers: tuple[NamedCount, ...]
    devices: tuple[NamedCount, ...]
