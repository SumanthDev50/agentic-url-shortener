"""Pydantic request/response models (the public API contract).

Request models are strict: unknown fields are rejected (``extra="forbid"``) and every field is
length/range constrained, so hostile payloads are stopped at the boundary before any business
logic runs.  Business rules that cannot be expressed declaratively live in
``shortener.domain.validation``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shortener.domain.models import AnalyticsReport, Link, LinkStatus, NamedCount
from shortener.domain.validation import (
    MAX_TTL_SECONDS,
    MAX_URL_LENGTH,
    MIN_TTL_SECONDS,
)

MAX_EXPIRES_AT_LENGTH = 64


class CreateLinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        min_length=1,
        max_length=MAX_URL_LENGTH,
        description=(
            "Destination. Only http/https; no credentials; loopback, private, link-local and "
            "other non-public addresses and internal host names are rejected. The canonical "
            "re-serialised form is stored and redirected to."
        ),
        examples=["https://example.com/docs?page=1"],
    )
    alias: str | None = Field(
        default=None,
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Custom code (case-sensitive). Reserved words are rejected; clash gives 409.",
        examples=["spring-sale"],
    )
    expires_at: datetime | None = Field(
        default=None,
        description=(
            "Absolute expiry: ISO-8601 with a UTC offset, in the future, at most five years "
            "ahead. Mutually exclusive with ttl_seconds."
        ),
        examples=["2030-01-01T00:00:00Z"],
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=MIN_TTL_SECONDS,
        le=MAX_TTL_SECONDS,
        strict=True,
        description="Relative expiry in seconds. Mutually exclusive with expires_at.",
        examples=[3600],
    )

    @field_validator("expires_at", mode="before")
    @classmethod
    def _parse_expires_at(cls, value: Any) -> datetime | None:
        """Accept only ISO-8601 strings (not epoch numbers, which would be ambiguous)."""
        if value is None:
            return None
        if not isinstance(value, str) or len(value) > MAX_EXPIRES_AT_LENGTH:
            raise ValueError("expires_at must be an ISO-8601 date-time string")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("expires_at must be an ISO-8601 date-time with a UTC offset") from None


class LinkResponse(BaseModel):
    code: str = Field(description="The short code (path segment of the short URL).")
    short_url: str = Field(description="Absolute short URL (the configured base URL + code).")
    original_url: str = Field(description="Canonical destination the short URL redirects to.")
    created_at: datetime
    expires_at: datetime | None = Field(description="Expiry instant, or null for none.")
    status: LinkStatus = Field(description="active, expired or disabled.")
    click_count: int = Field(
        description="Recorded clicks. Eventually consistent: updated by a background writer."
    )

    @classmethod
    def from_link(cls, link: Link, *, base_url: str, status: LinkStatus) -> LinkResponse:
        return cls(
            code=link.code,
            short_url=f"{base_url}/{link.code}",
            original_url=link.original_url,
            created_at=link.created_at,
            expires_at=link.expires_at,
            status=status,
            click_count=link.click_count,
        )


class LinkListResponse(BaseModel):
    items: list[LinkResponse]
    total: int = Field(description="Total links owned by the caller.")
    limit: int
    offset: int


class DayClicks(BaseModel):
    date: date
    clicks: int


class CountItem(BaseModel):
    name: str
    clicks: int

    @classmethod
    def from_named(cls, item: NamedCount) -> CountItem:
        return cls(name=item.name, clicks=item.clicks)


class AnalyticsResponse(BaseModel):
    code: str
    days: int = Field(description="Window length: the last N UTC calendar days, today included.")
    total_clicks: int
    unique_visitors: int = Field(
        description="Distinct daily-salted visitor hashes (a visitor on two days counts twice)."
    )
    first_click_at: datetime | None
    last_click_at: datetime | None
    clicks_by_day: list[DayClicks] = Field(description="One entry per day, zero-filled.")
    top_referrers: list[CountItem] = Field(
        description="Referrer scheme://host, most frequent first."
    )
    browsers: list[CountItem]
    devices: list[CountItem]

    @classmethod
    def from_report(cls, code: str, report: AnalyticsReport) -> AnalyticsResponse:
        return cls(
            code=code,
            days=report.days,
            total_clicks=report.summary.total_clicks,
            unique_visitors=report.summary.unique_visitors,
            first_click_at=report.summary.first_click_at,
            last_click_at=report.summary.last_click_at,
            clicks_by_day=[DayClicks(date=d.day, clicks=d.clicks) for d in report.clicks_by_day],
            top_referrers=[CountItem.from_named(item) for item in report.top_referrers],
            browsers=[CountItem.from_named(item) for item in report.browsers],
            devices=[CountItem.from_named(item) for item in report.devices],
        )


class ErrorDetail(BaseModel):
    location: str = Field(description="Dot-separated path of the offending field.")
    message: str


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable, static message (never reflects input).")
    request_id: str | None = Field(description="Correlates with the X-Request-ID header and logs.")
    details: list[ErrorDetail] | None = Field(
        default=None, description="Field-level problems (validation errors only)."
    )


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    auth: Literal["enabled", "disabled"]


class ReadyResponse(BaseModel):
    status: Literal["ready"]


ErrorSpec = dict[int | str, dict[str, Any]]

_ERROR_DESCRIPTIONS = {
    400: "Malformed request.",
    401: "Missing or invalid API key.",
    404: "Not found (also returned for links owned by another key).",
    409: "Conflict.",
    410: "The link is expired or disabled.",
    413: "Request body too large.",
    422: "Validation failed.",
    429: "Rate limit exceeded; see Retry-After and X-RateLimit-* headers.",
    503: "Temporarily unavailable; see Retry-After where present.",
}


def error_responses(*statuses: int) -> ErrorSpec:
    """OpenAPI ``responses`` entries using the common error model."""
    return {
        status: {"model": ErrorResponse, "description": _ERROR_DESCRIPTIONS[status]}
        for status in statuses
    }


#: Errors every authenticated management endpoint can produce.
AUTHENTICATED_ERRORS: ErrorSpec = error_responses(401, 429)
