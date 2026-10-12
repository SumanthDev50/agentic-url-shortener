"""Link use cases: create / get / list / disable / resolve.

``resolve`` is the redirect hot path.  It consults an LRU+TTL cache of ``code -> target, expiry,
disabled`` (negative results are cached briefly), evaluates expiry against the clock on *every*
call (a cached entry never outlives a link's expiry), and falls back to the repository on a
miss.  ``disable`` invalidates the cache entry immediately; other instances learn about it after
the positive-cache TTL (documented revocation lag).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from shortener.clock import Clock
from shortener.domain.codegen import CodeGenerator
from shortener.domain.errors import (
    AliasTakenError,
    CodeGenerationError,
    LinkGoneError,
    LinkNotFoundError,
)
from shortener.domain.models import Link, LinkStatus, NewLink
from shortener.domain.validation import (
    UrlPolicy,
    is_safe_redirect_target,
    is_valid_code_format,
    resolve_expiry,
    validate_alias,
    validate_destination_url,
)
from shortener.observability.metrics import Metrics
from shortener.services.cache import LruTtlCache
from shortener.storage.link_repository import CodeTakenError, LinkRepository


@dataclass(frozen=True, slots=True)
class CachedTarget:
    """What the redirect path needs to know about a link."""

    link_id: int
    url: str
    expires_at: datetime | None
    disabled: bool


class _NotFound:
    """Marker for a negatively cached (unknown) code."""


NOT_FOUND = _NotFound()
RedirectCache = LruTtlCache[str, CachedTarget | _NotFound]


@dataclass(frozen=True, slots=True)
class ResolvedLink:
    """Result of resolving a short code for redirection."""

    link_id: int
    url: str


@dataclass(frozen=True, slots=True)
class LinkPage:
    items: Sequence[Link]
    total: int
    limit: int
    offset: int


class LinkService:
    def __init__(
        self,
        *,
        links: LinkRepository,
        codes: CodeGenerator,
        cache: RedirectCache,
        clock: Clock,
        url_policy: UrlPolicy,
        metrics: Metrics,
        negative_ttl: float,
    ) -> None:
        self._links = links
        self._codes = codes
        self._cache = cache
        self._clock = clock
        self._url_policy = url_policy
        self._metrics = metrics
        self._negative_ttl = negative_ttl

    # ------------------------------------------------------------------ commands

    def create(
        self,
        *,
        url: str,
        owner: str,
        alias: str | None = None,
        expires_at: datetime | None = None,
        ttl_seconds: int | None = None,
    ) -> Link:
        """Validate the request and persist a new link owned by ``owner``."""
        now = self._clock.now()
        destination = validate_destination_url(url, self._url_policy)
        expiry = resolve_expiry(now=now, expires_at=expires_at, ttl_seconds=ttl_seconds)
        if alias is not None:
            link = self._insert_alias(validate_alias(alias), destination, now, expiry, owner)
        else:
            link = self._insert_generated(destination, now, expiry, owner)
        self._cache.invalidate(link.code)  # drop a negative entry created by an earlier probe
        return link

    def disable(self, code: str, owner: str) -> None:
        """Soft-delete a link (idempotent) and revoke it from the local cache at once."""
        link = self.get(code, owner)
        self._links.disable(link.code, self._clock.now())
        self._cache.invalidate(link.code)

    # ------------------------------------------------------------------ queries

    def get(self, code: str, owner: str) -> Link:
        """The owner's link; other owners' links are indistinguishable from missing ones."""
        link = self._links.get_by_code(code) if is_valid_code_format(code) else None
        if link is None or link.created_by != owner:
            raise LinkNotFoundError
        return link

    def list(self, owner: str, limit: int, offset: int) -> LinkPage:
        items = self._links.list_by_owner(owner, limit, offset)
        return LinkPage(
            items=items, total=self._links.count_by_owner(owner), limit=limit, offset=offset
        )

    def resolve(self, code: str) -> ResolvedLink:
        """Redirect hot path: 404 unknown, 410 expired/disabled, else the target."""
        if not is_valid_code_format(code):
            raise LinkNotFoundError
        target = self._lookup(code)
        if target.disabled:
            raise LinkGoneError(LinkStatus.DISABLED, "The link has been disabled.")
        if target.expires_at is not None and self._clock.now() >= target.expires_at:
            raise LinkGoneError(LinkStatus.EXPIRED, "The link has expired.")
        return ResolvedLink(link_id=target.link_id, url=target.url)

    def status_of(self, link: Link) -> LinkStatus:
        return link.status_at(self._clock.now())

    # ------------------------------------------------------------------ internals

    def _lookup(self, code: str) -> CachedTarget:
        cached = self._cache.get(code)
        if isinstance(cached, _NotFound):
            self._metrics.inc("shortener_redirect_cache_total", result="negative_hit")
            raise LinkNotFoundError
        if cached is not None:
            self._metrics.inc("shortener_redirect_cache_total", result="hit")
            return cached
        self._metrics.inc("shortener_redirect_cache_total", result="miss")
        link = self._links.get_by_code(code)
        if link is None or not is_safe_redirect_target(link.original_url):
            self._cache.put(code, NOT_FOUND, ttl=self._negative_ttl)
            raise LinkNotFoundError
        target = CachedTarget(
            link_id=link.id,
            url=link.original_url,
            expires_at=link.expires_at,
            disabled=link.disabled_at is not None,
        )
        self._cache.put(code, target)
        return target

    def _insert_alias(
        self, alias: str, url: str, now: datetime, expiry: datetime | None, owner: str
    ) -> Link:
        try:
            return self._links.insert(NewLink(alias, url, now, expiry, owner))
        except CodeTakenError:
            raise AliasTakenError from None

    def _insert_generated(
        self, url: str, now: datetime, expiry: datetime | None, owner: str
    ) -> Link:
        """Try bounded, unique-constraint-checked candidates; never loop without limit."""
        for code in self._codes.candidates():
            try:
                return self._links.insert(NewLink(code, url, now, expiry, owner))
            except CodeTakenError:
                continue
        raise CodeGenerationError
