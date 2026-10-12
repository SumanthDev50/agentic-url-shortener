"""Composition root: wires settings, storage, services and background workers.

Nothing here touches the filesystem or starts a thread until :meth:`Services.start` runs (from
the application lifespan), so ``create_app()`` stays cheap and side-effect free - which is what
lets ``create_app(...).openapi()`` generate API documentation without a database.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from shortener.clock import Clock, SystemClock
from shortener.config import RateLimit, Settings
from shortener.domain.codegen import CodeGenerator
from shortener.observability.metrics import Metrics
from shortener.services.analytics_service import AnalyticsService
from shortener.services.auth import ApiKeyAuthenticator, AuthService
from shortener.services.cache import LruTtlCache
from shortener.services.click_recorder import ClickRecorder, ClickWriter
from shortener.services.client_ip import ClientIpResolver
from shortener.services.idempotency import IdempotencyService
from shortener.services.link_service import LinkService, RedirectCache
from shortener.services.maintenance import MaintenanceJob, MaintenanceRunner
from shortener.services.rate_limiter import TokenBucketLimiter
from shortener.storage.click_repository import SqliteClickRepository
from shortener.storage.db import Database
from shortener.storage.idempotency_repository import SqliteIdempotencyRepository
from shortener.storage.link_repository import SqliteLinkRepository
from shortener.storage.migrations import migrate

logger = logging.getLogger(__name__)

ANALYTICS_SECRET_BYTES = 32


@dataclass(slots=True)
class Services:
    settings: Settings
    clock: Clock
    metrics: Metrics
    database: Database
    links: LinkService
    analytics: AnalyticsService
    idempotency: IdempotencyService
    auth: AuthService
    redirect_limiter: TokenBucketLimiter
    client_ips: ClientIpResolver
    recorder: ClickRecorder
    maintenance: MaintenanceRunner
    ephemeral_analytics_secret: bool

    def start(self) -> None:
        """Open the database, migrate (refusing to start on inconsistency), start workers."""
        self.database.open()
        applied = migrate(self.database)
        if applied:
            logger.info("migrations_applied", extra={"fields": {"versions": applied}})
        if self.settings.auth_disabled:
            logger.warning(
                "authentication_disabled",
                extra={"fields": {"detail": "management API is open; loopback bind only"}},
            )
        if self.ephemeral_analytics_secret:
            logger.warning(
                "analytics_secret_ephemeral",
                extra={
                    "fields": {
                        "detail": "SHORTENER_ANALYTICS_SECRET unset; visitor ids reset on restart"
                    }
                },
            )
        self.maintenance.start()
        self.recorder.start()

    def stop(self) -> None:
        """Flush queued clicks, stop workers and close every database connection."""
        self.recorder.stop()
        self.maintenance.stop()
        self.database.close()

    def is_ready(self) -> bool:
        return self.database.ping()


def build_services(
    settings: Settings,
    *,
    clock: Clock | None = None,
    code_generator: CodeGenerator | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Services:
    """Construct the object graph (``clock``, ``code_generator`` and ``sleep`` are test seams)."""
    clock = clock or SystemClock()
    metrics = Metrics()
    database = Database(settings.database_path)
    link_repository = SqliteLinkRepository(database)
    click_repository = SqliteClickRepository(database)

    cache: RedirectCache = LruTtlCache(
        max_entries=settings.cache_max_entries,
        default_ttl=settings.cache_ttl_seconds,
        clock=clock,
    )
    links = LinkService(
        links=link_repository,
        codes=code_generator or CodeGenerator(),
        cache=cache,
        clock=clock,
        url_policy=settings.url_policy,
        metrics=metrics,
        negative_ttl=settings.negative_cache_ttl_seconds,
    )
    writer = ClickWriter(click_repository, link_repository, database)
    recorder = ClickRecorder(
        writer.write,
        metrics=metrics,
        capacity=settings.click_queue_size,
        batch_size=settings.click_batch_size,
    )
    metrics.register_gauge("shortener_click_queue_depth", lambda: float(recorder.depth))
    analytics_secret = settings.analytics_secret
    analytics = AnalyticsService(
        clicks=click_repository,
        recorder=recorder,
        clock=clock,
        server_secret=analytics_secret or secrets.token_bytes(ANALYTICS_SECRET_BYTES),
        metrics=metrics,
        retention_days=settings.click_retention_days,
    )
    idempotency = IdempotencyService(
        repository=SqliteIdempotencyRepository(database),
        transactions=database,
        clock=clock,
        metrics=metrics,
    )
    auth = AuthService(
        authenticator=ApiKeyAuthenticator([(key.key_id, key.digest) for key in settings.api_keys]),
        disabled=settings.auth_disabled,
        api_limiter=_limiter(settings.api_rate, settings, clock),
        failure_limiter=_limiter(settings.auth_failure_rate, settings, clock),
        metrics=metrics,
        failure_delay_seconds=settings.auth_failure_delay_ms / 1000,
        sleep=sleep,
    )
    maintenance = MaintenanceRunner(
        [
            MaintenanceJob("click_retention", analytics.purge_expired),
            MaintenanceJob("idempotency_keys", idempotency.purge_expired),
        ]
    )
    return Services(
        settings=settings,
        clock=clock,
        metrics=metrics,
        database=database,
        links=links,
        analytics=analytics,
        idempotency=idempotency,
        auth=auth,
        redirect_limiter=_limiter(settings.redirect_rate, settings, clock),
        client_ips=ClientIpResolver(settings.trusted_proxies),
        recorder=recorder,
        maintenance=maintenance,
        ephemeral_analytics_secret=analytics_secret is None,
    )


def _limiter(rate: RateLimit, settings: Settings, clock: Clock) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        capacity=rate.burst,
        refill_per_second=rate.refill_per_second,
        clock=clock,
        max_entries=settings.rate_limit_max_keys,
    )
