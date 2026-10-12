"""Application factory.

``create_app(settings=None)`` builds the FastAPI app without side effects (no files, no threads):
the database is migrated and the background workers are started by the lifespan.  Run it with
``python -m shortener`` or ``uvicorn shortener.main:create_app --factory``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from shortener import __version__
from shortener.api import routes_health, routes_links, routes_redirect
from shortener.api.errors import install_error_handlers
from shortener.api.middleware import (
    AccessLogMiddleware,
    BodySizeLimitMiddleware,
    ErrorBoundaryMiddleware,
    RequestIdMiddleware,
    ResponseHeadersMiddleware,
)
from shortener.clock import Clock
from shortener.config import Settings, load_settings
from shortener.container import Services, build_services
from shortener.domain.codegen import CodeGenerator
from shortener.observability.logging import configure_logging

DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})

DESCRIPTION = (
    "URL shortener with custom aliases, expiry, idempotent creation, per-key ownership, "
    "privacy-preserving analytics, rate limiting and operational endpoints. "
    "All `/api/v1` endpoints require an `X-API-Key` header."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: migrate + start workers (refuses to start on schema inconsistency)."""
    services: Services = app.state.services
    try:
        services.start()
        yield
    finally:
        services.stop()


def create_app(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    code_generator: CodeGenerator | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FastAPI:
    """Build the application.  ``clock``, ``code_generator`` and ``sleep`` are test seams."""
    settings = settings if settings is not None else load_settings()
    configure_logging(settings.log_level)
    services = build_services(settings, clock=clock, code_generator=code_generator, sleep=sleep)

    docs: dict[str, Any] = (
        {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
        if settings.enable_docs
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(
        title="URL Shortener",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        **docs,
    )
    app.state.services = services
    install_error_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_links.router)
    app.include_router(routes_redirect.router)  # catch-all `/{code}`: must stay last
    _install_middleware(app, services)
    return app


def _install_middleware(app: FastAPI, services: Services) -> None:
    """Add middleware innermost-first (``add_middleware`` makes the last one outermost)."""
    settings = services.settings
    app.add_middleware(ErrorBoundaryMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(
        ResponseHeadersMiddleware,
        hsts=settings.base_url.startswith("https://"),
        relaxed_paths=DOCS_PATHS,
    )
    app.add_middleware(AccessLogMiddleware, metrics=services.metrics)
    app.add_middleware(RequestIdMiddleware)


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Lazily build ``shortener.main.app`` so importing the module has no side effects."""
    global _app
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _app is None:
        _app = create_app()
    return _app
