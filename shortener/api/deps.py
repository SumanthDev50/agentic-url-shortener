"""Dependency providers: service container access, client identity and API-key auth."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from shortener.api.middleware import set_response_headers
from shortener.container import Services
from shortener.services.auth import Principal

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    auto_error=False,  # we raise our own 401 so the error model stays consistent
    description="API key for the management API (configured via SHORTENER_API_KEYS).",
)


def get_services(request: Request) -> Services:
    return cast(Services, request.app.state.services)


def client_address(request: Request, services: Services) -> str:
    """Resolved client address (trust boundary for ``X-Forwarded-For`` applied)."""
    peer = request.client.host if request.client else None
    return services.client_ips.resolve(peer, request.headers.get("x-forwarded-for"))


def require_api_key(
    request: Request,
    services: Annotated[Services, Depends(get_services)],
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> Principal:
    """Authenticate the caller, apply the per-key rate limit and return the owner identity.

    Raises 401 (``unauthorized``) or 429 (``rate_limited``).  Rate-limit headers for the
    successful case are attached to the response by ``ResponseHeadersMiddleware``.
    """
    outcome = services.auth.authenticate(api_key, client_address(request, services))
    set_response_headers(request.scope, outcome.rate_limit.headers())
    return outcome.principal
