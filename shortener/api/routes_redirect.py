"""Public redirect endpoint: ``GET /{code}`` (FR-04).

This is the hot path and part of the public allow-list (no API key).  Hardening: 302 (so every
click is observable and links stay revocable), ``no-store`` caching, a fixed tiny body (nothing
from the request is reflected), per-IP rate limiting, and analytics that can never fail the
redirect.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from shortener.api.deps import client_address, get_services
from shortener.api.middleware import set_response_headers
from shortener.api.schemas import error_responses
from shortener.container import Services

router = APIRouter(tags=["redirect"])

REDIRECT_BODY = b"Redirecting"
_REDIRECT_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


@router.get(
    "/{code}",
    status_code=302,
    summary="Follow a short link",
    description=(
        "Redirects to the destination with a 302. Unknown codes answer 404; expired or "
        "disabled links answer 410. Rate limited per client address. No API key required."
    ),
    responses={
        302: {
            "description": "Redirect to the destination.",
            "headers": {
                "Location": {
                    "description": "Canonical destination URL.",
                    "schema": {"type": "string"},
                }
            },
        },
        **error_responses(404, 410, 422, 429),
    },
)
def follow_link(
    code: str,
    request: Request,
    services: Annotated[Services, Depends(get_services)],
) -> Response:
    client = client_address(request, services)
    decision = services.redirect_limiter.acquire(client)
    if not decision.allowed:
        services.metrics.inc("shortener_rate_limit_rejections_total", scope="redirect")
        raise decision.to_error()
    set_response_headers(request.scope, decision.headers())

    resolved = services.links.resolve(code)
    services.analytics.record_click(
        link_id=resolved.link_id,
        client_ip=client,
        user_agent=request.headers.get("user-agent"),
        referer=request.headers.get("referer"),
    )
    return Response(
        content=REDIRECT_BODY,
        status_code=302,
        media_type="text/plain",
        headers={"Location": resolved.url, **_REDIRECT_HEADERS},
    )
