"""Operational endpoints (NFR-05).

``/healthz`` (liveness, no dependencies) and ``/readyz`` (database check) are public;
``/metrics`` requires an API key because counters can reveal usage patterns.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from shortener import __version__
from shortener.api.deps import get_services, require_api_key
from shortener.api.schemas import (
    AUTHENTICATED_ERRORS,
    HealthResponse,
    ReadyResponse,
    error_responses,
)
from shortener.container import Services
from shortener.domain.errors import ServiceUnavailableError
from shortener.services.auth import Principal

router = APIRouter(tags=["health"])

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Liveness",
    description="Always 200 while the process runs; checks no dependency. Shows the auth mode.",
)
def healthz(services: Annotated[Services, Depends(get_services)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        auth="enabled" if services.auth.enabled else "disabled",
    )


@router.get(
    "/readyz",
    response_model=ReadyResponse,
    summary="Readiness",
    description="200 when the database answers, otherwise 503.",
    responses=error_responses(503),
)
def readyz(services: Annotated[Services, Depends(get_services)]) -> ReadyResponse:
    if not services.is_ready():
        raise ServiceUnavailableError("The database is not reachable.")
    return ReadyResponse(status="ready")


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus text exposition format. Requires an API key.",
    responses={
        200: {
            "description": "Metrics in the Prometheus text format.",
            "content": {PROMETHEUS_CONTENT_TYPE: {"schema": {"type": "string"}}},
        },
        **AUTHENTICATED_ERRORS,
    },
)
def metrics(
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
) -> PlainTextResponse:
    return PlainTextResponse(services.metrics.render(), media_type=PROMETHEUS_CONTENT_TYPE)
