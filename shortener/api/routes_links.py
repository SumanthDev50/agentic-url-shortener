"""Management API: ``/api/v1/links`` (FR-01, FR-02, FR-03, FR-05, FR-06, FR-07, FR-08, FR-10).

Every route depends on ``require_api_key`` and passes the authenticated key id to the service
as the *owner*; services scope all reads and writes to it, so another key's link is
indistinguishable from a missing one (404, no IDOR).
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Response

from shortener.api.deps import get_services, require_api_key
from shortener.api.schemas import (
    AUTHENTICATED_ERRORS,
    AnalyticsResponse,
    CreateLinkRequest,
    LinkListResponse,
    LinkResponse,
    error_responses,
)
from shortener.container import Services
from shortener.domain.models import Link
from shortener.domain.validation import validate_idempotency_key
from shortener.services.analytics_service import DEFAULT_DAYS, MAX_DAYS, MIN_DAYS
from shortener.services.auth import Principal
from shortener.services.idempotency import request_fingerprint
from shortener.storage.link_repository import MAX_OFFSET, MAX_PAGE_SIZE

router = APIRouter(prefix="/api/v1/links", tags=["links"])

DEFAULT_PAGE_SIZE = 20


def link_response(link: Link, services: Services) -> LinkResponse:
    return LinkResponse.from_link(
        link, base_url=services.settings.base_url, status=services.links.status_of(link)
    )


@router.post(
    "",
    status_code=201,
    response_model=LinkResponse,
    summary="Create a short link",
    response_description="The created link (or the replayed original response).",
    description=(
        "Creates a link owned by the calling API key. Send an `Idempotency-Key` header "
        "(max 128 printable ASCII characters) to make retries safe: the same key with the "
        "same payload replays the original response with `Idempotent-Replay: true` for 24 "
        "hours; the same key with a different payload is rejected with 422."
    ),
    responses={
        **AUTHENTICATED_ERRORS,
        **error_responses(409, 413, 422, 503),
    },
)
def create_link(
    body: CreateLinkRequest,
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            max_length=128,
            description="Makes creation idempotent for 24 hours (per API key).",
        ),
    ] = None,
) -> Response:
    key = validate_idempotency_key(idempotency_key) if idempotency_key is not None else None

    def create() -> str:
        link = services.links.create(
            url=body.url,
            owner=principal.key_id,
            alias=body.alias,
            expires_at=body.expires_at,
            ttl_seconds=body.ttl_seconds,
        )
        return link_response(link, services).model_dump_json()

    result = services.idempotency.execute(
        key=key,
        owner=principal.key_id,
        request_hash=request_fingerprint(body.model_dump(mode="json")),
        operation=create,
    )
    headers = {"Location": f"/api/v1/links/{json.loads(result.body)['code']}"}
    if result.replayed:
        headers["Idempotent-Replay"] = "true"
    else:
        services.metrics.inc("shortener_links_created_total")
    return Response(
        content=result.body, status_code=201, media_type="application/json", headers=headers
    )


@router.get(
    "",
    response_model=LinkListResponse,
    summary="List the caller's links",
    response_description="One page of links and the total count.",
    description="Newest first (stable ordering by creation sequence), with the total count.",
    responses={**AUTHENTICATED_ERRORS, **error_responses(422)},
)
def list_links(
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
) -> LinkListResponse:
    page = services.links.list(principal.key_id, limit, offset)
    return LinkListResponse(
        items=[link_response(link, services) for link in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/{code}",
    response_model=LinkResponse,
    summary="Get link details",
    response_description="Metadata, status and recorded click count.",
    responses={**AUTHENTICATED_ERRORS, **error_responses(404, 422)},
)
def get_link(
    code: str,
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
) -> LinkResponse:
    return link_response(services.links.get(code, principal.key_id), services)


@router.delete(
    "/{code}",
    status_code=204,
    summary="Disable a link",
    description=(
        "Soft delete: the short URL answers 410 afterwards. Idempotent: disabling an "
        "already disabled link also returns 204."
    ),
    response_description="The link was disabled.",
    responses={**AUTHENTICATED_ERRORS, **error_responses(404, 422)},
)
def disable_link(
    code: str,
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
) -> Response:
    services.links.disable(code, principal.key_id)
    return Response(status_code=204)


@router.get(
    "/{code}/analytics",
    response_model=AnalyticsResponse,
    summary="Link analytics",
    response_description="Click aggregates for the requested window.",
    description=(
        "Aggregates over the last N UTC calendar days (today included). Clicks are recorded "
        "asynchronously, so very recent clicks may not be visible yet. Visitor ids are "
        "daily-salted hashes: a visitor active on two days counts as two."
    ),
    responses={**AUTHENTICATED_ERRORS, **error_responses(404, 422)},
)
def link_analytics(
    code: str,
    principal: Annotated[Principal, Depends(require_api_key)],
    services: Annotated[Services, Depends(get_services)],
    days: Annotated[int, Query(ge=MIN_DAYS, le=MAX_DAYS)] = DEFAULT_DAYS,
) -> AnalyticsResponse:
    link = services.links.get(code, principal.key_id)
    return AnalyticsResponse.from_report(link.code, services.analytics.report(link, days))
