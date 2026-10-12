"""Exception handlers and helpers that produce the consistent error model (NFR-11).

Every error response has the shape ``{"error": {"code", "message", "request_id"}}`` (plus
``details`` for validation problems).  Messages are static: neither stack traces nor request
values are ever sent to clients; unexpected exceptions are logged server-side only.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from shortener.domain.errors import (
    AliasTakenError,
    AuthenticationError,
    CodeGenerationError,
    DomainError,
    IdempotencyConflictError,
    IdempotencyKeyReuseError,
    InvalidInputError,
    LinkGoneError,
    LinkNotFoundError,
    RateLimitedError,
    ServiceUnavailableError,
)
from shortener.observability.logging import current_request_id

MAX_DETAILS = 10
MAX_DETAIL_TEXT = 200

_STATUS_BY_ERROR: tuple[tuple[type[DomainError], int], ...] = (
    (InvalidInputError, 422),
    (IdempotencyKeyReuseError, 422),
    (AliasTakenError, 409),
    (IdempotencyConflictError, 409),
    (LinkNotFoundError, 404),
    (LinkGoneError, 410),
    (AuthenticationError, 401),
    (RateLimitedError, 429),
    (CodeGenerationError, 503),
    (ServiceUnavailableError, 503),
)
_CODE_BY_STATUS = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
}


def status_for(error: DomainError) -> int:
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status
    return 500


def headers_for(error: DomainError) -> dict[str, str]:
    """Protocol headers that accompany specific errors."""
    if isinstance(error, RateLimitedError):
        return {
            "Retry-After": str(max(error.retry_after, 1)),
            "X-RateLimit-Limit": str(error.limit),
            "X-RateLimit-Remaining": str(error.remaining),
            "X-RateLimit-Reset": str(error.reset_after),
        }
    if isinstance(error, CodeGenerationError):
        return {"Retry-After": str(error.retry_after_seconds)}
    if isinstance(error, AuthenticationError):
        return {"WWW-Authenticate": "ApiKey"}
    return {}


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    body: dict[str, object] = {
        "code": code,
        "message": message,
        "request_id": current_request_id(),
    }
    if details:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
    return error_response(status_for(exc), exc.code, exc.message, headers=headers_for(exc))


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {
            "location": ".".join(str(part)[:64] for part in problem["loc"]),
            "message": str(problem["msg"])[:MAX_DETAIL_TEXT],
        }
        for problem in jsonable_encoder(exc.errors())[:MAX_DETAILS]
    ]
    return error_response(422, "validation_error", "The request is not valid.", details=details)


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    phrase = _phrase(exc.status_code)
    code = _CODE_BY_STATUS.get(exc.status_code, "http_error")
    headers = {str(key): str(value) for key, value in (exc.headers or {}).items()}
    return error_response(exc.status_code, code, phrase, headers=headers)


def _phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers (Starlette types them for ``Exception``; ours are narrower)."""
    app.add_exception_handler(DomainError, cast(ExceptionHandler, handle_domain_error))
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, handle_validation_error)
    )
    app.add_exception_handler(StarletteHTTPException, cast(ExceptionHandler, handle_http_exception))
