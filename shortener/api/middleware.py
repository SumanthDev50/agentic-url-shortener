"""Pure-ASGI middleware (no ``BaseHTTPMiddleware``: it buffers, breaks contextvars and cannot
cancel cleanly).

Stack, outermost first (see ``shortener.main.create_app``):

``RequestIdMiddleware`` -> ``AccessLogMiddleware`` -> ``ResponseHeadersMiddleware`` ->
``BodySizeLimitMiddleware`` -> ``ErrorBoundaryMiddleware`` -> application.

Responses produced by inner layers (including 413 and 500) therefore always carry the request id
and the security headers, and are counted and logged.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from shortener.api.errors import error_response
from shortener.observability.logging import log_event, request_id_var
from shortener.observability.metrics import Metrics

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_HEADERS_STATE_KEY = "response_headers"

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9-]{8,64}")
_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_API_CSP = "default-src 'none'; frame-ancestors 'none'"

logger = logging.getLogger("shortener.access")
error_logger = logging.getLogger(__name__)


def accept_or_generate_request_id(inbound: str | None) -> str:
    """Accept a well-formed inbound id; otherwise mint one (prevents log forging)."""
    if inbound is not None and _REQUEST_ID_PATTERN.fullmatch(inbound):
        return inbound
    return str(uuid.uuid4())


def set_response_headers(scope: Scope, headers: Mapping[str, str]) -> None:
    """Ask ``ResponseHeadersMiddleware`` to add ``headers`` to this request's response."""
    scope.setdefault("state", {})[RESPONSE_HEADERS_STATE_KEY] = dict(headers)


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = accept_or_generate_request_id(Headers(scope=scope).get("x-request-id"))
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


class AccessLogMiddleware:
    """One structured log line and one counter increment per request.

    Only the route *template* is recorded (bounded metric cardinality); the path, query string,
    client address and headers are deliberately not logged.
    """

    def __init__(self, app: ASGIApp, *, metrics: Metrics) -> None:
        self.app = app
        self._metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def send_and_capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_and_capture)
        finally:
            self._record(scope, status, time.perf_counter() - started)

    def _record(self, scope: Scope, status: int, seconds: float) -> None:
        route = getattr(scope.get("route"), "path", None) or "unmatched"
        method = scope["method"] if scope["method"] in _KNOWN_METHODS else "OTHER"
        self._metrics.inc(
            "shortener_http_requests_total", method=method, route=route, status=str(status)
        )
        log_event(
            logger,
            logging.INFO,
            "request",
            method=method,
            route=route,
            status=status,
            duration_ms=round(seconds * 1000, 2),
        )


class ResponseHeadersMiddleware:
    """Adds security headers (without overriding ones a route set) and per-request extras."""

    def __init__(self, app: ASGIApp, *, hsts: bool, relaxed_paths: frozenset[str]) -> None:
        self.app = app
        base = {
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        if hsts:
            base["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        self._strict = {**base, "Content-Security-Policy": _API_CSP}
        self._relaxed = base  # interactive documentation pages need scripts and styles
        self._relaxed_paths = relaxed_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        defaults = self._relaxed if scope["path"] in self._relaxed_paths else self._strict

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in defaults.items():
                    if name not in headers:
                        headers[name] = value
                extras: Mapping[str, str] = scope.get("state", {}).get(
                    RESPONSE_HEADERS_STATE_KEY, {}
                )
                for name, value in extras.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes`` (declared or streamed) with 413."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            rejection = self._check_declared(declared)
            if rejection is not None:
                await rejection(scope, receive, send)
                return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    raise StarletteHTTPException(status_code=413)
            return message

        await self.app(scope, limited_receive, send)

    def _check_declared(self, declared: str) -> JSONResponse | None:
        if not declared.isascii() or not declared.isdigit():
            return error_response(400, "bad_request", "Invalid Content-Length header.")
        if int(declared) > self._max_bytes:
            return error_response(413, "payload_too_large", "Request body too large.")
        return None


class ErrorBoundaryMiddleware:
    """Last line of defence: any unhandled exception becomes a generic 500 error response.

    The exception (with traceback) is logged server-side only; the client sees neither.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            error_logger.exception("unhandled_error")
            if started:
                raise  # headers are gone; let the server abort the connection
            response = error_response(500, "internal_error", "Internal server error.")
            await response(scope, receive, send)
