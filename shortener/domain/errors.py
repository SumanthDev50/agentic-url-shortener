"""Domain error hierarchy.

Every error carries a stable machine-readable ``code`` (part of the public API contract) and a
static, client-safe ``message``.  Messages never reflect user input.  The HTTP status mapping
lives in ``shortener.api.errors`` so the domain stays transport-agnostic.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class of all errors that are reported to API clients."""

    code = "domain_error"
    default_message = "The request could not be processed."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class InvalidInputError(DomainError):
    """The caller supplied a value that violates a business rule."""

    code = "invalid_input"
    default_message = "The request contains an invalid value."


class InvalidUrlError(InvalidInputError):
    code = "invalid_url"
    default_message = "The destination URL is not acceptable."


class InvalidAliasError(InvalidInputError):
    code = "invalid_alias"
    default_message = "The alias is not acceptable."


class InvalidExpiryError(InvalidInputError):
    code = "invalid_expiry"
    default_message = "The expiry is not acceptable."


class InvalidIdempotencyKeyError(InvalidInputError):
    code = "invalid_idempotency_key"
    default_message = "The Idempotency-Key header is not acceptable."


class AliasTakenError(DomainError):
    code = "alias_taken"
    default_message = "The alias is already in use."


class LinkNotFoundError(DomainError):
    code = "not_found"
    default_message = "The link does not exist."


class LinkGoneError(DomainError):
    """The link existed but is expired or disabled (HTTP 410)."""

    code = "link_gone"
    default_message = "The link is no longer available."

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class CodeGenerationError(DomainError):
    """Bounded code generation retries were exhausted (HTTP 503 + Retry-After)."""

    code = "code_generation_failed"
    default_message = "A short code could not be allocated right now; retry shortly."
    retry_after_seconds = 1


class IdempotencyKeyReuseError(DomainError):
    """The same Idempotency-Key was used with a different request payload."""

    code = "idempotency_key_reuse"
    default_message = "The Idempotency-Key was already used with a different request."


class IdempotencyConflictError(DomainError):
    """The key is being processed by a concurrent request."""

    code = "idempotency_conflict"
    default_message = "A request with this Idempotency-Key is still being processed."


class AuthenticationError(DomainError):
    code = "unauthorized"
    default_message = "A valid API key is required."


class RateLimitedError(DomainError):
    """A rate limit was exceeded; carries the data for the ``X-RateLimit-*`` headers."""

    code = "rate_limited"
    default_message = "Too many requests."

    def __init__(self, *, limit: int, remaining: int, retry_after: int, reset_after: int) -> None:
        super().__init__()
        self.limit = limit
        self.remaining = remaining
        self.retry_after = retry_after
        self.reset_after = reset_after


class ServiceUnavailableError(DomainError):
    code = "service_unavailable"
    default_message = "The service is temporarily unavailable."
