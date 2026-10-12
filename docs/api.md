# URL Shortener API

> Generated from the OpenAPI schema by `python -m shortener.apidoc`. Do not edit by hand.

Version: `0.1.0`

URL shortener with custom aliases, expiry, idempotent creation, per-key ownership, privacy-preserving analytics, rate limiting and operational endpoints. All `/api/v1` endpoints require an `X-API-Key` header.

## Authentication

Every `/api/v1` endpoint and `/metrics` require the header `X-API-Key: <secret>`.
Keys are configured with `SHORTENER_API_KEYS` (`id:secret` pairs). A link belongs to the key
that created it; other keys receive `404` for it. `GET /{code}`, `/healthz` and `/readyz`
are public.

## Error model

All errors use one JSON shape. `code` is stable and machine-readable, `message` is static,
`request_id` matches the `X-Request-ID` response header, and `details` (validation errors
only) lists the offending fields.

```json
{"error": {"code": "not_found", "message": "The link does not exist.",
           "request_id": "6b00369f-e071-4e74-9721-d6956aa29276"}}
```

Rate limited responses (`429`) carry `Retry-After` and `X-RateLimit-Limit`,
`X-RateLimit-Remaining` and `X-RateLimit-Reset` (seconds until the bucket is full again).

## Endpoints

### `GET /healthz`

**Liveness**

Always 200 while the process runs; checks no dependency. Shows the auth mode.

Authentication: none (public)

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | `HealthResponse` (application/json) |

### `GET /readyz`

**Readiness**

200 when the database answers, otherwise 503.

Authentication: none (public)

| Status | Description | Body |
|---|---|---|
| 200 | Successful Response | `ReadyResponse` (application/json) |
| 503 | Temporarily unavailable; see Retry-After where present. | `ErrorResponse` (application/json) |

### `GET /metrics`

**Prometheus metrics**

Prometheus text exposition format. Requires an API key.

Authentication: `X-API-Key` header

| Status | Description | Body |
|---|---|---|
| 200 | Metrics in the Prometheus text format. | - |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

### `GET /api/v1/links`

**List the caller's links**

Newest first (stable ordering by creation sequence), with the total count.

Authentication: `X-API-Key` header

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `limit` | query | `integer` | no | Constraints: `minimum=1, maximum=100`. |
| `offset` | query | `integer` | no | Constraints: `minimum=0, maximum=1000000`. |

| Status | Description | Body |
|---|---|---|
| 200 | One page of links and the total count. | `LinkListResponse` (application/json) |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

### `POST /api/v1/links`

**Create a short link**

Creates a link owned by the calling API key. Send an `Idempotency-Key` header (max 128 printable ASCII characters) to make retries safe: the same key with the same payload replays the original response with `Idempotent-Replay: true` for 24 hours; the same key with a different payload is rejected with 422.

Authentication: `X-API-Key` header

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `Idempotency-Key` | header | `string or null` | no | Makes creation idempotent for 24 hours (per API key). Constraints: `maxLength=128`. |

Request body (JSON): `CreateLinkRequest`

| Status | Description | Body |
|---|---|---|
| 201 | The created link (or the replayed original response). | `LinkResponse` (application/json) |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 409 | Conflict. | `ErrorResponse` (application/json) |
| 413 | Request body too large. | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |
| 503 | Temporarily unavailable; see Retry-After where present. | `ErrorResponse` (application/json) |

### `GET /api/v1/links/{code}`

**Get link details**

Authentication: `X-API-Key` header

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `code` | path | `string` | yes |  |

| Status | Description | Body |
|---|---|---|
| 200 | Metadata, status and recorded click count. | `LinkResponse` (application/json) |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 404 | Not found (also returned for links owned by another key). | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

### `DELETE /api/v1/links/{code}`

**Disable a link**

Soft delete: the short URL answers 410 afterwards. Idempotent: disabling an already disabled link also returns 204.

Authentication: `X-API-Key` header

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `code` | path | `string` | yes |  |

| Status | Description | Body |
|---|---|---|
| 204 | The link was disabled. | - |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 404 | Not found (also returned for links owned by another key). | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

### `GET /api/v1/links/{code}/analytics`

**Link analytics**

Aggregates over the last N UTC calendar days (today included). Clicks are recorded asynchronously, so very recent clicks may not be visible yet. Visitor ids are daily-salted hashes: a visitor active on two days counts as two.

Authentication: `X-API-Key` header

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `code` | path | `string` | yes |  |
| `days` | query | `integer` | no | Constraints: `minimum=1, maximum=365`. |

| Status | Description | Body |
|---|---|---|
| 200 | Click aggregates for the requested window. | `AnalyticsResponse` (application/json) |
| 401 | Missing or invalid API key. | `ErrorResponse` (application/json) |
| 404 | Not found (also returned for links owned by another key). | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

### `GET /{code}`

**Follow a short link**

Redirects to the destination with a 302. Unknown codes answer 404; expired or disabled links answer 410. Rate limited per client address. No API key required.

Authentication: none (public)

| Parameter | In | Type | Required | Description |
|---|---|---|---|---|
| `code` | path | `string` | yes |  |

| Status | Description | Body |
|---|---|---|
| 302 | Redirect to the destination. | - |
| 404 | Not found (also returned for links owned by another key). | `ErrorResponse` (application/json) |
| 410 | The link is expired or disabled. | `ErrorResponse` (application/json) |
| 422 | Validation failed. | `ErrorResponse` (application/json) |
| 429 | Rate limit exceeded; see Retry-After and X-RateLimit-* headers. | `ErrorResponse` (application/json) |

## Schemas

### AnalyticsResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `code` | `string` | yes |  |
| `days` | `integer` | yes | Window length: the last N UTC calendar days, today included. |
| `total_clicks` | `integer` | yes |  |
| `unique_visitors` | `integer` | yes | Distinct daily-salted visitor hashes (a visitor on two days counts twice). |
| `first_click_at` | `string (date-time) or null` | yes |  |
| `last_click_at` | `string (date-time) or null` | yes |  |
| `clicks_by_day` | `array of DayClicks` | yes | One entry per day, zero-filled. |
| `top_referrers` | `array of CountItem` | yes | Referrer scheme://host, most frequent first. |
| `browsers` | `array of CountItem` | yes |  |
| `devices` | `array of CountItem` | yes |  |

### CountItem

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `string` | yes |  |
| `clicks` | `integer` | yes |  |

### CreateLinkRequest

| Field | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | yes | Destination. Only http/https; no credentials; loopback, private, link-local and other non-public addresses and internal host names are rejected. The canonical re-serialised form is stored and redirected to. Constraints: `minLength=1, maxLength=2048`. |
| `alias` | `string or null` | no | Custom code (case-sensitive). Reserved words are rejected; clash gives 409. Constraints: `minLength=3, maxLength=32, pattern='^[A-Za-z0-9_-]+$'`. |
| `expires_at` | `string (date-time) or null` | no | Absolute expiry: ISO-8601 with a UTC offset, in the future, at most five years ahead. Mutually exclusive with ttl_seconds. |
| `ttl_seconds` | `integer or null` | no | Relative expiry in seconds. Mutually exclusive with expires_at. Constraints: `minimum=60, maximum=157680000`. |

### DayClicks

| Field | Type | Required | Description |
|---|---|---|---|
| `date` | `string (date)` | yes |  |
| `clicks` | `integer` | yes |  |

### ErrorBody

| Field | Type | Required | Description |
|---|---|---|---|
| `code` | `string` | yes | Stable machine-readable error code. |
| `message` | `string` | yes | Human-readable, static message (never reflects input). |
| `request_id` | `string or null` | yes | Correlates with the X-Request-ID header and logs. |
| `details` | `array of ErrorDetail or null` | no | Field-level problems (validation errors only). |

### ErrorDetail

| Field | Type | Required | Description |
|---|---|---|---|
| `location` | `string` | yes | Dot-separated path of the offending field. |
| `message` | `string` | yes |  |

### ErrorResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `error` | `ErrorBody` | yes |  |

### HealthResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | `string` | yes |  |
| `version` | `string` | yes |  |
| `auth` | `string` | yes |  |

### LinkListResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `items` | `array of LinkResponse` | yes |  |
| `total` | `integer` | yes | Total links owned by the caller. |
| `limit` | `integer` | yes |  |
| `offset` | `integer` | yes |  |

### LinkResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `code` | `string` | yes | The short code (path segment of the short URL). |
| `short_url` | `string` | yes | Absolute short URL (the configured base URL + code). |
| `original_url` | `string` | yes | Canonical destination the short URL redirects to. |
| `created_at` | `string (date-time)` | yes |  |
| `expires_at` | `string (date-time) or null` | yes | Expiry instant, or null for none. |
| `status` | `LinkStatus` | yes | active, expired or disabled. |
| `click_count` | `integer` | yes | Recorded clicks. Eventually consistent: updated by a background writer. |

### LinkStatus

One of: `active`, `expired`, `disabled`.

### ReadyResponse

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | `string` | yes |  |
