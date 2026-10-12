# URL shortener - product notes

Companion to [`product-spec.md`](product-spec.md) (the behavioural contract) and
[`api.md`](api.md) (generated from the OpenAPI schema with `python -m shortener.apidoc`).
This file records how the v1 baseline is built, the trade-offs behind it, and the places later
changes are expected to touch.

## Running it

```text
python -m shortener                                   # loads SHORTENER_* from the environment
uvicorn shortener.main:create_app --factory           # same app, uvicorn CLI
python -m shortener.apidoc                            # regenerate docs/api.md
python -m shortener.apidoc --check                    # exit 1 if docs/api.md is stale
```

`python -m shortener` refuses to start (exit code 2, message without secret values) when the
configuration is invalid or unsafe: no API keys without an explicit
`SHORTENER_AUTH_DISABLED=true`, auth disabled on a non-loopback bind, docs enabled without
auth, placeholder or weak secrets, unknown `SHORTENER_*` variables. The default bind address is
`127.0.0.1`.

## Configuration reference

All variables are optional except `SHORTENER_API_KEYS` (unless auth is disabled). The block
below is the canonical content of `.env.example`; a test keeps it in sync with the loader
(every variable documented exactly once, placeholders rejected, valid once replaced).

> `.env.example` could not be created by the agent that built the product: the workspace
> permission rules deny writes to `.env*` paths. Copy this block to `.env.example` to finish
> that deliverable; `tests/shortener/test_shortener_config.py` then verifies the copy is
> identical to this block.

```env
# URL shortener configuration (copy to .env and edit; .env is git-ignored).
#
# The service reads SHORTENER_* variables from the process environment and validates them at
# startup (python -m shortener).  It refuses to start when a value is invalid, when no API key
# is configured (unless authentication is explicitly disabled), or when a placeholder such as
# the ones below is left in place.  Error messages name the variable but never print its value.
# Unknown SHORTENER_* variables are rejected so a typo cannot silently weaken a setting.
#
# Load it into your shell before starting, for example (POSIX):  set -a; . ./.env; set +a
#
# Generate secrets with:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"

# --- network -------------------------------------------------------------------------------
# Bind address: an IP literal or "localhost".  Defaults to loopback; bind a public interface
# only behind a reverse proxy with authentication enabled.
SHORTENER_HOST=127.0.0.1
SHORTENER_PORT=8000
# Public base URL used to build short URLs and to reject destinations that point back at this
# service (redirect loops).  Scheme + host (+ port) only.
SHORTENER_BASE_URL=http://127.0.0.1:8000
# Extra host names of this service (comma separated) that must never be accepted as destinations.
SHORTENER_OWN_HOSTS=
# Peers (IP or CIDR, comma separated) whose X-Forwarded-For header is trusted.  Empty = never.
SHORTENER_TRUSTED_PROXIES=

# --- storage -------------------------------------------------------------------------------
SHORTENER_DATABASE_PATH=data/shortener.db
# Click events older than this many days are purged at startup and daily.
SHORTENER_CLICK_RETENTION_DAYS=90

# --- authentication ------------------------------------------------------------------------
# Comma separated id:secret pairs.  id: 1-32 of A-Z a-z 0-9 _ . -   secret: 32-256 of A-Z a-z
# 0-9 _ - (use secrets.token_urlsafe(32)).  Links belong to the key id that created them.
# The value below is a PLACEHOLDER and is rejected by the loader: replace it.
SHORTENER_API_KEYS=local-dev:REPLACE_ME_with_output_of_secrets_token_urlsafe_32
# Development only: "true" disables authentication.  Allowed only with a loopback
# SHORTENER_HOST, cannot be combined with SHORTENER_API_KEYS, and is shown by /healthz.
SHORTENER_AUTH_DISABLED=false
# Serve /docs, /redoc and /openapi.json.  Requires authentication to be enabled.
SHORTENER_ENABLE_DOCS=false

# --- privacy -------------------------------------------------------------------------------
# Server secret for the daily-salted visitor hash.  If unset a random per-process secret is used
# (with a warning) and visitor ids reset on restart.  PLACEHOLDER below is rejected: replace or
# remove the line.
SHORTENER_ANALYTICS_SECRET=REPLACE_ME_with_output_of_secrets_token_urlsafe_32

# --- destination policy --------------------------------------------------------------------
# Comma separated domains.  Private/loopback/link-local addresses and internal names are always
# rejected; these lists can only restrict further.  ALLOW_HOSTS (when set) permits only those
# domains and their subdomains; DENY_HOSTS blocks the listed domains and their subdomains.
SHORTENER_ALLOW_HOSTS=
SHORTENER_DENY_HOSTS=

# --- rate limiting (token buckets, per process) --------------------------------------------
# Management API: per API key id.
SHORTENER_API_RATE_BURST=30
SHORTENER_API_RATE_PER_MINUTE=120
# Redirects: per client address.
SHORTENER_REDIRECT_RATE_BURST=120
SHORTENER_REDIRECT_RATE_PER_MINUTE=600
# Failed authentication attempts: per client address (stricter), plus a small delay per failure.
SHORTENER_AUTH_FAILURE_RATE_BURST=5
SHORTENER_AUTH_FAILURE_RATE_PER_MINUTE=10
SHORTENER_AUTH_FAILURE_DELAY_MS=200
# Upper bound of tracked buckets (older ones are evicted) so attackers cannot exhaust memory.
SHORTENER_RATE_LIMIT_MAX_KEYS=100000

# --- redirect cache and click queue --------------------------------------------------------
SHORTENER_CACHE_MAX_ENTRIES=10000
SHORTENER_CACHE_TTL_SECONDS=60
SHORTENER_NEGATIVE_CACHE_TTL_SECONDS=5
SHORTENER_CLICK_QUEUE_SIZE=10000
SHORTENER_CLICK_BATCH_SIZE=200

# --- logging -------------------------------------------------------------------------------
# DEBUG, INFO, WARNING, ERROR or CRITICAL.  Logs are JSON lines on stderr.
SHORTENER_LOG_LEVEL=INFO
```

## Design notes and trade-offs

**Layering.** `api -> services -> domain / storage`. Routes never touch SQL; `domain` is pure
(no I/O, not even DNS); services get repositories, cache, clock and metrics by constructor
injection. `shortener/container.py` is the only composition root and `create_app()` has no side
effects: the lifespan opens and migrates the database and starts the background threads, which
is also why `create_app(...).openapi()` can generate `docs/api.md` without a database.

**Redirects are 302.** Every click reaches analytics and links stay revocable (a 301 would be
cached by browsers for good). Cost: more redirect traffic. Responses are `no-store`, have a fixed
body, and `Referrer-Policy: no-referrer`.

**`click_count` is eventually consistent.** Clicks go through a bounded in-memory queue and a
single writer thread that batches events and the counter updates into one transaction. A
redirect therefore never waits for, or fails because of, analytics; under overload clicks are
dropped and counted (`shortener_clicks_dropped_total`), and shutdown flushes the queue. The
trade-off: the counter lags by the queue delay and is lossy under overload. A hard per-link click
limit (the brownfield `max-clicks` scenario) cannot be built on this: it needs an atomic
conditional `UPDATE ... WHERE click_count < max_clicks` with a `rowcount` check on the resolve
path and cache bypass for limited links (see the last section).

**Redirect cache.** LRU + TTL (default 60 s) of `code -> target, expiry, disabled`. Expiry is
evaluated against the clock on every hit, so a cached entry never outlives the link. Unknown
codes are cached negatively for a short time (default 5 s). Disabling invalidates the local entry
immediately; with several instances revocation lags by up to the positive TTL. Only codes that
match the alias pattern reach the cache or the database, so junk paths cannot churn the cache.

**Rate limiting is per process.** Token buckets: per API-key id for management routes, per client
address for redirects, and a stricter bucket for authentication failures. With N instances the
effective limit is up to N times the configured one (documented trade-off; a shared store would
fix it). The bucket map is bounded (default 100 000, least recently used evicted), so
attacker-chosen keys cannot exhaust memory. `X-Forwarded-For` is honoured only from
`SHORTENER_TRUSTED_PROXIES`, read from the right.

**Authentication and ownership.** Keys are `id:secret` pairs; only SHA-256 digests are kept and
every configured digest is compared with `hmac.compare_digest` (no early exit). Failed attempts
are delayed and throttled per client; once a client's failure bucket is empty its requests are
rejected before the key is evaluated, so guessing cannot continue at full speed (a correct key
from a locked-out client also waits for the bucket to refill). Every link stores `created_by`
(the key id) and every `/api/v1/links/{code}...` operation is scoped to it: other owners get 404,
never 403. With authentication disabled (loopback only) every link belongs to the owner
`anonymous`; such links are invisible once authentication is enabled.

**Idempotency.** The stored response and the created link are written in one transaction taken
with `BEGIN IMMEDIATE`; the key is looked up under that lock. Concurrent identical requests
therefore execute once and the rest replay the stored body (`Idempotent-Replay: true`). A reused
key with a different payload is 422. Failed requests store nothing and may be retried; keys are
scoped per API key and expire after 24 hours.

**Destination validation.** The raw URL is screened (control, whitespace, backslash and Unicode
format/bidi characters), parsed once, validated component by component and re-serialised; the
canonical ASCII form is what is stored and redirected to. Hosts are IDNA-encoded (mixed scripts
rejected); IPv4 in any legacy notation (`127.1`, `2130706433`, `0x7f.1`, octal, full-width digits)
and IPv6 (including IPv4-mapped) must be globally routable; `localhost`, `*.local`, `*.internal`,
`*.localdomain`, `*.home.arpa`, `*.lan`, `*.intranet` and single-label host names are rejected,
as are the service's own hosts. Limitation: validation is syntactic. A public host *name* that
resolves to a private address (DNS rebinding, `127.0.0.1.nip.io`) cannot be detected without
resolving it; the shortener never fetches destinations itself, but any component that does
(link previews, the planned destination health checker) must resolve and re-check addresses at
connect time.

**Privacy.** Raw IPs are never stored or logged. A visitor id is
`HMAC-SHA256(daily_salt, truncated_ip | ua_family)[:16]` with the IP truncated to /24 (IPv4) or
/48 (IPv6) first and `daily_salt = HMAC(server_secret, utc_date)` derived on the fly, so ids are
unlinkable across days (a returning visitor counts once per day). The user agent is reduced to a
fixed set of browser/device labels and the referrer to `scheme://host` (stored in the
`referrer_host` column). Click retention is enforced at startup and daily. Without
`SHORTENER_ANALYTICS_SECRET` a random per-process secret is used and a warning is logged.

**Storage.** SQLite with one connection per thread (`foreign_keys=ON`, `trusted_schema=OFF`,
`busy_timeout`, WAL), parameterised statements only (each statement is a complete literal; the
one dynamic choice, the breakdown dimension, goes through an allow-list mapping). Migrations are
forward-only SQL files applied in a transaction with their SHA-256 recorded in
`schema_migrations`; startup aborts if an applied file changed, if the database knows a migration
this build does not, or if applied versions have gaps.

**Error model.** `{"error": {"code", "message", "request_id"}}` for every error (validation errors
add `details`). Messages are static and never reflect input; unexpected exceptions become a
generic 500 and are logged with their traceback server-side only.

**Logging and metrics.** One JSON object per line on stderr, with the request id from a context
variable. `X-API-Key`, `Authorization` and cookies are redacted in structured fields and scrubbed
in free text. The access log records the route template, never paths, queries, client addresses
or headers. `/metrics` (API key required) exposes request counts by method/route/status,
redirect cache hits/misses, click queue depth and drops, rate-limit rejections and auth failures.

**Interactive docs.** `/docs`, `/redoc` and `/openapi.json` exist only with
`SHORTENER_ENABLE_DOCS=true` and authentication enabled. They are then served without a key
(browsers cannot attach one) and load their assets from a CDN, so keep them off in production.

## Where later changes land

| Change | Touch points |
|---|---|
| New request/response field (for example `max_clicks`) | `api/schemas.py` (`CreateLinkRequest`, `LinkResponse`), `LinkService.create`, `domain/models.py`, a new migration `003_*.sql`, repositories |
| New storage column / table | a *new* numbered migration (never edit 001/002: their checksums are enforced), then `storage/link_repository.py` |
| Redirect rules (limits, geo, schedules) | `LinkService.resolve` and `CachedTarget` in `services/link_service.py`; `routes_redirect.follow_link` stays a thin adapter |
| Hard click limits | atomic conditional `UPDATE` in `LinkRepository` on the resolve path plus cache bypass/invalidation for limited links; do not rely on the background writer |
| New analytics dimension | `ClickEvent`, `ClickRepository.breakdown` allow-list, `AnalyticsReport`, `AnalyticsResponse` |
| New endpoint | a router under `api/` that depends on `require_api_key` and passes `principal.key_id` as the owner |

## Tests

`tests/shortener/` holds unit tests per module and API integration tests with `TestClient`
(create, redirect, analytics, disable; auth; ownership; rate limiting; idempotency; error model;
request ids; health, readiness and metrics; analytics failure isolation; concurrent alias
creation; no raw IP in the database or logs). Time is driven by a fake clock; secrets are
generated at runtime. Run `python -m pytest tests/shortener --cov=shortener`.
