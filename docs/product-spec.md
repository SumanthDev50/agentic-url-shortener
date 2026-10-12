# URL shortener — product specification (v1 baseline)

This is the behavioural contract for the `shortener/` package. The brownfield and
ambiguous scenarios of the orchestrator evolve *this* baseline, so structure and
naming matter: keep the layering strict and the modules small and cohesive.

## 1. Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | Create a short link for a destination URL (`POST /api/v1/links`) → 201 with `code`, `short_url`, `original_url`, `created_at`, `expires_at`. |
| FR-02 | Optional **custom alias** (3–32 chars `[A-Za-z0-9_-]`, case-sensitive, reserved words rejected, unique → 409 on clash). |
| FR-03 | Optional **expiry**: `expires_at` (ISO-8601, future, ≤ 5 years) *or* `ttl_seconds` (60 … 157 680 000); mutually exclusive. |
| FR-04 | **Redirect** `GET /{code}` → 302 with `Location`; unknown → 404; expired/disabled → 410. |
| FR-05 | **Link details** `GET /api/v1/links/{code}` (metadata, status, click_count). |
| FR-06 | **List** links `GET /api/v1/links?limit&offset` (limit ≤ 100, stable ordering, total count). |
| FR-07 | **Disable** `DELETE /api/v1/links/{code}` → 204 (soft delete; redirect then 410; idempotent). |
| FR-08 | **Analytics** `GET /api/v1/links/{code}/analytics?days=N` (1–365, default 30): total clicks, unique visitors, first/last click, clicks by day (zero-filled), top referrers, browser and device breakdown. |
| FR-09 | URL validation: http/https only; ≤ 2048 chars; no embedded credentials; host required; **blocks loopback / private / link-local / reserved IP literals and `localhost`/`*.local`/`*.internal` names** (SSRF hardening, configurable allow/deny lists); IDN normalised to punycode; no control characters. Self-referencing short URLs are rejected (no redirect loops). |
| FR-10 | Idempotent creation: `Idempotency-Key` header (≤ 128 chars) replays the original 201 for the same key+payload (same response, `Idempotent-Replay: true`) and returns 422/409 for the same key with a different payload. Keys expire after 24 h. |

## 2. Non-functional / reliability requirements

| ID | Requirement |
|---|---|
| NFR-01 | **Redirect never fails because analytics fails.** Click events go through a bounded in-memory queue drained by a background writer (batching); queue full → drop + counter, never block. Flushed on graceful shutdown. |
| NFR-02 | **Low-latency redirects**: LRU+TTL cache of `code → target/expiry/status`; invalidated on disable; negative caching for 404 with short TTL. |
| NFR-03 | **Collision-safe code generation**: 7-char base62 from `secrets`; unique constraint is the source of truth; bounded retry (5), length bump after 3 collisions; failure → 503 with `Retry-After`, never an infinite loop. |
| NFR-04 | **Rate limiting**: token bucket per client (API key id for management endpoints, IP for redirects), configurable; 429 with `Retry-After` + `X-RateLimit-*` headers. In-memory (per instance) — documented trade-off. |
| NFR-05 | **Health**: `GET /healthz` (liveness, no dependencies), `GET /readyz` (DB check, 503 when unavailable), `GET /metrics` (Prometheus text: request counts by route/status, redirect cache hits/misses, click queue depth/drops, rate-limit rejections). |
| NFR-06 | **Observability**: JSON structured logs, `X-Request-ID` propagation (accept a valid inbound id, else generate), no secrets/PII in logs. |
| NFR-07 | **Security by default**: management endpoints (everything under `/api/v1`) require `X-API-Key`; keys come from `SHORTENER_API_KEYS` (comma-separated, compared in constant time, never logged). **The service refuses to start without keys unless `SHORTENER_AUTH_DISABLED=true`** (loud warning). Security headers (`X-Content-Type-Options`, `Referrer-Policy`, `Cache-Control: no-store` on API), request body size limit (16 KiB), strict pydantic models (`extra=forbid`). |
| NFR-08 | **Privacy**: raw IP never stored; visitor id = truncated HMAC-SHA256(salt, ip + user-agent); salt from `SHORTENER_ANALYTICS_SALT` (random per-process with a warning if unset); only the Referer **host** is stored; user-agent reduced to browser family + device class. |
| NFR-09 | **Storage**: SQLite (WAL, `busy_timeout`, foreign keys on) behind repository interfaces; versioned, forward-only SQL migrations with a `schema_migrations` table, each applied in a transaction, checksum-verified; parameterised SQL only. |
| NFR-10 | **Configuration** via environment (`SHORTENER_*`), validated at startup with clear errors that never echo secret values; `.env.example` documents all variables. |
| NFR-11 | Consistent **error model**: `{"error": {"code": "...", "message": "...", "request_id": "..."}}` with stable machine-readable codes; no stack traces to clients. |
| NFR-12 | Graceful lifecycle: startup migrations, `lifespan` start/stop of background writer, clean shutdown flush. |

## 3. Layering and package structure

```
shortener/
  __init__.py                 # __version__
  main.py                     # create_app(settings=None) -> FastAPI ; `app = create_app()` is built lazily for uvicorn factory use
  config.py                   # Settings (frozen dataclass) + load_settings(env)
  api/
    routes_links.py           # /api/v1/links...
    routes_redirect.py        # /{code}
    routes_health.py          # /healthz /readyz /metrics
    schemas.py                # pydantic request/response models
    deps.py                   # require_api_key, get_services (dependency providers)
    errors.py                 # exception handlers -> error model
    middleware.py             # request id, security headers, body-size limit, access log
  domain/
    models.py                 # Link, ClickEvent, LinkStatus (dataclasses)
    validation.py             # validate_destination_url, validate_alias, resolve_expiry
    codegen.py                # CodeGenerator (secrets, base62)
    errors.py                 # DomainError hierarchy with stable codes
  services/
    link_service.py           # create / get / list / disable / resolve
    analytics_service.py      # record click (queue), aggregate
    click_recorder.py         # bounded queue + background writer
    cache.py                  # LruTtlCache
    rate_limiter.py           # TokenBucketLimiter
    idempotency.py
  storage/
    db.py                     # connection factory (per-thread connections, pragmas)
    migrations.py             # runner
    migrations/001_init.sql   # links, clicks, schema_migrations is created by runner
    migrations/002_idempotency.sql   # (or fold into 001 — keep ≥ 2 migrations so a later one is "new")
    link_repository.py
    click_repository.py
  observability/
    logging.py                # JSON formatter + request-id contextvar
    metrics.py                # thread-safe counters/gauges + Prometheus renderer
tests/shortener/              # unit + integration (TestClient), < 20 s total
.env.example  README section  docs/api.md (generated from OpenAPI)
```

Rules: `api` never touches SQL; `domain` performs no I/O; services receive repositories/cache/clock via constructor injection (testable with fakes); a `Clock` abstraction is used wherever time matters; no module-level singletons except the lazily-created `app`.

## 4. Data model (initial migrations)

`links(id INTEGER PK, code TEXT UNIQUE NOT NULL, original_url TEXT NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NULL, disabled_at TEXT NULL, click_count INTEGER NOT NULL DEFAULT 0, created_by TEXT NULL)`
`clicks(id INTEGER PK, link_id INTEGER NOT NULL REFERENCES links(id), clicked_at TEXT NOT NULL, referrer_host TEXT NULL, browser TEXT NOT NULL, device TEXT NOT NULL, visitor_hash TEXT NOT NULL)` + indexes `(link_id, clicked_at)`.
`idempotency_keys(key TEXT, owner TEXT, request_hash TEXT, response_json TEXT, created_at TEXT, PRIMARY KEY(key, owner))`.
All timestamps are UTC ISO-8601 strings. `click_count` is incremented by the background writer (eventually consistent) — **document this**; it is a deliberate trade-off that a later scenario (click limits) must revisit.

## 5. Testing requirements

Unit tests per module (validation edge cases incl. SSRF vectors, alias rules, expiry rules, codegen collision/exhaustion with an injected RNG, cache TTL/LRU eviction with a fake clock, rate limiter refill, migration runner incl. checksum tamper, click recorder drop/flush behaviour), plus API integration tests with `TestClient` (create → redirect → analytics → disable; auth required/denied; rate limit 429; idempotency; error model; request id; health/ready/metrics; analytics failure does not break redirect; concurrent creation of the same alias yields exactly one 201). Target ≥ 90 % line coverage on `shortener/`. Whole suite must run in < 20 s (it is executed inside orchestrator sandboxes).

## 6. Hardening requirements (normative; added after the security design review)

**URL validation (`domain/validation.py`, FR-09).** Accept only `http`/`https` via `urllib.parse.urlsplit` on the *stripped* input;
reject if the raw string contains control characters (< 0x20, 0x7f), whitespace, backslashes, or Unicode format/bidi characters
(`javascript:`, `data:`, `file:`, `ftp:`, `java\tscript:` and `https:\evil` variants must all fail). No userinfo. Host is lowercased,
converted to IDNA/punycode (mixed-script labels rejected), ≤ 253 chars; reject the shortener's own configured hosts (loops). Reject
loopback / private / link-local / reserved / multicast / unspecified IP literals in **every notation** (`127.1`, `2130706433`,
`0x7f.1`, `[::ffff:127.0.0.1]`, `[::1]`, ULA/link-local IPv6) and `localhost`, `*.localhost`, `*.local`, `*.internal`, `*.localdomain`.
Allowed ports: any valid port except 0. **Store and redirect to the canonical re-serialised URL** (ASCII, percent-encoded), never the raw input.
Reserved aliases (rejected): `api`, `admin`, `docs`, `redoc`, `openapi.json`, `healthz`, `readyz`, `metrics`, `static`, `favicon.ico`, `robots.txt`.
Aliases are case-sensitive (documented: look-alike aliases are an abuse vector mitigated by auth + rate limiting).

**Auth & ownership (NFR-07).** `SHORTENER_API_KEYS` is a comma-separated list of `id:secret` pairs (secrets generated with
`secrets.token_urlsafe(32)`; never in the repo or `.env.example` — the example shows the *format* with an obvious placeholder
that the loader **rejects**). Only SHA-256 digests are kept in memory; comparison via `hmac.compare_digest`; `X-API-Key` and
`Authorization` are redacted in every log. **Every link stores `created_by` = key id, and every `/api/v1/links/{code}…` route is
scoped to the owner (404 for other keys — no IDOR).** `SHORTENER_AUTH_DISABLED=true` is allowed only when the bind host is loopback
(startup refuses otherwise), logs a loud warning and shows `auth: disabled` in `/healthz`. `/metrics` requires a key (or loopback);
`/healthz` and `/readyz` are public. FastAPI `/docs`, `/redoc`, `/openapi.json` are **disabled** unless `SHORTENER_ENABLE_DOCS=true` and auth is enabled.
The orchestrator's SEC-003 public allow-list is exactly `{GET /{code}, GET /healthz, GET /readyz}`.

**Abuse controls (NFR-04).** Rate-limit key = `request.client.host`; `X-Forwarded-For` is honoured only from `SHORTENER_TRUSTED_PROXIES`.
Per-key buckets for management routes, per-IP buckets for redirects, a stricter limiter + small delay for authentication failures.
The bucket map is **bounded** (default 100 000 entries, LRU/TTL eviction) so attacker-chosen keys cannot exhaust memory.
Inbound `X-Request-ID` is accepted only if it matches `^[A-Za-z0-9-]{8,64}$`, otherwise a fresh id is generated (no log forging).
Request bodies are capped (16 KiB) by middleware; pydantic constrains every field (lengths, ranges, `extra=forbid`).

**Redirect hardening (FR-04).** 302, `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`, a fixed tiny body
(no reflected input). Negative cache (404) with short TTL; disabling a link invalidates the cache entry immediately (multi-instance revocation
lag = the positive-cache TTL, documented).

**Privacy (NFR-08).** `visitor_id = HMAC-SHA256(daily_salt, truncated_ip || "|" || ua_family)[:16]` with `daily_salt = HMAC(server_secret, utc_date)`
derived on the fly and never stored (ids are unlinkable across days and not reversible); IPv4 truncated to /24, IPv6 to /48 *before* hashing;
only the user-agent **family** and the referrer **scheme + host** are stored (length-capped). `server_secret` comes from `SHORTENER_ANALYTICS_SECRET`
(random per-process with a warning if unset). Retention: `SHORTENER_CLICK_RETENTION_DAYS` (default 90) purge on startup and daily.
Raw IP, full user-agent and referrer paths/queries are never persisted or logged.

**SQL & storage (NFR-09).** Parameterised queries only; any dynamic `ORDER BY` goes through an allow-list mapping; `LIKE` wildcards escaped with `ESCAPE`;
`LIMIT/OFFSET` bounded ints. Pragmas: `foreign_keys=ON`, `trusted_schema=OFF`, `busy_timeout`, WAL. `schema_migrations(version, checksum)`:
**the service refuses to start if an applied migration's checksum differs** (immutability enforced at runtime, complementing orchestrator rule CHG-001).

**Atomic counters (design note for the brownfield scenario).** In v1 `click_count` is incremented by the background writer (eventually consistent) and the
redirect path reads through an LRU cache. A hard per-link click limit therefore requires an atomic conditional update on the resolve path
(`UPDATE links SET click_count = click_count + 1 WHERE code = ? AND (max_clicks IS NULL OR click_count < max_clicks)` with a `rowcount` check)
and cache bypass/invalidation for limited links — see the `max-clicks` ADR the orchestrator generates.
