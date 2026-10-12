-- 002_idempotency: stored responses for Idempotency-Key replay (FR-10).
-- Rows older than 24 hours are ignored on lookup and purged by the maintenance job.

CREATE TABLE idempotency_keys (
    key           TEXT NOT NULL,
    owner         TEXT NOT NULL,
    request_hash  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (key, owner)
);

CREATE INDEX idx_idempotency_created_at ON idempotency_keys (created_at);
