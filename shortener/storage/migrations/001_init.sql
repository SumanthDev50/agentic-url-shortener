-- 001_init: links and clicks.
-- Migrations are forward-only and immutable once applied: the runner stores each file's
-- SHA-256 in schema_migrations and refuses to start if an applied file has changed.
-- All timestamps are UTC, fixed-width ISO-8601 strings (lexicographic order == time order).

CREATE TABLE links (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    original_url TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    expires_at   TEXT,
    disabled_at  TEXT,
    click_count  INTEGER NOT NULL DEFAULT 0 CHECK (click_count >= 0),
    created_by   TEXT
);

CREATE INDEX idx_links_created_by_id ON links (created_by, id);

CREATE TABLE clicks (
    id            INTEGER PRIMARY KEY,
    link_id       INTEGER NOT NULL REFERENCES links (id),
    clicked_at    TEXT    NOT NULL,
    referrer_host TEXT,
    browser       TEXT    NOT NULL,
    device        TEXT    NOT NULL,
    visitor_hash  TEXT    NOT NULL
);

CREATE INDEX idx_clicks_link_clicked_at ON clicks (link_id, clicked_at);
CREATE INDEX idx_clicks_clicked_at ON clicks (clicked_at);
