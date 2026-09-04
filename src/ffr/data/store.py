"""SQLite persistence layer: schema, FTS5, and all DB access helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ffr.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,     -- nflverse gsis_id
    display_name  TEXT NOT NULL,
    position      TEXT,
    rookie_season INTEGER,
    last_season   INTEGER
);

CREATE TABLE IF NOT EXISTS player_aliases (
    alias_norm TEXT NOT NULL,
    player_id  TEXT NOT NULL REFERENCES players(player_id),
    source     TEXT,
    PRIMARY KEY (alias_norm, player_id)
);
CREATE INDEX IF NOT EXISTS idx_alias_norm ON player_aliases(alias_norm);

CREATE TABLE IF NOT EXISTS player_seasons (
    player_id TEXT NOT NULL,
    season    INTEGER NOT NULL,
    team      TEXT,
    position  TEXT,
    games     INTEGER,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id         INTEGER PRIMARY KEY,
    url            TEXT NOT NULL,
    source         TEXT NOT NULL,
    season         INTEGER NOT NULL,
    published_at   TEXT,                -- best-effort display date
    snapshot_at    TEXT,                -- wayback capture ts / live fetch ts
    effective_date TEXT NOT NULL,       -- time-lock key (YYYY-MM-DD)
    doc_type       TEXT NOT NULL,       -- article | rankings_page | adp_page
    categories     TEXT,                -- JSON array
    title          TEXT,
    text           TEXT,
    content_sha1   TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_docs_season_date ON documents(season, effective_date);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, text, content='documents', content_rowid='doc_id'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, text) VALUES (new.doc_id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, text)
    VALUES ('delete', old.doc_id, old.title, old.text);
END;

CREATE TABLE IF NOT EXISTS doc_players (
    doc_id        INTEGER NOT NULL REFERENCES documents(doc_id),
    player_id     TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (doc_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_doc_players_pid ON doc_players(player_id);

CREATE TABLE IF NOT EXISTS rankings (
    source      TEXT NOT NULL,
    season      INTEGER NOT NULL,
    scrape_date TEXT NOT NULL,          -- effective date of the snapshot
    position    TEXT NOT NULL,          -- QB/RB/WR/TE/K/DST or OVR
    rank        INTEGER NOT NULL,
    adp         REAL,
    player_id   TEXT,
    raw_name    TEXT NOT NULL,
    PRIMARY KEY (source, season, scrape_date, position, rank)
);
CREATE INDEX IF NOT EXISTS idx_rankings_season ON rankings(season, scrape_date);

CREATE TABLE IF NOT EXISTS weekly_points (
    player_id TEXT NOT NULL,
    season    INTEGER NOT NULL,
    week      INTEGER NOT NULL,
    half_ppr  REAL NOT NULL,
    PRIMARY KEY (player_id, season, week)
);

CREATE TABLE IF NOT EXISTS ground_truth (
    player_id           TEXT NOT NULL,
    season              INTEGER NOT NULL,
    total_ex_final_week REAL NOT NULL,
    games               INTEGER NOT NULL,
    PRIMARY KEY (player_id, season)
);

CREATE TABLE IF NOT EXISTS season_meta (
    season  INTEGER PRIMARY KEY,
    kickoff TEXT NOT NULL                -- date of first regular-season game (YYYY-MM-DD)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    started_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS harness_versions (
    run_id       TEXT NOT NULL,
    generation   INTEGER NOT NULL,
    agent_idx    INTEGER NOT NULL,
    text         TEXT NOT NULL,
    parent_sha1  TEXT,
    audit_status TEXT,                  -- clean | stripped | reverted
    audit_report TEXT,
    PRIMARY KEY (run_id, generation, agent_idx)
);

CREATE TABLE IF NOT EXISTS cost_ledger (
    run_id          TEXT NOT NULL,
    generation      INTEGER,
    phase           TEXT NOT NULL,      -- draft | rewrite | audit | classify
    model           TEXT NOT NULL,
    input_tok       INTEGER NOT NULL DEFAULT 0,
    cache_read_tok  INTEGER NOT NULL DEFAULT 0,
    cache_write_tok INTEGER NOT NULL DEFAULT 0,
    output_tok      INTEGER NOT NULL DEFAULT 0,
    usd             REAL NOT NULL DEFAULT 0
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize) the database. Returns a Row-factory connection."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
