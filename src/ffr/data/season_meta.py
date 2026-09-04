"""Season kickoff dates (first regular-season game) — defines the draft lock date."""

from __future__ import annotations

import datetime as dt
import sqlite3

from ffr.config import ensure_ca_bundle


def season_kickoff(conn: sqlite3.Connection, season: int) -> str:
    row = conn.execute(
        "SELECT kickoff FROM season_meta WHERE season = ?", (season,)
    ).fetchone()
    if row:
        return row["kickoff"]
    ensure_ca_bundle()
    import nflreadpy as nfl
    import polars as pl

    sched = nfl.load_schedules([season]).filter(pl.col("game_type") == "REG")
    kickoff = str(sched["gameday"].min())
    conn.execute(
        "INSERT OR REPLACE INTO season_meta VALUES (?,?)", (season, kickoff)
    )
    conn.commit()
    return kickoff


def lock_date(conn: sqlite3.Connection, season: int) -> str:
    """Draft-day information cutoff: the day before the season's first game."""
    kickoff = dt.date.fromisoformat(season_kickoff(conn, season))
    return str(kickoff - dt.timedelta(days=1))
