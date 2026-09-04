"""Fixed-starters scoring: the lineup locked at draft end scores all season.

Team score = sum of each starter's actual half-PPR total over the season's
scored weeks (final regular-season week excluded). Bench contributes nothing.
"""

from __future__ import annotations

import sqlite3


def player_season_total(conn: sqlite3.Connection, player_id: str, season: int) -> float:
    row = conn.execute(
        "SELECT total_ex_final_week FROM ground_truth WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    return row["total_ex_final_week"] if row else 0.0


def score_lineup(conn: sqlite3.Connection, starter_ids: list[str], season: int) -> float:
    return round(sum(player_season_total(conn, pid, season) for pid in starter_ids), 2)
