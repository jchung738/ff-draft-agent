"""Build draft player pools.

Until the rankings corpus exists (Phase 2), the pool is derived from the PRIOR
season's ground truth: ADP = overall rank by prior-season points, projection =
prior-season points. This is a naive but leakage-free prior for bot drafts.
"""

from __future__ import annotations

import sqlite3

from ffr.draft.engine import PoolPlayer

_TOP_N = {"QB": 30, "RB": 60, "WR": 60, "TE": 30, "K": 20, "DST": 20}


def pool_from_prior_season(conn: sqlite3.Connection, season: int) -> list[PoolPlayer]:
    prior = season - 1
    rows = conn.execute(
        """SELECT g.player_id, p.display_name,
                  COALESCE(s.position, p.position) AS position,
                  g.total_ex_final_week AS pts
           FROM ground_truth g
           JOIN players p ON p.player_id = g.player_id
           LEFT JOIN player_seasons s ON s.player_id = g.player_id AND s.season = ?
           WHERE g.season = ?
           ORDER BY g.total_ex_final_week DESC""",
        (prior, prior),
    ).fetchall()

    kept: list[PoolPlayer] = []
    counts: dict[str, int] = {}
    for rank, r in enumerate(rows, start=1):
        pos = r["position"]
        if pos not in _TOP_N:
            continue
        if counts.get(pos, 0) >= _TOP_N[pos]:
            continue
        counts[pos] = counts.get(pos, 0) + 1
        kept.append(
            PoolPlayer(
                player_id=r["player_id"],
                name=r["display_name"],
                position=pos,
                adp=float(rank),
                proj=r["pts"],
            )
        )
    return kept


def pool_from_rankings(
    conn: sqlite3.Connection, season: int, lock_date: str, source: str = "fantasypros"
) -> list[PoolPlayer]:
    """Pool from the latest pre-lock ADP snapshot (available after Phase 2)."""
    rows = conn.execute(
        """SELECT r.player_id, p.display_name,
                  COALESCE(s.position, p.position) AS position,
                  r.adp, r.rank,
                  COALESCE(g.total_ex_final_week, 0) AS prior_pts
           FROM rankings r
           JOIN players p ON p.player_id = r.player_id
           LEFT JOIN player_seasons s ON s.player_id = r.player_id AND s.season = ? - 1
           LEFT JOIN ground_truth g ON g.player_id = r.player_id AND g.season = ? - 1
           WHERE r.season = ? AND r.source = ? AND r.player_id IS NOT NULL
             AND r.scrape_date = (
                 SELECT MAX(scrape_date) FROM rankings
                 WHERE season = ? AND source = ? AND scrape_date <= ?
             )
           ORDER BY r.rank""",
        (season, season, season, source, season, source, lock_date),
    ).fetchall()
    return [
        PoolPlayer(
            player_id=r["player_id"],
            name=r["display_name"],
            position=r["position"],
            adp=r["adp"] if r["adp"] is not None else float(r["rank"]),
            proj=r["prior_pts"],
        )
        for r in rows
    ]
