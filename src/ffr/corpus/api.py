"""TimeLockedCorpus: the ONLY read path drafter agents get.

Every query is filtered server-side to effective_date <= lock_date for the draft
season, and outcome data is reachable only for seasons strictly before it.
Agents are never trusted to self-filter.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


def _fts_quote(query: str) -> str:
    """Quote user text as FTS5 phrase terms (prevents syntax injection)."""
    terms = [t.replace('"', "") for t in query.split()]
    return " OR ".join(f'"{t}"' for t in terms if t)


@dataclass
class TimeLockedCorpus:
    conn: sqlite3.Connection
    season: int
    lock_date: str  # YYYY-MM-DD; day before season kickoff

    # --- news ---------------------------------------------------------------

    def search_news(
        self,
        query: str,
        player_id: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        sql = """
            SELECT d.doc_id, d.title, d.source, d.effective_date, d.categories,
                   snippet(documents_fts, 1, '[', ']', ' … ', 30) AS snippet
            FROM documents_fts f
            JOIN documents d ON d.doc_id = f.rowid
            WHERE documents_fts MATCH ?
              AND d.season = ? AND d.effective_date <= ?
        """
        params: list = [_fts_quote(query), self.season, self.lock_date]
        if player_id:
            sql += " AND d.doc_id IN (SELECT doc_id FROM doc_players WHERE player_id = ?)"
            params.append(player_id)
        if category:
            sql += " AND d.categories LIKE ?"
            params.append(f'%"{category}"%')
        sql += " ORDER BY rank LIMIT ?"
        params.append(min(limit, 25))
        return [dict(r) for r in self.conn.execute(sql, params)]

    def read_document(self, doc_id: int) -> dict | None:
        row = self.conn.execute(
            """SELECT doc_id, url, source, effective_date, categories, title, text
               FROM documents
               WHERE doc_id = ? AND season = ? AND effective_date <= ?""",
            (doc_id, self.season, self.lock_date),
        ).fetchone()
        return dict(row) if row else None

    # --- rankings / ADP -----------------------------------------------------

    def get_adp(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Latest pre-lock ADP snapshot."""
        return [
            dict(r)
            for r in self.conn.execute(
                """SELECT r.rank, r.adp, r.player_id, r.raw_name AS name,
                          COALESCE(s.position, p.position) AS position, s.team
                   FROM rankings r
                   LEFT JOIN players p ON p.player_id = r.player_id
                   LEFT JOIN player_seasons s
                          ON s.player_id = r.player_id AND s.season = ? - 1
                   WHERE r.season = ? AND r.scrape_date = (
                       SELECT MAX(scrape_date) FROM rankings
                       WHERE season = ? AND scrape_date <= ?
                   )
                   ORDER BY r.rank LIMIT ? OFFSET ?""",
                (self.season, self.season, self.season, self.lock_date,
                 min(limit, 100), offset),
            )
        ]

    def get_adp_trend(self, player_id: str) -> list[dict]:
        """A player's ADP across all pre-lock snapshots (rising/falling signal)."""
        return [
            dict(r)
            for r in self.conn.execute(
                """SELECT scrape_date, rank, adp FROM rankings
                   WHERE season = ? AND player_id = ? AND scrape_date <= ?
                   ORDER BY scrape_date""",
                (self.season, player_id, self.lock_date),
            )
        ]

    # --- history (strictly prior seasons) ------------------------------------

    def get_player_history(self, player_id: str) -> list[dict]:
        return [
            dict(r)
            for r in self.conn.execute(
                """SELECT g.season, g.total_ex_final_week AS points, g.games,
                          s.team, s.position
                   FROM ground_truth g
                   LEFT JOIN player_seasons s
                          ON s.player_id = g.player_id AND s.season = g.season
                   WHERE g.player_id = ? AND g.season < ?
                   ORDER BY g.season""",
                (player_id, self.season),
            )
        ]

    def get_prior_season_results(
        self, season: int, position: str | None = None, limit: int = 50
    ) -> list[dict]:
        if season >= self.season:
            return []  # draft-season (or future) outcomes are never reachable
        sql = """
            SELECT g.player_id, p.display_name AS name,
                   COALESCE(s.position, p.position) AS position,
                   g.total_ex_final_week AS points, g.games
            FROM ground_truth g
            JOIN players p ON p.player_id = g.player_id
            LEFT JOIN player_seasons s
                   ON s.player_id = g.player_id AND s.season = g.season
            WHERE g.season = ?
        """
        params: list = [season]
        if position:
            sql += " AND COALESCE(s.position, p.position) = ?"
            params.append(position)
        sql += " ORDER BY g.total_ex_final_week DESC LIMIT ?"
        params.append(min(limit, 100))
        return [dict(r) for r in self.conn.execute(sql, params)]
