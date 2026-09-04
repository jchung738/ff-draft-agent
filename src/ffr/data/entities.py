"""Player entity resolution: name normalization, alias lookup, mention tagging."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter

import ahocorasick
from unidecode import unidecode

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm(name: str) -> str:
    """Normalize a player name for alias matching."""
    s = unidecode(name).lower()
    s = s.replace("'", "")  # Le'Veon == LeVeon; before punct->space
    s = _PUNCT.sub(" ", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    # Join runs of single-letter initials: "d j chark" -> "dj chark"
    merged: list[str] = []
    for t in tokens:
        if len(t) == 1 and merged and len(merged[-1]) <= 2 and merged[-1].isalpha():
            merged[-1] += t
        else:
            merged.append(t)
    return _WS.sub(" ", " ".join(merged)).strip()


# Curated aliases the nflverse master doesn't cover: nicknames and historical
# team names. nflverse team stats normalize relocated teams to modern
# abbreviations (STL Rams -> LA, SD Chargers -> LAC, OAK Raiders -> LV).
ALIAS_OVERRIDES: dict[str, str] = {
    "washington redskins dst": "DST_WAS",
    "redskins dst": "DST_WAS",
    "redskins": "DST_WAS",
    "washington football team dst": "DST_WAS",
    "football team dst": "DST_WAS",
    "oakland raiders dst": "DST_LV",
    "raiders dst": "DST_LV",
    "san diego chargers dst": "DST_LAC",
    "st louis rams dst": "DST_LA",
    "los angeles rams dst": "DST_LA",
    "rams dst": "DST_LA",
    # ADP tables often list DSTs as the bare team name
    "washington redskins": "DST_WAS",
    "washington football team": "DST_WAS",
    "oakland raiders": "DST_LV",
    "san diego chargers": "DST_LAC",
    "st louis rams": "DST_LA",
    "la rams defense": "DST_LA",
    "la rams dst": "DST_LA",
    "la chargers defense": "DST_LAC",
    "la chargers dst": "DST_LAC",
}

# Team abbreviations in load_teams that never appear in nflverse stats
# (superseded by the modern abbreviation above).
DEFUNCT_TEAM_ABBRS = {"LAR", "OAK", "SD", "STL"}

# Nicknames the press uses that differ from the nflverse display name.
# display_name -> extra alias forms
NICKNAME_FORMS: dict[str, list[str]] = {
    "Mitchell Trubisky": ["Mitch Trubisky"],
    "Robbie Anderson": ["Robby Anderson"],
    "Robbie Chosen": ["Robby Anderson", "Robbie Anderson"],
    "Joshua Palmer": ["Josh Palmer"],
    "Gabriel Davis": ["Gabe Davis"],
    "Kenneth Walker III": ["Ken Walker"],
    "Chigoziem Okonkwo": ["Chig Okonkwo"],
    "DJ Moore": ["D.J. Moore"],
    "Scott Miller": ["Scotty Miller"],
    "Deonte Harty": ["Deonte Harris"],
    "Joseph Fortson": ["Jody Fortson"],
    "Steven Hauschka": ["Stephen Hauschka"],
    "Marquise Brown": ["Hollywood Brown"],
    "Corey Brown": ["Philly Brown"],
    "Isiah Pacheco": ["Isaih Pacheco"],
    "Nathaniel Dell": ["Tank Dell"],
    "Cadillac Williams": ["Carnell Williams"],
    "Chris Herndon": ["Christopher Herndon"],
}


def resolve(
    conn: sqlite3.Connection,
    raw_name: str,
    season: int,
    position: str | None = None,
    team: str | None = None,
) -> str | None:
    """Resolve a scraped name to a player_id using season/position/team context.

    Returns None when ambiguous or unknown; callers keep raw_name for review.
    """
    alias = norm(raw_name)
    if not alias:
        return None
    rows = conn.execute(
        """SELECT a.player_id, p.position AS master_pos, s.position AS season_pos,
                  s.team, s.season IS NOT NULL AS active,
                  p.rookie_season, p.last_season,
                  (SELECT COALESCE(SUM(g.total_ex_final_week), 0)
                   FROM ground_truth g
                   WHERE g.player_id = a.player_id AND g.season < ?) AS prior_pts
           FROM player_aliases a
           JOIN players p ON p.player_id = a.player_id
           LEFT JOIN player_seasons s ON s.player_id = a.player_id AND s.season = ?
           WHERE a.alias_norm = ?""",
        (season, season, alias),
    ).fetchall()
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]["player_id"]

    def score(r: sqlite3.Row) -> tuple:
        pos = r["season_pos"] or r["master_pos"]
        in_career = (
            r["rookie_season"] is not None
            and r["last_season"] is not None
            and r["rookie_season"] <= season <= r["last_season"]
        )
        return (
            position is not None and pos == position,
            team is not None and r["team"] == team,
            bool(r["active"]),
            in_career,
            # Fantasy production in strictly PRIOR seasons (no future leakage):
            # rankings virtually always mean the fantasy-relevant namesake.
            r["prior_pts"],
        )

    ranked = sorted(rows, key=score, reverse=True)
    # Ambiguous if the top two candidates tie on all context signals.
    if score(ranked[0]) == score(ranked[1]):
        return None
    return ranked[0]["player_id"]


def build_automaton(conn: sqlite3.Connection) -> ahocorasick.Automaton:
    """Aho-Corasick automaton over all alias surface forms -> sets of player_ids.

    Used for document mention tagging and for the harness-audit pre-screen.
    """
    auto = ahocorasick.Automaton()
    by_alias: dict[str, set[str]] = {}
    for row in conn.execute("SELECT alias_norm, player_id FROM player_aliases"):
        by_alias.setdefault(row["alias_norm"], set()).add(row["player_id"])
    for alias, pids in by_alias.items():
        # Only multi-token aliases: single tokens ("smith") flood with false hits.
        if " " in alias:
            auto.add_word(alias, (alias, pids))
    auto.make_automaton()
    return auto


def tag_mentions(auto: ahocorasick.Automaton, text: str) -> Counter[str]:
    """Count player mentions in text. Unambiguous aliases only."""
    haystack = norm(text)
    counts: Counter[str] = Counter()
    for end, (alias, pids) in auto.iter(haystack):
        start = end - len(alias) + 1
        before = haystack[start - 1] if start > 0 else " "
        after = haystack[end + 1] if end + 1 < len(haystack) else " "
        # Require word boundaries so "art smith" doesn't hit inside "start smithing".
        if before == " " and after == " " and len(pids) == 1:
            counts[next(iter(pids))] += 1
    return counts
