"""Roster scoring with weekly bench substitution and bye-week waiver streaming.

Rules (all decisions use only backward-looking information):
- The lineup locked at draft end starts every scored week.
- INJURY/INACTIVE (starter's team plays, starter doesn't): the best eligible
  BENCH player who is active covers the slot — handcuffs earn their value.
- BYE (starter's team has no game that week — a schedule fact known preseason):
  the slot is covered by the best of bench OR waivers (undrafted players),
  matching how real managers stream bye weeks (especially K/DST).
- "Best" = season-to-date PPG through the prior week; draft order breaks
  week-1 ties for bench. Byes never occur in week 1, so waiver ranking always
  has real data.
- One player can cover only one slot per week. Waiver fills are week-scoped
  streams (no roster mutation); cross-team waiver contention is not modeled.
"""

from __future__ import annotations

import sqlite3

from ffr.config import lineup_config, week_window


def player_season_total(conn: sqlite3.Connection, player_id: str, season: int) -> float:
    row = conn.execute(
        "SELECT total_ex_final_week FROM ground_truth WHERE player_id = ? AND season = ?",
        (player_id, season),
    ).fetchone()
    return row["total_ex_final_week"] if row else 0.0


def score_lineup(conn: sqlite3.Connection, starter_ids: list[str], season: int) -> float:
    """Pure fixed-starters sum (no substitution). Kept for per-pick deltas."""
    return round(sum(player_season_total(conn, pid, season) for pid in starter_ids), 2)


def _season_data(conn: sqlite3.Connection, season: int):
    """All weekly points, positions, and teams for the season (small: ~7k rows)."""
    pts: dict[tuple[str, int], float] = {}
    for r in conn.execute(
        "SELECT player_id, week, half_ppr FROM weekly_points WHERE season = ?", (season,)
    ):
        pts[(r["player_id"], r["week"])] = r["half_ppr"]
    meta: dict[str, tuple[str | None, str | None]] = {}
    for r in conn.execute(
        """SELECT p.player_id, COALESCE(s.position, p.position) AS pos, s.team
           FROM players p
           LEFT JOIN player_seasons s
                  ON s.player_id = p.player_id AND s.season = ?""",
        (season,),
    ):
        meta[r["player_id"]] = (r["pos"], r["team"])
    # team played in week W iff its DST pseudo-player has a row that week
    team_weeks = {
        (pid.removeprefix("DST_"), wk) for (pid, wk) in pts if pid.startswith("DST_")
    }
    return pts, meta, team_weeks


def _assign_slots(
    starter_ids: list[str], meta: dict[str, tuple[str | None, str | None]]
) -> list[tuple[str, str]]:
    """Map the 9 starters to named slots. Fixed slots fill in draft order;
    the surplus RB/WR/TE becomes the FLEX."""
    cfg = lineup_config()
    remaining = dict(cfg.slots)
    out: list[tuple[str, str]] = []
    for pid in starter_ids:
        pos = (meta.get(pid) or (None, None))[0] or ""
        if remaining.get(pos, 0) > 0:
            remaining[pos] -= 1
            out.append((pos, pid))
        elif pos in cfg.flex_positions and remaining.get("FLEX", 0) > 0:
            remaining["FLEX"] -= 1
            out.append(("FLEX", pid))
        else:  # invalid lineups are engine-rejected; score the slot as-is
            out.append((pos, pid))
    return out


def score_roster(
    conn: sqlite3.Connection,
    starter_ids: list[str],
    bench_ids: list[str],
    season: int,
    drafted_ids: set[str] | None = None,
) -> float:
    """Weekly scoring with bench substitution + bye-week waiver streaming.

    bench_ids must be in draft order. drafted_ids = every rostered player in
    the league (defines the waiver pool); defaults to starters+bench only.
    """
    cfg = lineup_config()
    win = week_window(season)
    pts, meta, team_weeks = _season_data(conn, season)
    drafted = drafted_ids or set(starter_ids) | set(bench_ids)
    slots = _assign_slots(starter_ids, meta)
    draft_index = {pid: i for i, pid in enumerate(bench_ids)}

    def pos_of(pid: str) -> str | None:
        return (meta.get(pid) or (None, None))[0]

    def on_bye(pid: str, week: int) -> bool:
        team = (meta.get(pid) or (None, None))[1]
        return team is not None and (team, week) not in team_weeks

    # season-to-date (points, games) for everyone, updated week by week
    cum: dict[str, list[float]] = {}

    def ppg(pid: str) -> float:
        c = cum.get(pid)
        return c[0] / c[1] if c and c[1] else -1.0

    total = 0.0
    for week in range(win.first_week, win.last_week + 1):
        used: set[str] = set()
        for slot, starter in slots:
            occupant = None
            if (starter, week) in pts:
                occupant = starter
            else:
                eligible = cfg.flex_positions if slot == "FLEX" else (slot,)
                candidates = [
                    b for b in bench_ids
                    if b not in used and pos_of(b) in eligible and (b, week) in pts
                ]
                if on_bye(starter, week):
                    # bye: waivers are fair game too (undrafted, active this week)
                    candidates += [
                        pid for (pid, wk) in pts
                        if wk == week
                        and pid not in drafted
                        and pid not in used
                        and pos_of(pid) in eligible
                    ]
                if candidates:
                    occupant = max(
                        candidates,
                        key=lambda p: (ppg(p), -draft_index.get(p, 10_000)),
                    )
            if occupant is not None:
                used.add(occupant)
                total += pts[(occupant, week)]
        for (pid, wk), val in pts.items():  # update PPG after scoring the week
            if wk == week:
                c = cum.setdefault(pid, [0.0, 0])
                c[0] += val
                c[1] += 1
    return round(total, 2)
