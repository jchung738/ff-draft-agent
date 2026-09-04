"""Roster scoring with weekly bench substitution and bye-week waiver streaming.

Rules (all decisions use only backward-looking information):
- The lineup locked at draft end starts every scored week.
- PROMOTION: a bench player whose recent form (PPG over his last N played
  games) beats a slot occupant's by a clear margin takes the slot — drafted
  breakouts earn their real value. The demoted player goes to the bench and
  can win the job back symmetrically. Waivers never promote; K/DST excluded.
- INJURY/INACTIVE (occupant's team plays, occupant doesn't): the best eligible
  BENCH player who is active covers the slot — handcuffs earn their value.
- BYE (team has no game that week — a schedule fact known preseason): the slot
  is covered by the best of bench OR waivers (undrafted players), matching how
  real managers stream bye weeks.
- K/DST: these positions cannot be benched, so ANY absence (bye or injury)
  streams from waivers — as real managers do.
- "Best" = season-to-date PPG through the prior week; draft order breaks
  week-1 ties for bench. Byes never occur in week 1, so waiver ranking always
  has real data.
- One player can cover only one slot per week. Waiver fills are week-scoped
  streams (no roster mutation); cross-team waiver contention is not modeled.
"""

from __future__ import annotations

import sqlite3

from ffr.config import lineup_config, load_scoring_config, week_window


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


def simulate_roster(
    conn: sqlite3.Connection,
    starter_ids: list[str],
    bench_ids: list[str],
    season: int,
    drafted_ids: set[str] | None = None,
) -> dict:
    """Weekly simulation with bench substitution + bye-week waiver streaming.

    Returns the full season trace: per-week slot occupants (with substitution
    cause and source), weekly totals, and every rostered player's weekly points.
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

    promo = load_scoring_config().get("promotion", {})
    promo_on = promo.get("enabled", True)
    window = promo.get("window_games", 3)
    min_games = promo.get("min_games", 2)
    m_ratio = promo.get("margin_ratio", 1.2)
    m_pts = promo.get("margin_points", 2.0)
    roster_ids = list(dict.fromkeys(starter_ids + bench_ids))
    recent: dict[str, list[float]] = {}  # roster players' played-game points

    def form(pid: str) -> float | None:
        games = recent.get(pid, [])
        if len(games) < min_games:
            return None
        tail = games[-window:]
        return sum(tail) / len(tail)

    # slot owners are mutable: in-form bench players can take a slot
    owners: list[list] = [[slot, starter, starter] for slot, starter in slots]
    bench_now: list[str] = list(bench_ids)

    total = 0.0
    weeks: list[dict] = []
    for week in range(win.first_week, win.last_week + 1):
        # 1. promotion pass — backward-looking form only; waivers never promote
        if promo_on:
            for row in owners:
                slot, _orig, owner = row
                if slot in ("K", "DST"):
                    continue
                eligible = cfg.flex_positions if slot == "FLEX" else (slot,)
                best, best_form = None, None
                for b in bench_now:
                    if pos_of(b) not in eligible:
                        continue
                    bf = form(b)
                    if bf is not None and (best_form is None or bf > best_form):
                        best, best_form = b, bf
                if best is None:
                    continue
                owner_form = form(owner) or 0.0
                if best_form >= max(owner_form * m_ratio, owner_form + m_pts):
                    bench_now.remove(best)
                    bench_now.append(owner)  # demoted: coverage + comeback path
                    row[2] = best

        # 2. weekly coverage + scoring
        used: set[str] = set()
        week_total = 0.0
        slot_rows: list[dict] = []
        for slot, orig, owner in owners:
            occupant = None
            sub = None
            if (owner, week) in pts:
                occupant = owner
                if owner != orig:
                    sub = {"cause": "promotion", "from": "bench"}
            else:
                cause = "bye" if on_bye(owner, week) else "injury"
                eligible = cfg.flex_positions if slot == "FLEX" else (slot,)
                candidates = [
                    b for b in bench_now
                    if b not in used and pos_of(b) in eligible and (b, week) in pts
                ]
                if cause == "bye" or slot in ("K", "DST"):
                    # byes stream from waivers; K/DST always can (no bench allowed)
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
                    sub = {
                        "cause": cause,
                        "from": "bench" if occupant in drafted else "waiver",
                    }
                else:
                    sub = {"cause": cause, "from": "none"}
            points = pts.get((occupant, week), 0.0) if occupant else 0.0
            if occupant is not None:
                used.add(occupant)
                week_total += points
            slot_rows.append(
                {
                    "slot": slot,
                    "starter": orig,
                    "occupant": occupant,
                    "points": round(points, 2),
                    "sub": sub,
                }
            )
        total += week_total
        weeks.append({"week": week, "total": round(week_total, 2), "slots": slot_rows})
        for (pid, wk), val in pts.items():  # update trackers after scoring the week
            if wk == week:
                c = cum.setdefault(pid, [0.0, 0])
                c[0] += val
                c[1] += 1
                if pid in roster_ids:
                    recent.setdefault(pid, []).append(val)

    roster_ids = list(dict.fromkeys(starter_ids + bench_ids))
    occupant_ids = {
        s["occupant"] for w in weeks for s in w["slots"] if s["occupant"]
    }
    all_ids = list(dict.fromkeys(roster_ids + sorted(occupant_ids)))
    names = {
        r["player_id"]: r["display_name"]
        for r in conn.execute(
            f"""SELECT player_id, display_name FROM players
                WHERE player_id IN ({",".join("?" * len(all_ids))})""",
            all_ids,
        )
    } if all_ids else {}
    players = {
        pid: {
            "name": names.get(pid, pid),
            "position": pos_of(pid),
            "role": "starter" if pid in starter_ids else "bench",
            "weekly": {
                wk: round(pts[(pid, wk)], 2)
                for wk in range(win.first_week, win.last_week + 1)
                if (pid, wk) in pts
            },
        }
        for pid in roster_ids
    }
    return {
        "total": round(total, 2),
        "weeks": weeks,
        "players": players,
        "names": names,
    }


def score_roster(
    conn: sqlite3.Connection,
    starter_ids: list[str],
    bench_ids: list[str],
    season: int,
    drafted_ids: set[str] | None = None,
) -> float:
    return simulate_roster(conn, starter_ids, bench_ids, season, drafted_ids)["total"]
