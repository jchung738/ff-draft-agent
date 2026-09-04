"""Ingest nflverse data: players, aliases, weekly half-PPR points, season ground truth.

DST is modeled as pseudo-players with player_id 'DST_<team_abbr>'.
The season's final regular-season week is excluded per config/scoring.yaml.
"""

from __future__ import annotations

import sqlite3

import polars as pl

from ffr.config import ensure_ca_bundle, load_scoring_config, week_window
from ffr.data.entities import (
    ALIAS_OVERRIDES,
    DEFUNCT_TEAM_ABBRS,
    NICKNAME_FORMS,
    norm,
)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def _half_ppr_expr(s: dict) -> pl.Expr:
    """Half-PPR points for skill players from weekly stat columns."""
    return (
        pl.col("passing_yards").fill_null(0) * s["pass_yd"]
        + pl.col("passing_tds").fill_null(0) * s["pass_td"]
        + pl.col("passing_interceptions").fill_null(0) * s["interception"]
        + (pl.col("rushing_yards").fill_null(0) + pl.col("receiving_yards").fill_null(0)) * s["rush_yd"]
        + (pl.col("rushing_tds").fill_null(0) + pl.col("receiving_tds").fill_null(0)) * s["rush_td"]
        + pl.col("receptions").fill_null(0) * s["reception"]
        + (
            pl.col("sack_fumbles_lost").fill_null(0)
            + pl.col("rushing_fumbles_lost").fill_null(0)
            + pl.col("receiving_fumbles_lost").fill_null(0)
        )
        * s["fumble_lost"]
        + (
            pl.col("passing_2pt_conversions").fill_null(0)
            + pl.col("rushing_2pt_conversions").fill_null(0)
            + pl.col("receiving_2pt_conversions").fill_null(0)
        )
        * s["two_point"]
        + pl.col("special_teams_tds").fill_null(0) * s["rush_td"]
    )


def _kicker_expr(s: dict) -> pl.Expr:
    fg_short = (
        pl.col("fg_made_0_19").fill_null(0)
        + pl.col("fg_made_20_29").fill_null(0)
        + pl.col("fg_made_30_39").fill_null(0)
    )
    fg_40 = pl.col("fg_made_40_49").fill_null(0)
    fg_50 = pl.col("fg_made_50_59").fill_null(0) + pl.col("fg_made_60_").fill_null(0)
    return (
        fg_short * s["fg_0_39"]
        + fg_40 * s["fg_40_49"]
        + fg_50 * s["fg_50_plus"]
        + pl.col("fg_missed").fill_null(0) * s["fg_miss"]
        + pl.col("pat_made").fill_null(0) * s["xp_made"]
        + pl.col("pat_missed").fill_null(0) * s["xp_miss"]
    )


def _pa_points(pa: float, tiers: list[list[float]]) -> float:
    for lo, hi, pts in tiers:
        if lo <= pa <= hi:
            return pts
    return 0.0


def ingest_players(conn: sqlite3.Connection) -> int:
    """Load nflverse player master into players + player_aliases (plus DST pseudo-players)."""
    ensure_ca_bundle()
    import nflreadpy as nfl

    players = nfl.load_players().filter(pl.col("gsis_id").is_not_null())
    rows = []
    aliases = []
    for r in players.iter_rows(named=True):
        pid = r["gsis_id"]
        rows.append(
            (pid, r["display_name"], r["position"], r["rookie_season"], r["last_season"])
        )
        forms = {
            r["display_name"],
            r["football_name"],
            r["short_name"],
            f"{r['first_name']} {r['last_name']}",
            f"{r['common_first_name']} {r['last_name']}" if r["common_first_name"] else None,
        }
        forms.update(NICKNAME_FORMS.get(r["display_name"], []))
        for form in forms:
            if form and (a := norm(form)):
                aliases.append((a, pid, "nflverse"))

    teams = nfl.load_teams()
    for t in teams.unique(subset=["team_abbr"]).iter_rows(named=True):
        if t["team_abbr"] in DEFUNCT_TEAM_ABBRS:
            continue
        pid = f"DST_{t['team_abbr']}"
        name = f"{t['team_name']} DST"
        rows.append((pid, name, "DST", None, None))
        city = t["team_name"].rsplit(" ", 1)[0]  # "Denver Broncos" -> "Denver"
        for form in (
            name, t["team_name"], f"{t['team_nick']} DST", t["team_nick"],
            f"{t['team_name']} Defense", f"{city} Defense", f"{city} DST",
        ):
            if form and (a := norm(form)):
                aliases.append((a, pid, "nflverse"))

    # Drop defunct DST rows/aliases left by earlier ingests
    for abbr in DEFUNCT_TEAM_ABBRS:
        conn.execute("DELETE FROM player_aliases WHERE player_id = ?", (f"DST_{abbr}",))
        conn.execute("DELETE FROM players WHERE player_id = ?", (f"DST_{abbr}",))

    conn.executemany(
        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?)", rows
    )
    conn.executemany(
        "INSERT OR IGNORE INTO player_aliases VALUES (?,?,?)", aliases
    )
    known = {r[0] for r in rows}
    conn.executemany(
        "INSERT OR REPLACE INTO player_aliases VALUES (?,?,?)",
        [(a, pid, "override") for a, pid in ALIAS_OVERRIDES.items() if pid in known],
    )
    conn.commit()
    return len(rows)


def ingest_season_stats(conn: sqlite3.Connection, seasons: list[int]) -> None:
    """Compute weekly half-PPR points and season ground truth for the given seasons."""
    ensure_ca_bundle()
    import nflreadpy as nfl

    s = load_scoring_config()["scoring"]
    dst_cfg = load_scoring_config()["dst"]

    stats = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    skill = (
        stats.filter(pl.col("position").is_in(SKILL_POSITIONS))
        .with_columns(_half_ppr_expr(s).alias("half_ppr"))
        .select("player_id", "season", "week", "half_ppr", "team", "position")
    )
    kickers = (
        stats.filter(pl.col("position") == "K")
        .with_columns(_kicker_expr(s).alias("half_ppr"))
        .select("player_id", "season", "week", "half_ppr", "team", "position")
    )

    # DST: team defensive stats + points allowed from schedules
    ts = nfl.load_team_stats(seasons).filter(pl.col("season_type") == "REG")
    sched = nfl.load_schedules(seasons).filter(pl.col("game_type") == "REG")
    pa_home = sched.select(
        pl.col("season"), pl.col("week"),
        pl.col("home_team").alias("team"), pl.col("away_score").alias("points_allowed"),
    )
    pa_away = sched.select(
        pl.col("season"), pl.col("week"),
        pl.col("away_team").alias("team"), pl.col("home_score").alias("points_allowed"),
    )
    pa = pl.concat([pa_home, pa_away])
    dst = ts.join(pa, on=["season", "week", "team"], how="left").with_columns(
        (
            pl.col("def_sacks").fill_null(0) * dst_cfg["sack"]
            + pl.col("def_interceptions").fill_null(0) * dst_cfg["interception"]
            + pl.col("fumble_recovery_opp").fill_null(0) * dst_cfg["fumble_recovery"]
            + (pl.col("def_tds").fill_null(0) + pl.col("fumble_recovery_tds").fill_null(0))
            * dst_cfg["td"]
            + pl.col("def_safeties").fill_null(0) * dst_cfg["safety"]
        ).alias("base_pts")
    )

    wp_rows: list[tuple] = []
    season_rows: dict[tuple[str, int], tuple[float, int]] = {}
    ps_rows: dict[tuple[str, int], tuple[str | None, str | None, int]] = {}

    def account(pid: str, season: int, week: int, pts: float, team: str | None, pos: str | None):
        wp_rows.append((pid, season, week, round(pts, 2)))
        win = week_window(season)
        if win.first_week <= week <= win.last_week:
            tot, games = season_rows.get((pid, season), (0.0, 0))
            season_rows[(pid, season)] = (tot + pts, games + 1)
        _, _, n = ps_rows.get((pid, season), (None, None, 0))
        ps_rows[(pid, season)] = (team, pos, n + 1)

    for df in (skill, kickers):
        for r in df.iter_rows(named=True):
            account(r["player_id"], r["season"], r["week"], r["half_ppr"], r["team"], r["position"])

    tiers = dst_cfg["points_allowed_tiers"]
    for r in dst.iter_rows(named=True):
        pts = r["base_pts"] + _pa_points(r["points_allowed"] or 0, tiers)
        account(f"DST_{r['team']}", r["season"], r["week"], pts, r["team"], "DST")

    conn.executemany(
        "INSERT OR REPLACE INTO weekly_points VALUES (?,?,?,?)", wp_rows
    )
    conn.executemany(
        "INSERT OR REPLACE INTO ground_truth VALUES (?,?,?,?)",
        [
            (pid, season, round(tot, 2), games)
            for (pid, season), (tot, games) in season_rows.items()
        ],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_seasons VALUES (?,?,?,?,?)",
        [
            (pid, season, team, pos, games)
            for (pid, season), (team, pos, games) in ps_rows.items()
        ],
    )
    conn.commit()
