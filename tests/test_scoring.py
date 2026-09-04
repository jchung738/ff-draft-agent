"""Scoring math + excluded-week logic."""

import polars as pl
import pytest

from ffr.config import load_scoring_config, week_window
from ffr.data.ground_truth import _half_ppr_expr, _kicker_expr, _pa_points
from ffr.data.store import connect
from ffr.draft.scoring import score_lineup


def test_week_window_16_game_era():
    win = week_window(2019)
    assert (win.first_week, win.last_week) == (1, 16)  # wk 17 excluded


def test_week_window_17_game_era():
    win = week_window(2023)
    assert (win.first_week, win.last_week) == (1, 17)  # wk 18 excluded


def test_half_ppr_known_stat_line():
    # 300 pass yds, 2 pass TD, 1 INT, 50 rush yds, 1 rush TD => 12+8-2+5+6 = 29
    df = pl.DataFrame(
        {
            "passing_yards": [300.0], "passing_tds": [2], "passing_interceptions": [1],
            "rushing_yards": [50.0], "receiving_yards": [0.0],
            "rushing_tds": [1], "receiving_tds": [0], "receptions": [0],
            "sack_fumbles_lost": [0], "rushing_fumbles_lost": [0],
            "receiving_fumbles_lost": [0], "passing_2pt_conversions": [0],
            "rushing_2pt_conversions": [0], "receiving_2pt_conversions": [0],
            "special_teams_tds": [0],
        }
    )
    s = load_scoring_config()["scoring"]
    pts = df.with_columns(_half_ppr_expr(s).alias("p"))["p"][0]
    assert pts == pytest.approx(29.0)


def test_half_ppr_receiver_line():
    # 8 rec, 110 yds, 1 TD, 1 fumble lost => 4 + 11 + 6 - 2 = 19
    df = pl.DataFrame(
        {
            "passing_yards": [0.0], "passing_tds": [0], "passing_interceptions": [0],
            "rushing_yards": [0.0], "receiving_yards": [110.0],
            "rushing_tds": [0], "receiving_tds": [1], "receptions": [8],
            "sack_fumbles_lost": [0], "rushing_fumbles_lost": [0],
            "receiving_fumbles_lost": [1], "passing_2pt_conversions": [0],
            "rushing_2pt_conversions": [0], "receiving_2pt_conversions": [0],
            "special_teams_tds": [0],
        }
    )
    s = load_scoring_config()["scoring"]
    pts = df.with_columns(_half_ppr_expr(s).alias("p"))["p"][0]
    assert pts == pytest.approx(19.0)


def test_kicker_line():
    # 2 FG 0-39, 1 FG 40-49, 1 miss, 3 XP => 6 + 4 - 1 + 3 = 12
    df = pl.DataFrame(
        {
            "fg_made_0_19": [0], "fg_made_20_29": [1], "fg_made_30_39": [1],
            "fg_made_40_49": [1], "fg_made_50_59": [0], "fg_made_60_": [0],
            "fg_missed": [1], "pat_made": [3], "pat_missed": [0],
        }
    )
    s = load_scoring_config()["scoring"]
    pts = df.with_columns(_kicker_expr(s).alias("p"))["p"][0]
    assert pts == pytest.approx(12.0)


def test_dst_points_allowed_tiers():
    tiers = load_scoring_config()["dst"]["points_allowed_tiers"]
    assert _pa_points(0, tiers) == 10.0
    assert _pa_points(3, tiers) == 7.0
    assert _pa_points(21, tiers) == 0.0
    assert _pa_points(45, tiers) == -4.0


def test_fixed_starters_scoring(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    conn.executemany(
        "INSERT INTO ground_truth VALUES (?,?,?,?)",
        [("p1", 2023, 200.0, 17), ("p2", 2023, 150.5, 16)],
    )
    assert score_lineup(conn, ["p1", "p2"], 2023) == pytest.approx(350.5)
    assert score_lineup(conn, ["p1", "missing"], 2023) == pytest.approx(200.0)


# --- next-man-up substitution + bye-week waiver streaming --------------------


@pytest.fixture
def sub_conn(tmp_path):
    """Season 2019 (weeks 1-16 scored). Teams AAA (bye wk 2) and BBB.

    rb_star (AAA, starter): plays wk1, bye wk2, injured wk3 (team plays, no row)
    rb_hand (AAA, bench handcuff): plays wk1 and wk3, bye wk2
    rb_wav  (BBB, undrafted): plays wk1-3
    rb_b2   (BBB, second bench RB): plays wk1-3, weaker
    """
    conn = connect(tmp_path / "t.sqlite")
    players = [
        ("rb_star", "Star Back", "RB"), ("rb_hand", "Handcuff Back", "RB"),
        ("rb_wav", "Waiver Back", "RB"), ("rb_b2", "Bench Two", "RB"),
    ]
    conn.executemany(
        "INSERT INTO players VALUES (?,?,?,2015,2026)", players
    )
    conn.executemany(
        "INSERT INTO player_seasons VALUES (?,2019,?,'RB',16)",
        [("rb_star", "AAA"), ("rb_hand", "AAA"), ("rb_wav", "BBB"), ("rb_b2", "BBB")],
    )
    weekly = [
        # team-week markers (DST pseudo-players define which teams played)
        ("DST_AAA", 1, 0.0), ("DST_AAA", 3, 0.0),          # AAA bye in wk 2
        ("DST_BBB", 1, 0.0), ("DST_BBB", 2, 0.0), ("DST_BBB", 3, 0.0),
        ("rb_star", 1, 20.0),                                # then bye, then hurt
        ("rb_hand", 1, 5.0), ("rb_hand", 3, 15.0),
        ("rb_wav", 1, 8.0), ("rb_wav", 2, 10.0), ("rb_wav", 3, 12.0),
        ("rb_b2", 1, 2.0), ("rb_b2", 2, 3.0), ("rb_b2", 3, 4.0),
    ]
    conn.executemany(
        "INSERT INTO weekly_points VALUES (?,2019,?,?)", weekly
    )
    conn.commit()
    return conn


def test_injury_uses_bench_next_man_up(sub_conn):
    from ffr.draft.scoring import simulate_roster

    # wk1: star 20 | wk2: AAA bye -> waiver rb_wav 10 (hand also on bye)
    # wk3: injury (AAA played, star absent) -> bench only -> hand 15
    sim = simulate_roster(
        sub_conn, ["rb_star"], ["rb_hand"], 2019,
        drafted_ids={"rb_star", "rb_hand"},
    )
    assert sim["total"] == pytest.approx(20 + 10 + 15)
    wk2, wk3 = sim["weeks"][1]["slots"][0], sim["weeks"][2]["slots"][0]
    assert wk2["occupant"] == "rb_wav" and wk2["sub"] == {"cause": "bye", "from": "waiver"}
    assert wk3["occupant"] == "rb_hand" and wk3["sub"] == {"cause": "injury", "from": "bench"}
    assert sim["players"]["rb_hand"]["role"] == "bench"
    assert sim["players"]["rb_star"]["weekly"] == {1: 20.0}


def test_injury_never_pulls_from_waivers(sub_conn):
    from ffr.draft.scoring import score_roster

    # no bench at all: wk2 bye -> waiver 10; wk3 injury -> slot scores 0
    total = score_roster(
        sub_conn, ["rb_star"], [], 2019, drafted_ids={"rb_star"}
    )
    assert total == pytest.approx(20 + 10 + 0)


def test_bye_prefers_better_of_bench_and_waiver(sub_conn):
    from ffr.draft.scoring import score_roster

    # bench rb_b2 has ppg 2.0 after wk1; waiver rb_wav ppg 8.0 -> wk2 uses waiver
    total = score_roster(
        sub_conn, ["rb_star"], ["rb_b2"], 2019,
        drafted_ids={"rb_star", "rb_b2"},
    )
    # wk3 injury: bench-only -> rb_b2 4
    assert total == pytest.approx(20 + 10 + 4)


def test_bench_player_covers_only_one_slot(sub_conn):
    from ffr.draft.scoring import score_roster

    # two injured starters, one bench player: only one slot covered in wk3
    sub_conn.execute("INSERT INTO players VALUES ('rb_star2','Star Two','RB',2015,2026)")
    sub_conn.execute("INSERT INTO player_seasons VALUES ('rb_star2',2019,'AAA','RB',16)")
    sub_conn.execute("INSERT INTO weekly_points VALUES ('rb_star2',2019,1,18.0)")
    sub_conn.commit()
    total = score_roster(
        sub_conn, ["rb_star", "rb_star2"], ["rb_hand"], 2019,
        drafted_ids={"rb_star", "rb_star2", "rb_hand"},
    )
    # wk1: 20+18 | wk2: both on bye -> waivers: rb_wav 10 covers one, rb_b2 3 covers other
    # wk3: both injured -> bench only -> hand 15 covers one, other slot 0
    assert total == pytest.approx(38 + 13 + 15)
