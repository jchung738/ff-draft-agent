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
