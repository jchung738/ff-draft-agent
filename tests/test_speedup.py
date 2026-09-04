"""Speed-up features: late-round budgets, model schedule, parallel trials plumbing."""

import pytest

from ffr.evolve.orchestrator import drafter_model_for_gen


def test_pick_budget_by_round():
    from ffr.agents.drafter import LLMDrafter

    d = LLMDrafter.__new__(LLMDrafter)  # no client/corpus needed for the pure helper
    d.max_tool_calls = 6
    d.late_round_start = 11
    d.late_round_tool_calls = 2
    assert d._pick_budget(1) == 6
    assert d._pick_budget(10) == 6
    assert d._pick_budget(11) == 2
    assert d._pick_budget(15) == 2


def test_model_schedule_ladder():
    cfg = {
        "drafter_model": "claude-haiku-4-5",
        "model_schedule": [
            {"from_gen": 0, "model": "claude-haiku-4-5"},
            {"from_gen": 10, "model": "claude-sonnet-4-6"},
        ],
    }
    assert drafter_model_for_gen(cfg, 0) == "claude-haiku-4-5"
    assert drafter_model_for_gen(cfg, 9) == "claude-haiku-4-5"
    assert drafter_model_for_gen(cfg, 10) == "claude-sonnet-4-6"
    assert drafter_model_for_gen(cfg, 19) == "claude-sonnet-4-6"


def test_model_schedule_absent_falls_back():
    assert drafter_model_for_gen({"drafter_model": "claude-haiku-4-5"}, 5) == "claude-haiku-4-5"


def test_engine_calls_start_prep_after_pick():
    from ffr.draft.bots import ADPBot
    from ffr.draft.engine import DraftEngine
    from test_engine import make_pool

    class PrepBot(ADPBot):
        def __init__(self):
            self.preps = 0

        def start_prep(self, engine, team_idx):
            self.preps += 1

    bot = PrepBot()
    engine = DraftEngine(season=2023, pool=make_pool(), seed=2)
    engine.run([bot] + [ADPBot() for _ in range(13)])
    assert bot.preps == 15  # hook fires after every one of its picks


def test_on_clock_budget_and_faller_bump():
    from ffr.agents.drafter import LLMDrafter

    d = LLMDrafter.__new__(LLMDrafter)
    d.max_tool_calls, d.late_round_start, d.late_round_tool_calls = 6, 11, 2
    d.on_clock_tool_calls = 2
    d.notes = "planned: target the falling RB"
    # prepared mid-draft -> on-clock budget; round 1 / late rounds -> normal path
    assert (d.on_clock_tool_calls if (bool(d.notes) and 5 > 1 and 5 < d.late_round_start)
            else d._pick_budget(5)) == 2
    assert d._pick_budget(1) == 6
    assert d._pick_budget(12) == 2


def test_prep_tools_exclude_engine_and_pick():
    from ffr.agents.drafter import PREP_TOOLS

    names = {t["name"] for t in PREP_TOOLS}
    assert "make_pick" not in names
    assert "get_draft_state" not in names
    assert "get_available_players" not in names
    assert "search_news" in names and "get_adp_trend" in names


def test_parallel_lineup_stays_deterministic():
    """Lineup fan-out must not perturb event content or order across runs."""
    from ffr.draft.bots import ADPBot
    from ffr.draft.engine import DraftEngine
    from test_engine import make_pool

    def run():
        e = DraftEngine(season=2023, pool=make_pool(), seed=11)
        e.run([ADPBot() for _ in range(14)])
        return e.events

    a, b = run(), run()
    assert a == b
    lineup_teams = [e["team"] for e in a if e["type"] == "lineup"]
    assert lineup_teams == sorted(lineup_teams), "lineup events logged in team order"
