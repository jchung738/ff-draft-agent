"""Draft engine properties: snake order, legality, determinism, auto-pick fallback."""

import pytest

from ffr.config import lineup_config
from ffr.draft.bots import ADPBot, RandomBot, VORBot
from ffr.draft.engine import DraftEngine, PoolPlayer


def make_pool() -> list[PoolPlayer]:
    """Synthetic pool: enough at every position for 14 teams x 15 rounds."""
    sizes = {"QB": 30, "RB": 70, "WR": 70, "TE": 30, "K": 20, "DST": 20}
    pool, adp = [], 1.0
    # interleave positions so ADP order mixes positions realistically
    queues = {
        pos: [
            PoolPlayer(f"{pos}{i}", f"{pos} Player{i}", pos, 0.0, 300.0 - adpish)
            for i, adpish in zip(range(1, n + 1), range(n))
        ]
        for pos, n in sizes.items()
    }
    # Real ADP shape: skill positions interleaved, K/DST at the very end.
    order = ["RB", "WR", "RB", "WR", "QB", "TE", "RB", "WR", "QB", "TE"]
    while any(queues[pos] for pos in order):
        for pos in order:
            if queues[pos]:
                p = queues[pos].pop(0)
                pool.append(
                    PoolPlayer(p.player_id, p.name, p.position, adp, p.proj)
                )
                adp += 1.0
    for pos in ("K", "DST"):
        for p in queues[pos]:
            pool.append(PoolPlayer(p.player_id, p.name, p.position, adp, p.proj))
            adp += 1.0
    return pool


def run_draft(seed=7, drafters=None):
    engine = DraftEngine(season=2023, pool=make_pool(), seed=seed)
    drafters = drafters or [ADPBot() for _ in range(14)]
    engine.run(drafters)
    return engine


def test_snake_order_reverses():
    engine = DraftEngine(season=2023, pool=make_pool(), seed=3)
    order = engine.pick_order()
    assert order[:14] == list(reversed(order[14:28]))
    assert order[28:42] == order[:14]
    assert len(order) == 14 * 15


def test_full_draft_completes_legally():
    engine = run_draft()
    cfg = lineup_config()
    for team_idx, roster in enumerate(engine.rosters):
        assert roster.size == cfg.rounds
        ok, reason = roster.validate_lineup(engine.lineups[team_idx])
        assert ok, f"team {team_idx}: {reason}"
    picked = [e["player_id"] for e in engine.events if e["type"] == "pick"]
    assert len(picked) == len(set(picked)), "player drafted twice"


def test_determinism_same_seed():
    a, b = run_draft(seed=11), run_draft(seed=11)
    assert [e for e in a.events] == [e for e in b.events]


def test_different_seed_different_slots():
    a = DraftEngine(season=2023, pool=make_pool(), seed=1)
    b = DraftEngine(season=2023, pool=make_pool(), seed=2)
    assert a.slot_order != b.slot_order


class BrokenBot:
    def pick(self, engine, team_idx):
        return "nonexistent_player"

    def set_lineup(self, engine, team_idx):
        raise RuntimeError("boom")


def test_autopick_and_autolineup_fallback():
    drafters = [BrokenBot()] + [ADPBot() for _ in range(13)]
    engine = run_draft(drafters=drafters)
    forfeits = [
        e for e in engine.events if e["type"] == "pick" and e["team"] == 0 and e["forfeited"]
    ]
    assert len(forfeits) == 15, "every BrokenBot pick should forfeit to auto-pick"
    lineup_events = [e for e in engine.events if e["type"] == "lineup" and e["team"] == 0]
    assert lineup_events[0]["fallback"], "broken lineup should fall back"
    ok, _ = engine.rosters[0].validate_lineup(engine.lineups[0])
    assert ok


def test_vor_beats_random_on_projections():
    """VOR drafting should assemble higher-projected lineups than random picks."""

    def lineup_proj(engine, team_idx):
        proj = {p.player_id: p.proj for p in engine.pool}
        return sum(proj[pid] for pid in engine.lineups[team_idx])

    vor_scores, rnd_scores = [], []
    for seed in range(6):
        drafters = [VORBot() if i < 7 else RandomBot(seed * 100 + i) for i in range(14)]
        engine = DraftEngine(season=2023, pool=make_pool(), seed=seed)
        engine.run(drafters)
        for i in range(14):
            (vor_scores if i < 7 else rnd_scores).append(lineup_proj(engine, i))
    assert sum(vor_scores) / len(vor_scores) > sum(rnd_scores) / len(rnd_scores)
