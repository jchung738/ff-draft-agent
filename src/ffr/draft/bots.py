"""Rule-based baseline drafters: ADP-greedy, value-over-replacement, random."""

from __future__ import annotations

import random

from ffr.config import lineup_config
from ffr.draft.engine import DraftEngine, PoolPlayer


class ADPBot:
    """Always takes the best available player by ADP (legal picks only)."""

    def pick(self, engine: DraftEngine, team_idx: int) -> str:
        return engine.auto_pick(team_idx).player_id

    def set_lineup(self, engine: DraftEngine, team_idx: int) -> list[str]:
        return engine.auto_lineup(team_idx)


class RandomBot:
    """Picks a uniformly random legal player. Baseline floor."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def pick(self, engine: DraftEngine, team_idx: int) -> str:
        roster = engine.rosters[team_idx]
        legal = [
            p for p in engine.available.values() if roster.can_add(p.ref())[0]
        ]
        return self.rng.choice(legal).player_id

    def set_lineup(self, engine: DraftEngine, team_idx: int) -> list[str]:
        return engine.auto_lineup(team_idx)


# Replacement level: projection of the N-th best player at each position,
# N ~= number rostered as startable across a 14-team league.
_REPLACEMENT_RANK = {"QB": 14, "RB": 42, "WR": 42, "TE": 14, "K": 14, "DST": 14}


class VORBot:
    """Drafts the player with the largest projection above replacement level."""

    def __init__(self) -> None:
        self._repl: dict[str, float] | None = None

    def _replacement(self, engine: DraftEngine) -> dict[str, float]:
        if self._repl is None:
            self._repl = {}
            for pos, n in _REPLACEMENT_RANK.items():
                at_pos = sorted(
                    (p for p in engine.pool if p.position == pos),
                    key=lambda p: p.proj,
                    reverse=True,
                )
                self._repl[pos] = at_pos[n - 1].proj if len(at_pos) >= n else 0.0
        return self._repl

    def pick(self, engine: DraftEngine, team_idx: int) -> str:
        repl = self._replacement(engine)
        roster = engine.rosters[team_idx]

        def vor(p: PoolPlayer) -> float:
            return p.proj - repl.get(p.position, 0.0)

        # Fill K/DST only in the last two rounds regardless of VOR.
        picks_left = roster.max_size - roster.size
        counts = roster.position_counts()
        candidates = [
            p for p in engine.available.values() if roster.can_add(p.ref())[0]
        ]
        if picks_left > 2:
            skill = [p for p in candidates if p.position not in ("K", "DST")]
            candidates = skill or candidates
        else:
            cfg = lineup_config()
            for pos in ("K", "DST"):
                if counts.get(pos, 0) < cfg.slots.get(pos, 0):
                    forced = [p for p in candidates if p.position == pos]
                    if forced:
                        candidates = forced
                        break
        return max(candidates, key=vor).player_id

    def set_lineup(self, engine: DraftEngine, team_idx: int) -> list[str]:
        return engine.auto_lineup(team_idx)
