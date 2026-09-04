"""Snake-draft state machine: seeded slots, pick validation, auto-pick, JSONL log."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ffr.config import lineup_config
from ffr.draft.roster import PlayerRef, Roster


@dataclass(frozen=True)
class PoolPlayer:
    player_id: str
    name: str
    position: str
    adp: float          # average draft position (lower = earlier)
    proj: float         # pre-draft projection (e.g. prior-season points)

    def ref(self) -> PlayerRef:
        return PlayerRef(self.player_id, self.name, self.position)


class Drafter(Protocol):
    """A draft participant. Implementations must be stateless across trials."""

    def pick(self, engine: "DraftEngine", team_idx: int) -> str:
        """Return the player_id to draft."""
        ...

    def set_lineup(self, engine: "DraftEngine", team_idx: int) -> list[str]:
        """Return starter player_ids after the draft."""
        ...


@dataclass
class DraftEngine:
    season: int
    pool: list[PoolPlayer]
    seed: int = 0
    log_path: Path | None = None
    teams: int = field(init=False)
    rounds: int = field(init=False)
    rosters: list[Roster] = field(init=False)
    slot_order: list[int] = field(init=False)   # slot_order[draft_slot] = drafter index
    available: dict[str, PoolPlayer] = field(init=False)
    events: list[dict] = field(default_factory=list)
    lineups: dict[int, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cfg = lineup_config()
        self.teams = cfg.teams
        self.rounds = cfg.rounds
        self.rosters = [Roster() for _ in range(self.teams)]
        rng = random.Random(self.seed)
        self.slot_order = list(range(self.teams))
        rng.shuffle(self.slot_order)
        self.available = {p.player_id: p for p in self.pool}

    # --- mechanics -------------------------------------------------------

    def pick_order(self) -> list[int]:
        """Team index for every pick, snake order over seeded slots."""
        order = []
        for rnd in range(self.rounds):
            slots = range(self.teams) if rnd % 2 == 0 else reversed(range(self.teams))
            order.extend(self.slot_order[s] for s in slots)
        return order

    def draft_slot(self, team_idx: int) -> int:
        return self.slot_order.index(team_idx)

    def available_sorted(self, position: str | None = None) -> list[PoolPlayer]:
        players = [
            p for p in self.available.values() if position is None or p.position == position
        ]
        return sorted(players, key=lambda p: p.adp)

    def auto_pick(self, team_idx: int) -> PoolPlayer:
        """Best available by ADP, preferring players that fill an unfilled starting slot.

        Mirrors standard autodraft: starters before backups, and never a pick
        that makes the roster uncompletable.
        """
        roster = self.rosters[team_idx]
        legal = [p for p in self.available_sorted() if roster.can_add(p.ref())[0]]
        for p in legal:
            if p.position not in ("K", "DST") and roster.fills_need(p.ref()):
                return p
        if legal:
            # Bench best-ADP; K/DST arrive here once legality forces them.
            return legal[0]
        # Pool exhausted at some required position: take best available anyway.
        return self.available_sorted()[0]

    def _log(self, event: dict) -> None:
        self.events.append(event)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def _execute_pick(
        self, pick_no: int, rnd: int, team_idx: int, drafter: Drafter
    ) -> None:
        roster = self.rosters[team_idx]
        chosen: PoolPlayer | None = None
        forfeited = None
        reason: str | None = None
        sources: dict | None = None
        for attempt in range(2):
            try:
                pid = drafter.pick(self, team_idx)
                reason = getattr(drafter, "last_pick_reason", None)
                sources = getattr(drafter, "last_pick_sources", None)
            except Exception as e:  # drafter crash → auto-pick
                forfeited = f"drafter error: {e}"
                break
            player = self.available.get(pid)
            if player is None:
                forfeited = f"invalid or taken player_id: {pid}"
                continue
            ok, why_illegal = roster.can_add(player.ref())
            if not ok:
                forfeited = f"illegal pick {pid}: {why_illegal}"
                continue
            chosen, forfeited = player, None
            break
        if chosen is None:
            chosen = self.auto_pick(team_idx)
        if roster.can_add(chosen.ref())[0]:
            roster.add(chosen.ref())
        else:
            roster.force_add(chosen.ref())
            forfeited = (forfeited or "") + " [forced: pool exhausted]"
        del self.available[chosen.player_id]
        self._log(
            {
                "type": "pick",
                "pick": pick_no,
                "round": rnd + 1,
                "team": team_idx,
                "slot": self.draft_slot(team_idx),
                "player_id": chosen.player_id,
                "name": chosen.name,
                "position": chosen.position,
                "adp": chosen.adp,
                "forfeited": forfeited,
                "reason": reason if not forfeited else None,
                "sources": sources if not forfeited else None,
            }
        )

    def auto_lineup(self, team_idx: int) -> list[str]:
        """Greedy legal lineup maximizing projections."""
        cfg = lineup_config()
        roster = self.rosters[team_idx]
        proj = {p.player_id: p.proj for p in self.pool}
        remaining = sorted(
            roster.players, key=lambda p: proj.get(p.player_id, 0.0), reverse=True
        )
        starters: list[str] = []
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            for _ in range(cfg.slots.get(pos, 0)):
                pick = next((p for p in remaining if p.position == pos), None)
                if pick:
                    remaining.remove(pick)
                    starters.append(pick.player_id)
        for _ in range(cfg.slots.get("FLEX", 0)):
            pick = next((p for p in remaining if p.position in cfg.flex_positions), None)
            if pick:
                remaining.remove(pick)
                starters.append(pick.player_id)
        return starters

    # --- run -------------------------------------------------------------

    def run(self, drafters: list[Drafter]) -> None:
        assert len(drafters) == self.teams
        self._log(
            {
                "type": "start",
                "season": self.season,
                "seed": self.seed,
                "slot_order": self.slot_order,
            }
        )
        for pick_no, team_idx in enumerate(self.pick_order(), start=1):
            rnd = (pick_no - 1) // self.teams
            self._execute_pick(pick_no, rnd, team_idx, drafters[team_idx])

        for team_idx, drafter in enumerate(drafters):
            roster = self.rosters[team_idx]
            fallback = None
            try:
                starters = drafter.set_lineup(self, team_idx)
                ok, reason = roster.validate_lineup(starters)
                if not ok:
                    fallback = f"invalid lineup: {reason}"
            except Exception as e:
                fallback = f"lineup error: {e}"
            if fallback:
                starters = self.auto_lineup(team_idx)
            self.lineups[team_idx] = starters
            self._log(
                {
                    "type": "lineup",
                    "team": team_idx,
                    "starters": starters,
                    "fallback": fallback,
                }
            )
