"""Roster and lineup legality for 14-team half-PPR: QB/2RB/2WR/TE/FLEX/K/DST + 6 bench."""

from __future__ import annotations

from dataclasses import dataclass, field

from ffr.config import LineupConfig, lineup_config


@dataclass(frozen=True)
class PlayerRef:
    player_id: str
    name: str
    position: str  # QB/RB/WR/TE/K/DST


@dataclass
class Roster:
    cfg: LineupConfig = field(default_factory=lineup_config)
    players: list[PlayerRef] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.players)

    @property
    def max_size(self) -> int:
        return self.cfg.rounds

    def position_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.players:
            counts[p.position] = counts.get(p.position, 0) + 1
        return counts

    def _min_needed(self, counts: dict[str, int]) -> int:
        """Picks still required to be able to field a full starting lineup."""
        need = 0
        flex_surplus = 0
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            required = self.cfg.slots.get(pos, 0)
            have = counts.get(pos, 0)
            need += max(0, required - have)
            if pos in self.cfg.flex_positions:
                flex_surplus += max(0, have - required)
        if flex_surplus < self.cfg.slots.get("FLEX", 0):
            need += self.cfg.slots.get("FLEX", 0) - flex_surplus
        return need

    def fills_need(self, player: PlayerRef) -> bool:
        """True if drafting this player reduces the unfilled-starting-slot count."""
        counts = self.position_counts()
        before = self._min_needed(counts)
        counts[player.position] = counts.get(player.position, 0) + 1
        return self._min_needed(counts) < before

    def can_add(self, player: PlayerRef) -> tuple[bool, str]:
        """A pick is legal if the roster can still complete a starting lineup."""
        if self.size >= self.max_size:
            return False, "roster full"
        if any(p.player_id == player.player_id for p in self.players):
            return False, "already on roster"
        counts = self.position_counts()
        # K/DST cannot be benched: cap at required slots so league supply
        # (20-32 each for 14 required) can never be exhausted by hoarding.
        if player.position in ("K", "DST"):
            if counts.get(player.position, 0) >= self.cfg.slots.get(player.position, 0):
                return False, f"cannot bench extra {player.position}"
        counts[player.position] = counts.get(player.position, 0) + 1
        remaining_after = self.max_size - (self.size + 1)
        if self._min_needed(counts) > remaining_after:
            return False, (
                f"picking {player.position} leaves too few picks to fill required slots"
            )
        return True, ""

    def add(self, player: PlayerRef) -> None:
        ok, reason = self.can_add(player)
        if not ok:
            raise ValueError(reason)
        self.players.append(player)

    def force_add(self, player: PlayerRef) -> None:
        """Add bypassing legality — engine fallback when the pool is exhausted."""
        self.players.append(player)

    def validate_lineup(self, starter_ids: list[str]) -> tuple[bool, str]:
        """Check that starter_ids exactly fill QB/2RB/2WR/TE/FLEX/K/DST from this roster."""
        slots_total = sum(self.cfg.slots.values())
        if len(set(starter_ids)) != len(starter_ids):
            return False, "duplicate players in lineup"
        if len(starter_ids) != slots_total:
            return False, f"lineup needs exactly {slots_total} players"
        by_id = {p.player_id: p for p in self.players}
        missing = [pid for pid in starter_ids if pid not in by_id]
        if missing:
            return False, f"not on roster: {missing}"
        return self.assign_slots([by_id[pid] for pid in starter_ids])

    def assign_slots(self, starters: list[PlayerRef]) -> tuple[bool, str]:
        counts: dict[str, int] = {}
        for p in starters:
            counts[p.position] = counts.get(p.position, 0) + 1
        flex_needed = self.cfg.slots.get("FLEX", 0)
        flex_used = 0
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            required = self.cfg.slots.get(pos, 0)
            have = counts.get(pos, 0)
            if have < required:
                return False, f"lineup missing {pos}"
            surplus = have - required
            if surplus:
                if pos not in self.cfg.flex_positions:
                    return False, f"too many {pos}"
                flex_used += surplus
        if flex_used != flex_needed:
            return False, f"flex slots filled {flex_used}/{flex_needed}"
        return True, ""
