"""Tool schemas + dispatcher for drafter agents.

The dispatcher binds a TimeLockedCorpus and the DraftEngine; ALL filtering is
enforced here, server-side. The agent is never trusted to self-filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ffr.corpus.api import TimeLockedCorpus
from ffr.draft.engine import DraftEngine

TOOLS: list[dict] = [
    {
        "name": "search_news",
        "description": (
            "Full-text search over news articles published BEFORE draft day of the "
            "current season (training camp reports, injuries, rookies, rankings, "
            "off-field news). Call this to research players you are considering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. a player name or topic"},
                "player_id": {"type": "string", "description": "Optional: restrict to docs mentioning this player_id"},
                "category": {
                    "type": "string",
                    "enum": ["training_camp", "injury", "rookie", "rankings", "off_field", "trends"],
                },
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_document",
        "description": "Read the full text of a document returned by search_news.",
        "input_schema": {
            "type": "object",
            "properties": {"doc_id": {"type": "integer"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "get_adp",
        "description": (
            "Latest pre-draft ADP (average draft position) snapshot for the current "
            "season. Use this to gauge market consensus and find value vs ADP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}, "offset": {"type": "integer"}},
        },
    },
    {
        "name": "get_adp_trend",
        "description": "A player's ADP across pre-draft snapshots this summer (riser/faller signal).",
        "input_schema": {
            "type": "object",
            "properties": {"player_id": {"type": "string"}},
            "required": ["player_id"],
        },
    },
    {
        "name": "get_player_history",
        "description": "A player's per-season fantasy totals for PRIOR seasons only.",
        "input_schema": {
            "type": "object",
            "properties": {"player_id": {"type": "string"}},
            "required": ["player_id"],
        },
    },
    {
        "name": "get_prior_season_results",
        "description": (
            "Top fantasy scorers for a PRIOR season (strictly before the current "
            "draft season). Useful for studying historical positional trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {"type": "integer"},
                "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE", "K", "DST"]},
                "limit": {"type": "integer"},
            },
            "required": ["season"],
        },
    },
    {
        "name": "get_draft_state",
        "description": "Current round, your roster, your slot, picks until your next turn, and recent picks.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_available_players",
        "description": "Available (undrafted) players sorted by ADP. Call before every pick.",
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "enum": ["QB", "RB", "WR", "TE", "K", "DST"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "make_pick",
        "description": "Draft a player by player_id. Ends your turn if the pick is legal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "string"},
                "reasoning": {
                    "type": "string",
                    "description": "1-2 sentences: why this pick over the alternatives you considered",
                },
            },
            "required": ["player_id", "reasoning"],
        },
    },
]

SET_LINEUP_TOOL: dict = {
    "name": "set_lineup",
    "description": (
        "Lock your starting lineup: exactly 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX "
        "(RB/WR/TE), 1 K, 1 DST from your roster. Starters score weekly; when a "
        "starter misses a game your best active bench player at the position "
        "auto-covers (byes can also pull from waivers), so pick starters for "
        "output and keep your best insurance on the bench."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "starter_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exactly 9 player_ids from your roster",
            }
        },
        "required": ["starter_ids"],
    },
}


@dataclass
class ToolDispatcher:
    corpus: TimeLockedCorpus
    engine: DraftEngine
    team_idx: int
    pick_result: str | None = None      # set when make_pick is accepted
    pick_reason: str | None = None
    lineup_result: list[str] | None = None
    # research provenance for the current pick (auto-tracked, not model-claimed)
    searches: list[str] = None  # type: ignore[assignment]
    docs_read: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.searches = []
        self.docs_read = []

    def dispatch(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (result_json, is_error)."""
        try:
            result = self._run(name, args)
            return json.dumps(result, default=str), False
        except Exception as e:
            return json.dumps({"error": str(e)}), True

    def _run(self, name: str, args: dict):
        roster = self.engine.rosters[self.team_idx]
        if name == "search_news":
            self.searches.append(args["query"])
            return self.corpus.search_news(
                args["query"],
                player_id=args.get("player_id"),
                category=args.get("category"),
                limit=args.get("limit", 10),
            )
        if name == "read_document":
            doc = self.corpus.read_document(int(args["doc_id"]))
            if doc is None:
                return {"error": "document not found"}
            self.docs_read.append(
                {
                    "source": doc["source"],
                    "date": doc["effective_date"],
                    "title": doc["title"],
                    "url": doc["url"],
                }
            )
            doc["text"] = doc["text"][:8000]
            return doc
        if name == "get_adp":
            return self.corpus.get_adp(args.get("limit", 50), args.get("offset", 0))
        if name == "get_adp_trend":
            return self.corpus.get_adp_trend(args["player_id"])
        if name == "get_player_history":
            return self.corpus.get_player_history(args["player_id"])
        if name == "get_prior_season_results":
            return self.corpus.get_prior_season_results(
                int(args["season"]), args.get("position"), args.get("limit", 50)
            )
        if name == "get_draft_state":
            picks = [e for e in self.engine.events if e["type"] == "pick"]
            order = self.engine.pick_order()
            next_my_pick = next(
                (i - len(picks) for i in range(len(picks), len(order)) if order[i] == self.team_idx),
                None,
            )
            return {
                "round": len(picks) // self.engine.teams + 1,
                "pick_number": len(picks) + 1,
                "my_draft_slot": self.engine.draft_slot(self.team_idx) + 1,
                "picks_until_my_turn": next_my_pick,
                "my_roster": [
                    {"player_id": p.player_id, "name": p.name, "position": p.position}
                    for p in roster.players
                ],
                "recent_picks": [
                    {"name": e["name"], "position": e["position"], "team": e["team"]}
                    for e in picks[-14:]
                ],
            }
        if name == "get_available_players":
            players = self.engine.available_sorted(args.get("position"))
            return [
                {"player_id": p.player_id, "name": p.name, "position": p.position, "adp": p.adp}
                for p in players[: min(args.get("limit", 25), 60)]
            ]
        if name == "make_pick":
            pid = args["player_id"]
            player = self.engine.available.get(pid)
            if player is None:
                return {"error": f"player_id {pid} is invalid or already drafted"}
            ok, reason = roster.can_add(player.ref())
            if not ok:
                return {"error": f"illegal pick: {reason}"}
            self.pick_result = pid
            self.pick_reason = (args.get("reasoning") or "").strip()[:600] or None
            return {"ok": True, "drafted": player.name}
        if name == "set_lineup":
            starters = list(args["starter_ids"])
            ok, reason = roster.validate_lineup(starters)
            if not ok:
                return {"error": f"invalid lineup: {reason}"}
            self.lineup_result = starters
            return {"ok": True}
        raise ValueError(f"unknown tool {name}")
