"""LLM drafter: per-pick manual tool loop with a frozen, cached harness prefix.

Implements the Drafter protocol. Caching layout (prefix-stable):
[tools] -> [system: engine rules + harness, cache_control on last block] -> messages.
The harness is byte-frozen for the whole draft, so the prefix caches across all picks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import anthropic

from ffr.agents.tools import SET_LINEUP_TOOL, TOOLS, ToolDispatcher
from ffr.corpus.api import TimeLockedCorpus
from ffr.draft.engine import DraftEngine

ENGINE_RULES = """You are drafting a fantasy football team in a live 14-team snake draft.

League settings:
- Half-PPR scoring (0.5 per reception), 15 roster spots.
- Starting lineup (locked after the draft for the ENTIRE season): 1 QB, 2 RB, 2 WR,
  1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DST. Bench players score NOTHING, ever.
- Your team's score = the actual season points of your 9 locked starters
  (final regular-season week excluded). Highest score among the 14 teams wins.

Because bench players never score, depth only matters as insurance is worthless —
every pick should be evaluated by how it upgrades your locked starting 9.

You have research tools: news search (only pre-draft news is available), ADP,
player history, and prior-season results. You may use at most {max_tool_calls}
tool calls per pick, so be efficient. When ready, call make_pick. If you fail to
make a legal pick, the system auto-picks for you (badly) — always end with make_pick.

Your strategy document (written by you, refined across many drafts):
"""

MODEL_PRICES = {  # USD per MTok: (input, output, cache_read, cache_write)
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 6.25),
    "claude-fable-5": (10.0, 50.0, 1.0, 12.5),
}


def usage_cost(model: str, usage) -> float:
    inp, out, cread, cwrite = MODEL_PRICES.get(model, (3.0, 15.0, 0.3, 3.75))
    return (
        usage.input_tokens * inp
        + usage.output_tokens * out
        + (usage.cache_read_input_tokens or 0) * cread
        + (usage.cache_creation_input_tokens or 0) * cwrite
    ) / 1e6


def _model_params(model: str) -> dict:
    """Thinking/effort params per model family."""
    if model.startswith("claude-fable"):
        return {}  # thinking always on; omit the parameter entirely
    if model.startswith("claude-haiku"):
        return {}  # no adaptive thinking / effort support
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}


@dataclass
class LLMDrafter:
    model: str
    harness: str
    corpus: TimeLockedCorpus
    max_tool_calls: int = 6
    on_usage: Callable[[str, object], None] | None = None  # (model, usage) -> None
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    notes: str = ""  # within-trial scratchpad, re-injected each pick (bounded)

    def _system(self) -> list[dict]:
        return [
            {
                "type": "text",
                "text": ENGINE_RULES.format(max_tool_calls=self.max_tool_calls)
                + "\n<harness>\n" + self.harness + "\n</harness>",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _loop(
        self,
        dispatcher: ToolDispatcher,
        user_prompt: str,
        tools: list[dict],
        done: Callable[[], bool],
    ) -> None:
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        for _ in range(self.max_tool_calls):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self._system(),
                tools=tools,
                messages=messages,
                **_model_params(self.model),
            )
            if self.on_usage:
                self.on_usage(self.model, response.usage)
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break  # gave up without acting; engine fallback handles it
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                result, is_error = dispatcher.dispatch(tu.name, dict(tu.input))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})
            if done():
                return

    # --- Drafter protocol ---------------------------------------------------

    def pick(self, engine: DraftEngine, team_idx: int) -> str:
        dispatcher = ToolDispatcher(corpus=self.corpus, engine=engine, team_idx=team_idx)
        picks_made = len([e for e in engine.events if e["type"] == "pick"])
        rnd = picks_made // engine.teams + 1
        prompt = (
            f"Round {rnd}, overall pick {picks_made + 1}. It is your turn. "
            f"Research as needed, then call make_pick."
        )
        if self.notes:
            prompt += f"\n\nYour notes from earlier this draft:\n{self.notes[:2000]}"
        self._loop(
            dispatcher, prompt, TOOLS, done=lambda: dispatcher.pick_result is not None
        )
        if dispatcher.pick_result is None:
            raise RuntimeError("drafter did not make a pick")  # engine auto-picks
        return dispatcher.pick_result

    def set_lineup(self, engine: DraftEngine, team_idx: int) -> list[str]:
        dispatcher = ToolDispatcher(corpus=self.corpus, engine=engine, team_idx=team_idx)
        roster = engine.rosters[team_idx]
        roster_desc = "\n".join(
            f"- {p.player_id} {p.name} ({p.position})" for p in roster.players
        )
        prompt = (
            "The draft is over. Lock your starting lineup for the entire season "
            "(1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 DST). Research if needed, then "
            f"call set_lineup.\n\nYour roster:\n{roster_desc}"
        )
        self._loop(
            dispatcher,
            prompt,
            TOOLS[:-1] + [SET_LINEUP_TOOL],  # research tools + set_lineup, no make_pick
            done=lambda: dispatcher.lineup_result is not None,
        )
        if dispatcher.lineup_result is None:
            raise RuntimeError("drafter did not set a lineup")  # engine auto-lineups
        return dispatcher.lineup_result
