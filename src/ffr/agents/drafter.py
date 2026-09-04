"""LLM drafter: per-pick manual tool loop with a frozen, cached harness prefix.

Implements the Drafter protocol. Caching layout (prefix-stable):
[tools] -> [system: engine rules + harness, cache_control on last block] -> messages.
The harness is byte-frozen for the whole draft, so the prefix caches across all picks.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from ffr.agents.tools import SET_LINEUP_TOOL, TOOLS, ToolDispatcher
from ffr.corpus.api import TimeLockedCorpus
from ffr.draft.engine import DraftEngine

# Corpus-only research tools (safe off-thread: never touch live engine state)
PREP_TOOLS = [t for t in TOOLS if t["name"] not in
              ("get_draft_state", "get_available_players", "make_pick")]

ENGINE_RULES = """You are drafting a fantasy football team in a live 14-team snake draft.

League settings:
- Half-PPR scoring (0.5 per reception), 15 roster spots.
- Starting lineup (locked after the draft): 1 QB, 2 RB, 2 WR, 1 TE,
  1 FLEX (RB/WR/TE), 1 K, 1 DST.
- Weekly simulation over the season (final regular-season week excluded):
  * Starters score their actual points each week they play.
  * INJURY/INACTIVE: if a starter misses a game, your best active BENCH player
    at that position automatically covers the slot that week. Handcuffs and
    bench depth therefore have real value.
  * BYE weeks: the slot is covered by the best of your bench OR the best
    available waiver (undrafted) player, so byes are survivable.
  * K/DST: any absence (bye or injury) auto-streams from waivers, so never
    spend draft capital insuring those slots.
  * Substitutions are automatic next-man-up by season-to-date form — you make
    no in-season decisions, so draft the roster you'd want that engine to run.
- Your team's score = sum of weekly lineup points. Highest of 14 teams wins.

Evaluate early picks by starter quality; evaluate bench picks by how likely
they are to be needed (injury-prone starters, ambiguous backfields, handcuffs)
and how well they'd score if pressed into the lineup.

Each pick prompt already includes your draft state and the top available
players by ADP — do not spend tool calls re-fetching those. Use your limited
tool budget (stated each pick) on what the prompt cannot tell you: news search
(only pre-draft news exists), ADP trends, player history, prior-season results.
When ready, call make_pick. If you fail to make a legal pick, the system
auto-picks for you (badly) — always end with make_pick.

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
    late_round_start: int = 11      # from this round on, use the reduced budget
    late_round_tool_calls: int = 2
    prep_research: bool = True      # research the next pick between turns
    prep_tool_calls: int = 3
    on_clock_tool_calls: int = 2    # budget when a prep plan is in hand
    faller_threshold: float = 8.0   # ADP this far below the pick number = faller
    on_usage: Callable[[str, object], None] | None = None  # (model, usage) -> None
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    notes: str = ""  # within-trial scratchpad, re-injected each pick (bounded)
    last_pick_reason: str | None = None   # engine reads these into the pick event
    last_pick_sources: dict | None = None  # auto-tracked research provenance
    _prep_thread: threading.Thread | None = field(default=None, repr=False)
    _prep_error: Exception | None = field(default=None, repr=False)
    _prep_sources: dict | None = field(default=None, repr=False)

    def _system(self) -> list[dict]:
        return [
            {
                "type": "text",
                "text": ENGINE_RULES + "\n<harness>\n" + self.harness + "\n</harness>",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _pick_budget(self, rnd: int) -> int:
        return self.max_tool_calls if rnd < self.late_round_start else self.late_round_tool_calls

    # --- between-turn prep research ------------------------------------------

    def start_prep(self, engine: DraftEngine, team_idx: int) -> None:
        """Called by the engine right after this agent picks: research the NEXT
        pick in a background thread while the other 13 teams are on the clock.
        The snapshot is taken synchronously; the thread touches only the corpus."""
        if not self.prep_research:
            return
        picks_made = len([e for e in engine.events if e["type"] == "pick"])
        next_rnd = picks_made // engine.teams + 1
        if next_rnd > engine.rounds or next_rnd >= self.late_round_start:
            return  # late rounds run on the cheap budget without prep
        snap = ToolDispatcher(corpus=self.corpus, engine=engine, team_idx=team_idx)
        state, _ = snap.dispatch("get_draft_state", {})
        available, _ = snap.dispatch("get_available_players", {"limit": 40})
        prompt = (
            "PREPARATION (you are NOT on the clock). Your next pick is roughly "
            f"{json.loads(state).get('picks_until_my_turn', '?')} picks away.\n"
            f"<draft_state>\n{state}\n</draft_state>\n"
            f"<available_players>\n{available}\n</available_players>\n"
            f"Judge who will plausibly still be available at your next turn (players "
            f"above your pick number by ADP will likely be gone). Research the best "
            f"candidates with at most {self.prep_tool_calls - 1} tool calls, then "
            "write your plan as plain text: ranked targets with one-line whys, plus "
            "a contingency if your top target is gone. Keep it under 250 words."
        )
        if self.notes:
            prompt += f"\n\nYour previous notes:\n{self.notes[:1500]}"

        def work() -> None:
            try:
                prep_dispatcher = ToolDispatcher(
                    corpus=self.corpus, engine=engine, team_idx=team_idx
                )
                plan = self._prep_loop(prep_dispatcher, prompt)
                if plan:
                    self.notes = plan[:2000]
                self._prep_sources = {
                    "queries": prep_dispatcher.searches,
                    "docs": prep_dispatcher.docs_read,
                }
            except Exception as e:  # on-clock loop re-hits budget errors itself
                self._prep_error = e

        self._prep_thread = threading.Thread(target=work, daemon=True)
        self._prep_thread.start()

    def _prep_loop(self, dispatcher: ToolDispatcher, prompt: str) -> str:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        final = ""
        for _ in range(self.prep_tool_calls):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                system=self._system(),
                tools=PREP_TOOLS,
                messages=messages,
                **_model_params(self.model),
            )
            if self.on_usage:
                self.on_usage(self.model, response.usage)
            texts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
            if texts:
                final = " ".join(texts)
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for tu in tool_uses:
                result, is_error = dispatcher.dispatch(tu.name, dict(tu.input))
                results.append(
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": result, "is_error": is_error}
                )
            messages.append({"role": "user", "content": results})
        return final

    def _loop(
        self,
        dispatcher: ToolDispatcher,
        user_prompt: str,
        tools: list[dict],
        done: Callable[[], bool],
        budget: int | None = None,
        final_tool: str | None = None,
    ) -> None:
        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        self._last_text = ""
        limit = budget or self.max_tool_calls
        params = _model_params(self.model)
        for step in range(limit):
            # On the last allowed call, force the decision tool: an agent may
            # research until then, but it can never end its turn without acting.
            # (Forced tool_choice is incompatible with thinking, so thinking
            # models get a hard text nudge instead.)
            extra = {}
            if final_tool and step == limit - 1:
                if "thinking" in params:
                    nudge = f"FINAL CALL: you must call {final_tool} now — no more research."
                    last = messages[-1]
                    if isinstance(last["content"], list):
                        last["content"].append({"type": "text", "text": nudge})
                    else:
                        last["content"] += "\n\n" + nudge
                else:
                    extra["tool_choice"] = {"type": "tool", "name": final_tool}
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self._system(),
                tools=tools,
                messages=messages,
                **extra,
                **params,
            )
            if self.on_usage:
                self.on_usage(self.model, response.usage)
            texts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
            if texts:
                self._last_text = " ".join(texts)
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
        # join the between-turn prep (usually finished 13 picks ago)
        if self._prep_thread is not None:
            self._prep_thread.join(timeout=240)
            self._prep_thread = None
        self._prep_error = None  # on-clock loop re-raises budget errors itself

        dispatcher = ToolDispatcher(corpus=self.corpus, engine=engine, team_idx=team_idx)
        picks_made = len([e for e in engine.events if e["type"] == "pick"])
        pick_no = picks_made + 1
        rnd = picks_made // engine.teams + 1
        prepared = bool(self.notes) and rnd > 1 and rnd < self.late_round_start
        budget = self.on_clock_tool_calls if prepared else self._pick_budget(rnd)

        # engine-computed faller detection: don't trust the model to notice
        fallers = [
            p for p in engine.available_sorted()[:40]
            if p.adp + self.faller_threshold <= pick_no
        ]
        if fallers:
            budget += 1

        # Prefetch what agents always ask for first — saves 1-2 round-trips/pick.
        state, _ = dispatcher.dispatch("get_draft_state", {})
        available, _ = dispatcher.dispatch("get_available_players", {"limit": 25})
        prompt = (
            f"Round {rnd}, overall pick {pick_no}. It is your turn.\n"
            f"<draft_state>\n{state}\n</draft_state>\n"
            f"<available_players>\n{available}\n</available_players>\n"
        )
        if fallers:
            names = ", ".join(
                f"{p.name} ({p.position}, ADP {p.adp:.0f})" for p in fallers[:5]
            )
            prompt += (
                f"UNEXPECTED FALLERS still on the board: {names}. A faller can be a "
                "bargain or a red flag — one news check may be worth it.\n"
            )
        prompt += (
            f"You have at most {budget} tool calls this pick (make_pick included) — "
            f"research only what the context above cannot tell you, then call make_pick."
        )
        if self.notes:
            prompt += f"\n\nYour prepared plan/notes:\n{self.notes[:2000]}"
        self._loop(
            dispatcher, prompt, TOOLS,
            done=lambda: dispatcher.pick_result is not None,
            budget=budget,
            final_tool="make_pick",
        )
        # fall back to the agent's own commentary if it skipped the reasoning arg
        self.last_pick_reason = dispatcher.pick_reason or (
            self._last_text.strip()[:600] or None
        )
        sources = {"queries": dispatcher.searches, "docs": dispatcher.docs_read}
        if self._prep_sources:  # include the between-turn research provenance
            sources = {
                "queries": self._prep_sources["queries"] + sources["queries"],
                "docs": self._prep_sources["docs"] + sources["docs"],
            }
            self._prep_sources = None
        self.last_pick_sources = sources
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
            final_tool="set_lineup",
        )
        if dispatcher.lineup_result is None:
            raise RuntimeError("drafter did not set a lineup")  # engine auto-lineups
        return dispatcher.lineup_result
