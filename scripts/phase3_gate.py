"""Phase 3 gate: one LLM drafter vs 13 ADP bots on a real season.

Verifies: tool loop terminates within budgets, picks are legal, lineup is set,
cost per draft is sane, and prompt caching is actually hitting.
"""

from __future__ import annotations

import sys

from ffr.agents.drafter import LLMDrafter, usage_cost
from ffr.config import CONFIG_DIR
from ffr.corpus.api import TimeLockedCorpus
from ffr.data import store
from ffr.data.season_meta import lock_date
from ffr.draft.bots import ADPBot
from ffr.draft.engine import DraftEngine
from ffr.draft.pool import pool_from_rankings
from ffr.draft.scoring import score_lineup

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
MODEL = "claude-haiku-4-5"

conn = store.connect()
lock = lock_date(conn, SEASON)
corpus = TimeLockedCorpus(conn, SEASON, lock)

totals = {"usd": 0.0, "calls": 0, "cache_read": 0, "input": 0, "output": 0}


def on_usage(model: str, usage) -> None:
    totals["usd"] += usage_cost(model, usage)
    totals["calls"] += 1
    totals["cache_read"] += usage.cache_read_input_tokens or 0
    totals["input"] += usage.input_tokens
    totals["output"] += usage.output_tokens


harness = (CONFIG_DIR / "seed_harness.md").read_text()
llm = LLMDrafter(model=MODEL, harness=harness, corpus=corpus, on_usage=on_usage)
drafters = [llm] + [ADPBot() for _ in range(13)]

engine = DraftEngine(
    season=SEASON, pool=pool_from_rankings(conn, SEASON, lock), seed=7
)
engine.run(drafters)

scores = {i: score_lineup(conn, engine.lineups[i], SEASON) for i in range(14)}
rank = sorted(scores, key=lambda i: -scores[i]).index(0) + 1
llm_picks = [e for e in engine.events if e["type"] == "pick" and e["team"] == 0]
forfeits = [e for e in llm_picks if e["forfeited"]]
lineup_ev = next(e for e in engine.events if e["type"] == "lineup" and e["team"] == 0)

print(f"season {SEASON} | LLM ({MODEL}) drafted from slot {engine.draft_slot(0) + 1}")
print(f"LLM score: {scores[0]:.1f}  rank {rank}/14  (field: "
      f"{min(scores.values()):.1f}-{max(scores.values()):.1f})")
print("LLM picks:", [f"R{e['round']} {e['name']}" for e in llm_picks[:8]])
print(f"forfeited picks: {len(forfeits)}  lineup fallback: {lineup_ev['fallback']}")
print(f"api calls: {totals['calls']}  input tok: {totals['input']}  "
      f"cache-read tok: {totals['cache_read']}  output tok: {totals['output']}")
print(f"cost: ${totals['usd']:.3f}")
print("GATE:", "PASS" if rank <= 10 and len(forfeits) <= 3 else "REVIEW")
