# ff-draft-agent

Evolutionary system that produces an LLM "harness" (self-written strategy document)
that is extremely good at fantasy football drafting.

**How it works:** 14 LLM drafter agents compete in 14-team half-PPR snake drafts for
past seasons (2015–2025), seeing only news/rankings published before that season's
draft day (time-locked via Wayback Machine snapshot dates). Rosters are scored against
actual season results (fixed starting lineup, final regular-season week excluded).
Between trials each agent rewrites its own harness — the only artifact that persists;
an auditor agent strips any player-specific or year-specific leakage every generation.
The consistently top-ranked harness wins.

## Honest caveat on leakage
The base model has pretraining knowledge of past seasons' outcomes. The primary metric
is therefore **relative rank among agents sharing the same base model** and the
train (2015–2022) vs holdout (2023–2024) gap — not absolute points, which are inflated
for pre-cutoff seasons.

## Setup
```
uv sync --extra dev
uv run pytest
uv run ffr ingest-stats            # nflverse players + weekly half-PPR points
uv run ffr draft --season 2022     # bot-only sanity draft
```
Behind a TLS-intercepting proxy, build `data/ca_bundle.pem` (certifi + macOS keychain);
`ffr` picks it up automatically.

## Layout
- `config/` — scoring rules, week windows, lineup, crawl sources
- `src/ffr/data/` — SQLite store (FTS5), nflverse ground truth, Wayback crawler, entities
- `src/ffr/corpus/` — TimeLockedCorpus, the only read path agents get
- `src/ffr/draft/` — snake-draft engine, roster legality, fixed-starters scoring, bots
- `src/ffr/agents/` — drafter tool loop, harness rewriter, leakage pre-screen, auditor
- `src/ffr/evolve/` — generation orchestrator, lineage, cost ledger

## Scoring decisions
- Half-PPR; DST modeled as `DST_<team>` pseudo-players; K from distance-tiered FGs.
- Ground truth excludes the final regular-season week: wk 17 for seasons ≤2020,
  wk 18 for ≥2021 ("fantasy championship ends before the last week").
