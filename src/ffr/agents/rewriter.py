"""Post-trial harness self-rewrite.

Each agent sees only its OWN audited harness, its own draft log, its scores, and
per-pick outcome deltas — never other agents' harnesses. The rewrite is reminded
that player-specific content will be stripped by the audit, so durable strategy
is the only thing worth writing.
"""

from __future__ import annotations

import json

import anthropic

from ffr.agents.drafter import usage_cost

REWRITE_SYSTEM = """You are a fantasy football drafting agent improving your own
strategy document ("harness") between trials of a backtesting experiment.

Rules of the experiment:
- Next trial you start blank: the harness is the ONLY thing you keep.
- The harness will be audited: any mention of specific players, teams+seasons,
  season-specific ADP anchors, or specific outcomes will be STRIPPED. Writing them
  is wasted space. Obfuscating them gets the whole harness reverted.
- Future trials may be ANY past season, and eventually unseen seasons. Only
  season-agnostic strategy generalizes: positional value logic, roster
  construction, ADP-relative value rules, attribute-based heuristics (age,
  injury history, role ambiguity, rookie profiles), news-interpretation rules,
  and tool-usage policy (which tools to call, when, and what to look for).

Study your draft results: where you gained or lost value vs the field, which
picks busted and what pre-draft signals (news categories, ADP trends, history)
could have flagged them. Then rewrite the harness to draft better in ANY season.
Keep it under 1500 words. Output ONLY the new harness text."""


def rewrite_harness(
    client: anthropic.Anthropic,
    model: str,
    harness: str,
    trial_summary: dict,
    on_usage=None,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=6000,
        system=REWRITE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    "<current_harness>\n" + harness + "\n</current_harness>\n\n"
                    "<trial_results>\n" + json.dumps(trial_summary, default=str)
                    + "\n</trial_results>\n\nWrite the improved harness now."
                ),
            }
        ],
    )
    if on_usage:
        on_usage(model, response.usage)
    return next(b.text for b in response.content if b.type == "text").strip()
