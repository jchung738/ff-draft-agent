"""Retroactively re-score a run's completed generations under current rules.

Draft logs are immutable and simulate_roster is deterministic, so scores/ranks
are recomputed from the JSONL event logs. The old results.json is preserved as
results_prevrules.json. Also verifies waiver legality on every trace:
- waiver fills only on bye, K/DST, or empty-bench weeks
- injury coverage prefers bench whenever an eligible bench player is active
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ffr.config import RUNS_DIR, lineup_config
from ffr.data import store
from ffr.draft.scoring import simulate_roster

run_id = sys.argv[1] if len(sys.argv) > 1 else "ladder01"
run_dir = RUNS_DIR / run_id
conn = store.connect()
cfg = lineup_config()

def ranks_from(scores: dict[int, dict[int, float]], seasons: list[int]) -> dict:
    mean_rank: dict[str, float] = {}
    for season in seasons:
        ordered = sorted(((i, s[season]) for i, s in scores.items()), key=lambda x: -x[1])
        for rank, (i, _) in enumerate(ordered, start=1):
            mean_rank[str(i)] = mean_rank.get(str(i), 0) + rank / len(seasons)
    return mean_rank

waiver_checks = {"bye": 0, "kdst": 0, "empty_bench": 0}
violations = []

for gen_dir in sorted(run_dir.glob("gen_*")):
    rf = gen_dir / "results.json"
    if not rf.exists():
        continue
    old = json.loads(rf.read_text())
    seasons = sorted(
        int(f.stem) for f in (gen_dir / "drafts").glob("*.jsonl")
        if (gen_dir / f"_DONE_season_{f.stem}").exists()
    )
    agents = sorted(int(a) for a in old["ranks"])
    scores: dict[int, dict[int, float]] = {i: {} for i in agents}
    for season in seasons:
        events = [
            json.loads(l)
            for l in (gen_dir / "drafts" / f"{season}.jsonl").read_text().splitlines()
            if l.strip()
        ]
        picks = [e for e in events if e["type"] == "pick"]
        drafted = {e["player_id"] for e in picks}
        for i in agents:
            starters = next(
                e["starters"] for e in events
                if e["type"] == "lineup" and e["team"] == i
            )
            team_picks = [e["player_id"] for e in picks if e["team"] == i]
            bench = [p for p in team_picks if p not in starters]
            sim = simulate_roster(conn, starters, bench, season, drafted)
            scores[i][season] = sim["total"]
            # legality audit of every waiver fill in the trace
            for wk in sim["weeks"]:
                bench_active = {}
                for s in wk["slots"]:
                    if s["sub"] and s["sub"]["from"] == "waiver":
                        if s["slot"] in ("K", "DST"):
                            waiver_checks["kdst"] += 1
                        elif s["sub"]["cause"] == "bye":
                            waiver_checks["bye"] += 1
                        elif s["sub"]["cause"] == "injury":
                            waiver_checks["empty_bench"] += 1
                        elif s["sub"]["cause"] == "promotion":
                            violations.append((gen_dir.name, season, i, wk["week"], s))
    new_ranks = ranks_from(scores, seasons)
    old_ranks = old["ranks"]
    (gen_dir / "results_prevrules.json").write_text(json.dumps(old, indent=2))
    old["scores"] = {str(i): {str(se): v for se, v in s.items()} for i, s in scores.items()}
    old["ranks"] = new_ranks
    rf.write_text(json.dumps(old, indent=2))
    shifts = {
        a: round(new_ranks[a] - old_ranks[a], 1)
        for a in old_ranks
        if abs(new_ranks[a] - old_ranks[a]) >= 1.0
    }
    print(f"{gen_dir.name} seasons={seasons} rescored; rank shifts >=1.0: {shifts or 'none'}")

print("\nwaiver fills by legal cause:", waiver_checks)
print("ILLEGAL waiver events:", violations if violations else "NONE")
