"""Generation lifecycle: audit -> trials -> rank -> rewrite -> select -> persist.

State lives on disk under runs/<run_id>/gen_XXX/; every step writes a _DONE
marker so a killed run resumes where it stopped. Cost is tracked per response
and the run halts cleanly past the USD budget.

Parallelism: picks within a draft are inherently serial (shared board), but
seasons within a generation, the 14 lineup locks, 14 rewrites, and 14 audits
are all independent and fan out across threads (the work is API-bound; the
anthropic sync client is thread-safe). Each worker opens its own SQLite
connection; the shared cost ledger is guarded by a lock.
"""

from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import yaml
from rich.console import Console

from ffr.agents.auditor import audit_harness
from ffr.agents.drafter import LLMDrafter, usage_cost
from ffr.agents.prescreen import prescreen
from ffr.agents.rewriter import rewrite_harness
from ffr.config import CONFIG_DIR, RUNS_DIR
from ffr.corpus.api import TimeLockedCorpus
from ffr.data import store
from ffr.data.season_meta import lock_date
from ffr.draft.bots import ADPBot
from ffr.draft.engine import DraftEngine
from ffr.draft.pool import pool_from_prior_season, pool_from_rankings
from ffr.draft.scoring import score_lineup, score_roster

console = Console()


class BudgetExceeded(Exception):
    pass


def drafter_model_for_gen(cfg: dict, gen: int) -> str:
    """Model ladder: the last schedule entry whose from_gen <= gen wins."""
    model = cfg.get("drafter_model", "claude-haiku-4-5")
    for entry in sorted(cfg.get("model_schedule") or [], key=lambda e: e["from_gen"]):
        if gen >= entry["from_gen"]:
            model = entry["model"]
    return model


class Orchestrator:
    def __init__(self, run_id: str, config_path: Path | None = None):
        self.run_id = run_id
        cfg_file = config_path or CONFIG_DIR / "run_default.yaml"
        self.cfg = yaml.safe_load(cfg_file.read_text())["run"]
        self.run_dir = RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(cfg_file.read_text())
        self.conn = store.connect()  # main-thread connection
        # ledger connection is shared across worker threads, guarded by a lock
        self.ledger_conn = store.connect(check_same_thread=False)
        self.ledger_lock = threading.Lock()
        self.client = anthropic.Anthropic()
        self.spent_usd = self._ledger_total()
        self.rng = random.Random(self.cfg["seed"])

    # --- cost ----------------------------------------------------------------

    def _ledger_total(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(usd),0) t FROM cost_ledger WHERE run_id=?", (self.run_id,)
        ).fetchone()
        return row["t"]

    def _track(self, phase: str, generation: int):
        def on_usage(model: str, usage):
            usd = usage_cost(model, usage)
            with self.ledger_lock:
                self.spent_usd += usd
                self.ledger_conn.execute(
                    "INSERT INTO cost_ledger VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        self.run_id, generation, phase, model,
                        usage.input_tokens,
                        usage.cache_read_input_tokens or 0,
                        usage.cache_creation_input_tokens or 0,
                        usage.output_tokens, usd,
                    ),
                )
                self.ledger_conn.commit()
                over = self.spent_usd > self.cfg["budget_usd"]
            if over:
                raise BudgetExceeded(f"budget exceeded: ${self.spent_usd:.2f}")

        return on_usage

    # --- persistence helpers --------------------------------------------------

    def _gen_dir(self, gen: int) -> Path:
        d = self.run_dir / f"gen_{gen:03d}"
        (d / "harnesses").mkdir(parents=True, exist_ok=True)
        (d / "drafts").mkdir(exist_ok=True)
        return d

    @staticmethod
    def _done(path: Path, step: str) -> bool:
        return (path / f"_DONE_{step}").exists()

    @staticmethod
    def _mark(path: Path, step: str) -> None:
        (path / f"_DONE_{step}").touch()

    def _harnesses(self, gen: int) -> dict[int, str]:
        d = self._gen_dir(gen) / "harnesses"
        out = {}
        for i in range(self.cfg["llm_agents"]):
            f = d / f"agent_{i:02d}.md"
            if f.exists():
                out[i] = f.read_text()
        return out

    def _seed_harnesses(self) -> None:
        """Generation 0 harnesses: stylistic variants of the seed strategy doc."""
        seed = (CONFIG_DIR / "seed_harness.md").read_text()
        emphases = [
            "Zero-RB: prioritize elite WRs early.", "RB-heavy: lock 3 RBs in 4 rounds.",
            "Balanced best-player-available.", "Early elite TE.", "Late-round QB always.",
            "News-first: research every pick heavily.", "ADP-value purist: only take fallers.",
            "Upside-hunting: prefer high-variance profiles.", "Floor-first: durability above all.",
            "Rookie-heavy: chase breakout profiles.", "Veteran-only: proven roles.",
            "Contrarian: fade market consensus.", "Streamlined: minimal research, trust ADP.",
            "Injury-hawk: veto any injury flag.",
        ]
        d = self._gen_dir(0) / "harnesses"
        for i in range(self.cfg["llm_agents"]):
            (d / f"agent_{i:02d}.md").write_text(
                seed + f"\n## Personal emphasis\n- {emphases[i % len(emphases)]}\n"
            )

    # --- steps ----------------------------------------------------------------

    def _audit(self, gen: int) -> None:
        gen_dir = self._gen_dir(gen)
        if self._done(gen_dir, "audit"):
            return
        on_usage = self._track("audit", gen)
        harnesses = self._harnesses(gen)

        def audit_one(i: int, harness: str):
            wconn = store.connect()
            try:
                violations = prescreen(wconn, harness)
            finally:
                wconn.close()
            if gen == 0 and not violations:
                return i, harness, "clean", None, 0.0  # seed harnesses skip the LLM
            result = audit_harness(self.client, harness, violations, on_usage)
            status = "clean" if result.clean else "stripped"
            return i, result.stripped_harness, status, result.report, result.fraction_stripped

        with ThreadPoolExecutor(max_workers=len(harnesses) or 1) as pool:
            futures = [pool.submit(audit_one, i, h) for i, h in harnesses.items()]
            outcomes = [f.result() for f in futures]

        for i, final, status, report, frac in sorted(outcomes):
            if frac > self.cfg["audit_strip_threshold"]:
                final = self._last_clean_ancestor(gen, i)
                status = "reverted"
            if report is not None:
                (gen_dir / "harnesses" / f"agent_{i:02d}.audit.json").write_text(
                    json.dumps(report, indent=2, default=str)
                )
            (gen_dir / "harnesses" / f"agent_{i:02d}.md").write_text(final)
            self.conn.execute(
                "INSERT OR REPLACE INTO harness_versions VALUES (?,?,?,?,?,?,?)",
                (self.run_id, gen, i, final, None, status, None),
            )
            self.conn.commit()
            console.print(f"  audit agent {i}: {status}")
        self._mark(gen_dir, "audit")

    def _last_clean_ancestor(self, gen: int, agent_idx: int) -> str:
        for g in range(gen - 1, -1, -1):
            row = self.conn.execute(
                """SELECT text FROM harness_versions
                   WHERE run_id=? AND generation=? AND agent_idx=? AND audit_status='clean'""",
                (self.run_id, g, agent_idx),
            ).fetchone()
            if row:
                return row["text"]
        return (CONFIG_DIR / "seed_harness.md").read_text()

    @staticmethod
    def _pool(conn, season: int, lock: str):
        pool = pool_from_rankings(conn, season, lock)
        if len(pool) < 180:  # rankings corpus not ready for this season
            pool = pool_from_prior_season(conn, season)
        return pool

    def _run_season(
        self,
        gen: int,
        season: int,
        harnesses: dict[int, str],
        on_usage,
        model: str,
        out_dir: Path | None = None,
    ) -> list[dict]:
        """One complete draft for one season. Own SQLite connection (thread-safe)."""
        gen_dir = out_dir or self._gen_dir(gen)
        log_path = gen_dir / "drafts" / f"{season}.jsonl"
        if (gen_dir / f"_DONE_season_{season}").exists():
            return [json.loads(l) for l in log_path.read_text().splitlines()]
        wconn = store.connect()
        try:
            log_path.unlink(missing_ok=True)
            lock = lock_date(wconn, season)
            corpus = TimeLockedCorpus(wconn, season, lock)
            n_llm = self.cfg["llm_agents"]
            drafters = [
                LLMDrafter(
                    model=model,
                    harness=harnesses[i],
                    corpus=corpus,
                    max_tool_calls=self.cfg["max_tool_calls_per_pick"],
                    late_round_start=self.cfg.get("late_round_start", 11),
                    late_round_tool_calls=self.cfg.get("late_round_tool_calls", 2),
                    on_usage=on_usage,
                    client=self.client,
                )
                if i < n_llm
                else ADPBot()
                for i in range(self.cfg["agents"])
            ]
            engine = DraftEngine(
                season=season,
                pool=self._pool(wconn, season, lock),
                seed=self.cfg["seed"] * 1000 + gen * 100 + season % 100,
                log_path=log_path,
            )
            engine.run(drafters)
            (gen_dir / f"_DONE_season_{season}").touch()
            return engine.events
        finally:
            wconn.close()

    def _trials(self, gen: int, seasons: list[int]) -> dict:
        gen_dir = self._gen_dir(gen)
        results_file = gen_dir / "results.json"
        if self._done(gen_dir, "trials"):
            return json.loads(results_file.read_text())
        harnesses = self._harnesses(gen)
        on_usage = self._track("draft", gen)
        model = drafter_model_for_gen(self.cfg, gen)
        scores: dict[int, dict[int, float]] = {i: {} for i in harnesses}
        pick_logs: dict[int, dict[int, list]] = {i: {} for i in harnesses}

        # seasons are independent drafts over the same frozen harnesses: parallel
        with ThreadPoolExecutor(max_workers=len(seasons)) as pool:
            futures = {
                pool.submit(self._run_season, gen, season, harnesses, on_usage, model): season
                for season in seasons
            }
            events_by_season = {futures[f]: f.result() for f in futures}

        for season, events in events_by_season.items():
            all_picks = [e for e in events if e["type"] == "pick"]
            drafted = {e["player_id"] for e in all_picks}
            for i in list(scores):
                starters = next(
                    e["starters"] for e in events
                    if e["type"] == "lineup" and e["team"] == i
                )
                team_picks = [e["player_id"] for e in all_picks if e["team"] == i]
                bench = [p for p in team_picks if p not in starters]
                scores[i][season] = score_roster(
                    self.conn, starters, bench, season, drafted
                )
                pick_logs[i][season] = [e for e in all_picks if e["team"] == i]

        results = {
            # stringify ALL keys so fresh results match the JSON-loaded shape
            "scores": {
                str(i): {str(se): v for se, v in s.items()} for i, s in scores.items()
            },
            "ranks": self._ranks(scores, seasons),
            "picks": {str(i): {str(s): p for s, p in by.items()} for i, by in pick_logs.items()},
        }
        results_file.write_text(json.dumps(results, indent=2))
        self._mark(gen_dir, "trials")
        return results

    @staticmethod
    def _ranks(scores: dict[int, dict[int, float]], seasons: list[int]) -> dict:
        mean_rank: dict[str, float] = {}
        for season in seasons:
            season_scores = sorted(
                ((i, s[season]) for i, s in scores.items()), key=lambda x: -x[1]
            )
            for rank, (i, _) in enumerate(season_scores, start=1):
                mean_rank[str(i)] = mean_rank.get(str(i), 0) + rank / len(seasons)
        return mean_rank

    def _rewrite(self, gen: int, results: dict, seasons: list[int]) -> None:
        next_dir = self._gen_dir(gen + 1)
        if self._done(next_dir, "rewrite"):
            return
        on_usage = self._track("rewrite", gen)
        harnesses = self._harnesses(gen)
        ranks = results["ranks"]
        order = sorted(harnesses, key=lambda i: ranks[str(i)])
        top2, bottom2 = order[:2], order[-2:]

        def rewrite_one(i: int, harness: str) -> tuple[int, str]:
            wconn = store.connect()
            try:
                summary = {"your_mean_rank_of_14": ranks[str(i)], "seasons": {}}
                for season in seasons:
                    picks = results["picks"][str(i)][str(season)]
                    summary["seasons"][season] = {
                        "final_score": results["scores"][str(i)][str(season)],
                        "picks": [
                            {
                                "round": p["round"],
                                "position": p["position"],
                                "adp_at_pick": p["adp"],
                                "actual_season_points": score_lineup(
                                    wconn, [p["player_id"]], season
                                ),
                                "forfeited": p["forfeited"],
                            }
                            for p in picks
                        ],
                    }
            finally:
                wconn.close()
            source, note = harness, ""
            if i in bottom2:  # replaced by a mutated copy of a top harness
                source = harnesses[top2[bottom2.index(i)]]
                note = (
                    "\nYour previous strategy was eliminated. You inherit a stronger "
                    "harness; diverge from it meaningfully where you see weakness."
                )
            new_harness = rewrite_harness(
                self.client, self.cfg["rewriter_model"], source,
                {**summary, "note": note}, on_usage,
            )
            return i, new_harness

        with ThreadPoolExecutor(max_workers=len(harnesses) or 1) as pool:
            futures = [pool.submit(rewrite_one, i, h) for i, h in harnesses.items()]
            for f in futures:
                i, new_harness = f.result()
                (next_dir / "harnesses" / f"agent_{i:02d}.md").write_text(new_harness)
        self._mark(next_dir, "rewrite")

    def _score_events(self, events: list[dict], season: int, agent_ids) -> dict:
        all_picks = [e for e in events if e["type"] == "pick"]
        drafted = {e["player_id"] for e in all_picks}
        out = {}
        for i in agent_ids:
            starters = next(
                e["starters"] for e in events
                if e["type"] == "lineup" and e["team"] == i
            )
            team_picks = [e["player_id"] for e in all_picks if e["team"] == i]
            bench = [p for p in team_picks if p not in starters]
            out[i] = score_roster(self.conn, starters, bench, season, drafted)
        return out

    def evaluate_holdout(self) -> None:
        """Final exam: draft the holdout seasons with the last generation's audited
        harnesses (frozen; no rewrite). Reports train-vs-holdout mean rank."""
        gens = sorted(
            int(p.name.split("_")[1])
            for p in self.run_dir.glob("gen_*")
            if (p / "results.json").exists()
        )
        if not gens:
            console.print("[red]no completed generations to evaluate[/]")
            return
        last = gens[-1]
        harnesses = self._harnesses(last)
        hold_dir = self.run_dir / "holdout"
        (hold_dir / "drafts").mkdir(parents=True, exist_ok=True)
        on_usage = self._track("holdout", last)
        model = drafter_model_for_gen(self.cfg, last)
        seasons = list(self.cfg["holdout_seasons"])
        console.print(f"[bold]holdout eval[/] gen {last} harnesses on {seasons} ({model})")

        with ThreadPoolExecutor(max_workers=len(seasons)) as pool:
            futures = {
                pool.submit(
                    self._run_season, last, season, harnesses, on_usage, model, hold_dir
                ): season
                for season in seasons
            }
            events_by_season = {futures[f]: f.result() for f in futures}

        scores = {i: {} for i in harnesses}
        for season, events in events_by_season.items():
            for i, sc in self._score_events(events, season, list(harnesses)).items():
                scores[i][season] = sc
        holdout_ranks = self._ranks(scores, seasons)

        # train baseline: mean rank over the last up-to-3 completed generations
        recent = gens[-3:]
        train_ranks: dict[str, float] = {}
        for g in recent:
            r = json.loads((self._gen_dir(g) / "results.json").read_text())["ranks"]
            for a, v in r.items():
                train_ranks[a] = train_ranks.get(a, 0) + v / len(recent)

        report = {
            "generation": last,
            "holdout_seasons": seasons,
            "holdout_ranks": holdout_ranks,
            "train_ranks_recent": train_ranks,
            "scores": {str(i): {str(s): v for s, v in sc.items()} for i, sc in scores.items()},
        }
        (hold_dir / "results.json").write_text(json.dumps(report, indent=2))
        console.print("agent | train rank (last 3 gens) | holdout rank | gap")
        for a in sorted(holdout_ranks, key=lambda x: holdout_ranks[x]):
            t, h = train_ranks.get(a), holdout_ranks[a]
            gap = f"{h - t:+.1f}" if t is not None else "?"
            console.print(f"  {a:>2} | {t:.1f} | {h:.1f} | {gap}")

    # --- run ------------------------------------------------------------------

    def run(self) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO runs VALUES (?,?,datetime('now'))",
            (self.run_id, json.dumps(self.cfg)),
        )
        self.conn.commit()
        if not self._harnesses(0):
            self._seed_harnesses()
        try:
            for gen in range(self.cfg["generations"]):
                pool = list(self.cfg["train_seasons"])
                self.rng.seed(self.cfg["seed"] + gen)
                seasons = self.rng.sample(pool, self.cfg["seasons_per_generation"])
                console.print(
                    f"[bold]generation {gen}[/] seasons={seasons} "
                    f"model={drafter_model_for_gen(self.cfg, gen)} "
                    f"spent=${self.spent_usd:.2f}"
                )
                self._audit(gen)
                results = self._trials(gen, seasons)
                console.print(f"  mean ranks: {results['ranks']}")
                self._rewrite(gen, results, seasons)
        except BudgetExceeded as e:
            console.print(f"[red]{e} — halting cleanly; run is resumable[/]")
