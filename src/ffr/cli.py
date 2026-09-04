"""ffr command line: ingest-stats | crawl | draft | evolve | audit."""

from __future__ import annotations

import argparse

from ffr.config import ensure_ca_bundle


def cmd_ingest_stats(args: argparse.Namespace) -> None:
    from ffr.data import ground_truth, store

    conn = store.connect()
    n = ground_truth.ingest_players(conn)
    print(f"players ingested: {n}")
    seasons = list(range(args.start, args.end + 1))
    ground_truth.ingest_season_stats(conn, seasons)
    for row in conn.execute(
        "SELECT season, COUNT(*) n, MAX(week) max_wk FROM weekly_points GROUP BY season ORDER BY season"
    ):
        print(f"  season {row['season']}: {row['n']} player-weeks (max wk {row['max_wk']})")


def cmd_crawl_adp(args: argparse.Namespace) -> None:
    from ffr.data import store
    from ffr.data.crawl import crawl_adp, crawl_ffc_adp

    conn = store.connect()
    for season in range(args.start, args.end + 1):
        n = crawl_adp(conn, season) if not args.ffc_only else 0
        n += crawl_ffc_adp(conn, season)
        resolved = conn.execute(
            "SELECT COUNT(*) c FROM rankings WHERE season=? AND player_id IS NOT NULL",
            (season,),
        ).fetchone()["c"]
        total = conn.execute(
            "SELECT COUNT(*) c FROM rankings WHERE season=?", (season,)
        ).fetchone()["c"]
        pct = 100 * resolved / total if total else 0
        print(f"season {season}: {n} rows ingested, {resolved}/{total} resolved ({pct:.1f}%)")


def cmd_resolve(args: argparse.Namespace) -> None:
    """Re-run entity resolution for unresolved ranking rows; print a report."""
    from ffr.data import store
    from ffr.data.entities import resolve

    conn = store.connect()
    fixed = 0
    for r in conn.execute(
        "SELECT rowid, raw_name, season, position FROM rankings WHERE player_id IS NULL"
    ).fetchall():
        pos = r["position"] if r["position"] != "OVR" else None
        pid = resolve(conn, r["raw_name"], r["season"], position=pos)
        if pid:
            conn.execute("UPDATE rankings SET player_id=? WHERE rowid=?", (pid, r["rowid"]))
            fixed += 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) c FROM rankings").fetchone()["c"]
    unresolved = conn.execute(
        "SELECT COUNT(*) c FROM rankings WHERE player_id IS NULL"
    ).fetchone()["c"]
    print(f"re-resolved {fixed}; {unresolved}/{total} still unresolved")
    for r in conn.execute(
        """SELECT raw_name, COUNT(*) n FROM rankings WHERE player_id IS NULL
           GROUP BY raw_name ORDER BY n DESC LIMIT 20"""
    ):
        print(f"  {r['n']:4d}  {r['raw_name']}")


def cmd_crawl_articles(args: argparse.Namespace) -> None:
    from ffr.data import store
    from ffr.data.crawl import crawl_articles

    conn = store.connect()
    keys = args.sources.split(",") if args.sources else None
    for season in range(args.start, args.end + 1):
        n = crawl_articles(conn, season, source_keys=keys, max_docs_per_source=args.max_docs)
        print(f"season {season}: {n} articles stored")


def cmd_draft(args: argparse.Namespace) -> None:
    """Run one bot-only draft on a real season and score it against ground truth."""
    from ffr.data import store
    from ffr.data.season_meta import lock_date
    from ffr.draft.bots import ADPBot, RandomBot, VORBot
    from ffr.draft.engine import DraftEngine
    from ffr.draft.pool import pool_from_rankings
    from ffr.draft.scoring import score_lineup

    conn = store.connect()
    kinds = {"vor": VORBot, "adp": ADPBot}
    results: dict[str, list[float]] = {"vor": [], "adp": [], "random": []}
    for seed in range(args.trials):
        pool = pool_from_rankings(conn, args.season, lock_date(conn, args.season))
        engine = DraftEngine(season=args.season, pool=pool, seed=seed)
        labels = ["vor"] * 5 + ["adp"] * 5 + ["random"] * 4
        drafters = [
            kinds[l]() if l in kinds else RandomBot(seed * 100 + i)
            for i, l in enumerate(labels)
        ]
        engine.run(drafters)
        for i, label in enumerate(labels):
            score = score_lineup(conn, engine.lineups[i], args.season)
            results[label].append(score)
    for label, scores in results.items():
        print(f"{label:7s} mean={sum(scores)/len(scores):7.1f}  n={len(scores)}")


def cmd_evolve(args: argparse.Namespace) -> None:
    from pathlib import Path

    from ffr.evolve.orchestrator import Orchestrator

    config = Path(args.config) if args.config else None
    Orchestrator(args.run_id, config_path=config).run()


def main() -> None:
    ensure_ca_bundle()
    parser = argparse.ArgumentParser(prog="ffr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest-stats", help="ingest nflverse players + weekly points")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2025)
    p.set_defaults(func=cmd_ingest_stats)

    p = sub.add_parser("crawl-adp", help="ingest FantasyPros (Wayback) + FFC (API) ADP")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--ffc-only", action="store_true")
    p.set_defaults(func=cmd_crawl_adp)

    p = sub.add_parser("resolve", help="re-resolve unresolved ranking names")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("crawl-articles", help="ingest news articles via Wayback")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--sources", type=str, default=None, help="comma-separated source keys")
    p.add_argument("--max-docs", type=int, default=400)
    p.set_defaults(func=cmd_crawl_articles)

    p = sub.add_parser("draft", help="run bot-only drafts on a real season")
    p.add_argument("--season", type=int, default=2022)
    p.add_argument("--trials", type=int, default=10)
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("dash", help="local observer dashboard")
    p.add_argument("--port", type=int, default=8787)
    p.set_defaults(func=lambda a: __import__("ffr.dash", fromlist=["serve"]).serve(a.port))

    p = sub.add_parser("evolve", help="run the harness evolution loop")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="run config yaml")
    p.set_defaults(func=cmd_evolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
