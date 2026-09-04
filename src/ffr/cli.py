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


def cmd_draft(args: argparse.Namespace) -> None:
    """Run one bot-only draft on a real season and score it against ground truth."""
    from ffr.data import store
    from ffr.draft.bots import ADPBot, RandomBot, VORBot
    from ffr.draft.engine import DraftEngine
    from ffr.draft.pool import pool_from_prior_season
    from ffr.draft.scoring import score_lineup

    conn = store.connect()
    kinds = {"vor": VORBot, "adp": ADPBot}
    results: dict[str, list[float]] = {"vor": [], "adp": [], "random": []}
    for seed in range(args.trials):
        pool = pool_from_prior_season(conn, args.season)
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


def main() -> None:
    ensure_ca_bundle()
    parser = argparse.ArgumentParser(prog="ffr")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest-stats", help="ingest nflverse players + weekly points")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2025)
    p.set_defaults(func=cmd_ingest_stats)

    p = sub.add_parser("draft", help="run bot-only drafts on a real season")
    p.add_argument("--season", type=int, default=2022)
    p.add_argument("--trials", type=int, default=10)
    p.set_defaults(func=cmd_draft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
