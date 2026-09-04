"""Corpus crawlers: FantasyPros ADP time series + generic Wayback article pipeline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3

from rich.console import Console

from ffr.config import load_sources_config
from ffr.data import wayback
from ffr.data.entities import build_automaton, resolve, tag_mentions
from ffr.data.extract import classify, extract_article
from ffr.data.season_meta import season_kickoff
from ffr.data.sources.fantasypros import parse_adp

console = Console()

ADP_PAGES = (
    # processed in order; half-PPR second so it overwrites same-date rows
    "https://www.fantasypros.com/nfl/adp/overall.php",
    "https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php",
)


def _window(conn: sqlite3.Connection, season: int) -> tuple[str, str, str]:
    """(from_ts, to_ts, kickoff_date) for the pre-draft window: June 1 -> kickoff."""
    kickoff = season_kickoff(conn, season)
    to_ts = kickoff.replace("-", "") + "235959"
    return f"{season}0601", to_ts, kickoff


def _month_chunks(from_ts: str, to_ts: str) -> list[tuple[str, str]]:
    """Split a CDX window into month-sized chunks (big wildcard queries 504)."""
    start = dt.date(int(from_ts[:4]), int(from_ts[4:6]), int(from_ts[6:8]))
    end = dt.date(int(to_ts[:4]), int(to_ts[4:6]), int(to_ts[6:8]))
    chunks = []
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        chunk_end = min(end, nxt - dt.timedelta(days=1))
        chunks.append((cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d") + "235959"))
        cur = nxt
    return chunks


def _weekly_subsample(snaps: list[wayback.Snapshot]) -> list[wayback.Snapshot]:
    """At most one snapshot per ISO week, always keeping the latest snapshot."""
    by_week: dict[tuple, wayback.Snapshot] = {}
    for s in snaps:
        week = dt.date.fromisoformat(s.date).isocalendar()[:2]
        by_week[week] = s  # last snapshot of each week wins
    picked = sorted(by_week.values(), key=lambda s: s.timestamp)
    if snaps and (not picked or picked[-1].timestamp != snaps[-1].timestamp):
        picked.append(snaps[-1])
    return picked


def crawl_adp(conn: sqlite3.Connection, season: int) -> int:
    """Ingest weekly FantasyPros ADP snapshots for a season's pre-draft window.

    Both page variants are ingested; half-PPR is processed second so its rows
    overwrite the standard-scoring rows for the same snapshot date.
    """
    from_ts, to_ts, _ = _window(conn, season)
    inserted = 0
    for page in ADP_PAGES:
        try:
            snaps = wayback.cdx_snapshots(page, from_ts, to_ts)
        except Exception as e:
            console.print(f"[yellow]{season}: CDX failed for {page} ({e})[/]")
            continue
        for snap in _weekly_subsample(snaps):
            try:
                html = wayback.fetch_snapshot(snap)
            except Exception:
                continue
            rows = parse_adp(html)
            if len(rows) < 50:  # broken/partial capture
                continue
            for r in rows:
                pid = resolve(conn, r.name, season, position=r.position, team=r.team)
                conn.execute(
                    "INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?,?,?,?)",
                    ("fantasypros", season, snap.date, "OVR", r.rank, r.adp, pid, r.name),
                )
                inserted += 1
        conn.commit()
    if inserted == 0:
        console.print(f"[yellow]{season}: no parseable ADP snapshots[/]")
    return inserted


def crawl_ffc_adp(conn: sqlite3.Connection, season: int) -> int:
    """Secondary ADP source: FantasyFootballCalculator's historical API.

    Serves past seasons directly (no Wayback). The aggregation window's end_date
    becomes the scrape_date; it precedes kickoff, so time-locking holds.
    """
    import httpx

    from ffr.config import ensure_ca_bundle

    ensure_ca_bundle()
    data = None
    for scoring in ("half-ppr", "ppr", "standard"):
        r = httpx.get(
            f"https://fantasyfootballcalculator.com/api/v1/adp/{scoring}",
            params={"year": season, "teams": 14},
            timeout=60,
        )
        if r.status_code == 200 and r.json().get("players"):
            data = r.json()
            break
    if not data:
        console.print(f"[yellow]{season}: no FFC ADP available[/]")
        return 0
    scrape_date = data.get("meta", {}).get("end_date") or f"{season}-09-01"
    kickoff = season_kickoff(conn, season)
    if scrape_date >= kickoff:  # never ingest a window that crosses kickoff
        scrape_date = kickoff
    inserted = 0
    for rank, p in enumerate(sorted(data["players"], key=lambda x: x["adp"]), start=1):
        pos = {"DEF": "DST", "PK": "K"}.get(p["position"], p["position"])
        if pos not in ("QB", "RB", "WR", "TE", "K", "DST"):
            continue
        pid = resolve(conn, p["name"], season, position=pos, team=p.get("team"))
        conn.execute(
            "INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?,?,?,?)",
            ("ffc", season, scrape_date, "OVR", rank, p["adp"], pid, p["name"]),
        )
        inserted += 1
    conn.commit()
    return inserted


def _top_player_ids(conn: sqlite3.Connection, season: int) -> set[str]:
    """Players relevant this season: appear in any ranking, or scored last season."""
    ids = {
        r["player_id"]
        for r in conn.execute(
            "SELECT DISTINCT player_id FROM rankings WHERE season=? AND player_id IS NOT NULL",
            (season,),
        )
    }
    if not ids:
        ids = {
            r["player_id"]
            for r in conn.execute(
                """SELECT player_id FROM ground_truth WHERE season=?
                   ORDER BY total_ex_final_week DESC LIMIT 300""",
                (season - 1,),
            )
        }
    return ids


def crawl_articles(
    conn: sqlite3.Connection,
    season: int,
    source_keys: list[str] | None = None,
    max_docs_per_source: int = 400,
) -> int:
    """Generic Wayback article pipeline for one season's pre-draft window."""
    from_ts, to_ts, _ = _window(conn, season)
    sources = load_sources_config()
    keys = source_keys or list(sources)
    auto = build_automaton(conn)
    relevant = _top_player_ids(conn, season)
    inserted = 0

    for key in keys:
        cfg = sources.get(key)
        if not cfg:
            console.print(f"[red]unknown source {key}[/]")
            continue
        seen_urls: set[str] = set()
        for pattern in cfg.get("article_patterns", []):
            snaps: list[wayback.Snapshot] = []
            for chunk_from, chunk_to in _month_chunks(from_ts, to_ts):
                try:
                    snaps.extend(
                        wayback.cdx_snapshots(
                            pattern, chunk_from, chunk_to, collapse="digest", limit=1500
                        )
                    )
                except Exception as e:
                    console.print(f"[yellow]{key} {pattern} {chunk_from}: CDX failed ({e})[/]")
            if not snaps:
                continue
            count = 0
            for snap in snaps:
                if count >= max_docs_per_source:
                    break
                if snap.url in seen_urls:
                    continue
                seen_urls.add(snap.url)
                try:
                    html = wayback.fetch_snapshot(snap)
                except Exception:
                    continue
                title, text, published = extract_article(html, snap.url)
                if not text:
                    continue
                sha1 = hashlib.sha1(text.encode()).hexdigest()
                mentions = tag_mentions(auto, f"{title}\n{text}")
                hits = {pid: n for pid, n in mentions.items() if pid in relevant}
                if not hits:
                    continue
                cur = conn.execute(
                    """INSERT OR IGNORE INTO documents
                       (url, source, season, published_at, snapshot_at, effective_date,
                        doc_type, categories, title, text, content_sha1)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        snap.url, key, season, published, snap.timestamp, snap.date,
                        "article", json.dumps(classify(title or "", text)), title, text, sha1,
                    ),
                )
                if cur.rowcount == 0:  # duplicate content from another capture
                    continue
                doc_id = cur.lastrowid
                conn.executemany(
                    "INSERT OR IGNORE INTO doc_players VALUES (?,?,?)",
                    [(doc_id, pid, n) for pid, n in hits.items()],
                )
                inserted += 1
                count += 1
                if inserted % 20 == 0:
                    conn.commit()
                    console.print(f"  {key}: {inserted} docs stored (season {season})")
            conn.commit()
        console.print(f"  {key}: {inserted} docs total so far (season {season})")
    return inserted
