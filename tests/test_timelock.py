"""Anti-leakage invariants: no tool can return post-lock or draft-season data."""

import json

import pytest

from ffr.corpus.api import TimeLockedCorpus
from ffr.data.store import connect

LOCK = "2023-09-06"


@pytest.fixture
def corpus(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    docs = [
        # (url, source, season, published, snapshot, effective, type, cats, title, text, sha)
        ("u1", "espn", 2023, "2023-08-01", "20230801", "2023-08-01", "article",
         json.dumps(["injury"]), "Bijan Robinson camp report",
         "Bijan Robinson looked explosive at training camp today. " * 20, "s1"),
        ("u2", "espn", 2023, "2023-09-06", "20230906", "2023-09-06", "article",
         json.dumps(["rankings"]), "Final draft rankings",
         "Bijan Robinson rounds out the top five. " * 20, "s2"),
        # post-lock doc — must NEVER be visible
        ("u3", "espn", 2023, "2023-10-01", "20231001", "2023-10-01", "article",
         json.dumps(["injury"]), "Bijan Robinson week 4 recap",
         "Bijan Robinson scored twice in week 4. " * 20, "s3"),
        # other-season doc — must not bleed across seasons
        ("u4", "espn", 2022, "2022-08-01", "20220801", "2022-08-01", "article",
         json.dumps(["injury"]), "Bijan Robinson college update",
         "Bijan Robinson is still at Texas. " * 20, "s4"),
    ]
    conn.executemany(
        """INSERT INTO documents (url, source, season, published_at, snapshot_at,
           effective_date, doc_type, categories, title, text, content_sha1)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        docs,
    )
    conn.execute("INSERT INTO players VALUES ('p_bijan','Bijan Robinson','RB',2023,2025)")
    conn.executemany(
        "INSERT INTO ground_truth VALUES (?,?,?,?)",
        [("p_bijan", 2022, 0.0, 0), ("p_bijan", 2023, 250.0, 17), ("p_bijan", 2024, 300.0, 17)],
    )
    conn.executemany(
        "INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?)",
        [
            ("fantasypros", 2023, "2023-08-15", "OVR", 5, 6.2, "p_bijan", "Bijan Robinson"),
            ("fantasypros", 2023, "2023-09-06", "OVR", 4, 5.0, "p_bijan", "Bijan Robinson"),
            # post-lock snapshot — must not be selected
            ("fantasypros", 2023, "2023-09-20", "OVR", 1, 1.0, "p_bijan", "Bijan Robinson"),
        ],
    )
    conn.commit()
    return TimeLockedCorpus(conn=conn, season=2023, lock_date=LOCK)


def test_search_never_returns_post_lock(corpus):
    hits = corpus.search_news("Bijan Robinson", limit=25)
    assert hits, "pre-lock docs should be findable"
    dates = [h["effective_date"] for h in hits]
    assert all(d <= LOCK for d in dates)
    titles = " ".join(h["title"] for h in hits)
    assert "week 4" not in titles.lower()
    assert "college" not in titles.lower()  # other season excluded too


def test_read_document_blocks_post_lock(corpus):
    post_lock_id = corpus.conn.execute(
        "SELECT doc_id FROM documents WHERE url='u3'"
    ).fetchone()["doc_id"]
    assert corpus.read_document(post_lock_id) is None
    ok_id = corpus.conn.execute(
        "SELECT doc_id FROM documents WHERE url='u1'"
    ).fetchone()["doc_id"]
    assert corpus.read_document(ok_id) is not None


def test_adp_uses_latest_pre_lock_snapshot(corpus):
    adp = corpus.get_adp()
    assert adp[0]["rank"] == 4  # the 09-06 snapshot, not the post-lock 09-20 one
    trend = corpus.get_adp_trend("p_bijan")
    assert [t["scrape_date"] for t in trend] == ["2023-08-15", "2023-09-06"]


def test_history_excludes_draft_season_and_future(corpus):
    hist = corpus.get_player_history("p_bijan")
    assert [h["season"] for h in hist] == [2022]
    assert corpus.get_prior_season_results(2023) == []
    assert corpus.get_prior_season_results(2024) == []
    prior = corpus.get_prior_season_results(2022)
    assert prior and prior[0]["season" if "season" in prior[0] else "points"] is not None


def test_fts_query_injection_is_inert(corpus):
    # FTS5 syntax in user input must not crash or widen the query
    hits = corpus.search_news('Robinson" OR doc_id:*', limit=5)
    assert isinstance(hits, list)
