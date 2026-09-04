"""Article extraction and keyword-based category classification."""

from __future__ import annotations

import re

CATEGORY_LEXICONS = {
    "training_camp": ["training camp", "camp battle", "preseason", "otas", "minicamp", "depth chart", "first-team reps"],
    "injury": ["injury", "injured", "hamstring", "acl", "mcl", "concussion", "questionable", "pup list", "out for", "surgery", "sprain", "holdout"],
    "rookie": ["rookie", "draft pick", "first-round pick", "drafted in the", "draft class"],
    "rankings": ["rankings", "ranked", "tiers", "cheat sheet", "cheatsheet", "top 10", "top-10", "sleeper", "bust", "breakout", "adp"],
    "off_field": ["suspension", "suspended", "arrest", "charged", "contract dispute", "trade request", "personal conduct", "legal"],
    "trends": ["historical", "regression", "career year", "trend", "since 20", "over the last"],
}


def classify(title: str, text: str) -> list[str]:
    haystack = f"{title}\n{text[:5000]}".lower()
    return [
        cat
        for cat, terms in CATEGORY_LEXICONS.items()
        if any(t in haystack for t in terms)
    ] or ["general"]


def extract_article(html: str, url: str) -> tuple[str | None, str | None, str | None]:
    """Returns (title, text, published_date) or (None, None, None) if not an article."""
    import trafilatura

    text = trafilatura.extract(html, url=url, include_comments=False)
    if not text or len(text) < 400:  # too short to be a real article
        return None, None, None
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = meta.title if meta else None
    published = None
    try:
        from htmldate import find_date

        published = find_date(html, url=url)
    except Exception:
        pass
    if not title:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html)
        title = m.group(1).strip() if m else url
    return title, text, published
