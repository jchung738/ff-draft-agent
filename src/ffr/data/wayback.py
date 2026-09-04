"""Wayback Machine client: CDX snapshot enumeration + cached raw-bytes fetching."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ffr.config import HTML_CACHE_DIR, ensure_ca_bundle

CDX_URL = "https://web.archive.org/cdx/search/cdx"
_MIN_INTERVAL = 1.0  # seconds between archive.org requests
_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _retryable(e: BaseException) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(e, (httpx.TransportError, httpx.TimeoutException))


_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(6),
)


@dataclass(frozen=True)
class Snapshot:
    timestamp: str  # YYYYMMDDhhmmss
    url: str        # original URL

    @property
    def date(self) -> str:
        t = self.timestamp
        return f"{t[0:4]}-{t[4:6]}-{t[6:8]}"


@_retry
def cdx_snapshots(
    url_pattern: str,
    from_ts: str,
    to_ts: str,
    collapse: str = "timestamp:8",
    limit: int = 2000,
) -> list[Snapshot]:
    """Enumerate archived snapshots of a URL/pattern within a time range."""
    ensure_ca_bundle()
    _throttle()
    params = {
        "url": url_pattern,
        "from": from_ts,
        "to": to_ts,
        "filter": "statuscode:200",
        "collapse": collapse,
        "output": "json",
        "limit": str(limit),
    }
    r = httpx.get(CDX_URL, params=params, timeout=120)
    r.raise_for_status()
    if not r.text.strip():  # CDX returns an empty body when nothing matches
        return []
    rows = r.json()
    return [Snapshot(timestamp=row[1], url=row[2]) for row in rows[1:]]


def _cache_path(snap: Snapshot):
    key = hashlib.sha1(f"{snap.url}|{snap.timestamp}".encode()).hexdigest()
    return HTML_CACHE_DIR / key[:2] / f"{key}.html"


@_retry
def _fetch(snap: Snapshot) -> str:
    _throttle()
    # id_ returns the original bytes without wayback chrome/rewriting
    url = f"https://web.archive.org/web/{snap.timestamp}id_/{snap.url}"
    r = httpx.get(url, timeout=120, follow_redirects=True)
    r.raise_for_status()
    return r.text


def fetch_snapshot(snap: Snapshot) -> str:
    """Fetch a snapshot's raw HTML, using the on-disk cache."""
    path = _cache_path(snap)
    if path.exists():
        return path.read_text(errors="replace")
    html = _fetch(snap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, errors="replace")
    return html
