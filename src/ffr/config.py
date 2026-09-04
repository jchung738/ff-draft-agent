"""Config loading: scoring rules, week windows, lineup, sources."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

DB_PATH = DATA_DIR / "ffr.sqlite"
HTML_CACHE_DIR = DATA_DIR / "html_cache"

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


@dataclass(frozen=True)
class WeekWindow:
    first_week: int
    last_week: int  # inclusive; the season's final regular-season week is excluded


@dataclass(frozen=True)
class LineupConfig:
    teams: int
    rounds: int
    slots: dict[str, int]  # slot name -> count (FLEX included)
    bench: int
    flex_positions: tuple[str, ...]


@lru_cache
def load_scoring_config() -> dict:
    with open(CONFIG_DIR / "scoring.yaml") as f:
        return yaml.safe_load(f)


def week_window(season: int) -> WeekWindow:
    """Scored-week range for a season (final regular-season week excluded)."""
    for row in load_scoring_config()["week_windows"]:
        lo, hi = row["seasons"]
        if lo <= season <= hi:
            return WeekWindow(row["first_week"], row["last_week"])
    raise ValueError(f"no week window configured for season {season}")


@lru_cache
def lineup_config() -> LineupConfig:
    cfg = load_scoring_config()["lineup"]
    return LineupConfig(
        teams=cfg["teams"],
        rounds=cfg["rounds"],
        slots=dict(cfg["slots"]),
        bench=cfg["bench"],
        flex_positions=tuple(cfg["flex_positions"]),
    )


def ensure_ca_bundle() -> None:
    """Point Python HTTP clients at the combined certifi + macOS keychain bundle.

    Required behind the corporate TLS-intercepting proxy; no-op if absent.
    """
    import os

    bundle = DATA_DIR / "ca_bundle.pem"
    if bundle.exists():
        os.environ.setdefault("SSL_CERT_FILE", str(bundle))
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))


@lru_cache
def load_sources_config() -> dict:
    with open(CONFIG_DIR / "sources.yaml") as f:
        return yaml.safe_load(f)["sources"]
