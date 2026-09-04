"""Mechanical leakage pre-screen for harnesses: player names, year-outcome
patterns, obfuscation signals. Free and deterministic; runs before the LLM audit."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from ffr.data.entities import build_automaton, norm

_OUTCOME_WORDS = (
    r"(finish(ed)?|bust(ed)?|broke out|breakout|led the league|overall [A-Z]{2}\d|"
    r"top[- ]\d+|rb1|wr1|qb1|te1|league winner|won my league|smash(ed)?)"
)

RULES: list[tuple[str, re.Pattern]] = [
    (
        "year+outcome",
        re.compile(rf"\b(20[01]\d|202\d)\b.{{0,60}}{_OUTCOME_WORDS}|{_OUTCOME_WORDS}.{{0,60}}\b(20[01]\d|202\d)\b", re.I | re.S),
    ),
    (
        "team+year",
        re.compile(
            r"\b(chiefs|eagles|bills|49ers|niners|cowboys|rams|bengals|ravens|lions|"
            r"packers|dolphins|jets|giants|steelers|browns|texans|colts|jaguars|titans|"
            r"broncos|raiders|chargers|patriots|commanders|redskins|bears|vikings|"
            r"saints|falcons|panthers|buccaneers|cardinals|seahawks)\b.{0,40}\b20[12]\d\b",
            re.I | re.S,
        ),
    ),
    ("draft-slot-ref", re.compile(r"\bround \d+ pick \d+\b.{0,40}\b[A-Z][a-z]+ [A-Z][a-z]+", re.S)),
    ("base64-blob", re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")),
    ("auditor-address", re.compile(r"\b(auditor|reviewer|ignore (the|all|previous) (rules|instructions))\b", re.I)),
    ("hex-blob", re.compile(r"(?:\b[0-9a-fA-F]{2}\b[ ,]){12,}")),
]


@dataclass(frozen=True)
class Violation:
    rule: str
    span: str


def prescreen(conn: sqlite3.Connection, harness: str) -> list[Violation]:
    violations: list[Violation] = []
    # 1. Player-name scan over the full alias automaton (word-boundary checked)
    auto = build_automaton(conn)
    haystack = norm(harness)
    for end, (alias, _pids) in auto.iter(haystack):
        start = end - len(alias) + 1
        before = haystack[start - 1] if start > 0 else " "
        after = haystack[end + 1] if end + 1 < len(haystack) else " "
        if before == " " and after == " ":
            violations.append(Violation("player-name", alias))
    # 2. Regex rules
    for rule, pattern in RULES:
        for m in pattern.finditer(harness):
            violations.append(Violation(rule, m.group(0)[:120]))
    return violations
