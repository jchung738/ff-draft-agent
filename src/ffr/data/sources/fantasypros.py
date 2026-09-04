"""FantasyPros ADP/rankings table parser.

Era-agnostic: handles the 2015-2016 markup (no player-label class, unclosed
<tr>), the 2017-2023 markup, and the 2024+ markup (extra CSS classes). The
stable anchor across all eras is the fp-player-name attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

_TR_SPLIT = re.compile(r"<tr[^>]*>")
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_NAME_ATTR = re.compile(r'fp-player-name="([^"]+)"')
_NAME_ANCHOR = re.compile(r'class="player-name[^"]*"[^>]*>([^<]+)</a>')
_ANCHOR_TEXT = re.compile(r'<a href="[^"]*/nfl/players/[^"]*"[^>]*>([^<]+)</a>')
_TEAM = re.compile(r"<small[^>]*>\s*([A-Z]{2,3})\b")
_TAG = re.compile(r"<[^>]+>")
_POS = re.compile(r"([A-Z]{1,3})\d*")

_POS_MAP = {"DEF": "DST", "DST": "DST", "PK": "K"}
_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}


@dataclass(frozen=True)
class AdpRow:
    rank: int
    name: str
    team: str | None
    position: str  # QB/RB/WR/TE/K/DST
    adp: float | None


def parse_adp(html: str) -> list[AdpRow]:
    rows: list[AdpRow] = []
    for seg in _TR_SPLIT.split(html)[1:]:
        seg = seg.split("</table")[0]
        tds = _TD.findall(seg)
        if len(tds) < 3:
            continue
        first = _TAG.sub("", tds[0]).strip()
        if not first.isdigit():
            continue
        name_m = _NAME_ATTR.search(seg) or _NAME_ANCHOR.search(seg) or _ANCHOR_TEXT.search(seg)
        if not name_m:
            continue
        position = None
        for td in tds[1:]:
            text = _TAG.sub("", td).strip()
            m = _POS.fullmatch(text)
            if m:
                mapped = _POS_MAP.get(m.group(1), m.group(1))
                if mapped in _POSITIONS:
                    position = mapped
                    break
        if position is None:
            continue
        team_m = _TEAM.search(tds[1])
        adp = None
        for td in reversed(tds):
            t = _TAG.sub("", td).replace(",", "").replace("&nbsp;", "").strip()
            if re.fullmatch(r"\d+(\.\d+)?", t):
                adp = float(t)
                break
        rows.append(
            AdpRow(
                rank=int(first),
                name=unescape(name_m.group(1)).strip(),
                team=team_m.group(1) if team_m else None,
                position=position,
                adp=adp,
            )
        )
    return rows
