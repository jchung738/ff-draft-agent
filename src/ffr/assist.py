"""Live draft assistant: run the best evolved harness against a REAL draft.

You mirror your draft board here as picks happen; `rec` asks the harness-powered
agent (with the 2026 time-locked corpus) what you should take and why.

Commands:
  t <name>      someone else drafted <name>
  m <name>      YOU drafted <name>
  r             recommend my pick (research + reasoning + sources)
  b [pos]       show top available (optionally by position)
  ro            show my roster
  u             undo last entry
  q             quit
"""

from __future__ import annotations

import readline  # noqa: F401  (line editing for input())
import json
from pathlib import Path

from rich.console import Console

from ffr.config import RUNS_DIR
from ffr.corpus.api import TimeLockedCorpus
from ffr.data import store
from ffr.data.entities import resolve
from ffr.data.season_meta import lock_date
from ffr.draft.pool import pool_from_rankings
from ffr.draft.roster import Roster

console = Console()


class LiveDraft:
    """Duck-typed stand-in for DraftEngine backed by a REAL draft's state."""

    def __init__(self, season: int, pool, my_slot: int, teams: int = 14, rounds: int = 15):
        self.season = season
        self.teams = teams
        self.rounds = rounds
        self.pool = pool
        self.available = {p.player_id: p for p in pool}
        self.rosters = [Roster() for _ in range(teams)]
        self.slot_order = list(range(teams))  # team idx == draft slot
        self.my_team = my_slot - 1
        self.events: list[dict] = []
        self._undo: list[str] = []

    def pick_order(self) -> list[int]:
        order = []
        for rnd in range(self.rounds):
            slots = range(self.teams) if rnd % 2 == 0 else reversed(range(self.teams))
            order.extend(list(slots))
        return order

    def draft_slot(self, team_idx: int) -> int:
        return team_idx

    def available_sorted(self, position: str | None = None):
        players = [
            p for p in self.available.values()
            if position is None or p.position == position
        ]
        return sorted(players, key=lambda p: p.adp)

    def on_clock(self) -> int:
        n = len(self.events)
        order = self.pick_order()
        return order[n] if n < len(order) else -1

    def apply_pick(self, player_id: str) -> dict:
        n = len(self.events)
        team = self.on_clock()
        p = self.available.pop(player_id)
        self.rosters[team].force_add(p.ref())  # real drafts owe us no legality
        event = {
            "type": "pick", "pick": n + 1, "round": n // self.teams + 1,
            "team": team, "slot": team, "player_id": p.player_id,
            "name": p.name, "position": p.position, "adp": p.adp,
            "forfeited": None,
        }
        self.events.append(event)
        self._undo.append(player_id)
        return event

    def undo(self) -> str | None:
        if not self._undo:
            return None
        pid = self._undo.pop()
        event = self.events.pop()
        p = next(x for x in self.pool if x.player_id == pid)
        self.available[pid] = p
        roster = self.rosters[event["team"]]
        roster.players = [x for x in roster.players if x.player_id != pid]
        return p.name


def best_harness(run_id: str) -> tuple[str, str]:
    """Cumulative-best lineage's latest audited harness from a run."""
    run_dir = RUNS_DIR / run_id
    totals: dict[str, list[float]] = {}
    gens = []
    for d in sorted(run_dir.glob("gen_*")):
        rf = d / "results.json"
        if rf.exists():
            gens.append(d)
            for a, r in json.loads(rf.read_text())["ranks"].items():
                totals.setdefault(a, []).append(r)
    if not gens:
        raise SystemExit(f"no completed generations in {run_id}")
    means = {a: sum(v) / len(v) for a, v in totals.items()}
    best = min(means, key=lambda a: means[a])
    # latest generation whose audit is done has the freshest CLEAN harness
    for d in sorted(run_dir.glob("gen_*"), reverse=True):
        f = d / "harnesses" / f"agent_{int(best):02d}.md"
        if (d / "_DONE_audit").exists() and f.exists():
            console.print(
                f"[bold]harness:[/] agent {best} (mean rank {means[best]:.2f} over "
                f"{len(totals[best])} gens), text from {d.name}"
            )
            return f.read_text(), best
    raise SystemExit("no audited harness found")


def _find(conn, live: LiveDraft, raw: str) -> str | None:
    pid = resolve(conn, raw, live.season)
    if pid and pid in live.available:
        return pid
    # fuzzy fallback over the available pool
    needle = raw.lower()
    hits = [p for p in live.available.values() if needle in p.name.lower()]
    if len(hits) == 1:
        return hits[0].player_id
    if hits:
        console.print("[yellow]ambiguous — candidates:[/]")
        for p in hits[:6]:
            console.print(f"  {p.name} ({p.position}, adp {p.adp})")
        return None
    console.print(f"[red]no available player matching {raw!r}[/]")
    return None


def run_assist(season: int, my_slot: int, model: str, run_id: str, harness_path: str | None):
    from ffr.agents.drafter import LLMDrafter

    conn = store.connect()
    lock = lock_date(conn, season)
    pool = pool_from_rankings(conn, season, lock)
    if len(pool) < 180:
        raise SystemExit(
            f"2026 pool too small ({len(pool)}) — run: uv run ffr crawl-adp --start {season} --end {season}"
        )
    docs = conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE season=?", (season,)
    ).fetchone()["c"]
    console.print(f"[bold]{season} corpus:[/] {len(pool)} player pool, {docs} news docs, lock {lock}")

    if harness_path:
        harness = Path(harness_path).read_text()
    else:
        harness, _ = best_harness(run_id)

    corpus = TimeLockedCorpus(conn, season, lock)
    live = LiveDraft(season, pool, my_slot)
    drafter = LLMDrafter(
        model=model, harness=harness, corpus=corpus,
        max_tool_calls=8, late_round_start=99,  # real draft: full budget always
        prep_research=False, extension_tool_calls=4,
    )

    console.print(f"you are draft slot {my_slot}/14. Commands: t/m/r/b/ro/u/q")
    while True:
        n = len(live.events)
        if n >= live.teams * live.rounds:
            console.print("[bold]draft complete[/]")
            break
        clock = live.on_clock()
        tag = "[bold green]YOU are on the clock[/]" if clock == live.my_team else f"slot {clock + 1} on the clock"
        try:
            cmd = input(f"pick {n + 1} ({tag if clock != live.my_team else 'YOUR PICK'}) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        op, _, arg = cmd.partition(" ")
        op = op.lower()
        if op == "q":
            break
        elif op == "u":
            name = live.undo()
            console.print(f"undone: {name}" if name else "nothing to undo")
        elif op in ("t", "m"):
            pid = _find(conn, live, arg)
            if pid:
                e = live.apply_pick(pid)
                who = "you" if op == "m" else f"slot {e['team'] + 1}"
                console.print(f"  {e['name']} ({e['position']}) -> {who}  [pick {e['pick']}]")
        elif op == "b":
            for p in live.available_sorted(arg.upper() or None if arg else None)[:15]:
                console.print(f"  {p.name:26s} {p.position:3s} adp {p.adp}")
        elif op == "ro":
            for p in live.rosters[live.my_team].players:
                console.print(f"  {p.name} ({p.position})")
        elif op == "r":
            console.print("[dim]researching…[/]")
            try:
                pid = drafter.pick(live, live.my_team)
            except Exception as ex:
                console.print(f"[red]agent failed: {ex}[/] — board (b) still works")
                continue
            p = live.available[pid]
            console.print(f"\n[bold green]RECOMMENDATION: {p.name}[/] ({p.position}, adp {p.adp})")
            if drafter.last_pick_reason:
                console.print(f"  [italic]{drafter.last_pick_reason}[/]")
            src = drafter.last_pick_sources or {}
            if src.get("queries"):
                console.print(f"  [dim]searched: {' | '.join(src['queries'])}[/]")
            for d in src.get("docs", []):
                console.print(f"  [dim]read: [{d['source']} {d['date']}] {d['title']}[/]")
            console.print("  (enter 'm <name>' once you actually draft)\n")
        else:
            console.print("commands: t <name> | m <name> | r | b [pos] | ro | u | q")
