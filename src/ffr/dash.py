"""Local observer dashboard: live draft board, standings, harness diffs, costs.

Stdlib-only (http.server). Reads runs/<run_id>/ artifacts and the SQLite cost
ledger fresh on every request, so an in-flight draft updates live. Binds to
127.0.0.1 — this is a local viewer, not a deployable app.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from ffr.config import RUNS_DIR, lineup_config
from ffr.data import store

_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # partially-written last line during a live draft
    return out


def _runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())


def _generations(run_id: str) -> list[int]:
    run_dir = RUNS_DIR / run_id
    return sorted(
        int(p.name.split("_")[1]) for p in run_dir.glob("gen_*") if p.is_dir()
    )


def _gen_dir(run_id: str, gen: int) -> Path:
    return RUNS_DIR / run_id / f"gen_{gen:03d}"


def _run_state(run_id: str) -> dict:
    gens = []
    for g in _generations(run_id):
        d = _gen_dir(run_id, g)
        results = None
        rf = d / "results.json"
        if rf.exists():
            r = json.loads(rf.read_text())
            results = {"scores": r["scores"], "ranks": r["ranks"]}
        drafts = sorted(p.stem for p in (d / "drafts").glob("*.jsonl")) if (d / "drafts").exists() else []
        agents = sorted(
            int(p.stem.split("_")[1])
            for p in (d / "harnesses").glob("agent_*.md")
        ) if (d / "harnesses").exists() else []
        if results:
            rewrite_done = (_gen_dir(run_id, g + 1) / "_DONE_rewrite").exists()
            phase = "complete" if rewrite_done else "rewriting harnesses"
        elif (d / "_DONE_audit").exists():
            phase = "drafting" if drafts else "starting draft"
        elif agents:
            phase = "auditing harnesses"
        else:
            phase = "pending"
        gens.append(
            {"gen": g, "results": results, "drafts": drafts, "agents": agents, "phase": phase}
        )

    progress = _progress(run_id, gens)
    conn = store.connect()
    costs = [
        dict(r)
        for r in conn.execute(
            """SELECT phase, model, COUNT(*) calls, SUM(input_tok) input_tok,
                      SUM(cache_read_tok) cache_read_tok, SUM(output_tok) output_tok,
                      ROUND(SUM(usd), 3) usd
               FROM cost_ledger WHERE run_id = ? GROUP BY phase, model""",
            (run_id,),
        )
    ]
    total = round(sum(c["usd"] or 0 for c in costs), 2)
    conn.close()
    return {
        "run_id": run_id, "generations": gens, "costs": costs,
        "total_usd": total, "progress": progress,
    }


def _mtime_span(files: list[Path]) -> float:
    """Seconds elapsed since the earliest of these files was written."""
    if not files:
        return 0.0
    return time.time() - min(f.stat().st_mtime for f in files)


def _eta(done: int, total: int, elapsed: float) -> int | None:
    if done <= 0 or elapsed <= 0:
        return None
    return int((total - done) * elapsed / done)


def _progress(run_id: str, gens: list[dict]) -> dict | None:
    """Done/total + rate-based ETA for the run's current phase."""
    if not gens:
        return None
    cfg = lineup_config()
    n_agents = max((len(g["agents"]) for g in gens), default=14) or 14

    # rewriting: gen g writes harness files into gen g+1's dir one by one
    for g in gens:
        if g["phase"] == "rewriting harnesses":
            files = list((_gen_dir(run_id, g["gen"] + 1) / "harnesses").glob("agent_*.md"))
            done = len(files)
            return {
                "label": f"gen {g['gen']} → {g['gen'] + 1}: rewriting harnesses",
                "done": done, "total": n_agents,
                "eta_s": _eta(done, n_agents, _mtime_span(files)),
            }

    g = gens[-1]
    if g["phase"] == "auditing harnesses":
        files = list((_gen_dir(run_id, g["gen"]) / "harnesses").glob("*.audit.json"))
        done = len(files)
        return {
            "label": f"gen {g['gen']}: auditing harnesses",
            "done": done, "total": n_agents,
            "eta_s": _eta(done, n_agents, _mtime_span(files)),
        }
    if g["phase"] in ("drafting", "starting draft") and g["drafts"]:
        season = g["drafts"][-1]
        f = _gen_dir(run_id, g["gen"]) / "drafts" / f"{season}.jsonl"
        total = cfg.teams * cfg.rounds
        done = sum(1 for e in _json_lines(f) if e.get("type") == "pick")
        try:
            start = f.stat().st_birthtime  # macOS
        except AttributeError:
            start = f.stat().st_ctime
        return {
            "label": f"gen {g['gen']}: drafting {season}",
            "done": done, "total": total,
            "eta_s": _eta(done, total, time.time() - start),
        }
    return None


def _draft(run_id: str, gen: int, season: str) -> dict:
    events = _json_lines(_gen_dir(run_id, gen) / "drafts" / f"{season}.jsonl")
    start = next((e for e in events if e["type"] == "start"), {})
    picks = [e for e in events if e["type"] == "pick"]
    lineups = {e["team"]: e for e in events if e["type"] == "lineup"}
    return {
        "slot_order": start.get("slot_order", []),
        "picks": picks,
        "lineups": lineups,
        "done": bool(lineups),
    }


def _teamlog(run_id: str, gen: int, season: str, team: int) -> dict:
    """Full season simulation trace for one team's drafted roster."""
    from ffr.draft.scoring import simulate_roster

    events = _json_lines(_gen_dir(run_id, gen) / "drafts" / f"{season}.jsonl")
    lineup = next(
        (e for e in events if e["type"] == "lineup" and e["team"] == team), None
    )
    if lineup is None:
        return {"ready": False}
    all_picks = [e for e in events if e["type"] == "pick"]
    starters = lineup["starters"]
    team_picks = [e["player_id"] for e in all_picks if e["team"] == team]
    bench = [p for p in team_picks if p not in starters]
    drafted = {e["player_id"] for e in all_picks}
    conn = store.connect()
    sim = simulate_roster(conn, starters, bench, int(season), drafted)
    conn.close()
    sim["ready"] = True
    return sim


def _harness(run_id: str, gen: int, agent: int) -> dict:
    d = _gen_dir(run_id, gen) / "harnesses"
    f = d / f"agent_{agent:02d}.md"
    text = f.read_text() if f.exists() else ""
    audit_f = d / f"agent_{agent:02d}.audit.json"
    audit = json.loads(audit_f.read_text()) if audit_f.exists() else None
    conn = store.connect()
    row = conn.execute(
        """SELECT audit_status FROM harness_versions
           WHERE run_id = ? AND generation = ? AND agent_idx = ?""",
        (run_id, gen, agent),
    ).fetchone()
    conn.close()
    status = row["audit_status"] if row else None
    prev_f = _gen_dir(run_id, gen - 1) / "harnesses" / f"agent_{agent:02d}.md"
    diff = ""
    if gen > 0 and prev_f.exists():
        diff = "\n".join(
            difflib.unified_diff(
                prev_f.read_text().splitlines(),
                text.splitlines(),
                fromfile=f"gen {gen - 1}",
                tofile=f"gen {gen}",
                lineterm="",
            )
        )
    return {"text": text, "audit": audit, "diff": diff, "status": status}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ff-draft-agent observer</title>
<style>
 body{font-family:-apple-system,Helvetica,sans-serif;margin:0;background:#101418;color:#dde3ea}
 header{padding:10px 16px;background:#1a2027;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;color:#7ec8ff} select,button{background:#232b34;color:#dde3ea;border:1px solid #39434e;border-radius:5px;padding:4px 8px}
 main{padding:14px 16px} .row{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#171d24;border:1px solid #2a333d;border-radius:8px;padding:12px;margin-bottom:14px}
 .card h2{font-size:13px;margin:0 0 8px;color:#9fb3c8;text-transform:uppercase;letter-spacing:.06em}
 table{border-collapse:collapse;font-size:11.5px} td,th{border:1px solid #2a333d;padding:3px 6px;text-align:left}
 th{background:#1d242c;color:#9fb3c8}
 .board td{min-width:86px;max-width:86px;height:30px;overflow:hidden;font-size:10.5px;vertical-align:top}
 .QB{background:#3d2f4f}.RB{background:#1f3d2e}.WR{background:#1e3547}.TE{background:#4a3a20}.K{background:#3a2430}.DST{background:#31383f}
 .forfeit{outline:2px solid #c0392b} .cur{outline:2px solid #f1c40f}
 .pos{opacity:.65;font-size:9px} .adp{opacity:.5;font-size:9px}
 td[title]{cursor:help} .why{color:#7ec8ff;font-size:9px}
 pre{background:#0d1116;border:1px solid #2a333d;border-radius:6px;padding:10px;font-size:11px;overflow:auto;max-height:480px;white-space:pre-wrap}
 .dadd{color:#7ee787}.ddel{color:#ff7b72}.dhead{color:#79c0ff}
 .rank1{color:#f1c40f;font-weight:bold} .muted{opacity:.55} .live{color:#7ee787}
</style></head><body>
<header><h1>ff-draft-agent observer</h1>
 run <select id="run"></select> gen <select id="gen"></select> draft <select id="season"></select>
 <span id="status" class="muted"></span>
 <span id="pwrap" style="display:none;align-items:center;gap:6px">
  <span id="plabel" class="muted"></span>
  <span style="display:inline-block;width:150px;height:10px;background:#232b34;border:1px solid #39434e;border-radius:5px;vertical-align:middle"><span id="pfill" style="display:block;height:10px;background:#7ec8ff;border-radius:5px;width:0%"></span></span>
  <span id="peta" class="muted"></span></span>
 <span style="flex:1"></span>
 <span id="cost" class="muted"></span></header>
<main>
 <div class="card"><h2>Draft board <span id="live" class="live"></span></h2><div style="overflow:auto"><table class="board" id="board"></table></div></div>
 <div class="row">
  <div class="card" style="flex:1;min-width:340px"><h2>Standings (this gen)</h2><table id="standings"></table></div>
  <div class="card" style="flex:1;min-width:340px"><h2>Mean rank by generation (lineages)</h2><table id="trajectory"></table></div>
 </div>
 <div class="card"><h2>Team season log — agent <select id="logteam"></select></h2>
  <div id="teamlog"></div></div>
 <div class="card"><h2>Harness — agent <select id="agent"></select>
   <label><input type="checkbox" id="showdiff" checked> diff vs prev gen</label></h2>
  <div id="auditnote" class="muted"></div><pre id="harness"></pre></div>
 <div class="card"><h2>Cost ledger</h2><table id="costs"></table></div>
</main>
<script>
const $=id=>document.getElementById(id); let state=null;
async function j(u){const r=await fetch(u);return r.json()}
function opt(sel,vals,keep){const old=sel.value;const svals=vals.map(String);sel.innerHTML='';svals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o)});if(keep&&svals.includes(old))sel.value=old}
async function refreshRuns(){const runs=await j('/api/runs');opt($('run'),runs,true);if(!$('run').value&&runs.length)$('run').value=runs[runs.length-1]}
async function refreshState(){if(!$('run').value)return;state=await j('/api/run/'+$('run').value);
 opt($('gen'),state.generations.map(g=>g.gen),true);
 const g=state.generations.find(x=>String(x.gen)===$('gen').value)||state.generations[state.generations.length-1];
 if(g){$('gen').value=g.gen;opt($('season'),g.drafts,true);opt($('agent'),g.agents,true);opt($('logteam'),g.agents,true);
  $('status').textContent='gen '+g.gen+': '+(g.phase||'');
  if(!g.drafts.length){$('board').innerHTML=`<tr><td style="padding:14px" class="muted">no draft yet for generation ${g.gen} — ${g.phase}. The board fills in when its draft starts.</td></tr>`;
   $('live').textContent='';$('teamlog').innerHTML='<span class="muted">available once this generation drafts</span>'}}
 $('cost').textContent='spend $'+state.total_usd;
 const pr=state.progress;
 if(pr){$('pwrap').style.display='inline-flex';
  $('plabel').textContent=pr.label+' '+pr.done+'/'+pr.total;
  $('pfill').style.width=Math.round(100*pr.done/pr.total)+'%';
  $('peta').textContent=pr.eta_s==null?'':'~'+(pr.eta_s>=60?Math.round(pr.eta_s/60)+'m':pr.eta_s+'s')+' left'}
 else{$('pwrap').style.display='none'}
 renderCosts();renderTrajectory();renderStandings()}
function renderCosts(){const t=$('costs');t.innerHTML='<tr><th>phase</th><th>model</th><th>calls</th><th>input</th><th>cache-read</th><th>output</th><th>usd</th></tr>';
 state.costs.forEach(c=>{t.insertAdjacentHTML('beforeend',`<tr><td>${c.phase}</td><td>${c.model}</td><td>${c.calls}</td><td>${c.input_tok}</td><td>${c.cache_read_tok}</td><td>${c.output_tok}</td><td>$${c.usd}</td></tr>`)})}
function renderTrajectory(){const t=$('trajectory');const gens=state.generations.filter(g=>g.results);
 t.innerHTML='<tr><th>agent</th>'+gens.map(g=>`<th>g${g.gen}</th>`).join('')+'</tr>';
 if(!gens.length)return;const agents=Object.keys(gens[0].results.ranks);
 agents.sort((a,b)=>gens[gens.length-1].results.ranks[a]-gens[gens.length-1].results.ranks[b]);
 agents.forEach(a=>{t.insertAdjacentHTML('beforeend','<tr><td>'+a+'</td>'+gens.map(g=>{const r=g.results.ranks[a];const cls=r<=2?'rank1':'';return `<td class="${cls}">${r?r.toFixed(1):''}</td>`}).join('')+'</tr>')})}
function renderStandings(){const g=state.generations.find(x=>String(x.gen)===$('gen').value);const t=$('standings');
 t.innerHTML='<tr><th>agent</th><th>mean rank</th><th>scores</th></tr>';if(!g||!g.results)return;
 Object.entries(g.results.ranks).sort((a,b)=>a[1]-b[1]).forEach(([a,r])=>{
  const sc=Object.entries(g.results.scores[a]||{}).map(([s,v])=>`${s}: ${v}`).join('  ');
  t.insertAdjacentHTML('beforeend',`<tr><td>${a}</td><td>${r.toFixed(2)}</td><td>${sc}</td></tr>`)})}
async function refreshDraft(){if(!$('run').value||$('season').value==='')return;
 const d=await j('/api/run/'+$('run').value+'/draft/'+$('gen').value+'/'+$('season').value);
 const teams=14,rounds=15,t=$('board');const bySlot={};(d.slot_order||[]).forEach((team,slot)=>bySlot[slot]=team);
 $('live').textContent=d.done?'':'● live — pick '+(d.picks.length+1);
 let html='<tr><th></th>';for(let s=0;s<teams;s++)html+=`<th>slot ${s+1}<br>agent ${bySlot[s]??''}</th>`;html+='</tr>';
 const grid={};d.picks.forEach(p=>{grid[p.round+'-'+p.slot]=p});
 for(let r=1;r<=rounds;r++){html+=`<tr><th>R${r}</th>`;
  for(let s=0;s<teams;s++){const p=grid[r+'-'+s];
   if(p){const cls=p.position+(p.forfeited?' forfeit':'');
    let tipText=p.reason||'';
    if(p.sources){const s=p.sources;
     if(s.queries&&s.queries.length)tipText+='\\n\\nsearched: '+s.queries.join(' | ');
     if(s.docs&&s.docs.length)tipText+='\\nread:\\n'+s.docs.map(d=>`  [${d.source} ${d.date}] ${d.title||d.url}`).join('\\n')}
    const why=tipText.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
    const tip=why?` title="${why}"`:'';const mark=why?' <span class="why">&#9432;</span>':'';
    html+=`<td class="${cls}"${tip}>${p.name}${mark}<br><span class="pos">${p.position}</span> <span class="adp">adp ${p.adp??''}</span></td>`}
   else{const isNext=d.picks.length&&!d.done&&nextCell(d)===r+'-'+s;html+=`<td class="${isNext?'cur':''}"></td>`}}
  html+='</tr>'}
 t.innerHTML=html}
function nextCell(d){const n=d.picks.length;const r=Math.floor(n/14)+1;const idx=n%14;const s=(r%2===1)?idx:13-idx;return r+'-'+s}
async function refreshTeamlog(){if(!$('run').value||$('logteam').value===''||$('season').value==='')return;
 const t=await j('/api/run/'+$('run').value+'/teamlog/'+$('gen').value+'/'+$('season').value+'/'+$('logteam').value);
 const el=$('teamlog');
 if(!t.ready){el.innerHTML='<span class="muted">draft not finished — log appears when lineups lock</span>';return}
 const nm=pid=>t.names[pid]||pid;
 // weekly lineup table: 9 slots + a column per bench player
 const benchIds=Object.entries(t.players).filter(([,p])=>p.role==='bench').map(([pid])=>pid);
 let html='<div style="overflow:auto"><table><tr><th>wk</th>';
 t.weeks[0].slots.forEach(s=>html+=`<th>${s.slot}</th>`);
 html+='<th>total</th>';
 benchIds.forEach(b=>html+=`<th class="muted">BN ${nm(b).split(' ').pop()}</th>`);
 html+='</tr>';
 t.weeks.forEach(w=>{html+=`<tr><td>${w.week}</td>`;
  const usedThisWeek=new Set(w.slots.map(s=>s.occupant).filter(Boolean));
  w.slots.forEach(s=>{
   if(!s.occupant){html+=`<td class="muted" title="${s.sub?nm(s.starter)+' out ('+s.sub.cause+'), no replacement':''}">—</td>`;return}
   const subbed=s.occupant!==s.starter;
   const style=subbed?(s.sub.cause==='bye'?'background:#1d3a52':'background:#4a2525'):'';
   const tip=subbed?` title="in for ${nm(s.starter)} (${s.sub.cause}) — from ${s.sub.from}"`:'';
   html+=`<td style="${style}"${tip}>${nm(s.occupant)}<br><span class="adp">${s.points}</span></td>`});
  html+=`<td><b>${w.total}</b></td>`;
  benchIds.forEach(b=>{const v=t.players[b].weekly[w.week];
   const active=usedThisWeek.has(b);
   const style=active?'background:#1f3d2e':'opacity:.55';
   const tip=active?' title="pulled into the lineup this week"':'';
   html+=`<td style="${style}"${tip}>${v===undefined?'·':v}</td>`});
  html+='</tr>'});
 html+='</table></div>';
 // roster moves list
 const moves=[];t.weeks.forEach(w=>w.slots.forEach(s=>{
  if(s.occupant&&s.occupant!==s.starter)moves.push(`wk ${w.week}: ${s.slot} — ${nm(s.starter)} out (${s.sub.cause}) → ${nm(s.occupant)} [${s.sub.from}] ${s.points} pts`);
  if(!s.occupant&&s.sub)moves.push(`wk ${w.week}: ${s.slot} — ${nm(s.starter)} out (${s.sub.cause}) → no eligible replacement, 0 pts`)}));
 html+='<h2 style="margin-top:12px">Roster moves</h2>'+(moves.length?'<pre style="max-height:200px">'+moves.join('\\n')+'</pre>':'<span class="muted">none — every starter played every week</span>');
 // player-by-week grid
 const wks=t.weeks.map(w=>w.week);
 html+='<h2 style="margin-top:12px">Player season grid</h2><div style="overflow:auto"><table><tr><th>player</th><th>role</th>'+wks.map(w=>`<th>${w}</th>`).join('')+'<th>total</th></tr>';
 Object.entries(t.players).forEach(([pid,p])=>{
  const tot=Object.values(p.weekly).reduce((a,b)=>a+b,0).toFixed(1);
  html+=`<tr><td>${p.name}</td><td class="muted">${p.position||''} ${p.role}</td>`+
   wks.map(w=>{const v=p.weekly[w];return `<td class="${v===undefined?'muted':''}">${v===undefined?'·':v}</td>`}).join('')+`<td><b>${tot}</b></td></tr>`});
 html+='</table></div>';
 el.innerHTML=html}
async function refreshHarness(){if(!$('run').value||$('agent').value==='')return;
 const h=await j('/api/run/'+$('run').value+'/harness/'+$('gen').value+'/'+$('agent').value);
 const esc=x=>String(x).replace(/&/g,'&amp;').replace(/</g,'&lt;');
 const noteKey=[$('run').value,$('gen').value,$('agent').value,h.status,JSON.stringify(h.audit||null)].join('|');
 if($('auditnote').dataset.key!==noteKey){
  const wasOpen=!!$('auditnote').querySelector('details[open]');
  let note='';
  if(h.status){const col={clean:'#7ee787',stripped:'#f1c40f',reverted:'#ff7b72'}[h.status]||'#9fb3c8';
   note+=`<b style="color:${col}">audit: ${h.status}</b>`}
  if(h.audit){const v=(h.audit.llm&&h.audit.llm.violations)||[];const pre=h.audit.prescreen||[];
   note+=` — ${v.length} LLM violation(s), ${pre.length} prescreen hit(s)`;
   if(v.length||pre.length){note+=`<details${wasOpen?' open':''} style="margin-top:4px"><summary style="cursor:pointer">show what was flagged</summary><table style="margin-top:4px">`;
    pre.forEach(p=>note+=`<tr><td class="muted">prescreen: ${esc(p.rule)}</td><td class="ddel">"${esc(p.span)}"</td></tr>`);
    v.forEach(x=>note+=`<tr><td class="muted">${esc(x.reason||'')}</td><td class="ddel">"${esc(x.span)}"</td></tr>`);
    note+='</table></details>'}}
  $('auditnote').innerHTML=note;$('auditnote').dataset.key=noteKey}
 const el=$('harness');
 const bodyKey=noteKey+'|'+$('showdiff').checked+'|'+(h.diff||'').length+'|'+(h.text||'').length;
 if(el.dataset.key===bodyKey)return;
 el.dataset.key=bodyKey;
 if($('showdiff').checked&&h.diff){el.innerHTML=h.diff.split('\\n').map(l=>{
   const esc=l.replace(/&/g,'&amp;').replace(/</g,'&lt;');
   if(l.startsWith('+'))return `<span class="dadd">${esc}</span>`;
   if(l.startsWith('-'))return `<span class="ddel">${esc}</span>`;
   if(l.startsWith('@@'))return `<span class="dhead">${esc}</span>`;return esc}).join('\\n')}
 else{el.textContent=h.text}}
async function tick(){try{await refreshRuns();await refreshState();await refreshDraft();await refreshTeamlog();await refreshHarness()}catch(e){$('status').textContent='… '+e}}
['run','gen','season','agent','logteam','showdiff'].forEach(id=>$(id).addEventListener('change',tick));
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj, default=str).encode())

    def do_GET(self):  # noqa: N802
        parts = [unquote(p) for p in self.path.strip("/").split("/") if p]
        try:
            if not parts:
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if parts[0] != "api":
                return self._send(b"not found", "text/plain")
            if parts[1] == "runs":
                return self._json(_runs())
            run_id = parts[2]
            if not _SAFE.match(run_id):
                return self._json({"error": "bad run id"})
            if len(parts) == 3:
                return self._json(_run_state(run_id))
            if parts[3] == "draft":
                gen, season = int(parts[4]), parts[5]
                if not _SAFE.match(season):
                    return self._json({"error": "bad season"})
                return self._json(_draft(run_id, gen, season))
            if parts[3] == "teamlog":
                gen, season, team = int(parts[4]), parts[5], int(parts[6])
                if not _SAFE.match(season):
                    return self._json({"error": "bad season"})
                return self._json(_teamlog(run_id, gen, season, team))
            if parts[3] == "harness":
                return self._json(_harness(run_id, int(parts[4]), int(parts[5])))
            return self._json({"error": "unknown endpoint"})
        except Exception as e:
            return self._json({"error": str(e)})


def serve(port: int = 8787) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"observer running at http://127.0.0.1:{port}  (ctrl-c to stop)")
    server.serve_forever()
