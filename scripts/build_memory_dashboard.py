"""Build a self-contained operator-memory dashboard (hand-drawn paper style).

    python scripts/build_memory_dashboard.py
    python scripts/build_memory_dashboard.py --out dashboard/memory.html

Reads rsi/knowledge/{facts,failures,skills,strategies}.jsonl plus conflicts,
embeds everything as JSON, and writes one offline-safe HTML page.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE = os.path.join(PROJECT, "rsi", "knowledge")
DEFAULT_OUT = os.path.join(PROJECT, "dashboard", "memory.html")


def read_jsonl(path: str) -> list:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def age_days(stamp: str) -> float:
    if not stamp:
        return 0.0
    try:
        value = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 86400.0)
    except ValueError:
        return 0.0


def load_data() -> dict:
    facts = read_jsonl(os.path.join(KNOWLEDGE, "facts.jsonl"))
    failures = read_jsonl(os.path.join(KNOWLEDGE, "failures.jsonl"))
    skills = read_jsonl(os.path.join(KNOWLEDGE, "skills.jsonl"))
    strategies = read_jsonl(os.path.join(KNOWLEDGE, "strategies.jsonl"))
    for row in strategies:
        cond = row.get("conditions") or {}
        row["_scope"] = {
            "op_family": cond.get("op_class") or [],
            "soc": cond.get("hardware") or [],
            "dtype": cond.get("dtypes") or [],
        }
    try:
        from rsi.memory import MemoryStore
        conflicts = MemoryStore(KNOWLEDGE).conflicts()
    except Exception as exc:
        conflicts = {"strategies": [], "skills": [], "failure_vs_skill": [],
                     "total": 0, "error": str(exc)}
    stale = [r for r in facts + failures + skills
             if age_days(r.get("updated") or r.get("last_seen") or r.get("created") or "") > 90]
    summary = {
        "facts": len(facts),
        "facts_active": sum(1 for r in facts if r.get("status", "active") == "active"),
        "failures": len(failures),
        "failures_open": sum(1 for r in failures if r.get("status") == "open"),
        "skills": len(skills),
        "skills_verified": sum(1 for r in skills if r.get("confidence") == "verified"),
        "strategies": len(strategies),
        "conflicts": conflicts.get("total", 0),
        "stale": len(stale),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return {
        "summary": summary,
        "facts": facts,
        "failures": failures,
        "skills": skills,
        "strategies": strategies,
        "conflicts": conflicts,
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Operator Memory · kernel-agent</title>
<style>
  :root{
    --ink:#2b2b2b; --paper:#fbf8ef; --muted:#6b6459;
    --yellow:#ffe066; --green:#c9f2c7; --pink:#ffd6e0; --blue:#cfe8ff; --orange:#ffd8a8;
    --red:#d64545; --line:#2b2b2b;
    --mono:'JetBrains Mono','Cascadia Mono',Consolas,'Courier New',monospace;
    --hand:"Kaiti SC","KaiTi","STKaiti","Comic Sans MS",cursive,sans-serif;
  }
  *{box-sizing:border-box}
  body{
    margin:0;padding:26px 18px 64px;color:var(--ink);
    background:radial-gradient(#00000008 1px,transparent 1px),
               linear-gradient(180deg,#f6f1e4 0%,#fbf8ef 42%,#f4efe2 100%);
    background-size:22px 22px,100% 100%;
    font-family:var(--hand);line-height:1.55;
  }
  .wrap{max-width:1240px;margin:0 auto}
  h1{font-size:clamp(34px,5vw,58px);line-height:1.05;display:inline-block;margin:0;
    padding:2px 10px 6px;background:linear-gradient(transparent 62%,var(--yellow) 62%);transform:rotate(-.7deg)}
  h2{font-size:clamp(20px,2.6vw,28px);display:inline-block;margin:0 0 12px;padding:0 6px;
    background:linear-gradient(transparent 58%,#ffe06688 58%);transform:rotate(-.3deg)}
  h3{font-size:17px;margin:0 0 6px}
  p{margin:6px 0}
  .sub{color:var(--muted);font-size:16px;margin-top:10px;max-width:900px}
  .head{position:relative;border:2.5px solid var(--ink);border-radius:255px 15px 225px 15px/15px 225px 15px 255px;
    background:#fffdf6;padding:24px 26px 20px;box-shadow:5px 6px 0 #00000012;transform:rotate(.2deg);margin-bottom:26px}
  .tape{position:absolute;width:146px;height:32px;background:#f7e79a99;top:-15px;left:56px;transform:rotate(-4deg);
    border-left:1px dashed #00000022;border-right:1px dashed #00000022}
  .tape.r{left:auto;right:70px;transform:rotate(3deg);background:#cfe8ff99}
  .pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
  .pill{border:2px solid var(--ink);border-radius:16px 5px 14px 6px;background:#fff;padding:7px 13px;min-width:104px;
    box-shadow:3px 3px 0 #00000012;transform:rotate(-.4deg)}
  .pill .k{font-family:var(--mono);font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  .pill .v{font-family:var(--mono);font-size:21px;font-weight:700}
  nav{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 22px}
  nav button{font-family:var(--hand);font-size:15px;background:#fff;border:2px solid var(--ink);border-radius:14px 5px 13px 5px;
    padding:6px 15px;cursor:pointer;box-shadow:2px 2px 0 #00000012;transform:rotate(-.5deg);color:var(--ink)}
  nav button:nth-child(even){transform:rotate(.5deg)}
  nav button.on{background:var(--yellow);font-weight:700;transform:rotate(0)}
  section{border:2.2px solid var(--ink);border-radius:18px 230px 20px 220px/210px 18px 230px 20px;
    background:#fffdf7cc;padding:20px 22px;margin:22px 0;box-shadow:4px 5px 0 #0000000d}
  .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0 16px}
  .toolbar input,.toolbar select{font-family:var(--hand);font-size:14px;border:2px solid var(--ink);border-radius:10px 4px 10px 4px;
    padding:6px 10px;background:#fff;color:var(--ink)}
  .toolbar input{min-width:260px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
  .card{position:relative;border:2.2px solid var(--ink);border-radius:16px 120px 18px 110px/110px 16px 120px 18px;
    background:#fff;padding:14px 15px 12px;box-shadow:3px 4px 0 #00000010}
  .card:nth-child(odd){transform:rotate(-.35deg)}
  .card:nth-child(even){transform:rotate(.35deg)}
  .badge{font-family:var(--mono);font-size:11px;color:#fff;background:var(--ink);border-radius:9px;padding:2px 8px;
    display:inline-block;margin-bottom:8px}
  .badge.fact{background:#3b5bdb}.badge.failure{background:#c92a2a}.badge.skill{background:#2b8a3e}
  .badge.strategy{background:#5f3dc4}.badge.open{background:#c92a2a}.badge.resolved{background:#2b8a3e}
  .badge.candidate{background:#b45309}.badge.verified{background:#2b8a3e}.badge.stale{background:#868e96}
  .chips{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
  .chip{font-family:var(--mono);font-size:11px;background:#f1efe8;border:1.5px solid #00000022;border-radius:9px;padding:1px 7px}
  .meta{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:6px}
  .meta.break{word-break:break-all}
  .stmt{font-size:15px}
  details{margin-top:8px}
  summary{cursor:pointer;font-size:13px;color:var(--muted)}
  pre{font-family:var(--mono);font-size:11.5px;background:#231f20;color:#f4efe6;border-radius:9px;
    padding:10px;overflow:auto;max-height:280px;white-space:pre-wrap;word-break:break-word}
  .sticky{border:1.8px solid #00000033;padding:14px 15px;box-shadow:3px 5px 10px #00000014;margin:12px 0;position:relative}
  .sticky::before{content:"";position:absolute;top:-9px;left:46px;width:70px;height:18px;background:#ffffff99;
    border:1px solid #00000012;transform:rotate(-2deg)}
  .sticky.yellow{background:var(--yellow);transform:rotate(-.9deg)}
  .sticky.green{background:var(--green);transform:rotate(.7deg)}
  .sticky.pink{background:var(--pink);transform:rotate(-.5deg)}
  .sticky.blue{background:var(--blue);transform:rotate(.5deg)}
  .empty{color:var(--muted);font-size:15px;padding:10px 2px}
  .flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;align-items:stretch;margin:6px 0 4px}
  .step{border:2px solid var(--ink);border-radius:110px 10px 100px 10px/10px 100px 10px 110px;background:#fff;
    padding:12px 10px;text-align:center;position:relative;box-shadow:2px 3px 0 #00000010}
  .step b{display:block;font-size:15px;margin-bottom:2px}
  .step .n{font-family:var(--mono);font-size:11px;color:#fff;background:var(--ink);border-radius:50%;
    width:20px;height:20px;line-height:20px;display:inline-block;margin-bottom:4px}
  .step:not(:last-child)::after{content:"→";position:absolute;right:-14px;top:50%;transform:translateY(-50%);font-size:18px}
  .form{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
  .form label{font-size:13px;color:var(--muted);display:block;margin-bottom:2px}
  .form input,.form select{width:100%;font-family:var(--mono);font-size:12.5px;border:2px solid var(--ink);
    border-radius:9px 4px 9px 4px;padding:6px 8px;background:#fff;color:var(--ink)}
  .btn{font-family:var(--hand);font-size:15px;background:var(--yellow);border:2px solid var(--ink);border-radius:13px 5px 12px 5px;
    padding:7px 18px;cursor:pointer;box-shadow:3px 3px 0 #00000015;margin-top:12px}
  .btn:active{transform:translate(1px,1px);box-shadow:1px 1px 0 #00000015}
  .result{border-left:5px solid var(--ink);padding:8px 12px;margin:10px 0;background:#fff}
  .score{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
  footer{text-align:center;color:var(--muted);font-size:13px;margin-top:34px}
  footer a{color:var(--ink);text-decoration:none;border-bottom:2px solid var(--yellow)}
  @media(max-width:900px){.flow{grid-template-columns:1fr 1fr}.step::after{display:none}}
  @media(max-width:560px){.flow{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header class="head">
    <div class="tape"></div><div class="tape r"></div>
    <h1>Operator Memory</h1>
    <p class="sub">kernel-agent 的算子开发记忆：facts / failures / skills / strategies。<br>
      每条都带 scope 和 provenance；不匹配当前 SOC/dtype/shape 的先验不会注入。</p>
    <div class="pills" id="pills"></div>
  </header>

  <nav id="tabs">
    <button data-tab="overview" class="on">总览</button>
    <button data-tab="facts">Facts</button>
    <button data-tab="failures">Failures</button>
    <button data-tab="skills">Skills</button>
    <button data-tab="conflicts">冲突</button>
    <button data-tab="retrieve">检索</button>
  </nav>

  <main id="view"></main>
  <footer><a href="index.html">← 成绩面板</a> · generated by <code>scripts/build_memory_dashboard.py</code> · __GENERATED__ · 离线可打开</footer>
</div>

<script>
const D = __DATA__;
const S = D.summary;
let state = {tab:"overview", q:"", soc:"", status:""};

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const norm = (s) => String(s == null ? "" : s).toLowerCase().replace(/[-_\s]+/g, "");
const list = (v) => Array.isArray(v) ? v : (v == null || v === "" ? [] : [v]);
const canonDtype = (d) => ({fp16:"float16",half:"float16",bf16:"bfloat16",fp32:"float32",float:"float32"})[norm(d)] || norm(d);
const dtypeBytes = (d) => ({"float16":2,"bfloat16":2,"float32":4,"int8":1})[canonDtype(d)] || 0;

function chips(scope){
  if(!scope) return "";
  return Object.entries(scope).map(([k,v]) => {
    let val = Array.isArray(v) ? v.join(",") : (typeof v === "object" ? JSON.stringify(v) : v);
    return `<span class="chip">${esc(k)}=${esc(val)}</span>`;
  }).join("");
}
function ageDays(stamp){
  if(!stamp) return 0;
  const t = Date.parse(stamp);
  return isNaN(t) ? 0 : Math.max(0, (Date.now() - t) / 86400000);
}
function pill(k,v){ return `<div class="pill"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`; }

function renderPills(){
  $("#pills").innerHTML = [
    pill("facts", S.facts), pill("open failures", S.failures_open),
    pill("verified skills", S.skills_verified), pill("conflicts", S.conflicts),
    pill("stale", S.stale), pill("strategies", S.strategies),
  ].join("");
}

function overview(){
  const open = D.failures.filter(f => f.status === "open");
  const cand = D.skills.filter(s => s.confidence === "candidate");
  const attention = []
    .concat(open.map(f => `<li><b>${esc(f.id)}</b> ${esc(f.signature)} <span class="meta">open · ${esc(f.stage)}</span></li>`))
    .concat(cand.map(s => `<li><b>${esc(s.id)}</b> ${esc(s.name)} <span class="meta">candidate · needs 2nd pass</span></li>`));
  return `
  <section>
    <h2>① 记忆怎么用</h2>
    <div class="flow">
      <div class="step"><span class="n">1</span><b>Episode</b>run / case / log</div>
      <div class="step"><span class="n">2</span><b>蒸馏</b>fact / failure / strategy</div>
      <div class="step"><span class="n">3</span><b>晋升</b>2 次 pass → skill</div>
      <div class="step"><span class="n">4</span><b>检索</b>scope 硬过滤</div>
      <div class="step"><span class="n">5</span><b>下一轮</b>注入 prompt</div>
    </div>
    <div class="sticky yellow" style="margin-top:18px">
      <b>规则：</b>没有 episode 的 performance claim 不进 memory；CPU mechanics run 只能写 <code>soc=cpu</code>；
      scope 不匹配的先验宁可不给。
    </div>
  </section>
  <section>
    <h2>② 现在最需要关注</h2>
    ${attention.length ? `<ul>${attention.join("")}</ul>` : `<div class="empty">没有 open failure 或 candidate skill。</div>`}
    ${S.conflicts ? `<div class="sticky pink"><b>冲突 ${S.conflicts} 条。</b> 去「冲突」页看是策略正负证据、技能 verified/refuted，还是 open failure 撞上 verified skill。</div>` : ""}
  </section>
  <section>
    <h2>③ 最近的事实</h2>
    <div class="grid">${D.facts.slice(0,3).map(factCard).join("")}</div>
  </section>`;
}

function factCard(r){
  return `<article class="card">
    <span class="badge fact">FACT · ${esc(r.type||"")}</span>
    <div class="stmt">${esc(r.statement)}</div>
    <div class="chips">${chips(r.scope)}</div>
    <div class="meta">authority=${esc(r.authority||"-")} · ${esc(r.confidence||"")} · ${esc(r.status||"")}</div>
    <details><summary>scope / evidence</summary><pre>${esc(JSON.stringify({scope:r.scope,evidence:r.evidence,invalidators:r.invalidators},null,1))}</pre></details>
  </article>`;
}
function failureCard(r){
  return `<article class="card">
    <span class="badge failure">FAIL · ${esc(r.stage||"")}</span>
    <span class="badge ${esc(r.status||"")}">${esc(r.status||"")}</span>
    <div class="stmt">${esc(r.signature)}</div>
    <div class="chips">${chips(r.scope)}</div>
    <div class="meta">count=${esc(r.count||1)} · last=${esc((r.last_seen||"").slice(0,19))}</div>
    ${r.minimal_repro ? `<details><summary>minimal repro</summary><pre>${esc(JSON.stringify(r.minimal_repro,null,1))}</pre></details>` : ""}
    ${r.workaround ? `<p><b>workaround:</b> ${esc(r.workaround)}</p>` : ""}
  </article>`;
}
function skillCard(r){
  const params = (r.parameters||[]).map(p => `${p.name}=${p.default ?? p.value ?? "?"}`).join(", ");
  const measured = r.measured && Object.keys(r.measured).length ? JSON.stringify(r.measured) : "-";
  const runs = (r.validation && r.validation.runs) ? r.validation.runs.length : 0;
  return `<article class="card">
    <span class="badge skill">SKILL · ${esc(r.language||"")}</span>
    <span class="badge ${esc(r.confidence||"")}">${esc(r.confidence||"")}</span>
    <h3>${esc(r.name)}</h3>
    <div class="chips">${chips(r.scope)}</div>
    <div class="meta">params: ${esc(params||"-")} · validation runs=${runs}</div>
    <div class="meta">measured: ${esc(measured)}</div>
    ${r.template_ref ? `<div class="meta break">template: ${esc(r.template_ref)}</div>` : ""}
    <details><summary>preconditions / validation</summary><pre>${esc(JSON.stringify({preconditions:r.preconditions,validation:r.validation,invalidators:r.invalidators},null,1))}</pre></details>
  </article>`;
}
function strategyCard(r){
  const st = r.stats || {};
  return `<article class="card">
    <span class="badge strategy">STRATEGY</span>
    <span class="badge ${esc(r.confidence||"")}">${esc(r.confidence||"")}</span>
    <div class="stmt">${esc(r.statement)}</div>
    <div class="chips">${chips({"op_class":(r.conditions||{}).op_class,"hardware":(r.conditions||{}).hardware,"dtype":(r.conditions||{}).dtypes})}</div>
    <div class="meta">applied=${esc(st.applied||0)} · success=${esc(st.succeeded||0)} · EMA=${esc(st.ema_speedup==null?"-":st.ema_speedup)}</div>
  </article>`;
}

function filterRows(rows, extra){
  const q = state.q.toLowerCase();
  return rows.filter(r => {
    if(state.soc && !JSON.stringify(r.scope||r._scope||r.conditions||{}).toLowerCase().includes(state.soc)) return false;
    if(extra && !extra(r)) return false;
    return !q || JSON.stringify(r).toLowerCase().includes(q);
  });
}
function toolbar(showStatus){
  return `<div class="toolbar">
    <input id="q" placeholder="搜索 id / statement / scope..." value="${esc(state.q)}">
    <select id="soc"><option value="">全部 SOC</option>
      ${[...new Set(D.facts.concat(D.failures,D.skills).flatMap(r => list((r.scope||{}).soc)))]
        .map(s => `<option ${state.soc===s?"selected":""}>${esc(s)}</option>`).join("")}
    </select>
    ${showStatus ? `<select id="status"><option value="">全部状态</option>
      <option ${state.status==="open"?"selected":""}>open</option>
      <option ${state.status==="resolved"?"selected":""}>resolved</option></select>` : ""}
  </div>`;
}
function bindToolbar(){
  const q = $("#q"); if(q) q.oninput = e => { state.q = e.target.value; render(); };
  const soc = $("#soc"); if(soc) soc.onchange = e => { state.soc = e.target.value; render(); };
  const st = $("#status"); if(st) st.onchange = e => { state.status = e.target.value; render(); };
}

function factsView(){
  const rows = filterRows(D.facts, r => !state.status || (r.status||"active") === state.status);
  return `<section><h2>Facts · ${rows.length}</h2>${toolbar(true)}
    <div class="grid">${rows.map(factCard).join("") || `<div class="empty">没有匹配的 fact。</div>`}</div></section>`;
}
function failuresView(){
  const rows = filterRows(D.failures, r => !state.status || r.status === state.status);
  return `<section><h2>Failures · ${rows.length}</h2>${toolbar(true)}
    <div class="grid">${rows.map(failureCard).join("") || `<div class="empty">没有匹配的 failure。</div>`}</div></section>`;
}
function skillsView(){
  const rows = filterRows(D.skills, r => !state.status || r.confidence === state.status);
  return `<section><h2>Skills · ${rows.length}</h2>${toolbar(false)}
    <div class="grid">${rows.map(skillCard).join("") || `<div class="empty">没有匹配的 skill。</div>`}</div></section>`;
}
function conflictsView(){
  const c = D.conflicts || {};
  if(!c.total){
    return `<section><h2>冲突 · 0</h2>
      <div class="sticky green"><b>目前没有冲突。</b> 策略正负证据、技能 verified/refuted、open failure vs verified skill 都没有重叠。</div></section>`;
  }
  const strategies = (c.strategies||[]).map(x => `<div class="result"><b>${esc(x.id)}</b> ${esc(x.statement||"")}
    <pre>${esc(JSON.stringify(x.groups,null,1))}</pre></div>`).join("");
  const skills = (c.skills||[]).map(x => `<div class="result"><b>${esc(x.name)}</b> ${esc(x.a)} vs ${esc(x.b)}</div>`).join("");
  const fvs = (c.failure_vs_skill||[]).map(x => `<div class="result"><b>${esc(x.failure)}</b> ${esc(x.signature||"")}
    ↔ <b>${esc(x.skill)}</b> ${esc(x.skill_name||"")}</div>`).join("");
  return `<section><h2>冲突 · ${c.total}</h2>
    ${strategies ? `<h3>策略正负证据</h3>${strategies}` : ""}
    ${skills ? `<h3>技能 verified / refuted</h3>${skills}` : ""}
    ${fvs ? `<h3>open failure vs verified skill</h3>${fvs}` : ""}
  </section>`;
}

// ---- retrieve (client-side mirror of the Python scope filter) ----
function shapeSig(shapeText, dim, dtype){
  const dims = String(shapeText||"").match(/-?\d+/g)?.map(Number) || [];
  if(!dims.length) return {};
  const nd = dims.length;
  let d = Number(dim); if(d < 0) d += nd;
  if(d < 0 || d >= nd) return {};
  let inner = 1; for(let i=d+1;i<nd;i++) inner *= dims[i];
  let outer = 1; for(let i=0;i<d;i++) outer *= dims[i];
  const half = Math.floor(dims[d]/2)*inner;
  const eb = dtypeBytes(dtype);
  let prime = null;
  if(outer <= 10000000){
    if(outer < 2) prime = false;
    else if(outer % 2 === 0) prime = outer === 2;
    else { prime = true; for(let i=3;i*i<=outer;i+=2){ if(outer%i===0){ prime=false; break; } } }
  }
  const bucket = outer <= 1 ? "1" : (outer <= 64 ? "2-64" : (outer <= 1024 ? "65-1024" : "1025+"));
  return {rank:nd, dim:d, outer, inner, half, half_is_1:half===1,
          half_aligned_32b: !!eb && (half*eb)%32===0, outer_bucket:bucket, prime_outer:prime};
}
function verOk(actual, wanted){
  wanted = String(wanted||"").trim();
  if(!wanted || wanted === "*" ) return true;
  const m = wanted.match(/^(>=|<=|>|<|==|=)?\s*(.+)$/);
  const op = m[1] || "==", t = (m[2].match(/\d+/g)||[]).map(Number);
  const a = String(actual||"").match(/\d+/g)?.map(Number) || [];
  if(!a.length || !t.length) return norm(actual) === norm(m[2]);
  const n = Math.max(a.length,t.length);
  const A = a.concat(Array(n-a.length).fill(0)), T = t.concat(Array(n-t.length).fill(0));
  for(let i=0;i<n;i++){ if(A[i]!==T[i]) return op.startsWith(">") ? A[i]>T[i] : (op.startsWith("<") ? A[i]<T[i] : false); }
  return op === "=" || op === "==" || op === ">=" || op === "<=";
}
function matchScope(scope, f){
  scope = scope || {};
  const why = [];
  for(const key of ["op_family","op","soc","form","task"]){
    const want = list(scope[key]).map(norm).filter(Boolean);
    if(!want.length) continue;
    const actual = list(f[key]).map(norm).filter(Boolean);
    if(!actual.length) return {ok:false, why:["missing "+key]};
    if(!want.some(w => actual.some(a => w === a || a.includes(w) || w.includes(a)))) return {ok:false, why:[key+" mismatch"]};
    why.push(key+"="+actual.join(","));
  }
  const wantDt = list(scope.dtype).map(canonDtype).filter(Boolean);
  if(wantDt.length){
    const actual = list(f.dtype).map(canonDtype).filter(Boolean);
    if(!actual.length || !wantDt.some(x => actual.includes(x))) return {ok:false, why:["dtype mismatch"]};
    why.push("dtype="+actual.join(","));
  }
  for(const key of ["cann","bisheng","triton"]){
    const want = list(scope[key]); if(!want.length) continue;
    if(!f[key] || !want.some(w => verOk(f[key], w))) return {ok:false, why:[key+" mismatch"]};
    why.push(key+"="+f[key]);
  }
  const sig = f.shape_signature || {};
  for(const [key, want] of Object.entries(scope.shape||{})){
    const actual = sig[key];
    const ok = Array.isArray(want) ? want.includes(actual) : actual === want;
    if(!ok) return {ok:false, why:["shape."+key+" mismatch"]};
    why.push("shape."+key+"="+actual);
  }
  return {ok:true, why};
}
function retrieveView(){
  return `<section><h2>按 scope 检索</h2>
    <p class="meta">不匹配的 SOC / dtype / shape 不会被返回。这里的前端过滤与 <code>rsi.memory</code> 的规则一致。</p>
    <div class="form">
      <div><label>op</label><input id="f_op" value="swi_glu"></div>
      <div><label>op_family</label><input id="f_family" value="elementwise"></div>
      <div><label>soc</label><input id="f_soc" value="ascend-950pr"></div>
      <div><label>dtype</label><input id="f_dtype" value="bf16"></div>
      <div><label>form</label><input id="f_form" value="ascendc"></div>
      <div><label>shape</label><input id="f_shape" value="4096,8192"></div>
      <div><label>dim</label><input id="f_dim" value="-1"></div>
    </div>
    <button class="btn" id="doRetrieve">检索记忆 →</button>
    <div id="retrieve_out"></div>
  </section>`;
}
function runRetrieve(){
  const f = {
    op: $("#f_op").value.trim(),
    op_family: $("#f_family").value.trim(),
    soc: $("#f_soc").value.trim(),
    dtype: [$("#f_dtype").value.trim()].filter(Boolean),
    form: [$("#f_form").value.trim()].filter(Boolean),
    shape_signature: shapeSig($("#f_shape").value, $("#f_dim").value, $("#f_dtype").value),
  };
  const rows = [];
  const add = (kind, rec, scope, weight) => {
    const m = matchScope(scope, f);
    if(!m.ok) return;
    let conf = rec.confidence || (rec.status === "open" ? "verified" : "candidate");
    const cw = {verified:1.0,candidate:0.6,seed:0.35,refuted:0.05}[conf] ?? 0.4;
    rows.push({kind, rec, score: weight*(0.55+0.45*Math.min(1,0.55+0.08*m.why.length))*cw, why:m.why});
  };
  D.facts.forEach(r => add("FACT", r, r.scope, 1.0));
  D.failures.forEach(r => add("FAILURE", r, r.scope, r.status === "open" ? 0.92 : 0.5));
  D.skills.forEach(r => add("SKILL", r, r.scope, 0.86));
  D.strategies.forEach(r => add("STRATEGY", r, r._scope, 0.80));
  rows.sort((a,b) => b.score - a.score);
  const top = rows.slice(0, 12);
  $("#retrieve_out").innerHTML = top.length ? `
    <div class="sticky blue" style="margin-top:16px">命中 ${rows.length} 条，显示前 ${top.length} 条。</div>
    ${top.map(x => `<div class="result">
      <span class="badge ${x.kind.toLowerCase()}">${x.kind}</span>
      <span class="score">score=${x.score.toFixed(3)}</span>
      <div class="stmt">${esc(x.rec.statement || x.rec.signature || x.rec.name || "")}</div>
      <div class="meta">why: ${esc(x.why.join(", "))}</div>
    </div>`).join("")}` : `<div class="empty" style="margin-top:14px">没有命中。检查 SOC/dtype/form/shape 是否写对。</div>`;
}

function render(){
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === state.tab));
  const view = $("#view");
  const map = {overview, facts:factsView, failures:failuresView, skills:skillsView,
               conflicts:conflictsView, retrieve:retrieveView};
  view.innerHTML = (map[state.tab] || overview)();
  bindToolbar();
  if(state.tab === "retrieve"){ $("#doRetrieve").onclick = runRetrieve; }
}
document.querySelectorAll("#tabs button").forEach(b => {
  b.onclick = () => { state.tab = b.dataset.tab; render(); };
});
renderPills();
render();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    data = load_data()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__DATA__", payload).replace("__GENERATED__", data["summary"]["generated_at"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    s = data["summary"]
    print("wrote %s" % args.out)
    print("facts=%d open_failures=%d verified_skills=%d conflicts=%d stale=%d" % (
        s["facts"], s["failures_open"], s["skills_verified"], s["conflicts"], s["stale"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())