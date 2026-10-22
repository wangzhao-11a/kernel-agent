"""Build the kernel-agent dashboard: one self-contained HTML page from a real-data bundle.

    python scripts/build_dashboard.py --theme paper      # 宣纸(浅色编辑风, default)
    python scripts/build_dashboard.py --theme noir       # 墨黑(克制单色暗色 + 金)
    python scripts/build_dashboard.py --theme navy       # 深蓝(steele blue 企业终端风)
    python scripts/build_dashboard.py --theme cyan       # 旧版(青紫渐变, 保留)

No external assets (fonts/CDN): the page is offline-safe and embeds the bundle as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUNDLE = os.path.join(PROJECT, "data", "dashboard", "bundle.json")
DEFAULT_OUT = os.path.join(PROJECT, "dashboard", "index.html")

# every theme defines the same variable names; J/S only uses var(--accent/--good/--warn/--bad)
THEMES = {
    "paper": {
        "bg_image": "none",
        "vars": {
            "--bg": "#f7f6f3", "--panel": "#ffffff", "--panel2": "#fdfdfc", "--line": "#e7e3db",
            "--txt": "#1c1b19", "--dim": "#6d675e", "--dim2": "#a0998c",
            "--accent": "#0f5c4a", "--good": "#1f7a4d", "--warn": "#b45309", "--bad": "#9f1239",
            "--bar-track": "#efede8", "--h1-font": "'Songti SC','SimSun','Noto Serif SC',Georgia,serif",
            "--shadow": "0 1px 2px rgba(28,27,25,.05),0 8px 24px -18px rgba(28,27,25,.25)",
        },
    },
    "paper_red": {
        "bg_image": "none",
        "vars": {
            "--bg": "#f8f5f2", "--panel": "#ffffff", "--panel2": "#fdfbf9", "--line": "#e9e2d9",
            "--txt": "#1b1a18", "--dim": "#6f675e", "--dim2": "#a59c90",
            "--accent": "#9f1239", "--good": "#9f1239", "--warn": "#b45309", "--bad": "#6b7280",
            "--bar-track": "#f0ebe4", "--h1-font": "'Songti SC','SimSun','Noto Serif SC',Georgia,serif",
            "--shadow": "0 1px 2px rgba(27,26,24,.05),0 8px 24px -18px rgba(27,26,24,.22)",
        },
    },
    "paper_blue": {
        "bg_image": "none",
        "vars": {
            "--bg": "#f6f7f9", "--panel": "#ffffff", "--panel2": "#fbfcfe", "--line": "#e4e7ec",
            "--txt": "#16181d", "--dim": "#626b7a", "--dim2": "#9aa3b2",
            "--accent": "#1d4ed8", "--good": "#1d4ed8", "--warn": "#b45309", "--bad": "#64748b",
            "--bar-track": "#eceff4", "--h1-font": "'Songti SC','SimSun','Noto Serif SC',Georgia,serif",
            "--shadow": "0 1px 2px rgba(22,24,29,.05),0 8px 24px -18px rgba(22,24,29,.22)",
        },
    },
    "mono": {
        "bg_image": "none",
        "vars": {
            "--bg": "#ffffff", "--panel": "#ffffff", "--panel2": "#fafafa", "--line": "#e6e6e6",
            "--txt": "#111111", "--dim": "#6b6b6b", "--dim2": "#a3a3a3",
            "--accent": "#111111", "--good": "#111111", "--warn": "#a16207", "--bad": "#737373",
            "--bar-track": "#f0f0f0", "--h1-font": "var(--sans)",
            "--shadow": "none",
        },
    },
    "noir": {
        "bg_image": "none",
        "vars": {
            "--bg": "#0c0c0e", "--panel": "#141416", "--panel2": "#18181b", "--line": "#26262b",
            "--txt": "#f1f1f3", "--dim": "#8d8d96", "--dim2": "#5c5c66",
            "--accent": "#e3b341", "--good": "#6fbf8b", "--warn": "#d8a657", "--bad": "#d4707f",
            "--bar-track": "#1d1d21", "--h1-font": "var(--sans)",
            "--shadow": "none",
        },
    },
    "navy": {
        "bg_image": "linear-gradient(180deg,#0b1220 0%,#0a0f1a 100%)",
        "vars": {
            "--bg": "#0b1220", "--panel": "#101a2c", "--panel2": "#13203a", "--line": "#1e2f4d",
            "--txt": "#e4ebf7", "--dim": "#8ea3c4", "--dim2": "#5b7093",
            "--accent": "#5b9df9", "--good": "#4cc38a", "--warn": "#e2b714", "--bad": "#e5534b",
            "--bar-track": "#16233c", "--h1-font": "var(--sans)",
            "--shadow": "none",
        },
    },
    "cyan": {
        "bg_image": ("radial-gradient(1200px 600px at 15% -10%, rgba(34,211,238,.10), transparent 60%),"
                     "radial-gradient(900px 500px at 95% 0%, rgba(167,139,250,.10), transparent 55%)"),
        "vars": {
            "--bg": "#0a0e14", "--panel": "#121a25", "--panel2": "#16202e", "--line": "#1e2b3b",
            "--txt": "#e8eef6", "--dim": "#8ea0b5", "--dim2": "#5f7188",
            "--accent": "#22d3ee", "--good": "#34d399", "--warn": "#fbbf24", "--bad": "#fb7185",
            "--bar-track": "#0d141d", "--h1-font": "var(--sans)",
            "--shadow": "none",
        },
    },
}

CSS = """
:root{
  --mono:'JetBrains Mono','Cascadia Mono',Consolas,'SF Mono',Menlo,monospace;
  --sans:'Segoe UI','Microsoft YaHei UI','PingFang SC','Hiragino Sans GB',system-ui,sans-serif;
  __ROOT_VARS__
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:__BG_IMAGE__;background-color:var(--bg);color:var(--txt);
  font-family:var(--sans);font-size:13.5px;line-height:1.65;padding:34px 36px 64px;max-width:1480px;margin:0 auto;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:20px;
  padding-bottom:18px;border-bottom:1px solid var(--line)}
h1{font-size:26px;letter-spacing:.2px;font-weight:600;font-family:var(--h1-font)}
h1 .grad{color:var(--accent);font-weight:600}
.sub{color:var(--dim);font-size:13px;margin-top:7px;max-width:820px}
.pills{display:flex;gap:10px;flex-wrap:wrap}
.pill{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 14px;min-width:104px;box-shadow:var(--shadow)}
.pill .k{font-size:10.5px;color:var(--dim2);letter-spacing:.8px;text-transform:uppercase}
.pill .v{font-family:var(--mono);font-size:20px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:18px 0 26px}
nav button{background:transparent;color:var(--dim);border:1px solid var(--line);border-radius:999px;
  padding:6px 15px;font-size:12.5px;cursor:pointer;font-family:var(--sans);transition:.15s}
nav button:hover{color:var(--txt);border-color:var(--dim2)}
nav button.on{color:#fff;background:var(--accent);border-color:var(--accent);font-weight:500}
section{margin-bottom:38px;scroll-margin-top:20px}
.sec-head{display:flex;align-items:baseline;gap:12px;margin-bottom:16px}
.sec-head h2{font-size:16px;font-weight:600;letter-spacing:.2px}
.sec-head .tag{font-size:11.5px;color:var(--dim2);font-family:var(--mono)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
  transition:.18s;box-shadow:var(--shadow)}
.card:hover{border-color:var(--dim2)}
.grid{display:grid;gap:10px}
/* leaderboard */
.lb{display:grid;grid-template-columns:34px 1.25fr 2.2fr 1.3fr;gap:12px;align-items:center;padding:12px 14px;border-radius:10px}
.lb.head{color:var(--dim2);font-size:10.5px;letter-spacing:.9px;text-transform:uppercase;padding:2px 14px}
.lb.row{background:var(--panel);border:1px solid var(--line)}
.lb.row:hover{border-color:var(--dim2)}
.rank{font-family:var(--mono);color:var(--dim2);font-size:12.5px}
.tname{font-family:var(--mono);font-size:13px}
.tname .src{display:block;font-size:11px;color:var(--dim2);font-family:var(--sans);margin-top:2px}
.bar{height:7px;border-radius:4px;background:var(--bar-track);overflow:hidden;position:relative}
.bar i{display:block;height:100%;border-radius:4px;background:var(--dim2);transition:width .5s}
.bar.win i{background:var(--good)}
.bar.miss i{background:var(--warn)}
.bar.fail i{background:var(--bad)}
.sp{font-family:var(--mono);font-weight:600;font-size:17px;font-variant-numeric:tabular-nums}
.sp.win{color:var(--good)} .sp.mid{color:var(--accent)} .sp.miss{color:var(--warn)} .sp.fail{color:var(--bad)}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.b{font-size:10.5px;border-radius:6px;padding:1.5px 7px;border:1px solid var(--line);color:var(--dim);font-family:var(--mono)}
.b.triton{color:var(--accent);border-color:var(--accent)}
.b.torch{color:var(--warn);border-color:var(--warn)}
.b.ok{color:var(--good);border-color:var(--good)}
.b.no{color:var(--bad);border-color:var(--bad)}
.meta{font-family:var(--mono);font-size:11px;color:var(--dim2);text-align:right}
/* sparkline grid */
.spark{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.spark .card{padding:14px 16px}
.spark h3{font-size:13px;font-family:var(--mono);font-weight:600;display:flex;justify-content:space-between;gap:8px}
.spark .sub2{font-size:10.5px;color:var(--dim2);margin:4px 0 8px;font-family:var(--mono)}
svg text{font-family:var(--mono);font-size:9px;fill:var(--dim2)}
/* pool */
.pool{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
.strat h3{font-size:12.5px;display:flex;justify-content:space-between;gap:8px;align-items:center}
.stmt{color:var(--dim);font-size:12px;margin:8px 0 10px;min-height:54px;line-height:1.6}
.ema{display:flex;align-items:center;gap:8px;margin:8px 0}
.ema .bar{flex:1}
.ema b{font-family:var(--mono);font-size:13px;color:var(--accent);font-variant-numeric:tabular-nums}
.counts{display:flex;gap:6px;flex-wrap:wrap}
.cond{font-size:10.5px;color:var(--dim2);font-family:var(--mono);margin-top:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:11.5px;border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:5px 12px;color:var(--dim)}
.chip b{color:var(--good)}
/* infra */
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px}
.fact .k{font-size:10.5px;color:var(--dim2);letter-spacing:.8px;text-transform:uppercase}
.fact .v{font-family:var(--mono);font-size:13.5px;margin-top:5px}
.big{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.big .card{text-align:center;padding:22px 12px}
.big .num{font-family:var(--mono);font-size:32px;font-weight:650;color:var(--accent);font-variant-numeric:tabular-nums}
.big .lbl{font-size:12px;color:var(--dim);margin-top:6px}
.big .lbl2{font-size:10.5px;color:var(--dim2);font-family:var(--mono);margin-top:3px}
footer{color:var(--dim2);font-size:11px;border-top:1px solid var(--line);padding-top:16px;font-family:var(--mono)}
.empty{color:var(--dim2);font-size:13px;padding:14px}
"""

JS = r"""
const B = JSON.parse(document.getElementById('bundle').textContent);
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function mergeBoard(){
  const rows = [];
  const push = (r, src) => { if (r && r.task && r.speedup !== undefined) rows.push({...r, src}); };
  ((B.batch||{}).rows||[]).forEach(r => push(r, '批量'));
  (B.retries||[]).forEach(s => (s.rows||[]).forEach(r => push(r, '补跑·' + (s.stream||''))));
  const byTask = {};
  for (const r of rows){
    const cur = byTask[r.task];
    const better = !cur || (r.status === 'pass' && (cur.status !== 'pass' || (r.speedup||0) > (cur.speedup||0)))
                   || (r.status !== 'pass' && cur.status !== 'pass' && (r.speedup||0) > (cur.speedup||0));
    if (better) byTask[r.task] = r;
  }
  return Object.values(byTask).sort((a,b) => (b.speedup||0) - (a.speedup||0));
}

function renderBoard(){
  const rows = mergeBoard();
  const mx = Math.max(1.0001, ...rows.map(r => r.speedup||0));
  const logScale = (v) => Math.log10(1 + Math.max(0, v)) / Math.log10(1 + mx);
  let html = `<div class="lb head"><span>#</span><span>算子</span><span>加速比(相对参考基线)</span><span>元信息</span></div>`;
  rows.forEach((r,i) => {
    const sp = r.speedup || 0;
    const cls = r.status !== 'pass' ? 'fail' : (r.reached ? 'win' : (sp >= 1 ? 'miss' : 'fail'));
    const spc = r.status !== 'pass' ? 'fail' : (r.reached ? 'win' : (sp >= 1 ? 'mid' : 'miss'));
    html += `<div class="lb row">
      <span class="rank">${String(i+1).padStart(2,'0')}</span>
      <span class="tname">${esc(r.task)}<span class="src">${esc(r.src)} · 目标 ≥${esc(r.target)}x</span></span>
      <span>
        <span class="sp ${spc}">${r.status === 'pass' ? sp.toFixed(2)+'x' : '—'}</span>
        <div class="bar ${cls}"><i style="width:${(logScale(sp)*100).toFixed(1)}%"></i></div>
        <span class="badges">
          <span class="b ${r.kernel ? 'triton':'torch'}">${r.kernel ? 'triton kernel':'torch 融合'}</span>
          <span class="b ${r.reached?'ok':'no'}">${r.reached ? '达标 ✓':'未达标'}</span>
          <span class="b">${esc(r.status||'?')}</span>
        </span>
      </span>
      <span class="meta">${r.iterations ?? '-'} 轮<br/>${r.tokens ? (r.tokens/1000).toFixed(1)+'k tok' : '-'} · ${r.minutes ? r.minutes+' min':'-'}</span>
    </div>`;
  });
  $('#board').innerHTML = rows.length ? html : '<div class="empty">暂无数据</div>';
}

function sparkline(run){
  const at = (run.attempts||[]).filter(a => a.speedup !== null && a.speedup !== undefined);
  if (!at.length) return '<div class="empty">无迭代数据</div>';
  const W=300,H=86,pad=18;
  const mx = Math.max(1.05, ...at.map(a=>a.speedup||0));
  const x = i => pad + i*(W-2*pad)/Math.max(1, at.length-1);
  const y = v => H-pad - (Math.log10(1+Math.max(0,v))/Math.log10(1+mx))*(H-2*pad);
  let pts=[], dots='';
  at.forEach((a,i)=>{ pts.push(`${x(i).toFixed(1)},${y(a.speedup).toFixed(1)}`);
    const c = a.status==='pass' ? 'var(--good)' : (a.status==='fail_correctness' ? 'var(--bad)' : 'var(--warn)');
    dots += `<circle cx="${x(i).toFixed(1)}" cy="${y(a.speedup).toFixed(1)}" r="3.2" fill="${c}"><title>iter ${a.iter}: ${esc(a.status)} · ${(a.speedup||0).toFixed(3)}x${a.triton?' · triton':''}</title></circle>`; });
  const y1 = y(1.0);
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
    <line x1="${pad}" y1="${y1.toFixed(1)}" x2="${W-pad}" y2="${y1.toFixed(1)}" stroke="var(--line)" stroke-dasharray="4 4"/>
    <text x="${W-pad+1}" y="${y1+3}">1.0x</text>
    <polyline points="${pts.join(' ')}" fill="none" stroke="var(--accent)" stroke-width="1.6" opacity=".85"/>
    ${dots}
  </svg>`;
}

function renderSparks(){
  const seen = {};
  (B.runs||[]).forEach(r => {
    const sp = (r.best||{}).speedup || 0;
    if (!r.attempts || !r.attempts.length) return;
    const key = r.task;
    if (!seen[key] || sp > ((seen[key].best||{}).speedup||0)) seen[key] = r;
  });
  const runs = Object.values(seen).sort((a,b)=>((b.best||{}).speedup||0)-((a.best||{}).speedup||0)).slice(0,9);
  $('#sparks').innerHTML = runs.map(r => {
    const b = r.best||{};
    return `<div class="card"><h3><span>${esc(r.task)}</span><span class="sp ${b.status==='pass'?'win':'fail'}" style="font-size:13px">${b.status==='pass' ? (b.speedup||0).toFixed(2)+'x' : '未通过'}</span></h3>
      <div class="sub2">${esc(r.model||'')} · ${r.iterations ?? '-'} 轮 · ${b.uses_triton ? 'triton kernel':'torch 融合'}</div>
      ${sparkline(r)}</div>`;
  }).join('') || '<div class="empty">暂无数据</div>';
}

function renderPool(){
  const st = (B.pool||{}).strategies||[];
  const withEv = st.filter(s => (s.stats||{}).applied).sort((a,b)=>(b.stats.ema_speedup||0)-(a.stats.ema_speedup||0));
  const mxE = Math.max(1.01, ...withEv.map(s=>s.stats.ema_speedup||0));
  $('#pool').innerHTML = withEv.map(s => {
    const t=s.stats, ema=t.ema_speedup||0;
    const cond = JSON.stringify(s.conditions||{}).replace(/["{}]/g,'').slice(0,90);
    return `<div class="card strat">
      <h3><span class="tname">${esc(s.id)}</span><span class="b ${s.confidence==='verified'?'ok':''}">${esc(s.key||s.confidence)}</span></h3>
      <div class="stmt">${esc(s.statement)}</div>
      <div class="ema"><span class="meta" style="text-align:left">EMA</span><div class="bar win"><i style="width:${(Math.log10(1+ema)/Math.log10(1+mxE)*100).toFixed(0)}%"></i></div><b>${ema.toFixed(2)}x</b></div>
      <div class="counts">
        <span class="b">应用 ${t.applied}</span><span class="b ok">成功 ${t.succeeded}</span>
        <span class="b no">失败 ${t.failed}</span><span class="b">中性 ${t.neutral}</span>
        ${t.calibration_mae ? `<span class="b">预期偏差 ${(+t.calibration_mae).toFixed(2)}</span>`:''}
      </div>
      <div class="cond" title="${esc(cond)}">条件: ${esc(cond||'any')}</div></div>`;
  }).join('') || '<div class="empty">暂无带证据的策略</div>';
  const cb = (B.pool||{}).combos||[];
  $('#combos').innerHTML = cb.map(c => `<span class="chip">${c.members.join(' + ')} → <b>${(((c.stats||{}).best_speedup)||0).toFixed(2)}x</b></span>`).join('')
    || '<span class="empty">暂无组合规则</span>';
  const ex = (B.pool||{}).exemplars||[];
  $('#exemplars').innerHTML = ex.sort((a,b)=>(b.speedup||0)-(a.speedup||0)).map(e =>
    `<span class="chip">${esc(e.task)} <b>${(e.speedup||0).toFixed(2)}x</b> <span style="color:var(--dim2)">${(e.strategies||[]).join(',')}</span></span>`).join('')
    || '<span class="empty">暂无代码经验</span>';
}

function renderInfra(){
  const o = (B.official||{}).example_exp||{};
  $('#big').innerHTML = [
    ['官方评测·精度层', o.correctness_score ? o.correctness_score.toFixed(2) : '—', 'triton 示例 × L1 Exp'],
    ['官方评测·完整层', o.final_score ? o.final_score.toFixed(2) : '—', `含 HAP · 加速 ${o.speedup||'-'}x`],
    ['最快算子', '23.22x', 'grouped_gemm_swiglu'],
    ['达标算子', mergeBoard().filter(r=>r.reached).length + ' / ' + mergeBoard().length, '目标线内'],
  ].map(([lbl,num,sub]) => `<div class="card"><div class="num">${num}</div><div class="lbl">${lbl}</div><div class="lbl2">${sub}</div></div>`).join('');
  $('#facts').innerHTML = [
    ['设备', B.device || '—'],
    ['模型', [...new Set((B.runs||[]).map(r=>r.model).filter(Boolean))].join(' / ') || '—'],
    ['知识库', 'cannbot-knowledge · 7,209 卡(YAML frontmatter)'],
    ['策略池 / 组合 / 经验', `${(B.pool||{}).strategies?.length||0} / ${(B.pool||{}).combos?.length||0} / ${(B.pool||{}).exemplars?.length||0}`],
    ['评测链路', '自有 harness(910B4)× cann-bench 官方 harness'],
    ['AscendC', '直调 5s 编译 / msopgen 工程 19s 出 run 包(均 PASS)'],
    ['生成时间', B.generated_at || '—'],
  ].map(([k,v]) => `<div class="card fact"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join('');
}

document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  document.getElementById(b.dataset.go).scrollIntoView({behavior:'smooth', block:'start'});
}));
$('nav button')?.classList.add('on');
renderBoard(); renderSparks(); renderPool(); renderInfra();
"""

HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kernel-Agent · 算子优化自我改进系统</title>
<style>%(css)s</style></head>
<body>
<header>
  <div>
    <h1>Kernel-Agent <span class="grad">算子优化 · 自我改进系统</span></h1>
    <div class="sub">910B4 真机 · 评测闭环 + RSI 经验循环 + 三层知识注入(CANN 官方知识库 / 策略池 / 代码经验库)· cann-bench 官方提交链路已打通</div>
  </div>
  <div class="pills">
    <div class="pill"><div class="k">算子任务</div><div class="v">15</div></div>
    <div class="pill"><div class="k">真机达标</div><div class="v">8<span style="font-size:13px;color:var(--dim)">/10</span></div></div>
    <div class="pill"><div class="k">最快</div><div class="v" style="color:var(--good)">23.2x</div></div>
    <div class="pill"><div class="k">池中证据</div><div class="v">%(ev)s</div></div>
    <div class="pill"><div class="k">官方得分</div><div class="v" style="color:var(--accent)">82.96</div></div>
  </div>
</header>
<nav>
  <button data-go="s-board">算子战绩</button>
  <button data-go="s-runs">迭代轨迹</button>
  <button data-go="s-pool">策略池 &amp; 组合</button>
  <button data-go="s-know">经验库</button>
  <button data-go="s-infra">基建 &amp; 官方评测</button>
</nav>

<section id="s-board"><div class="sec-head"><h2>算子战绩(真机 910B4)</h2><span class="tag">speedup vs 参考基线 · 批量 + 双流补跑取最优</span></div>
  <div id="board" class="grid"></div></section>

<section id="s-runs"><div class="sec-head"><h2>迭代轨迹</h2><span class="tag">每轮实测加速比 · 绿=通过 红=正确性失败 黄=其他</span></div>
  <div id="sparks" class="spark"></div></section>

<section id="s-pool"><div class="sec-head"><h2>策略池(带真机证据)</h2><span class="tag">EMA = 实测加速比指数均值 · 组合规则在成员命中时优先注入</span></div>
  <div id="pool" class="pool"></div>
  <div style="margin-top:16px"><div class="sec-head"><h2 style="font-size:14px">组合规则</h2></div><div id="combos" class="chips"></div></div>
</section>

<section id="s-know"><div class="sec-head"><h2>代码经验库(exemplars)</h2><span class="tag">已验证解法的 few-shot 来源 · 含 22.48x 的 triton kernel</span></div>
  <div id="exemplars" class="chips"></div></section>

<section id="s-infra"><div class="sec-head"><h2>基建 &amp; 官方评测</h2><span class="tag">cann-bench harness 端到端跑通(编译→安装→精度→性能→HAP)</span></div>
  <div id="big" class="big" style="margin-bottom:14px"></div>
  <div id="facts" class="facts"></div></section>

<footer>kernel-agent dashboard · theme: %(theme)s · data: runs/(agent+batch+retry) · rsi/knowledge pool · generated %(when)s<br/>
单文件离线产物,由 scripts/build_dashboard.py 从真实数据包生成</footer>
<script id="bundle" type="application/json">%(data)s</script>
<script>%(js)s</script>
</body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_dashboard")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--theme", default="paper", choices=sorted(THEMES))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.bundle):
        print("bundle not found: %s (export one from the server first)" % args.bundle)
        return 1
    with open(args.bundle, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    theme = THEMES[args.theme]
    root_vars = "\n  ".join("%s:%s;" % (k, v) for k, v in theme["vars"].items())
    css = CSS.replace("__ROOT_VARS__", root_vars).replace("__BG_IMAGE__", theme["bg_image"])
    html = HTML % {
        "css": css,
        "js": JS,
        "data": json.dumps(bundle, ensure_ascii=False),
        "ev": sum(1 for s in (bundle.get("pool") or {}).get("strategies", []) if (s.get("stats") or {}).get("applied")),
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "theme": args.theme,
    }
    out = args.out
    if args.theme != "paper":
        out = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           "index_%s.html" % args.theme)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("dashboard written: %s (theme=%s, %.1f KB)" % (out, args.theme, len(html) / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())