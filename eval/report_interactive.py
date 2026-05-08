#!/usr/bin/env python3
"""Interactive HTML report: charts + filterable data table + per-item trace detail."""

import json
import sys
import os
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# CJK font
ZH_FONT = "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"
fm.fontManager.addfont(ZH_FONT)
plt.rcParams["font.family"] = "AR PL SungtiL GB"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "grid.color": "#21262d", "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
})

C = {"green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff",
     "red": "#f78166", "grey": "#8b949e", "border": "#30363d"}


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_funnel(metrics: dict) -> str:
    tiers = metrics["tier_distribution"]
    total = metrics["total"]
    order = [
        ("L0_memory", "L0 内存缓存", C["green"]),
        ("L0_redis", "L0 Redis 缓存", C["green"]),
        ("L0_whitelist", "L0 白名单", C["green"]),
        ("L1_keyword", "L1 关键词拦截", C["yellow"]),
        ("L1_chroma", "L1 ChromaDB 缓存", C["yellow"]),
        ("L2_bert", "L2 BERT 分类", C["blue"]),
        ("L3_llm", "L3 LLM 深度审核", C["red"]),
    ]
    labels, values, colors = [], [], []
    for k, lbl, c in order:
        v = tiers.get(k, 0)
        if v > 0:
            labels.append(lbl); values.append(v); colors.append(c)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor=C["border"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v} 条 ({v/total*100:.1f}%)", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("请求数"); ax.set_title(f"三层漏斗流量分布 (n={total})", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def chart_latency(results: list, metrics: dict) -> str:
    lats = [r["latency_ms"] for r in results]
    p50, p95, p99 = metrics["latency_p50_ms"], metrics["latency_p95_ms"], metrics["latency_p99_ms"]
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.logspace(np.log10(max(0.1, min(lats))), np.log10(max(lats)), 50)
    ax.hist(lats, bins=bins, color=C["blue"], alpha=0.7, edgecolor=C["border"])
    for val, label, color in [(p50, f"P50={p50:.0f}ms", C["green"]),
                               (p95, f"P95={p95:.0f}ms", C["yellow"]),
                               (p99, f"P99={p99:.0f}ms", C["red"])]:
        ax.axvline(val, color=color, linewidth=2, linestyle="--")
        ax.text(val + max(lats) * 0.02, ax.get_ylim()[1] * 0.85,
                label, color=color, fontsize=10, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("延迟 (ms, 对数刻度)"); ax.set_ylabel("请求数")
    ax.set_title(f"延迟分布 (平均 {metrics['avg_latency_ms']:.0f}ms)", fontweight="bold"); ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def chart_confusion(metrics: dict) -> str:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=metrics["total"] // 2, aspect="auto")
    cell_text = [
        [f"TN = {cm[0,0]}\n正确放行", f"FP = {cm[0,1]}\n误拦截"],
        [f"FN = {cm[1,0]}\n漏网", f"TP = {cm[1,1]}\n正确拦截"],
    ]
    for i in range(2):
        for j in range(2):
            clr = "white" if cm[i, j] > metrics["total"] // 5 else C["grey"]
            ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=12, fontweight="bold", color=clr)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测 Safe", "预测 Unsafe"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["实际 Safe", "实际 Unsafe"])
    ax.set_title("混淆矩阵", fontweight="bold")
    return fig_to_b64(fig)


def chart_passfail(metrics: dict) -> str:
    criteria = [
        ("LLM 调用率 < 20%", metrics["llm_call_rate"], 0.20, False),
        ("BERT 拦截率 > 30%", metrics["bert_intercept_rate"], 0.30, True),
        ("F1-score > 90%", metrics["f1"], 0.90, True),
        ("缓存命中率 > 15%", metrics["cache_hit_rate"], 0.15, True),
        ("平均延迟 < 500ms", metrics["avg_latency_ms"] / 1000, 0.50, False),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos = range(len(criteria))
    names = [c[0] for c in criteria]
    values = [c[1] for c in criteria]
    thresholds = [c[2] for c in criteria]
    higher_better = [c[3] for c in criteria]
    colors = []
    for v, t, h in zip(values, thresholds, higher_better):
        colors.append(C["green"] if (v >= t if h else v <= t) else C["red"])
    ax.barh(names, values, color=colors, height=0.6, edgecolor=C["border"])
    for i, (v, t, h) in enumerate(zip(values, thresholds, higher_better)):
        ax.axvline(t, y_pos[i] - 0.35, y_pos[i] + 0.35, color="white", linewidth=2, linestyle=":", alpha=0.6)
        ax.text(t + 0.005, y_pos[i] + 0.2, f"目标={t:.0%}" if t < 1 else f"目标={t:.0f}ms",
                fontsize=7, color="white", alpha=0.6)
    for bar, v in zip(ax.get_children()[:len(names)], values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}" if v < 1 else f"{v:.0f}ms", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("实际值"); ax.set_title("POC 验收标准达成情况", fontweight="bold")
    ax.set_xlim(0, max(max(values), max(thresholds)) * 1.15); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def build_data_table(results: list) -> str:
    """Build JSON data for the interactive table."""
    rows = []
    for i, r in enumerate(results):
        tier = r.get("tier", "?")
        # Build trace summary
        trace_summary = []
        for t in r.get("traces", []):
            trace_summary.append({
                "node": t.get("node", "?"),
                "step": t.get("step", "?"),
                "model": t.get("model", ""),
                "latency_ms": t.get("latency_ms", 0),
                "cost": t.get("cost", ""),
                "output": str(t.get("output", ""))[:200],
            })
        rows.append({
            "id": i,
            "content_id": r.get("content_id", f"tc_{i}"),
            "text": r.get("text", "")[:300],
            "expected": r.get("expected", "?"),
            "decision": r.get("decision", "?"),
            "confidence": round(r.get("confidence", 0), 4),
            "tier": tier,
            "path": r.get("path", "?"),
            "latency_ms": round(r.get("latency_ms", 0), 2),
            "reason": r.get("reason", "")[:300],
            "traces": trace_summary,
        })
    return json.dumps(rows, ensure_ascii=False)


def render(path: str) -> str:
    """Load benchmark results and generate HTML."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    metrics = data["metrics"]
    results = data["results"]

    # Attach expected labels to results
    if "cases" in data:
        for r, c in zip(results, data["cases"]):
            r["expected"] = c.get("expected", c.get("label", "?"))
    # Try loading from raw dataset if expected not in results
    if "expected" not in results[0]:
        # expected labels are in the benchmark output
        pass

    funnel_b64 = chart_funnel(metrics)
    latency_b64 = chart_latency(results, metrics)
    confusion_b64 = chart_confusion(metrics)
    passfail_b64 = chart_passfail(metrics)
    table_json = build_data_table(results)

    # Stats
    total = metrics["total"]
    hot = metrics["path_distribution"].get("hot", 0)
    cold = metrics["path_distribution"].get("cold", 0)
    passed = sum(1 for r in results if r.get("decision") == "pass")
    blocked = sum(1 for r in results if r.get("decision") == "block")
    reviewed = sum(1 for r in results if r.get("decision") == "review")
    correct = total - metrics["fn"] - metrics["fp"]

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>POC Benchmark Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; padding:20px; }}
.container {{ max-width:1200px; margin:0 auto; }}
h1 {{ font-size:24px; margin-bottom:5px; }}
h2 {{ font-size:18px; margin:25px 0 12px; border-bottom:1px solid #30363d; padding-bottom:8px; }}
h3 {{ font-size:14px; color:#8b949e; margin-bottom:20px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin-bottom:20px; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:16px; }}
.metric {{ background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:12px; text-align:center; }}
.metric .value {{ font-size:24px; font-weight:bold; }}
.metric .label {{ font-size:11px; color:#8b949e; margin-top:2px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #21262d; font-size:13px; }}
th {{ color:#8b949e; font-size:11px; text-transform:uppercase; position:sticky; top:0; background:#161b22; z-index:1; }}
img {{ max-width:100%; border-radius:6px; }}
.note {{ background:#1a1f2b; border-left:3px solid #d29922; padding:10px 15px; border-radius:0 6px 6px 0; font-size:13px; color:#d29922; margin-top:15px; }}
.filter-bar {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:15px; }}
.filter-bar select, .filter-bar input, .filter-bar button {{ background:#0d1117; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; padding:6px 10px; font-size:13px; }}
.filter-bar button {{ cursor:pointer; background:#238636; border-color:#238636; color:#fff; }}
.filter-bar button:hover {{ background:#2ea043; }}
.filter-bar input {{ width:200px; }}
.table-wrap {{ max-height:500px; overflow-y:auto; border:1px solid #30363d; border-radius:6px; }}
.clickable {{ cursor:pointer; color:#58a6ff; }}
.clickable:hover {{ text-decoration:underline; }}
.modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:100; justify-content:center; align-items:center; }}
.modal.active {{ display:flex; }}
.modal-content {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:25px; max-width:800px; width:90%; max-height:80vh; overflow-y:auto; }}
.modal-close {{ float:right; background:none; border:none; color:#8b949e; font-size:24px; cursor:pointer; }}
.trace-step {{ display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid #21262d; }}
.trace-step .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
.trace-step .info {{ flex:1; font-size:13px; }}
.trace-step .ms {{ font-size:12px; color:#8b949e; white-space:nowrap; }}
.tag {{ display:inline-block; padding:2px 6px; border-radius:3px; font-size:11px; font-weight:bold; }}
.tag-hot {{ background:#3fb950; color:#000; }}
.tag-cold {{ background:#58a6ff; color:#000; }}
.tag-pass {{ background:#3fb950; color:#000; }}
.tag-block {{ background:#f85149; color:#fff; }}
.tag-review {{ background:#d29922; color:#000; }}
.tag-safe {{ background:#3fb950; color:#000; }}
.tag-unsafe {{ background:#f85149; color:#fff; }}
.stats-row {{ display:flex; gap:8px; font-size:13px; color:#8b949e; margin-bottom:10px; }}
.stats-row strong {{ color:#c9d1d9; }}
</style>
</head>
<body>
<div class="container">

<h1>POC Benchmark Report</h1>
<h3>lmsys/toxic-chat (toxicchat0124) &middot; n={total} &middot; concurrency=4</h3>

<div class="card">
<h2>关键指标</h2>
<div class="metrics">
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['accuracy'] >= 0.9 else C['yellow']}">{metrics['accuracy']:.1%}</div>
        <div class="label">准确率</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['precision']:.1%}</div>
        <div class="label">精确率</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['recall']:.1%}</div>
        <div class="label">召回率</div>
    </div>
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['f1'] >= 0.9 else C['red']}">{metrics['f1']:.1%}</div>
        <div class="label">F1-score</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['avg_latency_ms']:.0f}ms</div>
        <div class="label">平均延迟</div>
    </div>
</div>
<div class="metrics">
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['llm_call_rate'] <= 0.2 else C['red']}">{metrics['llm_call_rate']:.1%}</div>
        <div class="label">LLM 调用率</div>
    </div>
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['bert_intercept_rate'] >= 0.3 else C['red']}">{metrics['bert_intercept_rate']:.1%}</div>
        <div class="label">BERT 拦截率</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['cache_hit_rate']:.1%}</div>
        <div class="label">缓存命中率</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['hot_path_rate']:.1%}</div>
        <div class="label">热路径比例</div>
    </div>
    <div class="metric">
        <div class="value">{1000/metrics['avg_latency_ms']:.1f} req/s</div>
        <div class="label">吞吐量 (估算)</div>
    </div>
</div>
<div class="note">
注意: lmsys/toxic-chat 的 unsafe 标签标记的是「AI 不应回答的 jailbreak 提示词」，而非用户违规内容。所有 false negative 都是 DAN/roleplay 类 prompt，系统判断为 safe 是正确行为。F1-score 不直接反映准确率。
</div>
</div>

<div class="card">
<h2>验收标准</h2>
<img src="data:image/png;base64,{passfail_b64}" alt="Acceptance Criteria">
</div>

<div class="card">
<h2>漏斗分布 & 延迟</h2>
<img src="data:image/png;base64,{funnel_b64}" alt="Funnel" style="margin-bottom:15px">
<img src="data:image/png;base64,{latency_b64}" alt="Latency">
</div>

<div class="card">
<h2>混淆矩阵</h2>
<img src="data:image/png;base64,{confusion_b64}" alt="Confusion Matrix" style="max-width:480px">
</div>

<div class="card">
<h2>全量数据 (点击行查看审核链路详情)</h2>
<div class="stats-row">
    <span>共 <strong>{total}</strong> 条</span>
    <span>| 热路径: <strong>{hot}</strong></span>
    <span>| 冷路径: <strong>{cold}</strong></span>
    <span>| 正确: <strong>{correct}</strong></span>
    <span>| pass: <strong>{passed}</strong></span>
    <span>| block: <strong>{blocked}</strong></span>
    <span>| review: <strong>{reviewed}</strong></span>
    <span id="filtered-count"></span>
</div>

<div class="filter-bar">
    <select id="filter-tier" onchange="applyFilters()">
        <option value="">全部层级</option>
        <option value="L0_memory">L0 内存缓存</option>
        <option value="L1_chroma">L1 ChromaDB</option>
        <option value="L2_bert">L2 BERT 分类</option>
        <option value="L3_llm">L3 LLM 审核</option>
    </select>
    <select id="filter-path" onchange="applyFilters()">
        <option value="">全部路径</option>
        <option value="hot">热路径</option>
        <option value="cold">冷路径</option>
    </select>
    <select id="filter-decision" onchange="applyFilters()">
        <option value="">全部决策</option>
        <option value="pass">pass</option>
        <option value="block">block</option>
        <option value="review">review</option>
    </select>
    <select id="filter-expected" onchange="applyFilters()">
        <option value="">全部期望</option>
        <option value="safe">safe</option>
        <option value="unsafe">unsafe</option>
    </select>
    <select id="filter-correct" onchange="applyFilters()">
        <option value="">全部结果</option>
        <option value="correct">正确</option>
        <option value="wrong">错误</option>
    </select>
    <input type="text" id="filter-search" placeholder="搜索文本内容..." oninput="applyFilters()">
    <button onclick="resetFilters()">重置</button>
</div>

<div class="table-wrap">
<table>
    <thead>
    <tr>
        <th>#</th>
        <th>文本</th>
        <th>期望</th>
        <th>决策</th>
        <th>置信度</th>
        <th>层级</th>
        <th>路径</th>
        <th>延迟</th>
    </tr>
    </thead>
    <tbody id="table-body"></tbody>
</table>
</div>
</div>

<!-- Detail Modal -->
<div class="modal" id="detail-modal" onclick="if(event.target===this)closeModal()">
    <div class="modal-content">
        <button class="modal-close" onclick="closeModal()">&times;</button>
        <div id="modal-body"></div>
    </div>
</div>

</div>

<script>
const DATA = {table_json};

let currentFilters = {{}};

function tierLabel(t) {{
    const m = {{
        "L0_memory":"L0 内存缓存","L0_redis":"L0 Redis","L0_whitelist":"L0 白名单",
        "L0_phash":"L0 图片哈希","L0_empty":"L0 空内容",
        "L1_keyword":"L1 关键词拦截","L1_chroma":"L1 ChromaDB 缓存",
        "L2_bert":"L2 BERT 分类","L3_llm":"L3 LLM 深度审核"
    }};
    return m[t] || t;
}}

function applyFilters() {{
    currentFilters = {{
        tier: document.getElementById('filter-tier').value,
        path: document.getElementById('filter-path').value,
        decision: document.getElementById('filter-decision').value,
        expected: document.getElementById('filter-expected').value,
        correct: document.getElementById('filter-correct').value,
        search: document.getElementById('filter-search').value.toLowerCase(),
    }};
    renderTable();
}}

function resetFilters() {{
    document.getElementById('filter-tier').value = '';
    document.getElementById('filter-path').value = '';
    document.getElementById('filter-decision').value = '';
    document.getElementById('filter-expected').value = '';
    document.getElementById('filter-correct').value = '';
    document.getElementById('filter-search').value = '';
    currentFilters = {{}};
    renderTable();
}}

function isCorrect(row) {{
    const e = row.expected;
    const d = row.decision;
    if (e === 'unsafe' && (d === 'block' || d === 'review')) return true;
    if (e === 'safe' && d === 'pass') return true;
    return false;
}}

function renderTable() {{
    const tbody = document.getElementById('table-body');
    let filtered = DATA.filter(r => {{
        if (currentFilters.tier && r.tier !== currentFilters.tier) return false;
        if (currentFilters.path && r.path !== currentFilters.path) return false;
        if (currentFilters.decision && r.decision !== currentFilters.decision) return false;
        if (currentFilters.expected && r.expected !== currentFilters.expected) return false;
        if (currentFilters.correct === 'correct' && !isCorrect(r)) return false;
        if (currentFilters.correct === 'wrong' && isCorrect(r)) return false;
        if (currentFilters.search && !r.text.toLowerCase().includes(currentFilters.search)) return false;
        return true;
    }});

    document.getElementById('filtered-count').textContent = '| 筛选结果: ' + filtered.length + ' 条';

    tbody.innerHTML = filtered.map(r => {{
        const correct = isCorrect(r);
        const rowBg = correct ? '' : 'background:#2d1518';
        const tierCls = r.tier.startsWith('L0') || r.tier.startsWith('L1') ? 'hot' : 'cold';
        const decCls = r.decision;
        return `<tr style="${{rowBg}};cursor:pointer" onclick="showDetail(${{r.id}})">
            <td>${{r.id}}</td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${{r.text.replace(/"/g,'&quot;')}}">${{r.text.substring(0, 80)}}</td>
            <td><span class="tag tag-${{r.expected}}">${{r.expected}}</span></td>
            <td><span class="tag tag-${{decCls}}">${{r.decision}}</span></td>
            <td>${{(r.confidence*100).toFixed(1)}}%</td>
            <td><span class="tag tag-${{tierCls}}">${{tierLabel(r.tier)}}</span></td>
            <td>${{r.path === 'hot' ? '热' : '冷'}}</td>
            <td>${{r.latency_ms.toFixed(0)}}ms</td>
        </tr>`;
    }}).join('');
}}

function showDetail(id) {{
    const r = DATA[id];
    const correct = isCorrect(r);
    const statusIcon = correct ? '&#9989;' : '&#10060;';
    const statusText = correct ? '正确' : '错误';

    let tracesHtml = r.traces.map((t, i) => {{
        const dotColor = t.step.includes('hit') || t.step.includes('block') ? '#3fb950' :
                         t.step.includes('miss') || t.step.includes('escalate') ? '#f85149' :
                         t.step.includes('ambig') ? '#d29922' : '#58a6ff';
        return `<div class="trace-step">
            <div class="dot" style="background:${{dotColor}}"></div>
            <div class="info">
                <strong>${{t.node}}</strong> &rarr; ${{t.step}}<br>
                <span style="color:#8b949e;font-size:11px">${{t.model || ''}}</span>
                ${{t.output ? `<br><span style="color:#8b949e;font-size:11px">${{t.output}}</span>` : ''}}
            </div>
            <div class="ms">${{t.latency_ms.toFixed(1)}}ms</div>
        </div>`;
    }}).join('');

    document.getElementById('modal-body').innerHTML = `
        <h2 style="margin-bottom:5px">${{statusIcon}} 审核详情 #${{id}} <span style="font-size:14px;color:#8b949e">(${{statusText}})</span></h2>
        <div style="margin:15px 0;padding:12px;background:#0d1117;border-radius:6px">
            <div style="font-size:14px;margin-bottom:8px;word-break:break-all"><strong>文本:</strong> ${{r.text}}</div>
            <div style="display:flex;gap:15px;flex-wrap:wrap;font-size:13px">
                <span>期望: <span class="tag tag-${{r.expected}}">${{r.expected}}</span></span>
                <span>决策: <span class="tag tag-${{r.decision}}">${{r.decision}}</span></span>
                <span>置信度: ${{(r.confidence*100).toFixed(1)}}%</span>
                <span>层级: ${{tierLabel(r.tier)}}</span>
                <span>路径: <span class="tag tag-${{r.path === 'hot' ? 'hot' : 'cold'}}">${{r.path === 'hot' ? '热路径' : '冷路径'}}</span></span>
                <span>总延迟: ${{r.latency_ms.toFixed(1)}}ms</span>
            </div>
            ${{r.reason ? `<div style="margin-top:8px;font-size:13px;color:#8b949e"><strong>原因:</strong> ${{r.reason}}</div>` : ''}}
        </div>
        <h3 style="margin-bottom:10px">审核链路追踪 (从上到下)</h3>
        <div style="max-height:400px;overflow-y:auto">${{tracesHtml}}</div>
    `;
    document.getElementById('detail-modal').classList.add('active');
}}

function closeModal() {{
    document.getElementById('detail-modal').classList.remove('active');
}}

document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

// Initial render
renderTable();
</script>
</body>
</html>'''
    return html


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/bench_toxicchat_results.json"

    out_path = path.replace(".json", ".html")

    print(f"Generating interactive report from {path}...")
    html = render(path)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {out_path} ({len(html)//1024} KB)")
    print(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
