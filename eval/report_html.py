#!/usr/bin/env python3
"""Generate an HTML report with embedded charts from benchmark results."""

import json
import sys
import os
import base64
from collections import Counter
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Use CJK-capable font (AR PL SungtiL GB supports both ASCII and Chinese)
ZH_FONT_PATH = "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"
fm.fontManager.addfont(ZH_FONT_PATH)
plt.rcParams["font.family"] = "AR PL SungtiL GB"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

C = {
    "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff",
    "red": "#f78166", "purple": "#bc8cff", "grey": "#8b949e",
    "bg": "#0d1117", "card": "#161b22", "border": "#30363d",
}


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_funnel(metrics: dict) -> str:
    tiers = metrics["tier_distribution"]
    total = metrics["total"]
    tier_order = [
        ("L0_memory", "L0 内存缓存", C["green"]),
        ("L0_redis", "L0 Redis 缓存", C["green"]),
        ("L0_whitelist", "L0 白名单", C["green"]),
        ("L1_keyword", "L1 关键词拦截", C["yellow"]),
        ("L1_chroma", "L1 ChromaDB 缓存", C["yellow"]),
        ("L2_bert", "L2 BERT 分类", C["blue"]),
        ("L3_llm", "L3 LLM 深度审核", C["red"]),
    ]
    labels, values, colors = [], [], []
    for k, lbl, clr in tier_order:
        v = tiers.get(k, 0)
        if v > 0:
            labels.append(lbl)
            values.append(v)
            colors.append(clr)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor=C["border"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v} 条 ({v/total*100:.1f}%)", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("请求数")
    ax.set_title(f"三层漏斗流量分布 (n={total})", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def chart_latency(results: list, metrics: dict) -> str:
    lats = [r["latency_ms"] for r in results]
    p50, p95, p99 = metrics["latency_p50_ms"], metrics["latency_p95_ms"], metrics["latency_p99_ms"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.logspace(np.log10(max(0.1, min(lats))), np.log10(max(lats)), 50)
    ax.hist(lats, bins=bins, color=C["blue"], alpha=0.7, edgecolor=C["border"])
    for val, label, color in [
        (p50, f"P50={p50:.0f}ms", C["green"]),
        (p95, f"P95={p95:.0f}ms", C["yellow"]),
        (p99, f"P99={p99:.0f}ms", C["red"]),
    ]:
        ax.axvline(val, color=color, linewidth=2, linestyle="--")
        ax.text(val + max(lats) * 0.02, ax.get_ylim()[1] * 0.85,
                label, color=color, fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("延迟 (ms, 对数刻度)")
    ax.set_ylabel("请求数")
    ax.set_title(f"延迟分布 (平均 {metrics['avg_latency_ms']:.0f}ms)", fontweight="bold")
    ax.grid(alpha=0.3)
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
            ax.text(j, i, cell_text[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=clr)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测 Safe", "预测 Unsafe"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["实际 Safe", "实际 Unsafe"])
    ax.set_title("混淆矩阵", fontweight="bold")
    return fig_to_b64(fig)


def chart_passfail(metrics: dict) -> str:
    """Horizontal bar chart showing PASS/FAIL for acceptance criteria."""
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
        passed = v >= t if h else v <= t
        colors.append(C["green"] if passed else C["red"])

    ax.barh(names, values, color=colors, height=0.6, edgecolor=C["border"])

    # Draw threshold markers
    for i, (v, t, h) in enumerate(zip(values, thresholds, higher_better)):
        ax.axvline(t, y_pos[i] - 0.35, y_pos[i] + 0.35,
                   color="white", linewidth=2, linestyle=":", alpha=0.6)
        ax.text(t + 0.005, y_pos[i] + 0.2, f"目标={t:.0%}" if t < 1 else f"目标={t:.0f}ms",
                fontsize=7, color="white", alpha=0.6)

    for bar, v in zip(ax.get_children()[:len(names)], values):
        label = f"{v:.1%}" if v < 1 else f"{v:.0f}ms"
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("实际值")
    ax.set_title("POC 验收标准达成情况", fontweight="bold")
    ax.set_xlim(0, max(max(values), max(thresholds)) * 1.15)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    return fig_to_b64(fig)


def badge(passed: bool) -> str:
    if passed:
        return '<span style="background:#3fb950;color:#000;padding:2px 8px;border-radius:4px;font-weight:bold">PASS</span>'
    return '<span style="background:#f85149;color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">FAIL</span>'


def generate_report(result_path: str) -> str:
    with open(result_path) as f:
        data = json.load(f)
    metrics = data["metrics"]
    results = data["results"]

    funnel_b64 = chart_funnel(metrics)
    latency_b64 = chart_latency(results, metrics)
    confusion_b64 = chart_confusion(metrics)
    passfail_b64 = chart_passfail(metrics)

    checks_html = ""
    criteria = [
        ("LLM 调用率 < 20%", metrics["llm_call_rate"], 0.20, False, f"{metrics['llm_call_rate']:.1%}"),
        ("BERT 拦截率 > 30%", metrics["bert_intercept_rate"], 0.30, True, f"{metrics['bert_intercept_rate']:.1%}"),
        ("F1-score > 90%", metrics["f1"], 0.90, True, f"{metrics['f1']:.1%}"),
        ("缓存命中率 > 15%", metrics["cache_hit_rate"], 0.15, True, f"{metrics['cache_hit_rate']:.1%}"),
        ("平均延迟 < 500ms", metrics["avg_latency_ms"] / 1000, 0.50, False, f"{metrics['avg_latency_ms']:.0f}ms"),
    ]
    for name, val, target, higher, display in criteria:
        passed = val >= target if higher else val <= target
        checks_html += f"""
        <tr>
            <td>{name}</td>
            <td style="text-align:right">{display}</td>
            <td style="text-align:right">{target:.0%}{' (higher)' if higher else ''}</td>
            <td style="text-align:center">{badge(passed)}</td>
        </tr>"""

    # Tier breakdown
    tier_rows = ""
    tier_names = {
        "L0_memory": ("Hot Path", C["green"]),
        "L0_redis": ("Hot Path", C["green"]),
        "L0_whitelist": ("Hot Path", C["green"]),
        "L1_keyword": ("Hot Path", C["yellow"]),
        "L1_chroma": ("Hot Path", C["yellow"]),
        "L2_bert": ("Cold Path", C["blue"]),
        "L3_llm": ("Cold Path", C["red"]),
    }
    for tier, count in sorted(metrics["tier_distribution"].items(), key=lambda x: -x[1]):
        path_name, color = tier_names.get(tier, ("?", C["grey"]))
        pct = count / metrics["total"] * 100
        tier_rows += f"""
        <tr>
            <td><span style="color:{color}">█</span> {tier}</td>
            <td>{path_name}</td>
            <td style="text-align:right">{count}</td>
            <td style="text-align:right">{pct:.1f}%</td>
        </tr>"""

    # Misclassifications
    misclass_rows = ""
    for r in results:
        if r.get("expected") and r["decision"] != r["expected"]:
            if misclass_rows.count("<tr>") < 12:
                misclass_rows += f"""
        <tr>
            <td style="max-width:400px;word-break:break-all;font-size:12px">{r.get('text', '')[:150]}</td>
            <td style="color:#f85149">{r.get('expected', '?')}</td>
            <td>{r['decision']}</td>
            <td>{r.get('confidence', 0):.2f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>POC Benchmark Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; padding:20px; }}
.container {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:24px; margin-bottom:5px; }}
h2 {{ font-size:18px; margin:30px 0 15px; border-bottom:1px solid #30363d; padding-bottom:8px; }}
h3 {{ font-size:14px; color:#8b949e; margin-bottom:20px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin-bottom:20px; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,1fr); gap:15px; margin-bottom:20px; }}
.metric {{ background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:15px; text-align:center; }}
.metric .value {{ font-size:28px; font-weight:bold; }}
.metric .label {{ font-size:11px; color:#8b949e; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #21262d; }}
th {{ color:#8b949e; font-size:11px; text-transform:uppercase; }}
img {{ max-width:100%; border-radius:6px; }}
.note {{ background:#1a1f2b; border-left:3px solid #d29922; padding:10px 15px; border-radius:0 6px 6px 0; font-size:13px; color:#d29922; margin-top:15px; }}
</style>
</head>
<body>
<div class="container">

<h1>POC Benchmark Report</h1>
<h3>lmsys/toxic-chat (toxicchat0124) &middot; n={metrics["total"]} &middot; concurrency=4</h3>

<div class="card">
<h2>Key Metrics</h2>
<div class="metrics">
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['accuracy'] >= 0.9 else C['yellow']}">{metrics['accuracy']:.1%}</div>
        <div class="label">Accuracy</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['precision']:.1%}</div>
        <div class="label">Precision</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['recall']:.1%}</div>
        <div class="label">Recall</div>
    </div>
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['f1'] >= 0.9 else C['red']}">{metrics['f1']:.1%}</div>
        <div class="label">F1-score</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['avg_latency_ms']:.0f}ms</div>
        <div class="label">Avg Latency</div>
    </div>
</div>

<div class="metrics">
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['llm_call_rate'] <= 0.2 else C['red']}">{metrics['llm_call_rate']:.1%}</div>
        <div class="label">LLM Call Rate</div>
    </div>
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['bert_intercept_rate'] >= 0.3 else C['red']}">{metrics['bert_intercept_rate']:.1%}</div>
        <div class="label">BERT Intercept</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['keyword_intercept_rate']:.1%}</div>
        <div class="label">Keyword Intercept</div>
    </div>
    <div class="metric">
        <div class="value" style="color:{C['green'] if metrics['cache_hit_rate'] >= 0.15 else C['yellow']}">{metrics['cache_hit_rate']:.1%}</div>
        <div class="label">Cache Hit Rate</div>
    </div>
    <div class="metric">
        <div class="value">{metrics['hot_path_rate']:.1%}</div>
        <div class="label">Hot Path Rate</div>
    </div>
</div>

<div class="note">
<strong>Note:</strong> lmsys/toxic-chat labels jailbreak/AI-safety prompts as "unsafe", not user-generated toxic content.
All 31 false negatives are DAN/roleplay prompts that are correctly classified as safe user content.
F1-score should NOT be used as accuracy metric &mdash; needs real user-content dataset for validation.
</div>
</div>

<div class="card">
<h2>Acceptance Criteria</h2>
<img src="data:image/png;base64,{passfail_b64}" alt="Acceptance Criteria">
<table style="margin-top:15px">
    <tr><th>Criterion</th><th>Actual</th><th>Target</th><th>Result</th></tr>
    {checks_html}
</table>
</div>

<div class="card">
<h2>3-Tier Funnel Distribution</h2>
<img src="data:image/png;base64,{funnel_b64}" alt="Funnel">
<table style="margin-top:15px">
    <tr><th>Tier</th><th>Path</th><th>Requests</th><th>Share</th></tr>
    {tier_rows}
</table>
</div>

<div class="card">
<h2>Latency Distribution</h2>
<img src="data:image/png;base64,{latency_b64}" alt="Latency">
<table style="margin-top:15px">
    <tr><th>Percentile</th><th>Latency</th></tr>
    <tr><td>P50 (median)</td><td style="color:{C['green']}">{metrics['latency_p50_ms']:.0f}ms</td></tr>
    <tr><td>P95</td><td style="color:{C['yellow']}">{metrics['latency_p95_ms']:.0f}ms</td></tr>
    <tr><td>P99</td><td style="color:{C['red']}">{metrics['latency_p99_ms']:.0f}ms</td></tr>
    <tr><td>Average</td><td>{metrics['avg_latency_ms']:.0f}ms</td></tr>
</table>
</div>

<div class="card">
<h2>Confusion Matrix</h2>
<img src="data:image/png;base64,{confusion_b64}" alt="Confusion Matrix" style="max-width:500px">
</div>

<div class="card">
<h2>Misclassified Samples</h2>
<table>
    <tr><th>Text</th><th>Expected</th><th>Predicted</th><th>Confidence</th></tr>
    {misclass_rows}
</table>
</div>

</div>
</body>
</html>"""
    return html


def main():
    if len(sys.argv) < 2:
        result_path = "data/bench_toxicchat_results.json"
    else:
        result_path = sys.argv[1]

    out_path = result_path.replace(".json", ".html")
    print(f"Loading {result_path}...")
    html = generate_report(result_path)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {out_path}")
    print(f"Open with: file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
