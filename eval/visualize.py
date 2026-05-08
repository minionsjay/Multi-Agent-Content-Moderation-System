#!/usr/bin/env python3
"""Generate charts from benchmark results JSON."""

import json
import sys
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np

# Use CJK-capable font
for fname in ["Droid Sans Fallback", "AR PL SungtiL GB", "AR PL UMing TW MBE"]:
    try:
        fm.findfont(fname, fallback_to_default=False)
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue

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


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def chart_funnel(metrics: dict, outdir: str):
    """Horizontal bar: tier distribution with flow arrows."""
    tiers = metrics["tier_distribution"]
    total = metrics["total"]

    tier_order = [
        ("L0_memory", "L0 内存缓存", "#3fb950"),
        ("L0_redis", "L0 Redis", "#3fb950"),
        ("L0_whitelist", "L0 白名单", "#3fb950"),
        ("L1_keyword", "L1 关键词拦截", "#d29922"),
        ("L1_chroma", "L1 ChromaDB 缓存", "#d29922"),
        ("L2_bert", "L2 BERT 分类", "#58a6ff"),
        ("L3_llm", "L3 LLM 深度审核", "#f78166"),
    ]

    labels, values, colors, pcts = [], [], [], []
    for key, label, color in tier_order:
        v = tiers.get(key, 0)
        if v > 0:
            labels.append(label)
            values.append(v)
            colors.append(color)
            pcts.append(f"{v} ({v / total * 100:.1f}%)")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor="#30363d")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                pct, va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("请求数")
    ax.set_title(f"三层漏斗流量分布 (n={total})", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir}/funnel.png")


def chart_latency(metrics: dict, results: list, outdir: str):
    """Histogram of latencies with P50/P95/P99 markers."""
    lats = [r["latency_ms"] for r in results]
    p50 = metrics["latency_p50_ms"]
    p95 = metrics["latency_p95_ms"]
    p99 = metrics["latency_p99_ms"]
    avg = metrics["avg_latency_ms"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.logspace(np.log10(max(1, min(lats))), np.log10(max(lats)), 60)
    ax.hist(lats, bins=bins, color="#58a6ff", alpha=0.7, edgecolor="#30363d")

    for val, label, color in [
        (p50, f"P50={p50:.0f}ms", "#3fb950"),
        (p95, f"P95={p95:.0f}ms", "#d29922"),
        (p99, f"P99={p99:.0f}ms", "#f78166"),
    ]:
        ax.axvline(val, color=color, linewidth=2, linestyle="--")
        ax.text(val + max(lats) * 0.01, ax.get_ylim()[1] * 0.85,
                label, color=color, fontsize=10, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("延迟 (ms, log scale)")
    ax.set_ylabel("请求数")
    ax.set_title(f"请求延迟分布 (avg={avg:.0f}ms)", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir}/latency.png")


def chart_confusion(metrics: dict, outdir: str):
    """Confusion matrix as heatmap."""
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=metrics["total"] // 2)

    labels = [["TN\n(正确放行)", "FP\n(误拦截)"],
              ["FN\n(漏网)", "TP\n(正确拦截)"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > metrics["total"] // 4 else "#c9d1d9"
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["预测 Safe", "预测 Unsafe"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["实际 Safe", "实际 Unsafe"])
    ax.set_title("混淆矩阵", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{outdir}/confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir}/confusion.png")


def chart_acceptance(metrics: dict, outdir: str):
    """Acceptance criteria pass/fail gauge."""
    criteria = [
        ("LLM 调用率 < 20%", metrics["llm_call_rate"], 0.20, False),
        ("BERT 拦截率 > 30%", metrics["bert_intercept_rate"], 0.30, True),
        ("F1-score > 90%", metrics["f1"], 0.90, True),
        ("缓存命中率 > 15%", metrics["cache_hit_rate"], 0.15, True),
        ("平均延迟 < 500ms", metrics["avg_latency_ms"] / 1000, 0.50, False),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(criteria))
    names = [c[0] for c in criteria]
    values = [c[1] for c in criteria]
    thresholds = [c[2] for c in criteria]
    higher_better = [c[3] for c in criteria]

    colors = []
    for v, t, h in zip(values, thresholds, higher_better):
        passed = v >= t if h else v <= t
        colors.append("#3fb950" if passed else "#f85149")

    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor="#30363d")

    # Draw threshold lines
    for i, (v, t, h) in enumerate(zip(values, thresholds, higher_better)):
        # Map threshold to bar scale
        ax.axvline(t, y_pos[i] - 0.35, y_pos[i] + 0.35,
                   color="white", linewidth=2, linestyle=":", alpha=0.7)
        ax.text(t + 0.01, y_pos[i] + 0.2, f"目标 {t:.0%}" if t < 1 else f"目标 {t:.0f}ms",
                fontsize=8, color="white", alpha=0.7)

    for bar, v in zip(bars, values):
        label = f"{v:.1%}" if v < 1 else f"{v:.0f}ms"
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=12, fontweight="bold")

    ax.set_xlabel("实际值")
    ax.set_title("POC 验收标准达成情况", fontweight="bold")
    ax.set_xlim(0, max(max(values), max(thresholds)) * 1.2)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/acceptance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir}/acceptance.png")


def chart_overview(metrics: dict, outdir: str):
    """Combined overview: 4 charts in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # -- Top-left: Funnel pie --
    ax = axes[0, 0]
    tier_labels = {
        "L0_memory": "L0 缓存",
        "L2_bert": "L2 BERT",
        "L3_llm": "L3 LLM",
    }
    tier_colors = {"L0_memory": "#3fb950", "L2_bert": "#58a6ff", "L3_llm": "#f78166"}
    tiers = metrics["tier_distribution"]
    pie_labels, pie_values, pie_colors = [], [], []
    for k, v in sorted(tiers.items(), key=lambda x: -x[1]):
        if k in tier_labels:
            pie_labels.append(f"{tier_labels[k]}\n{v}条 ({v/metrics['total']*100:.1f}%)")
            pie_values.append(v)
            pie_colors.append(tier_colors.get(k, "#8b949e"))
    wedges, texts = ax.pie(pie_values, colors=pie_colors, startangle=90,
                           textprops={"fontsize": 10})
    ax.set_title("漏斗流量分布", fontweight="bold")

    # -- Top-right: Latency histogram --
    ax = axes[0, 1]
    # Latency data is in results, but we only have metrics here
    ax.text(0.5, 0.7, f"P50: {metrics['latency_p50_ms']:.0f}ms", transform=ax.transAxes,
            fontsize=28, ha="center", color="#3fb950", fontweight="bold")
    ax.text(0.5, 0.45, f"P95: {metrics['latency_p95_ms']:.0f}ms", transform=ax.transAxes,
            fontsize=18, ha="center", color="#d29922")
    ax.text(0.5, 0.25, f"P99: {metrics['latency_p99_ms']:.0f}ms  |  Avg: {metrics['avg_latency_ms']:.0f}ms",
            transform=ax.transAxes, fontsize=14, ha="center", color="#f78166")
    ax.set_title("请求延迟", fontweight="bold")
    ax.axis("off")

    # -- Bottom-left: Confusion matrix --
    ax = axes[1, 0]
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=metrics["total"] // 2, aspect="auto")
    cell_labels = [
        [f"TN\n{cm[0,0]} 正确放行", f"FP\n{cm[0,1]} 误拦截"],
        [f"FN\n{cm[1,0]} 漏网", f"TP\n{cm[1,1]} 正确拦截"],
    ]
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i, j] > metrics["total"] // 5 else "#c9d1d9"
            ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                    fontsize=13, fontweight="bold", color=c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测 Safe", "预测 Unsafe"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["实际 Safe", "实际 Unsafe"])
    ax.set_title("混淆矩阵", fontweight="bold")

    # -- Bottom-right: Key metrics table --
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"准确率: {metrics['accuracy']:.1%}",
        f"精确率: {metrics['precision']:.1%}",
        f"召回率: {metrics['recall']:.1%}",
        f"F1-score: {metrics['f1']:.1%}",
        "",
        f"吞吐量: {metrics['total']}/{metrics['avg_latency_ms']:.0f}ms ≈ {1000/metrics['avg_latency_ms']:.1f} req/s",
        f"热路径率: {metrics['hot_path_rate']:.1%}",
        f"LLM 调用率: {metrics['llm_call_rate']:.1%}",
        f"BERT 拦截率: {metrics['bert_intercept_rate']:.1%}",
    ]
    for i, line in enumerate(lines):
        color = "#c9d1d9"
        if "LLM" in line:
            passed = metrics["llm_call_rate"] <= 0.20
        elif "BERT" in line:
            passed = metrics["bert_intercept_rate"] >= 0.30
        elif "F1" in line:
            passed = metrics["f1"] >= 0.90
        else:
            passed = None
        if passed is True:
            color = "#3fb950"
        elif passed is False:
            color = "#f85149"
        ax.text(0.05, 0.95 - i * 0.09, line, transform=ax.transAxes,
                fontsize=13, color=color, fontfamily="monospace",
                fontweight="bold" if passed is not None else "normal")
    ax.set_title("关键指标", fontweight="bold")

    fig.suptitle("POC Benchmark — lmsys/toxic-chat (English)", fontsize=18,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{outdir}/overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outdir}/overview.png")


def main():
    if len(sys.argv) < 2:
        result_path = "data/bench_toxicchat_results.json"
    else:
        result_path = sys.argv[1]

    outdir = os.path.dirname(result_path) or "data"

    print(f"Loading {result_path}...")
    data = load_results(result_path)
    metrics = data["metrics"]
    results = data["results"]

    print("Generating charts...")
    chart_funnel(metrics, outdir)
    chart_latency(metrics, results, outdir)
    chart_confusion(metrics, outdir)
    chart_acceptance(metrics, outdir)
    chart_overview(metrics, outdir)

    print(f"\nDone! Charts saved to {outdir}/")


if __name__ == "__main__":
    main()
