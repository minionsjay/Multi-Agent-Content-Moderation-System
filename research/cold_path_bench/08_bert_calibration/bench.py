#!/usr/bin/env python3
"""Benchmark: BERT Confidence Calibration

Tests whether BERT's confidence scores are well-calibrated.
A well-calibrated model means:
  - When it says "95% confident", it's correct 95% of the time
  - The confidence score can be used as an actual probability

This matters because the L2→L3 short-circuit decision relies on
confidence ≥ 0.95 to skip LLM. If BERT is overconfident, we'll
skip LLM on cases where BERT is wrong.

Tests:
  1. Reliability diagram (actual accuracy vs predicted confidence)
  2. Expected Calibration Error (ECE)
  3. Overconfidence analysis (BERT vs LLM disagreement rate)

Usage:
  python bench.py
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.bert_classify import bert_classifier
from src.skills.llm_audit import llm_auditor
import asyncio


# Diverse test cases covering edge cases
CALIBRATION_CASES = [
    # Clear cases
    ("today is a beautiful sunny day perfect for a picnic", "safe"),
    ("i genuinely appreciate all the help you have given me", "safe"),
    ("you are a worthless piece of human garbage go die", "unsafe"),
    ("i will personally make sure you regret being born", "unsafe"),
    # Ambiguous / edge cases
    ("what the hell were you thinking when you did that", "unsafe"),
    ("you are being absolutely ridiculous right now", "unsafe"),
    ("i hate how this stupid thing never works properly", "safe"),
    ("this product is complete garbage don't buy it", "safe"),
    ("i'm going to lose my mind if this keeps happening", "safe"),
    ("seriously what is wrong with you today", "unsafe"),
    # Chinese edge cases
    ("你说的也不是没有道理，但态度确实有点问题", "safe"),
    ("这个人真的是无语了，怎么会有这种人", "safe"),
    ("你这样说话真的有点过分了知道吗", "unsafe"),
    ("虽然你说的有道理，但说话方式让人不舒服", "safe"),
    ("有些人真的就是欠骂，给脸不要脸", "unsafe"),
]


def bench_calibration():
    """Compute calibration metrics."""
    print("=" * 60)
    print("Test 1: Reliability Diagram")
    print("=" * 60)

    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
    bin_stats = {b: {"total": 0, "correct": 0, "confs": []} for b in bins}

    all_results = []
    for text, expected in CALIBRATION_CASES:
        result = bert_classifier.classify(text)
        conf = result["confidence"]
        predicted = result["label"]
        correct = predicted == expected

        all_results.append({
            "text": text[:50], "expected": expected,
            "predicted": predicted, "confidence": conf, "correct": correct,
        })

        for (lo, hi), stats in bin_stats.items():
            if lo <= conf < hi or (hi == 1.0 and conf >= lo):
                stats["total"] += 1
                if correct: stats["correct"] += 1
                stats["confs"].append(conf)
                break

    print(f"\n  {'Bin':<15} {'Count':>6} {'Accuracy':>10} {'AvgConf':>10} {'Gap':>10}")
    print(f"  {'─'*15} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    ece = 0.0
    total = len(all_results)
    for (lo, hi), stats in bin_stats.items():
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        avg_conf = sum(stats["confs"]) / len(stats["confs"])
        gap = avg_conf - acc
        weight = stats["total"] / total
        ece += weight * abs(gap)

        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  [{lo:.2f}-{hi:.2f})   {stats['total']:>6} {acc:>10.1%} {avg_conf:>10.3f} {gap:>+10.3f}  {bar}")

    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    if ece < 0.05:
        print(f"  ✓ Well-calibrated (ECE < 0.05)")
    elif ece < 0.10:
        print(f"  ~ Marginally calibrated (ECE 0.05-0.10)")
    else:
        print(f"  ✗ Poorly calibrated (ECE > 0.10)")

    # Overconfidence analysis
    high_conf_wrong = [r for r in all_results if r["confidence"] >= 0.90 and not r["correct"]]
    if high_conf_wrong:
        print(f"\n  High-confidence errors (conf≥0.90 but wrong): {len(high_conf_wrong)}")
        for r in high_conf_wrong:
            print(f"    conf={r['confidence']:.3f} expected={r['expected']} got={r['predicted']} | {r['text']}")
    else:
        print(f"\n  ✓ No high-confidence errors")

    return ece, all_results


async def bench_bert_vs_llm():
    """Compare BERT and LLM predictions on edge cases."""
    print(f"\n{'='*60}")
    print("Test 2: BERT vs LLM Disagreement Analysis")
    print(f"{'='*60}")

    edge_cases = CALIBRATION_CASES[-8:]  # last 8 are edge cases
    disagreements = 0
    bert_correct = 0
    llm_correct = 0

    for text, expected in edge_cases:
        bert = bert_classifier.classify(text)
        llm = await llm_auditor.audit(text)

        bert_ok = bert["label"] == expected
        llm_ok = llm["label"] == expected
        disagree = bert["label"] != llm["label"]

        if disagree: disagreements += 1
        if bert_ok: bert_correct += 1
        if llm_ok: llm_correct += 1

        icon = "≠" if disagree else "="
        print(f"  [{icon}] BERT={bert['label']:6s}({bert['confidence']:.3f}) "
              f"LLM={llm['label']:6s}({llm['confidence']:.3f}) "
              f"expected={expected:6s} | {text[:50]}")

    n = len(edge_cases)
    print(f"\n  BERT accuracy: {bert_correct}/{n}")
    print(f"  LLM accuracy:  {llm_correct}/{n}")
    print(f"  Disagreements: {disagreements}/{n} ({disagreements/n*100:.0f}%)")

    if disagreements > 0:
        print(f"  ⚠  When BERT and LLM disagree, current code trusts LLM.")
        print(f"     If LLM is wrong more often, this hurts accuracy.")


if __name__ == "__main__":
    ece, _ = bench_calibration()
    asyncio.run(bench_bert_vs_llm())
    print(f"\n✅ BERT calibration benchmark complete")
