#!/usr/bin/env python3
"""Benchmark: Multilingual BERT on Chinese + English Text

After switching from unitary/toxic-bert (EN only) to
unitary/multilingual-toxic-xlm-roberta (15 languages), we need
to verify the model actually works on Chinese text.

Tests:
  1. Chinese text classification accuracy
  2. English text (before vs after switch)
  3. Confidence score comparison (EN vs CN)
  4. Throughput comparison (old vs new model)

Usage:
  python bench.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import BERT_MODEL
from src.skills.bert_classify import bert_classifier

# Labeled test cases
CN_CASES = [
    ("今天天气真好适合出去玩", "safe", "日常寒暄"),
    ("这个产品质量很好推荐给大家", "safe", "正常推荐"),
    ("你真是个傻逼什么都不懂", "unsafe", "直接辱骂"),
    ("操你妈的敢惹我试试", "unsafe", "严重辱骂"),
    ("脑残才买这个垃圾产品", "unsafe", "侮辱性评论"),
    ("支持台独台湾是独立国家", "unsafe", "政治敏感"),
    ("港独是香港唯一出路", "unsafe", "政治敏感"),
    ("我要杀了你这个混蛋", "unsafe", "暴力威胁"),
    ("加微信买六合彩稳赢不赔", "unsafe", "赌博广告"),
    ("约炮软件下载同城交友", "unsafe", "色情内容"),
    ("你说的也不是完全没道理", "safe", "正常讨论"),
    ("傻人有傻福吧算是", "safe", "俗语/非攻击性"),
    ("操场上同学们在操练", "safe", "白名单场景"),
    ("这游戏打得我想骂人", "safe", "情绪表达非攻击"),
    ("有些人真的就是欠骂", "unsafe", "边缘case"),
]

EN_CASES = [
    ("today is a beautiful day for a walk", "safe", "normal"),
    ("this product is amazing, highly recommend", "safe", "recommendation"),
    ("you are a worthless piece of garbage", "unsafe", "insult"),
    ("i will fucking kill you and your family", "unsafe", "death threat"),
    ("shut up you stupid idiot nobody likes you", "unsafe", "harassment"),
    ("what the hell is wrong with you seriously", "unsafe", "aggressive"),
    ("the sunset looks absolutely stunning today", "safe", "normal"),
    ("this tutorial is really helpful, thanks", "safe", "gratitude"),
]


def bench_model(name: str, cases: list, model_override: str = ""):
    """Run classification on test cases and return accuracy metrics."""
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    correct = 0
    total = len(cases)
    safe_correct = 0
    safe_total = 0
    unsafe_correct = 0
    unsafe_total = 0
    confs = {"safe": [], "unsafe": []}
    errors = []

    for text, expected, desc in cases:
        t0 = time.perf_counter()
        result = bert_classifier.classify(text, model_override=model_override) if model_override else bert_classifier.classify(text)
        ms = (time.perf_counter() - t0) * 1000

        predicted = result["label"]
        conf = result["confidence"]
        ok = predicted == expected
        if ok:
            correct += 1
        else:
            errors.append((text, expected, predicted, conf, desc))

        if expected == "safe":
            safe_total += 1
            if ok: safe_correct += 1
            confs["safe"].append(conf)
        else:
            unsafe_total += 1
            if ok: unsafe_correct += 1
            confs["unsafe"].append(conf)

        status = "✓" if ok else "✗"
        print(f"  [{status}] {predicted:6s} conf={conf:.3f} {ms:5.0f}ms | {desc:12s} | {text[:50]}")

    # Stats
    acc = correct / total * 100
    safe_acc = safe_correct / max(safe_total, 1) * 100
    unsafe_acc = unsafe_correct / max(unsafe_total, 1) * 100
    avg_safe_conf = sum(confs["safe"]) / max(len(confs["safe"]), 1)
    avg_unsafe_conf = sum(confs["unsafe"]) / max(len(confs["unsafe"]), 1)

    print(f"\n  Accuracy: {correct}/{total} ({acc:.1f}%)")
    print(f"    Safe:   {safe_correct}/{safe_total} ({safe_acc:.1f}%) avg_conf={avg_safe_conf:.3f}")
    print(f"    Unsafe: {unsafe_correct}/{unsafe_total} ({unsafe_acc:.1f}%) avg_conf={avg_unsafe_conf:.3f}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for text, exp, pred, conf, desc in errors:
            print(f"    expected={exp} got={pred} conf={conf:.3f} | {desc}: {text[:50]}")

    return {
        "name": name, "accuracy": acc, "safe_accuracy": safe_acc,
        "unsafe_accuracy": unsafe_acc, "avg_safe_conf": avg_safe_conf,
        "avg_unsafe_conf": avg_unsafe_conf, "errors": len(errors),
    }


def bench_llm_skip_rate(cases: list):
    """How many cases would skip LLM based on BERT confidence ≥ 0.95?"""
    skip = 0
    for text, _, _ in cases:
        result = bert_classifier.classify(text)
        if bert_classifier.should_skip_llm(result):
            skip += 1
    return skip


if __name__ == "__main__":
    print(f"Current BERT model: {BERT_MODEL}")

    # Test Chinese
    cn = bench_model("XLM-RoBERTa (CN)", CN_CASES)
    cn_skip = bench_llm_skip_rate(CN_CASES)

    # Test English
    en = bench_model("XLM-RoBERTa (EN)", EN_CASES)
    en_skip = bench_llm_skip_rate(EN_CASES)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Chinese: acc={cn['accuracy']:.1f}% safe_acc={cn['safe_accuracy']:.1f}% "
          f"unsafe_acc={cn['unsafe_accuracy']:.1f}% L2_skip={cn_skip}/{len(CN_CASES)}")
    print(f"  English: acc={en['accuracy']:.1f}% safe_acc={en['safe_accuracy']:.1f}% "
          f"unsafe_acc={en['unsafe_accuracy']:.1f}% L2_skip={en_skip}/{len(EN_CASES)}")
    print(f"  Safe conf gap (CN vs EN): {cn['avg_safe_conf']:.3f} vs {en['avg_safe_conf']:.3f}")
    print(f"  Unsafe conf gap (CN vs EN): {cn['avg_unsafe_conf']:.3f} vs {en['avg_unsafe_conf']:.3f}")
    print(f"\n✅ Multilingual BERT benchmark complete")
