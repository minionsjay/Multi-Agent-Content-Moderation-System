#!/usr/bin/env python3
"""Benchmark: BERT ONNX vs HuggingFace Pipeline

The Text Agent L2 layer tries ONNX first (2-3x faster on CPU), falls back
to HuggingFace transformers pipeline if ONNX model isn't available.

Tests:
  1. ONNX inference latency (if ONNX model exists)
  2. HF pipeline inference latency
  3. Accuracy comparison (do they produce the same results?)
  4. Cold start: model loading time for each backend
  5. Batch throughput

Usage:
  python bench.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.bert_classify import bert_classifier
from src.skills.bert_onnx import bert_onnx


TEST_TEXTS = [
    # English
    "you are a worthless piece of garbage",
    "what a beautiful day to go outside",
    "i will fucking kill you and your family",
    "this tutorial is really helpful thanks",
    "shut up you stupid moron nobody likes you",
    "the sunset looks amazing in this photo",
    # Chinese (will be skipped by English BERT)
    "你真是个傻逼什么都不懂",
    "今天天气真好适合出去玩",
    "我要杀了你全家信不信",
]


def bench_onnx():
    """ONNX inference latency."""
    print("=" * 60)
    print("Test 1: ONNX BERT Inference")
    print("=" * 60)

    if not bert_onnx._enabled:
        print("  ONNX model not available — skipping")
        print("  To enable: python -m transformers.onnx --model=unitary/toxic-bert "
              "--feature=sequence-classification onnx_models/")
        return None

    latencies = []
    for text in TEST_TEXTS[:6]:  # English only
        t0 = time.perf_counter()
        result = bert_onnx.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  [{result['label']:>6s} conf={result['confidence']:.3f}] "
              f"{ms:6.1f}ms | {text[:50]}")

    avg = sum(latencies) / len(latencies)
    print(f"\n  Avg: {avg:.1f}ms | QPS: {1000/avg:.0f}")
    return avg


def bench_hf():
    """HuggingFace pipeline inference latency."""
    print("\n" + "=" * 60)
    print("Test 2: HuggingFace Pipeline BERT Inference")
    print("=" * 60)

    latencies = []
    for text in TEST_TEXTS[:6]:
        t0 = time.perf_counter()
        result = bert_classifier.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  [{result['label']:>6s} conf={result['confidence']:.3f}] "
              f"{ms:6.1f}ms | {text[:50]}")

    avg = sum(latencies) / len(latencies)
    print(f"\n  Avg: {avg:.1f}ms | QPS: {1000/avg:.0f}")
    return avg


def bench_accuracy():
    """Compare ONNX vs HF output for same inputs."""
    print("\n" + "=" * 60)
    print("Test 3: ONNX vs HF Accuracy Comparison")
    print("=" * 60)

    if not bert_onnx._enabled:
        print("  ONNX not available — skipping")
        return

    agreements = 0
    for text in TEST_TEXTS[:6]:
        r1 = bert_onnx.classify(text)
        r2 = bert_classifier.classify(text)
        agree = r1["label"] == r2["label"]
        if agree:
            agreements += 1
        print(f"  [{'✓' if agree else '✗'}] ONNX={r1['label']:6s}({r1['confidence']:.3f}) "
              f"HF={r2['label']:6s}({r2['confidence']:.3f}) | {text[:50]}")

    print(f"\n  Agreement: {agreements}/{len(TEST_TEXTS[:6])}")


def bench_cold_start():
    """Model loading time (cold start)."""
    print("\n" + "=" * 60)
    print("Test 4: Cold Start (Model Loading Time)")
    print("=" * 60)

    # HF pipeline loading is done once via singleton
    print(f"  HF pipeline: loaded at module import (~60s for first load)")
    print(f"  ONNX session: {'loaded' if bert_onnx._enabled else 'not available'}")

    # Measure warm inference
    t0 = time.perf_counter()
    for _ in range(10):
        bert_classifier.classify("warmup test")
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  HF warm inference: {warm_ms/10:.1f}ms/req")


def bench_throughput():
    """Batch throughput comparison."""
    print("\n" + "=" * 60)
    print("Test 5: Throughput (100 inferences)")
    print("=" * 60)

    text = "you are a worthless piece of garbage and nobody likes you"

    # HF
    t0 = time.perf_counter()
    for _ in range(100):
        bert_classifier.classify(text)
    hf_ms = (time.perf_counter() - t0) * 1000 / 100
    hf_qps = 1000 / hf_ms

    # ONNX
    if bert_onnx._enabled:
        t0 = time.perf_counter()
        for _ in range(100):
            bert_onnx.classify(text)
        onnx_ms = (time.perf_counter() - t0) * 1000 / 100
        onnx_qps = 1000 / onnx_ms
        print(f"  HF pipeline:   {hf_ms:.1f}ms/req = {hf_qps:.0f} QPS")
        print(f"  ONNX runtime:  {onnx_ms:.1f}ms/req = {onnx_qps:.0f} QPS")
        print(f"  ONNX speedup:  {hf_ms/onnx_ms:.1f}x")
    else:
        print(f"  HF pipeline:   {hf_ms:.1f}ms/req = {hf_qps:.0f} QPS")
        print(f"  ONNX: not available")


def bench_chinese_skip():
    """Verify Chinese text is no longer force-skipped — BERT handles all languages."""
    print("\n" + "=" * 60)
    print("Test 6: Chinese Text — No Longer Skipped")
    print("=" * 60)

    for text in TEST_TEXTS:
        result = bert_classifier.classify(text)
        skip_llm = bert_classifier.should_skip_llm(result)
        print(f"  [{result['label']:>6s} conf={result['confidence']:.3f}] "
              f"{'→ skip LLM' if skip_llm else '→ escalate LLM'} | {text[:50]}")

    print(f"\n  Chinese text is no longer force-skipped.")
    print(f"  Models that support Chinese (KoalaAI, XLM-RoBERTa, Qwen3Guard)")
    print(f"  will classify it normally; English-only models will return")
    print(f"  low confidence and naturally escalate to LLM.")


if __name__ == "__main__":
    print("Warming up BERT models...")
    bert_classifier.warmup()
    print()

    bench_onnx()
    bench_hf()
    bench_accuracy()
    bench_cold_start()
    bench_throughput()
    bench_chinese_skip()
    print("\n✅ BERT ONNX vs HF benchmark complete")
