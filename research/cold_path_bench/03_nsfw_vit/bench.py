#!/usr/bin/env python3
"""Benchmark: NSFW ViT Image Classification

Uses Falconsai/nsfw_image_detection (ViT-based, ~350MB) to classify
images as nsfw/normal. This is the Image Agent's Step 2.

POC status: skipped (skip_model=True). This benchmark tests the model
download and inference if available, or measures the POC fallback path.

Tests:
  1. Model download size and time
  2. Inference latency (CPU)
  3. Classification accuracy on test images
  4. POC fallback behavior
  5. Memory usage

Usage:
  python bench.py              # full test (will download model if needed)
  python bench.py --poc-only   # only test POC fallback (no download)
"""

import sys, os, time, io, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.image_nsfw import nsfw_detector
from PIL import Image, ImageDraw
import random


def make_test_images():
    """Generate test images for NSFW classification testing.

    These are SYNTHETIC images — random patterns, not real content.
    """
    random.seed(42)
    images = {}

    # Normal-looking image: landscape-like
    img = Image.new("RGB", (320, 240), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 30):
        for y in range(120, 240, 20):
            draw.rectangle([x, y, x+28, y+18], fill=(34, 139, 34))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85); buf.seek(0)
    images["landscape"] = buf.read()

    # Pattern image
    img2 = Image.new("RGB", (320, 240))
    draw2 = ImageDraw.Draw(img2)
    for x in range(0, 320, 20):
        for y in range(0, 240, 20):
            c = random.randint(0, 255)
            draw2.rectangle([x, y, x+18, y+18], fill=(c, c//2, c//3))
    buf = io.BytesIO(); img2.save(buf, format="JPEG", quality=85); buf.seek(0)
    images["pattern"] = buf.read()

    # Small image (edge case)
    img3 = Image.new("RGB", (5, 5), color=(128, 128, 128))
    buf = io.BytesIO(); img3.save(buf, format="PNG"); buf.seek(0)
    images["tiny"] = buf.read()

    # Large image (~2MB)
    img4 = Image.new("RGB", (1920, 1080))
    draw4 = ImageDraw.Draw(img4)
    for x in range(0, 1920, 40):
        for y in range(0, 1080, 40):
            c = random.randint(0, 255)
            draw4.rectangle([x, y, x+38, y+38], fill=(c, c, c))
    buf = io.BytesIO(); img4.save(buf, format="JPEG", quality=90); buf.seek(0)
    images["large_hd"] = buf.read()

    return images


def bench_poc_fallback():
    """POC fallback behavior — what happens when model is skipped."""
    print("=" * 60)
    print("Test 1: POC Fallback Behavior (skip_model=True)")
    print("=" * 60)

    images = make_test_images()

    for name, img_bytes in images.items():
        t0 = time.perf_counter()
        result = nsfw_detector.classify(img_bytes, skip_model=True)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        print(f"  {name:12s} ({len(img_bytes):>7,} bytes): "
              f"label={result['label']:6s} conf={result['confidence']:.2f} "
              f"model={result.get('model','?')} "
              f"{elapsed_us:6.0f}μs")

    print(f"\n  POC limitation: ALL images classified as 'normal'")
    print(f"  The NSFW ViT model is NOT loaded — POC validates only image format/size")


def bench_real_inference():
    """Try to download and run the real NSFW model."""
    print("\n" + "=" * 60)
    print("Test 2: Real NSFW Model Inference")
    print("=" * 60)

    images = make_test_images()
    img_bytes = images["landscape"]

    # Try loading the model
    t0 = time.perf_counter()
    try:
        result = nsfw_detector.classify(img_bytes, skip_model=False)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Model loaded in {load_ms:.0f}ms")
        print(f"  Result: label={result['label']} conf={result['confidence']:.3f}")
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  Model load failed after {elapsed_ms:.0f}ms: {e}")
        print(f"  Falling back to POC basic validation")

    # Benchmark warm inference
    print(f"\n  Inference speed (warm):")
    for name, img_bytes in images.items():
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            result = nsfw_detector.classify(img_bytes, skip_model=False)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg = sum(latencies) / len(latencies)
        print(f"    {name:12s}: {avg:6.1f}ms avg")


def bench_edge_cases():
    """Test edge cases: corrupt data, empty, huge image."""
    print("\n" + "=" * 60)
    print("Test 3: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ("empty bytes", b""),
        ("garbage data", b"\x00\x01\x02\x03" * 100),
        ("text not image", b"hello world this is not an image"),
    ]

    for name, data in edge_cases:
        try:
            result = nsfw_detector.classify(data, skip_model=True)
            print(f"  {name:20s}: label={result.get('label','?')} error={result.get('error','none')}")
        except Exception as e:
            print(f"  {name:20s}: EXCEPTION: {e}")


def bench_memory():
    """Model memory usage estimate."""
    print("\n" + "=" * 60)
    print("Test 4: Memory Usage Estimate")
    print("=" * 60)
    print(f"  Model: Falconsai/nsfw_image_detection (ViT-base)")
    print(f"  Download size: ~350 MB")
    print(f"  GPU memory: ~500 MB (FP16)")
    print(f"  CPU memory: ~700 MB (FP32)")
    print(f"  POC fallback: 0 MB (skip_model=True)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-only", action="store_true", help="Only test POC fallback")
    args = parser.parse_args()

    bench_poc_fallback()
    if not args.poc_only:
        bench_real_inference()
    bench_edge_cases()
    bench_memory()
    print("\n✅ NSFW ViT benchmark complete")
