#!/usr/bin/env python3
"""Benchmark: EasyOCR Text Extraction from Images

Image Agent Step 3: Extract embedded text from images using EasyOCR
(ch_sim + en). This is critical for catching text-in-image violations
(e.g., "加微信" watermarked on an otherwise normal image).

Tests:
  1. Model loading time (cold start)
  2. Extraction latency (warm)
  3. Accuracy on synthetic text images
  4. Chinese + English mixed text
  5. No-text images (should return empty)
  6. Edge cases: rotated, blurry, low contrast

Usage:
  python bench.py
"""

import sys, os, time, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.image_ocr import image_ocr
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


def make_text_image(text: str, size=(400, 200), font_size=20) -> bytes:
    """Create an image with embedded text."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Try Chinese-capable fonts in order
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Chinese
        "/usr/share/fonts/truetype/arphic/uming.ttc",                 # Chinese
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",            # English fallback
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, size[1] // 2 - font_size), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_noise_image(size=(400, 200)) -> bytes:
    """Create a no-text control image."""
    random.seed(42)
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 20):
        for y in range(0, size[1], 20):
            c = random.randint(0, 255)
            draw.rectangle([x, y, x+18, y+18], fill=(c, c//2, c//3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bench_cold_start():
    """Measure EasyOCR model loading time."""
    print("=" * 60)
    print("Test 1: EasyOCR Cold Start (Model Loading)")
    print("=" * 60)

    # Generate a simple test image
    img_bytes = make_text_image("测试")

    t0 = time.perf_counter()
    result = image_ocr.extract(img_bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result.get("error"):
        print(f"  EasyOCR not loaded: {result['error']}")
        print(f"  Attempt took: {elapsed_ms:.0f}ms")
        print(f"  Download: ~450MB (ch_sim + en models)")
        print(f"  Install: pip install easyocr")
        return False

    print(f"  First extraction: {elapsed_ms:.0f}ms (includes model loading)")
    print(f"  Result: '{result.get('text', '')}' (conf={result.get('confidence', 0):.2f})")
    return True


def bench_warm_latency():
    """Measure warm extraction latency for various text lengths."""
    print("\n" + "=" * 60)
    print("Test 2: Warm Extraction Latency")
    print("=" * 60)

    test_cases = [
        ("short (3 chars)", "违规"),
        ("medium (10 chars)", "这是一个测试文本"),
        ("long (30 chars)", "这是比较长的测试文本内容用于评估OCR的识别速度和准确性"),
        ("English only", "This is a test sentence for OCR"),
        ("Mixed CN+EN", "加微信WeChat看更多内容"),
    ]

    for label, text in test_cases:
        img_bytes = make_text_image(text)
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            result = image_ocr.extract(img_bytes)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = sum(latencies) / len(latencies)
        extracted = result.get("text", "")
        accuracy = _text_match_ratio(text, extracted)
        print(f"  {label:20s}: {avg:6.0f}ms avg | extracted='{extracted[:40]}' "
              f"match={accuracy:.0%} conf={result.get('confidence', 0):.2f}")


def bench_no_text():
    """Verify OCR correctly returns empty for no-text images."""
    print("\n" + "=" * 60)
    print("Test 3: No-Text Images (should return empty)")
    print("=" * 60)

    img_bytes = make_noise_image()

    t0 = time.perf_counter()
    result = image_ocr.extract(img_bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    extracted = result.get("text", "")
    blocks = len(result.get("blocks", []))

    print(f"  No-text image: '{extracted}' in {elapsed_ms:.0f}ms ({blocks} text blocks)")
    if extracted.strip():
        print(f"  ⚠ OCR found text in no-text image — possible false positive")
    else:
        print(f"  ✓ Correctly returned empty")


def bench_robustness():
    """Test OCR robustness: blur, rotation, contrast."""
    print("\n" + "=" * 60)
    print("Test 4: Robustness to Image Degradation")
    print("=" * 60)

    base_text = "违规内容检测"
    base_img = Image.open(io.BytesIO(make_text_image(base_text)))

    variants = [
        ("original", base_img),
        ("blur 2px", base_img.filter(ImageFilter.GaussianBlur(radius=2))),
        ("blur 4px", base_img.filter(ImageFilter.GaussianBlur(radius=4))),
        ("low contrast", Image.fromarray(
            (__import__('numpy').array(base_img) * 0.5).astype('uint8'))),
    ]

    for label, img in variants:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        t0 = time.perf_counter()
        result = image_ocr.extract(img_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        extracted = result.get("text", "")
        match = _text_match_ratio(base_text, extracted)
        print(f"  {label:15s}: {elapsed_ms:5.0f}ms | '{extracted[:30]}' match={match:.0%}")


def _text_match_ratio(expected: str, actual: str) -> float:
    """Simple character-level match ratio."""
    if not expected or not actual:
        return 0.0 if expected else 1.0
    expected_clean = expected.replace(" ", "").lower()
    actual_clean = actual.replace(" ", "").lower()
    matches = sum(1 for c in expected_clean if c in actual_clean)
    return matches / len(expected_clean) if expected_clean else 1.0


if __name__ == "__main__":
    print("EasyOCR Benchmark\n")

    loaded = bench_cold_start()
    if loaded:
        bench_warm_latency()
        bench_no_text()
        bench_robustness()

    print("\n✅ EasyOCR benchmark complete")
