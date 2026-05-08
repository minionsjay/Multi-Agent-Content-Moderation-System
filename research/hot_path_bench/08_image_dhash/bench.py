#!/usr/bin/env python3
"""Benchmark: dHash Perceptual Image Hashing

dHash (difference hash) is the core image hot-path technology. It generates
a 64-bit fingerprint of an image that is robust to resizing, mild compression,
and slight color shifts. Critical for known CSAM/illegal imagery detection.

Tests:
  1. Computation speed (QPS)
  2. Robustness: resize, brightness, watermark
  3. Hamming distance distribution (similar vs different images)
  4. Known harmful hash DB matching
  5. Hash uniqueness across random patterns

Usage:
  python bench.py
"""

import sys, os, time, io, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from PIL import Image, ImageDraw
from src.skills.image_phash import image_phash


def make_pattern_image(seed: int, size=(320, 240)) -> Image.Image:
    """Generate a textured image from a seed. NOT real content."""
    random.seed(seed)
    img = Image.new("RGB", size, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 15 + seed % 20):
        for y in range(0, size[1], 15 + seed % 20):
            c = random.randint(0, 255)
            if seed % 3 == 0:
                draw.rectangle([x, y, x+12, y+12], fill=(c, c//2, c//3))
            elif seed % 3 == 1:
                draw.ellipse([x, y, x+12, y+12], fill=(c//3, c, c//2))
            else:
                draw.line([x, y, x+12, y+12], fill=(c, c//2, c//3), width=2)
    return img


def to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bench_speed():
    """dHash computation speed at different image sizes."""
    print("=" * 60)
    print("Test 1: dHash Computation Speed")
    print("=" * 60)

    for size, label in [((320, 240), "320×240"), ((640, 480), "640×480"), ((1280, 720), "1280×720")]:
        img = make_pattern_image(42, size)
        img_bytes = to_bytes(img)
        iterations = 200
        t0 = time.perf_counter()
        for _ in range(iterations):
            image_phash.dhash(img_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_hash_us = elapsed_ms / iterations * 1000
        qps = 1000 / (elapsed_ms / iterations)
        print(f"  {label:>10s}: {elapsed_ms:7.1f}ms for {iterations} hashes = {per_hash_us:6.0f}μs/hash ≈ {qps:6.0f} QPS")


def bench_robustness():
    """Test dHash robustness to common image manipulations."""
    print("\n" + "=" * 60)
    print("Test 2: Robustness to Manipulations")
    print("=" * 60)

    original = make_pattern_image(42)
    orig_bytes = to_bytes(original)
    orig_hash = image_phash.dhash(orig_bytes)
    print(f"  Original hash: {orig_hash}")

    tests = [
        ("resize 50%", to_bytes(original.resize((160, 120), Image.LANCZOS).resize((320, 240), Image.LANCZOS))),
        ("resize 200%", to_bytes(original.resize((640, 480), Image.LANCZOS).resize((320, 240), Image.LANCZOS))),
        ("brightness +10", _adjust_brightness(original, 10)),
        ("brightness -10", _adjust_brightness(original, -10)),
        ("contrast x1.2", _adjust_contrast(original, 1.2)),
        ("watermark text", _add_watermark(original, "HELLO WORLD")),
        ("JPEG compress", _jpeg_compress(original, 50)),
        ("JPEG compress low", _jpeg_compress(original, 20)),
    ]

    all_good = True
    for label, img_bytes in tests:
        h = image_phash.dhash(img_bytes)
        d = image_phash.hamming(orig_hash, h)
        status = "✓" if d <= 10 else "✗ FAIL"
        if d > 10:
            all_good = False
        print(f"  [{status}] {label:20s}: Hamming distance = {d:>2}/64")

    # Different image
    diff = make_pattern_image(99)
    diff_bytes = to_bytes(diff)
    diff_hash = image_phash.dhash(diff_bytes)
    diff_dist = image_phash.hamming(orig_hash, diff_hash)
    diff_ok = diff_dist > 20
    print(f"  [{'✓' if diff_ok else '✗'}x] {'different image':20s}: Hamming distance = {diff_dist:>2}/64")

    if all_good and diff_ok:
        print(f"\n  ✓ dHash robust to all tested manipulations, distinguishes different images")


def bench_hamming_distribution():
    """Hamming distance histogram for similar vs different pairs."""
    print("\n" + "=" * 60)
    print("Test 3: Hamming Distance Distribution")
    print("=" * 60)

    # Generate 10 "original" images
    originals = [make_pattern_image(i) for i in range(10)]
    orig_hashes = [image_phash.dhash(to_bytes(img)) for img in originals]

    # Similar pairs: same image vs its variations
    similar_dists = []
    for i, img in enumerate(originals):
        variants = [
            _adjust_brightness(img, 5),
            _add_watermark(img, "X"),
            _jpeg_compress(img, 80),
        ]
        for variant_bytes in variants:
            dist = image_phash.hamming(orig_hashes[i], image_phash.dhash(variant_bytes))
            similar_dists.append(dist)

    # Different pairs: different images
    different_dists = []
    for i in range(10):
        for j in range(i+1, 10):
            dist = image_phash.hamming(orig_hashes[i], orig_hashes[j])
            different_dists.append(dist)

    print(f"  Similar pairs  (n={len(similar_dists)}):   min={min(similar_dists)} max={max(similar_dists)} avg={sum(similar_dists)/len(similar_dists):.1f}")
    print(f"  Different pairs (n={len(different_dists)}): min={min(different_dists)} max={max(different_dists)} avg={sum(different_dists)/len(different_dists):.1f}")

    # Histogram
    print(f"\n  Histogram (d=0..63):")
    for threshold in [2, 5, 8, 10, 12, 15, 20]:
        sim_matches = sum(1 for d in similar_dists if d <= threshold)
        diff_matches = sum(1 for d in different_dists if d <= threshold)
        sim_rate = sim_matches / len(similar_dists) * 100
        fp_rate = diff_matches / len(different_dists) * 100
        print(f"    threshold≤{threshold:>2}: similar_recovery={sim_rate:5.1f}%  false_positive={fp_rate:5.1f}%")

    print(f"\n  Recommended: threshold = 10 (catch similar, avoid false positives)")


def bench_known_db():
    """Test known harmful hash database matching."""
    print("\n" + "=" * 60)
    print("Test 4: Known Harmful Hash Database Matching")
    print("=" * 60)

    # Build a synthetic "known harmful" database
    known = {}
    for i in range(5):
        img = make_pattern_image(i)
        h = image_phash.dhash(to_bytes(img))
        known[h] = {"category": f"test_cat_{i}", "source": "bench", "action": "block"}
    image_phash.load_known_hashes(known)
    print(f"  Database size: {len(known)} hashes")

    # Test matching
    tests = [
        ("exact match", make_pattern_image(0), True),
        ("resized match", make_pattern_image(1).resize((160,120)).resize((320,240)), True),
        ("different image", make_pattern_image(99), False),
    ]

    for label, img, should_match in tests:
        h = image_phash.dhash(to_bytes(img))
        result = image_phash.check_known(h)
        matched = result is not None
        status = "✓" if matched == should_match else "✗"
        detail = f"category={result['category']}" if result else "no match"
        print(f"  [{status}] {label:20s}: {detail}")


def bench_uniqueness():
    """Verify dHash produces unique hashes for different images."""
    print("\n" + "=" * 60)
    print("Test 5: Hash Uniqueness")
    print("=" * 60)

    hashes = set()
    collisions = 0
    n = 500

    for seed in range(n):
        img = make_pattern_image(seed * 137 + 42, (64, 48))
        h = image_phash.dhash(to_bytes(img))
        if h in hashes:
            collisions += 1
        hashes.add(h)

    uniqueness = (1 - collisions / n) * 100
    print(f"  Generated: {n} unique images (different seeds)")
    print(f"  Unique hashes: {len(hashes)}")
    print(f"  Collisions: {collisions}")
    print(f"  Uniqueness: {uniqueness:.1f}%")


def _adjust_brightness(img: Image.Image, delta: int) -> bytes:
    img2 = img.copy()
    pixels = img2.load()
    for x in range(img2.width):
        for y in range(img2.height):
            r, g, b = pixels[x, y]
            pixels[x, y] = (min(255, max(0, r+delta)), min(255, max(0, g+delta)), min(255, max(0, b+delta)))
    return to_bytes(img2)


def _adjust_contrast(img: Image.Image, factor: float) -> bytes:
    from PIL import ImageEnhance
    return to_bytes(ImageEnhance.Contrast(img).enhance(factor))


def _add_watermark(img: Image.Image, text: str) -> bytes:
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.text((10, 10), text, fill=(255, 255, 255))
    return to_bytes(img2)


def _jpeg_compress(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    bench_speed()
    bench_robustness()
    bench_hamming_distribution()
    bench_known_db()
    bench_uniqueness()
    print("\n✅ Image dHash benchmark complete")
