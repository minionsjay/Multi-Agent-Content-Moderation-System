#!/usr/bin/env python3
"""Benchmark: pHash-Based Image Cache Lookup

Image hot path layer I2: use dHash as cache key to find previously-moderated
images that are visually identical or near-identical. This catches re-uploads
of the same image at different URLs, with different compression, etc.

Tests:
  1. pHash cache lookup speed
  2. Hit rate for near-duplicate images
  3. Hamming distance threshold tuning
  4. Memory footprint per cached entry

Usage:
  python bench.py
"""

import sys, os, time, io, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from PIL import Image, ImageDraw
from src.skills.image_phash import image_phash
from src.skills.memory_cache import memory_cache


def make_image(seed: int) -> bytes:
    """Generate a random textured image."""
    random.seed(seed)
    img = Image.new("RGB", (320, 240))
    draw = ImageDraw.Draw(img)
    for x in range(0, 320, 20):
        for y in range(0, 240, 20):
            c = random.randint(0, 255)
            draw.rectangle([x, y, x+18, y+18], fill=(c, (c*seed)%256, (c+seed)%256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bench_cache_speed():
    """pHash cache lookup latency."""
    print("=" * 60)
    print("Test 1: pHash Cache Lookup Speed")
    print("=" * 60)

    img_bytes = make_image(42)
    phash = image_phash.dhash(img_bytes)

    # Store
    cache_key = f"[PH]{phash}"
    memory_cache.set(cache_key, "block", 0.95, "test")

    # Lookup
    iterations = 10000
    t0 = time.perf_counter()
    hits = 0
    for _ in range(iterations):
        result = memory_cache.get(cache_key)
        if result:
            hits += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  Cache key: [PH]{phash}")
    print(f"  {iterations} lookups: {elapsed_ms:.1f}ms = {elapsed_ms/iterations*1000:.1f}μs/lookup")
    print(f"  Hits: {hits}/{iterations}")
    print(f"  Effective QPS: {iterations/(elapsed_ms/1000):,.0f}")


def bench_near_duplicate_detection():
    """Test that near-identical images get similar hashes and cache hits."""
    print("\n" + "=" * 60)
    print("Test 2: Near-Duplicate Image Cache Hits")
    print("=" * 60)

    # Store original
    orig_bytes = make_image(100)
    orig_hash = image_phash.dhash(orig_bytes)
    memory_cache.set(f"[PH]{orig_hash}", "block", 0.95, "original")

    # Generate variations
    random.seed(100)
    img = Image.open(io.BytesIO(orig_bytes))
    variants = [
        ("resize 80%", img.resize((256, 192), Image.LANCZOS).resize((*img.size,), Image.LANCZOS)),
        ("JPEG q=70", _to_jpeg(img, 70)),
        ("brightness +8", _brighten(img, 8)),
        ("rotated 1°", img.rotate(1)),
    ]

    print(f"  Original hash: {orig_hash}")
    for label, variant in variants:
        variant_bytes = _to_png(variant)
        variant_hash = image_phash.dhash(variant_bytes)
        dist = image_phash.hamming(orig_hash, variant_hash)

        # Try to find via exact hash match
        exact_hit = memory_cache.get(f"[PH]{variant_hash}") is not None

        hamming_matches = 0
        if dist <= 10:
            # In production we'd scan the DB; here we show the concept
            hamming_matches += 1

        print(f"  {label:15s}: hash={variant_hash} d={dist:>2} "
              f"exact_cache={'HIT' if exact_hit else 'MISS'}"
              f"  hamming≤10={'✓' if hamming_matches else '✗'}")


def bench_threshold_tradeoff():
    """Trade-off between cache recall and false positives at different thresholds."""
    print("\n" + "=" * 60)
    print("Test 3: Hamming Threshold Trade-off")
    print("=" * 60)

    # Generate reference images and their hashes
    ref_hashes = {}
    for i in range(20):
        ref_hashes[i] = image_phash.dhash(make_image(i))

    # Match each ref against all others at different thresholds
    for threshold in [0, 2, 5, 8, 10, 15, 20]:
        true_positives = 0   # same image matched itself
        false_positives = 0  # different image matched

        for i in range(20):
            for j in range(20):
                dist = image_phash.hamming(ref_hashes[i], ref_hashes[j])
                if dist <= threshold:
                    if i == j:
                        true_positives += 1
                    else:
                        false_positives += 1

        print(f"  threshold≤{threshold:>2}: self_match={true_positives}/20  "
              f"cross_match={false_positives}/380 ({false_positives/380*100:.1f}%)")


def bench_memory_per_entry():
    """Memory footprint per cached image hash entry."""
    print("\n" + "=" * 60)
    print("Test 4: Memory per Cached Entry")
    print("=" * 60)

    phash = "a3f7c9b01e4d82f6"
    cache_key = f"[PH]{phash}"
    key_size = len(cache_key.encode())  # ~20 bytes
    # Value: {"decision": "block", "confidence": 0.95, "reason": "..."}
    value_size = 200  # estimated dict overhead

    print(f"  Key: {cache_key} ({key_size} bytes)")
    print(f"  Value: ~{value_size} bytes (estimated)")
    print(f"  Per entry: ~{key_size + value_size} bytes")
    print(f"  100K entries: ~{(key_size + value_size) * 100000 / 1024 / 1024:.0f} MB")
    print(f"  1M entries:   ~{(key_size + value_size) * 1000000 / 1024 / 1024:.0f} MB")

    # Actual measurement
    for i in range(100):
        h = image_phash.dhash(make_image(i))
        memory_cache.set(f"[PH]{h}", "block", 0.95, f"reason_{i}")

    print(f"  Cache entries after insert: {memory_cache.size}")


def _to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _to_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def _brighten(img: Image.Image, delta: int) -> Image.Image:
    img2 = img.copy()
    pixels = img2.load()
    for x in range(img2.width):
        for y in range(img2.height):
            r, g, b = pixels[x, y]
            pixels[x, y] = (min(255, r+delta), min(255, g+delta), min(255, b+delta))
    return img2


if __name__ == "__main__":
    bench_cache_speed()
    bench_near_duplicate_detection()
    bench_threshold_tradeoff()
    bench_memory_per_entry()
    print("\n✅ Image pHash cache benchmark complete")
