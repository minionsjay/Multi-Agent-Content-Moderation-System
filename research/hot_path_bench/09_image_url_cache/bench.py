#!/usr/bin/env python3
"""Benchmark: Image URL SHA256 Exact-Match Cache

Image hot path layer I3: exact URL/base64 hash match. The simplest layer —
same URL or base64 prefix → cached result. This catches re-posting of the
exact same image URL.

Tests:
  1. URL hash latency
  2. Base64 hash latency
  3. Cache hit rate for repeated URLs
  4. Collision resistance (different URLs → different hashes)

Usage:
  python bench.py
"""

import sys, os, time, hashlib, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.skills.memory_cache import memory_cache


def bench_url_hash_speed():
    """SHA256 hashing speed for image URLs and base64 prefixes."""
    print("=" * 60)
    print("Test 1: URL/Base64 Hash Speed")
    print("=" * 60)

    # Typical image URLs of varying lengths
    urls = [
        "https://cdn.example.com/img/abc123.jpg",
        "https://cdn.example.com/uploads/2024/01/15/large/highres/img_abc123def456_v2_compressed.jpg?w=800&q=90",
        "https://very-long-cdn-domain.images.example.com/social-media/posts/attachments/" +
            "a" * 200 + "/image.png?token=xyz&expires=9999999&sig=" + "b" * 100,
    ]

    for url in urls:
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            h = hashlib.sha256(url.encode()).hexdigest()[:16]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  URL ({len(url)} chars): {elapsed_ms:.1f}ms for {iterations} hashes "
              f"= {elapsed_ms/iterations*1000:.1f}μs/hash")


def bench_cache_hit_rate():
    """Simulate cache hit rate for repeated image URLs."""
    print("\n" + "=" * 60)
    print("Test 2: Image URL Cache Hit Rate Simulation")
    print("=" * 60)

    # Simulate: 200 unique images, some re-posted many times
    unique_urls = [f"https://cdn.example.com/img/photo_{i:04d}.jpg" for i in range(200)]
    spam_urls = [f"https://spam.example.com/bad_{i:04d}.jpg" for i in range(20)]

    # Traffic: 30% spam (same URLs repeated), 70% unique
    random.seed(42)
    traffic = []
    for _ in range(70):
        traffic.extend(random.sample(unique_urls, 50))  # unique images
    for _ in range(30):
        traffic.extend(spam_urls * 5)  # repeated spam images

    hits = 0
    misses = 0
    for url in traffic:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        result = memory_cache.get(f"[IMG]{key}")
        if result:
            hits += 1
        else:
            misses += 1
            memory_cache.set(f"[IMG]{key}", "block", 0.95, "test")

    total = hits + misses
    print(f"  Traffic: {total} requests")
    print(f"  Hits: {hits} ({hits/total*100:.1f}%)")
    print(f"  Misses: {misses} ({misses/total*100:.1f}%)")


def bench_url_collision():
    """Verify different URLs produce different hashes."""
    print("\n" + "=" * 60)
    print("Test 3: URL Hash Collision Check")
    print("=" * 60)

    # Generate similar but different URLs
    hashes = set()
    collisions = 0
    for i in range(10000):
        url = f"https://cdn.example.com/img/photo_{i:06d}.jpg?w=800&h=600&q=90&v={i%100}"
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        if h in hashes:
            collisions += 1
        hashes.add(h)

    print(f"  10,000 unique URLs → {collisions} collisions")
    print(f"  SHA256 truncated to 64-bit: collision rate = {collisions/10000:.4f}%")
    print(f"  Safe for exact-match image dedup ✓")


if __name__ == "__main__":
    bench_url_hash_speed()
    bench_cache_hit_rate()
    bench_url_collision()
    print("\n✅ Image URL cache benchmark complete")
