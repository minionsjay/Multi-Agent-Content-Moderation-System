#!/usr/bin/env python3
"""Benchmark: Full Image Gateway Hot Path

End-to-end test of the complete image hot path:
  I1: dHash → known harmful DB check
  I2: pHash → memory cache lookup (visually similar images)
  I3: URL exact match → memory cache lookup

Tests:
  1. Latency breakdown per layer
  2. Traffic distribution (hot block / hot pass / cold escalate)
  3. Throughput
  4. With and without known harmful DB

Usage:
  python bench.py
"""

import sys, os, time, io, base64, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from PIL import Image, ImageDraw
from src.gateway import gateway
from src.skills.embedder import embedder
from src.skills.image_phash import image_phash


def make_image(seed: int, size=(320, 240)) -> bytes:
    """Generate random textured image as base64 string."""
    random.seed(seed)
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)
    for x in range(0, size[0], 20):
        for y in range(0, size[1], 20):
            c = random.randint(0, 255)
            draw.rectangle([x, y, x+18, y+18], fill=(c, (c*seed)%256, (c+seed)%256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def bench_latency_breakdown():
    """Per-layer latency for different image request types."""
    print("=" * 60)
    print("Test 1: Image Hot Path Latency Breakdown")
    print("=" * 60)

    embedder.embed("warmup")

    # Load known harmful hash
    harmful_img = make_image(666)
    harmful_hash = image_phash.dhash(base64.b64decode(harmful_img))
    image_phash.load_known_hashes({
        harmful_hash: {"category": "test_harmful", "source": "bench", "action": "block"}
    })

    normal_img = make_image(42)  # not in known DB

    cases = [
        ("known harmful (dHash block)", "", harmful_img),
        ("normal image (escalate)", "", normal_img),
        ("no image (text only)", "normal text", ""),
        ("URL only (escalate)", "", "no_base64_here", "https://example.com/img.jpg"),
    ]

    for label, text, img_b64, *rest in cases:
        img_url = rest[0] if rest else ""
        t0 = time.perf_counter()
        result = gateway.check(text, img_url, img_b64)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        path = "hot" if result["decision"] is not None else "cold"
        tier = result["decision"].get("tier", "escalated") if result["decision"] else "escalated"
        traces = [t["step"] for t in result["traces"]]

        print(f"  [{path:4s}] {label:35s}: {elapsed_ms:6.2f}ms | tier={tier:15s} | {traces}")


def bench_traffic_distribution():
    """Send 300 image requests and measure hot/cold distribution."""
    print("\n" + "=" * 60)
    print("Test 2: Image Traffic Distribution (300 requests)")
    print("=" * 60)

    # Load 3 hashes as "known harmful"
    known_hashes = {}
    for i in range(3):
        img_b64 = make_image(1000 + i)
        h = image_phash.dhash(base64.b64decode(img_b64))
        known_hashes[h] = {"category": "test", "source": "bench", "action": "block"}
    image_phash.load_known_hashes(known_hashes)

    # Generate traffic: 5% known harmful, 30% repeated, 65% unique
    random.seed(42)
    traffic = []

    # Known harmful (5%)
    for _ in range(15):
        traffic.append(("harmful", make_image(1000 + random.randint(0, 2))))

    # Repeated (30%) — same images re-sent
    repeated = [make_image(2000 + i) for i in range(10)]
    for _ in range(90):
        traffic.append(("repeated", random.choice(repeated)))

    # Unique (65%)
    for i in range(195):
        traffic.append(("unique", make_image(3000 + i)))

    random.shuffle(traffic)

    results = {"hot_block": 0, "hot_pass": 0, "cold_escalated": 0}
    total_ms = 0.0

    for tag, img_b64 in traffic:
        t0 = time.perf_counter()
        result = gateway.check("", "", img_b64)
        total_ms += (time.perf_counter() - t0) * 1000

        if result["decision"] is not None:
            if result["decision"].get("decision") == "block":
                results["hot_block"] += 1
            else:
                results["hot_pass"] += 1
        else:
            results["cold_escalated"] += 1

    total = len(traffic)
    print(f"  Total:     {total}")
    print(f"  Hot block: {results['hot_block']:>4} ({results['hot_block']/total*100:5.1f}%)  ← dHash known DB")
    print(f"  Hot pass:  {results['hot_pass']:>4} ({results['hot_pass']/total*100:5.1f}%)  ← pHash cache hit")
    print(f"  Cold esc:  {results['cold_escalated']:>4} ({results['cold_escalated']/total*100:5.1f}%)  ← needs NSFW ViT + OCR")
    print(f"  Avg lat:   {total_ms/total:.2f}ms")


def bench_throughput():
    """Measure image hot path throughput."""
    print("\n" + "=" * 60)
    print("Test 3: Image Hot Path Throughput")
    print("=" * 60)

    images = [make_image(i) for i in range(100)]
    images += [make_image(i) for i in range(50)] * 2  # 50 repeated

    t0 = time.perf_counter()
    for img_b64 in images:
        gateway.check("", "", img_b64)
    elapsed = time.perf_counter() - t0

    print(f"  Total:  {len(images)} requests")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  QPS:    {len(images)/elapsed:.0f} req/s")
    print(f"  Avg:    {elapsed/len(images)*1000:.2f}ms/req")


def bench_compare_text_vs_image():
    """Compare text hot path latency vs image hot path latency."""
    print("\n" + "=" * 60)
    print("Test 4: Text vs Image Hot Path Comparison")
    print("=" * 60)

    # Text: keyword block
    t0 = time.perf_counter()
    for _ in range(100):
        gateway.check("你真是个傻逼", "", "")
    text_ms = (time.perf_counter() - t0) * 1000 / 100

    # Image: pHash computation (normal, no DB match)
    img_b64 = make_image(9999)
    # Warm up
    gateway.check("", "", img_b64)

    t0 = time.perf_counter()
    for _ in range(100):
        gateway.check("", "", img_b64)
    img_ms = (time.perf_counter() - t0) * 1000 / 100

    print(f"  Text keyword block:  {text_ms:.3f}ms/req")
    print(f"  Image dHash escalate: {img_ms:.3f}ms/req")
    print(f"  Image/text ratio:     {img_ms/text_ms:.1f}x")
    print(f"  Image overhead is dHash computation (~1ms per 320×240 image)")


if __name__ == "__main__":
    print("Warming up...")
    embedder.embed("warmup")
    gateway.check("warmup", "", "")
    print("Ready.\n")

    bench_latency_breakdown()
    bench_traffic_distribution()
    bench_throughput()
    bench_compare_text_vs_image()
    print("\n✅ Image Gateway full flow benchmark complete")
