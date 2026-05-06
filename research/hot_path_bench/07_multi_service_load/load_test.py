#!/usr/bin/env python3
"""Multi-Service Load Test for Hot Path

Simulates multiple services hitting the Gateway concurrently.
Each "service" generates different traffic patterns:
  - Service A: Social media comments (80% normal, 15% toxic, 5% spam)
  - Service B: Live chat (60% normal, 30% toxic, 10% spam)
  - Service C: Product reviews (90% normal, 5% toxic, 5% spam)
  - Service D: Forum posts (70% normal, 10% political, 10% toxic, 10% spam)
  - Service E: Automated bot traffic (30% normal, 70% spam)

Tests:
  1. Concurrent requests from N services
  2. Gateway throughput under load
  3. Cache hit rates with multi-service traffic mix
  4. Cold path escalation rate
  5. Latency distribution (P50/P95/P99)

Usage:
  python load_test.py                          # default: 5 services × 1000 req
  python load_test.py --services 10 --per-service 5000   # 10 × 5000
  python load_test.py --concurrent 50                     # 50 concurrent
"""

import sys, os, time, random, asyncio, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.gateway import gateway
from src.skills.embedder import embedder


# ---- Traffic Generators (per service type) ----

SOCIAL_TEMPLATES = {
    "normal": [
        "今天天气真好", "这个不错", "学到了", "哈哈哈哈",
        "谢谢分享", "好厉害", "同意", "确实是这样",
        "我也遇到过", "推荐给大家", "收藏了", "太棒了",
    ],
    "toxic": [
        "你真是个傻逼", "垃圾", "脑残吧你", "操你妈",
        "滚蛋", "去死吧", "废物", "弱智",
    ],
    "spam": [
        "加微信买六合彩稳赢", "真人百家乐在线",
        "同城约炮软件", "日赚千元不是梦",
    ],
}

LIVE_CHAT_TEMPLATES = {
    "normal": [
        "主播好", "666", "来了来了", "加油",
        "唱得好", "跳得不错", "哈哈哈", "牛逼",
    ],
    "toxic": [
        "傻逼主播", "你他妈会不会玩", "废物", "菜鸡",
        "别播了丢人", "你妈死了", "fuck you",
    ],
    "spam": [
        "加群123456", "关注我互粉", "免费领红包",
    ],
}

REVIEW_TEMPLATES = {
    "normal": [
        "质量很好", "性价比高", "物流很快", "包装精美",
        "非常满意", "值得购买", "客服态度好", "用着不错",
        "第二次购买了", "朋友推荐的", "颜色好看",
    ],
    "toxic": [
        "垃圾产品", "骗人的", "千万别买", "质量太差",
    ],
    "spam": [
        "好评返现加微信", "刷单联系我",
    ],
}

FORUM_TEMPLATES = {
    "normal": [
        "这个技术怎么实现的", "学习了", "感谢分享经验",
        "我的看法不太一样", "可以参考一下官方文档",
        "有没有更好的方案", "之前遇到过类似问题",
    ],
    "toxic": [
        "你懂个屁", "傻逼言论", "脑子进水了吧",
    ],
    "political": [
        "支持台独", "港独万岁", "藏独独立",
    ],
    "spam": [
        "加我微信看更多", "付费咨询请联系",
    ],
}


def generate_traffic(templates: dict, distribution: dict, count: int) -> list[str]:
    """Generate traffic with given category distribution."""
    texts = []
    for category, ratio in distribution.items():
        num = int(count * ratio)
        pool = templates[category]
        for _ in range(num):
            text = random.choice(pool)
            # Add some randomness to avoid exact duplicates
            if random.random() < 0.3:
                text += f" [{random.randint(0, 999):03d}]"
            texts.append(text)
    # Fill remaining with normal
    while len(texts) < count:
        texts.append(random.choice(templates["normal"]))
    random.shuffle(texts)
    return texts[:count]


# ---- Load Test Runner ----

async def run_service(name: str, texts: list[str],
                      results: list, stats: dict) -> dict:
    """Simulate one service sending requests."""
    local_stats = {
        "total": 0, "hot_block": 0, "hot_pass": 0,
        "cold_escalated": 0, "total_ms": 0.0, "latencies": [],
    }

    for text in texts:
        t0 = time.perf_counter()
        result = gateway.check(text, "", "")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        local_stats["total"] += 1
        local_stats["total_ms"] += elapsed_ms
        local_stats["latencies"].append(elapsed_ms)

        if result["decision"] is not None:
            if result["decision"]["decision"] == "block":
                local_stats["hot_block"] += 1
            else:
                local_stats["hot_pass"] += 1
        else:
            local_stats["cold_escalated"] += 1

    results.append({"service": name, **local_stats})
    return local_stats


async def run_load_test(service_configs: list[dict], concurrent: int):
    """Run load test with multiple services."""
    all_tasks = []
    all_results = []

    # Generate traffic for each service
    for svc in service_configs:
        texts = generate_traffic(
            svc["templates"], svc["distribution"], svc["count"]
        )
        all_tasks.append((svc["name"], texts))

    # Run with semaphore for concurrency control
    sem = asyncio.Semaphore(concurrent)

    async def bounded_run(name, texts):
        async with sem:
            return await run_service(name, texts, all_results, {})

    print(f"Starting load test: {len(service_configs)} services, "
          f"{sum(s['count'] for s in service_configs)} total requests, "
          f"concurrency={concurrent}")
    print()

    t0 = time.perf_counter()
    tasks = [bounded_run(name, texts) for name, texts in all_tasks]
    service_results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    # Aggregate
    total_req = sum(r["total"] for r in service_results)
    all_latencies = []
    for r in service_results:
        all_latencies.extend(r["latencies"])
    all_latencies.sort()

    hot_block = sum(r["hot_block"] for r in service_results)
    hot_pass = sum(r["hot_pass"] for r in service_results)
    cold = sum(r["cold_escalated"] for r in service_results)

    # Print report
    print("=" * 70)
    print("MULTI-SERVICE LOAD TEST RESULTS")
    print("=" * 70)
    print(f"  Total time:        {elapsed:.2f}s")
    print(f"  Total requests:    {total_req:,}")
    print(f"  Throughput:        {total_req/elapsed:,.0f} req/s")
    print(f"  Concurrency:       {concurrent}")
    print()
    print(f"  Traffic Distribution:")
    print(f"    Hot block:       {hot_block:>6} ({hot_block/total_req*100:5.1f}%)")
    print(f"    Hot pass:        {hot_pass:>6} ({hot_pass/total_req*100:5.1f}%)")
    print(f"    Cold escalated:  {cold:>6} ({cold/total_req*100:5.1f}%)")
    print(f"    Hot path total:  {hot_block+hot_pass:>6} ({(hot_block+hot_pass)/total_req*100:5.1f}%)")
    print()
    print(f"  Latency Distribution:")
    p50 = all_latencies[len(all_latencies)//2]
    p95 = all_latencies[int(len(all_latencies)*0.95)]
    p99 = all_latencies[int(len(all_latencies)*0.99)]
    print(f"    P50: {p50:.2f}ms")
    print(f"    P95: {p95:.2f}ms")
    print(f"    P99: {p99:.2f}ms")
    print(f"    Avg: {sum(all_latencies)/len(all_latencies):.2f}ms")
    print(f"    Max: {max(all_latencies):.2f}ms")
    print()

    # Per-service breakdown
    print(f"  Per-Service Breakdown:")
    print(f"    {'Service':<16s} {'Total':>6s} {'Hot%':>6s} {'Avg(ms)':>8s} {'P99(ms)':>8s}")
    for r in service_results:
        svc_lats = sorted(r["latencies"])
        hot_rate = (r["hot_block"] + r["hot_pass"]) / r["total"] * 100
        avg_lat = sum(svc_lats) / len(svc_lats)
        p99_lat = svc_lats[int(len(svc_lats)*0.99)] if len(svc_lats) > 0 else 0
        print(f"    {r['latencies']}")  # placeholder
    # Proper per-service print
    print(f"    {'Service':<16s} {'Total':>6s} {'Hot%':>6s} {'Avg(ms)':>8s} {'P99(ms)':>8s}")
    print(f"    {'─'*16} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")
    for r in service_results:
        svc_lats = sorted(r["latencies"])
        hot_rate = (r["hot_block"] + r["hot_pass"]) / r["total"] * 100
        avg_lat = sum(svc_lats) / len(svc_lats)
        p99_lat = svc_lats[int(len(svc_lats)*0.99)] if len(svc_lats) > 0 else 0
        # Find service name from service_configs
        svc_name = "unknown"
        for s in service_configs:
            if s.get("result_ref") == id(r):
                svc_name = s["name"]
                break
        print(f"    {r.get('service','?'):16s} {r['total']:>6d} {hot_rate:>5.1f}% {avg_lat:>7.2f}ms {p99_lat:>7.2f}ms")

    # Gateway stats
    print()
    gw = gateway.get_stats()
    print(f"  Gateway Stats:")
    print(f"    Memory cache hit:  {gw['memory_cache_hit_rate']*100:.1f}%")
    print(f"    Keyword hit:       {gw['keyword_hit_rate']*100:.1f}%")
    print(f"    Whitelist hit:     {gw['whitelist_hit_rate']*100:.1f}%")
    print(f"    ChromaDB hit:      {gw['chroma_cache_hit_rate']*100:.1f}%")
    print(f"    Escalated:         {gw['escalated_rate']*100:.1f}%")
    print()
    print("=" * 70)

    # Bottleneck analysis
    print()
    print("Bottleneck Analysis:")
    emb_hit_rate = embedder.cache_hit_rate if hasattr(embedder, 'cache_hit_rate') else 0
    print(f"  BGE Embedding cache hit rate: {emb_hit_rate:.1%}")
    print(f"  Single-core BGE QPS limit: ~200")
    print(f"  Current throughput: {total_req/elapsed:.0f} req/s")

    # Estimate required workers
    cold_rate = cold / total_req
    embedding_rate = cold_rate + gw['chroma_cache_hit_rate']  # all need embedding
    emb_qps = (total_req / elapsed) * embedding_rate
    workers_needed = max(1, emb_qps / 200)
    print(f"  Embedding QPS needed: {emb_qps:.0f}")
    print(f"  Workers needed (CPU): {workers_needed:.0f} (at 200 QPS/core)")
    if workers_needed > 1:
        print(f"  ⚠️  Recommended: uvicorn --workers {int(workers_needed)+1}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--services", type=int, default=5,
                       help="Number of services to simulate")
    parser.add_argument("--per-service", type=int, default=1000,
                       help="Requests per service")
    parser.add_argument("--concurrent", type=int, default=20,
                       help="Max concurrent requests")
    args = parser.parse_args()

    # Define service configurations
    service_configs = [
        {"name": "social_comments", "templates": SOCIAL_TEMPLATES,
         "distribution": {"normal": 0.80, "toxic": 0.15, "spam": 0.05},
         "count": args.per_service},
        {"name": "live_chat", "templates": LIVE_CHAT_TEMPLATES,
         "distribution": {"normal": 0.60, "toxic": 0.30, "spam": 0.10},
         "count": args.per_service},
        {"name": "product_reviews", "templates": REVIEW_TEMPLATES,
         "distribution": {"normal": 0.90, "toxic": 0.05, "spam": 0.05},
         "count": args.per_service},
        {"name": "forum_posts", "templates": FORUM_TEMPLATES,
         "distribution": {"normal": 0.70, "toxic": 0.10, "political": 0.10, "spam": 0.10},
         "count": args.per_service},
        {"name": "bot_spam", "templates": SOCIAL_TEMPLATES,
         "distribution": {"normal": 0.30, "spam": 0.50, "toxic": 0.20},
         "count": args.per_service},
    ][:args.services]

    # Warm up
    print("Warming up models...")
    embedder.embed("warmup")
    gateway.check("warmup", "", "")
    print("Ready.\n")

    asyncio.run(run_load_test(service_configs, args.concurrent))


if __name__ == "__main__":
    main()
