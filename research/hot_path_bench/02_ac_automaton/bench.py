#!/usr/bin/env python3
"""Benchmark: AC Automaton Keyword Matching

Tests:
  1. Latency vs text length (10 → 10,000 chars)
  2. Latency vs keyword dictionary size (10 → 10,000 words)
  3. AC automaton vs naive substring matching fallback
  4. Memory usage of the automaton
  5. Chinese-specific: mixed CJK + ASCII matching

Key questions:
  - Is AC automaton truly O(n) independent of dictionary size?
  - At what dictionary size does the fallback become unacceptable?
  - How fast is it compared to a simple 'if word in text' loop?
"""

import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.skills.keyword_filter import keyword_filter, DEFAULT_KEYWORDS


def bench_vs_text_length():
    """AC automaton latency vs text length."""
    print("=" * 60)
    print("Test 1: Latency vs Text Length")
    print("=" * 60)

    for length in [20, 50, 100, 200, 500, 1000, 2000, 5000]:
        # Generate random Chinese text with embedded keywords
        chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情最什么"
        text = "".join(random.choice(chars) for _ in range(length))

        # Inject a keyword
        text = text[:length//2] + "傻逼" + text[length//2:]

        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            keyword_filter.match(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"  text_length={length:>5}: {elapsed_ms:7.2f}ms for {iterations} calls "
              f"= {elapsed_ms/iterations*1000:6.2f}μs/call")


def bench_vs_dict_size():
    """Build automaton with different sizes, measure latency."""
    print("\n" + "=" * 60)
    print("Test 2: Latency vs Dictionary Size")
    print("=" * 60)

    # Note: current KeywordFilter singleton is built once. We test
    # by importing ahocorasick directly and building fresh automatons.
    try:
        import ahocorasick
    except ImportError:
        print("  pyahocorasick not installed — skipping")
        return

    text = "这是一个包含傻逼和脑残的测试文本，非常垃圾的产品"

    for num_words in [10, 50, 100, 500, 1000, 5000, 10000]:
        # Build automaton with synthetic words
        automaton = ahocorasick.Automaton()
        for i in range(num_words):
            word = f"word_{i:05d}"
            automaton.add_word(word, (word, "test"))
        automaton.make_automaton()

        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            list(automaton.iter(text))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"  dict_size={num_words:>6}: {elapsed_ms:7.2f}ms for {iterations} calls "
              f"= {elapsed_ms/iterations*1000:6.2f}μs/call")


def bench_ac_vs_naive():
    """Compare AC automaton vs 'if word in text' loop."""
    print("\n" + "=" * 60)
    print("Test 3: AC Automaton vs Naive Substring Matching")
    print("=" * 60)

    try:
        import ahocorasick
    except ImportError:
        print("  pyahocorasick not installed — skipping")
        return

    text = "这是一个很长的测试文本" * 100  # ~1000 chars

    for num_words in [10, 50, 100, 500]:
        words = {f"keyword_{i:04d}" for i in range(num_words)}

        # AC automaton
        automaton = ahocorasick.Automaton()
        for w in words:
            automaton.add_word(w, w)
        automaton.make_automaton()

        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            list(automaton.iter(text))
        ac_ms = (time.perf_counter() - t0) * 1000

        # Naive
        t0 = time.perf_counter()
        for _ in range(iterations):
            matches = []
            for w in words:
                if w in text:
                    matches.append(w)
        naive_ms = (time.perf_counter() - t0) * 1000

        ratio = naive_ms / ac_ms if ac_ms > 0 else float('inf')
        print(f"  words={num_words:>4}: AC={ac_ms:6.2f}ms  Naive={naive_ms:7.2f}ms  "
              f"AC is {ratio:.1f}x faster")


def bench_real_matches():
    """Test with real keyword dictionary on varied inputs."""
    print("\n" + "=" * 60)
    print("Test 4: Real Keyword Matching (Current Dictionary)")
    print("=" * 60)

    test_cases = [
        ("正常文本", "今天天气真好适合出去玩", None),
        ("明显辱骂", "你真是个傻逼什么都不懂", "toxic"),
        ("政治敏感", "支持台独台湾是一个独立国家", "politics"),
        ("暴力威胁", "我要杀了你这个混蛋", "violence"),
        ("色情内容", "加我微信看裸照约炮一夜情", "porn"),
        ("赌博广告", "真人百家乐在线赌博日赚千元", "gambling"),
        ("白名单通过", "操场上同学们在操练队列", None),
        ("英文辱骂", "you are a fucking idiot and stupid", "toxic"),
        ("长文本(1000字)", ("今天天气真好" * 100) + "傻逼", "toxic"),
        ("空文本", "", None),
    ]

    for label, text, expected in test_cases:
        t0 = time.perf_counter()
        result = keyword_filter.match(text)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        match = "✓" if (result["label"] == expected) or \
                       (expected is None and result["confidence"] < 0.9) else "✗"
        rlabel = str(result['label'] or 'none')
        print(f"  [{match}] {label:12s}: label={rlabel:10s} "
              f"conf={result['confidence']:.1f}  {elapsed_us:.0f}μs")


def bench_combo_detection():
    """Test multi-keyword combo detection."""
    print("\n" + "=" * 60)
    print("Test 5: Multi-Keyword Combo Detection")
    print("=" * 60)

    test_cases = [
        # (label, text, expected_label, expected_min_conf, description)
        ("单关键词-辱骂", "你真是个傻逼", "toxic", 1.0, "single toxic word"),
        ("单关键词-威胁", "我要杀了你", "violence", 1.0, "single violence word"),
        ("单关键词-政治", "支持台独", "politics", 1.0, "single politics word"),
        # Combo: predefined dangerous combinations
        ("预定义组合-辱骂+威胁", "你这个傻逼我要杀了你", "violence", 1.0,
         "combo: toxic+violence → violence override"),
        ("预定义组合-分裂主张", "支持台独和港独", "politics", 1.0,
         "combo: 台独+港独 → multi-separatism"),
        ("预定义组合-色情组合", "加微信约炮发裸照给你", "porn", 1.0,
         "combo: 约炮+裸照 → porn combo"),
        # Multi-category (implicit): different categories → boosted confidence
        ("多类别组合-辱骂+色情", "你这个傻逼加我微信约炮", "porn", 0.9,
         "multi-cat: toxic+porn → boosted, porn is higher severity"),
        ("多类别组合-辱骂+威胁", "你妈的他妈的我要弄死你", "violence", 1.0,
         "multi-cat: toxic+violence → very high, predefined combo matches"),
        ("多类别组合-赌博+色情", "赌博约炮一夜情在线下注", "gambling", 0.9,
         "multi-cat: gambling+porn → boosted by multi-keyword, combo gambling+下注 triggers"),
        # Multi-keyword same category (implicit boosting)
        ("同类别多词-辱骂", "你这个傻逼脑残白痴弱智", "toxic", 1.0,
         "same-cat: 4 toxic words → max confidence, combo 傻逼+脑残"),
        ("同类别多词-政治", "支持台独港独藏独", "politics", 1.0,
         "same-cat: 3 politics words → max (combo 台独+港独 hits)"),
        ("同类别双词-嵌入", "接口交换技术性交朋友", None, 0.0,
         "both '接口交换' and '性交朋友' are whitelisted → suppressed"),
        # Multiple categories (3+) → extreme signal
        ("三类组合-极端危险", "傻逼台独我要杀了你", "violence", 1.0,
         "3 categories: toxic+politics+violence → combo 傻逼+杀了你 triggers violence"),
        # No combo: single embedded keyword still shows as embedded
        ("单嵌入词-被白名单拦截", "接口交换技术讨论", None, 0.0,
         "'接口交换' whitelisted → suppressed to 0.0"),
        # Single embedded keyword in English word
        ("单嵌入词-英文混合", "这个fucking单词嵌入在英文中", "toxic", 0.6,
         "'fuck' embedded in 'fucking' → 0.6 (embedded in larger token)"),
        # 2 embedded keywords not whitelisted → should be boosted
        ("双嵌入词-无白名单提升", "这个系统有漏洞和bug需要修复", None, 0.0,
         "no keywords matched → 0.0"),
        # Edge cases
        ("空文本", "", None, 0.0, "empty → no match"),
        ("正常文本", "今天天气真好适合出去玩", None, 0.0, "clean text → no match"),
        ("白名单优先", "操场上同学们在操练队列", None, 0.0,
         "whitelist suppresses false positives"),
    ]

    passed = 0
    failed = 0

    for label, text, expected_label, min_conf, desc in test_cases:
        t0 = time.perf_counter()
        result = keyword_filter.match(text)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        # Checks
        label_ok = result["label"] == expected_label or (expected_label is None and result["confidence"] < 0.9)
        conf_ok = result["confidence"] >= min_conf
        combo_info = ""
        if result.get("combo_hit"):
            combo_info = f" combo={result['combo_hit']['note']}"
        if result.get("multi_hit"):
            combo_info += f" multi_hit={result['multi_hit']} cats={result.get('category_count', 0)}"

        ok = label_ok and conf_ok
        if ok:
            passed += 1
        else:
            failed += 1

        status = "✓" if ok else "✗"
        rlabel = str(result['label'] or 'none')
        print(f"  [{status}] {label:22s}: label={rlabel:10s} "
              f"conf={result['confidence']:.2f} (≥{min_conf})  "
              f"{elapsed_us:.0f}μs{combo_info}")
        if not ok:
            print(f"        {RED if 'RED' in dir() else ''}FAIL: {desc}"
                  f"{'  expected_label=' + str(expected_label) if not label_ok else ''}"
                  f"{'  expected_conf≥' + str(min_conf) if not conf_ok else ''}")

    print(f"\n  Passed: {passed}/{passed+failed}"
          f"{'  FAILED: ' + str(failed) if failed > 0 else ''}")


if __name__ == "__main__":
    bench_vs_text_length()
    bench_vs_dict_size()
    bench_ac_vs_naive()
    bench_real_matches()
    bench_combo_detection()
    print("\n✅ AC Automaton benchmark complete")
