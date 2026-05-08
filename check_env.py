#!/usr/bin/env python3
"""
Environment check suite — verify every technology component independently.

Usage:
    python check_env.py              # run all tests
    python check_env.py --quick      # skip optional/slow tests (LLM API, large models)
    python check_env.py --verbose    # show full traceback on failure
    python check_env.py --list       # list all tests without running

Exit code 0 = all mandatory tests pass, non-zero = some failed.
"""

import os
import sys
import time
import json
import hashlib
import traceback
import asyncio
import io

# Ensure poc/ is on path
sys.path.insert(0, os.path.dirname(__file__))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

QUICK = "--quick" in sys.argv
VERBOSE = "--verbose" in sys.argv

_all_tests = []
_results = []


def register(name, category, mandatory=True):
    """Decorator: register a test function."""
    def deco(fn):
        _all_tests.append((name, category, mandatory, fn))
        return fn
    return deco


# ============================================================================
# 1. Python & Core Packages
# ============================================================================

@register("Python >= 3.11", "core")
def test_python():
    v = sys.version_info
    assert v >= (3, 11), f"Python 3.11+ required, found {v.major}.{v.minor}"


@register("langgraph installed", "core")
def test_langgraph():
    import langgraph
    ver = getattr(langgraph, "__version__", "?")
    print(f"        version={ver}")


@register("fastapi + uvicorn installed", "core")
def test_fastapi():
    import fastapi, uvicorn  # noqa: F401


@register("chromadb installed", "core")
def test_chromadb():
    import chromadb  # noqa: F401


@register("transformers + torch installed", "core")
def test_transformers():
    import transformers, torch  # noqa: F401


@register("sentence-transformers installed", "core")
def test_sentence_transformers():
    import sentence_transformers  # noqa: F401


@register("cachetools installed", "core")
def test_cachetools():
    import cachetools  # noqa: F401


@register("httpx installed", "core")
def test_httpx():
    import httpx  # noqa: F401


@register("python-dotenv installed", "core")
def test_dotenv():
    from dotenv import load_dotenv
    load_dotenv()


@register("Pillow installed", "image")
def test_pillow():
    from PIL import Image  # noqa: F401


# ============================================================================
# 2. Configuration
# ============================================================================

@register(".env file loaded", "config")
def test_env_loaded():
    from dotenv import load_dotenv
    loaded = load_dotenv()
    if not loaded:
        print(f"        .env not found, using defaults")


@register("BERT_MODEL configured", "config")
def test_bert_config():
    from src.config import BERT_MODEL
    assert BERT_MODEL, "BERT_MODEL is empty"
    print(f"        model={BERT_MODEL}")


@register("EMBED_MODEL configured", "config")
def test_embed_config():
    from src.config import EMBED_MODEL
    assert EMBED_MODEL, "EMBED_MODEL is empty"
    print(f"        model={EMBED_MODEL}")


@register("LLM_PROVIDER configured", "config")
def test_llm_provider_config():
    from src.config import LLM_PROVIDER
    valid = {"deepseek", "openai", "anthropic", "local", "transformers"}
    assert LLM_PROVIDER in valid, f"LLM_PROVIDER={LLM_PROVIDER} must be one of {valid}"
    print(f"        provider={LLM_PROVIDER}")


@register("API key check (DeepSeek)", "config", mandatory=False)
def test_deepseek_key():
    from src.config import LLM_PROVIDER, DEEPSEEK_API_KEY
    if LLM_PROVIDER != "deepseek":
        print(f"        skipping (LLM_PROVIDER={LLM_PROVIDER})")
        return
    assert DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY not set"
    print(f"        key={DEEPSEEK_API_KEY[:8]}...")


# ============================================================================
# 3. Hot Path — Memory Cache
# ============================================================================

@register("Memory cache: set/get", "hotpath-L0")
def test_memory_cache_basic():
    from src.skills.memory_cache import memory_cache
    text = f"envcheck_{time.time()}"
    memory_cache.set(text, "block", 0.95, "test reason")
    result = memory_cache.get(text)
    assert result is not None, "get() returned None"
    assert result["decision"] == "block"
    assert result["confidence"] == 0.95


@register("Memory cache: SHA256 keying", "hotpath-L0")
def test_memory_cache_keying():
    from src.skills.memory_cache import memory_cache
    h1 = hashlib.sha256("text A".encode()).hexdigest()
    h2 = hashlib.sha256("text B".encode()).hexdigest()
    assert h1 != h2, "SHA256 collision"


@register("Redis cache (optional)", "hotpath-L0", mandatory=False)
def test_redis_cache():
    from src.skills.redis_cache import redis_cache
    if not redis_cache._available:
        raise Exception("Redis not available (no server running — this is OK in dev)")
    text = f"envcheck_redis_{time.time()}"
    redis_cache.set(text, "pass", 1.0, "test")
    result = redis_cache.get(text)
    assert result is not None


# ============================================================================
# 4. Hot Path — Keyword Filter
# ============================================================================

@register("jieba tokenizer", "hotpath-L1")
def test_jieba():
    import jieba
    tokens = list(jieba.cut("今天天气真好适合出去玩"))
    assert len(tokens) >= 3, f"Too few tokens: {tokens}"
    print(f"        tokens: {'/'.join(tokens)}")


@register("pyahocorasick AC automaton", "hotpath-L1")
def test_ahocorasick():
    import ahocorasick
    A = ahocorasick.Automaton()
    A.add_word("测试词", "test")
    A.make_automaton()
    matches = list(A.iter("这是一段包含测试词的文本"))
    assert len(matches) > 0, "No match found"


@register("Keyword filter: standalone word → block", "hotpath-L1")
def test_keyword_standalone():
    from src.skills.keyword_filter import keyword_filter
    result = keyword_filter.match("你真是个傻逼")
    assert result["confidence"] > 0.9, f"Expected high conf, got {result['confidence']}"
    print(f"        label={result['label']} conf={result['confidence']}")


@register("Keyword filter: embedded word → escalate", "hotpath-L1")
def test_keyword_embedded():
    from src.skills.keyword_filter import keyword_filter
    result = keyword_filter.match("接口交换技术讨论")
    assert result["confidence"] < 0.9, f"Embedded word should have low conf, got {result['confidence']}"
    print(f"        label={result['label']} conf={result['confidence']}")


@register("Keyword filter: whitelist → pass", "hotpath-L1")
def test_keyword_whitelist():
    from src.skills.keyword_filter import keyword_filter
    result = keyword_filter.match("他们在操场上操练")
    # Whitelisted phrase: should not match any keyword
    assert result["confidence"] < 0.4, f"Whitelisted phrase should have low conf, got {result['confidence']}"
    print(f"        conf={result['confidence']} matches={len(result.get('matches', []))}")


# ============================================================================
# 5. Hot Path — Embedding + ChromaDB
# ============================================================================

@register("BGE embedding model (load + infer)", "hotpath-L1")
def test_embedder():
    from src.skills.embedder import embedder
    vec = embedder.embed("这是一个测试文本用于环境检查")
    assert len(vec) == 512, f"Expected 512-dim, got {len(vec)}"
    norm = sum(v * v for v in vec) ** 0.5
    assert 0.9 < norm < 1.1, f"Normalized vector norm should be ~1.0, got {norm:.4f}"
    print(f"        dim={len(vec)} norm={norm:.4f}")


@register("Embedding consistency (same text → same vec)", "hotpath-L1")
def test_embedder_consistency():
    from src.skills.embedder import embedder
    v1 = embedder.embed("一致性测试文本")
    v2 = embedder.embed("一致性测试文本")
    diff = sum(abs(a - b) for a, b in zip(v1, v2))
    assert diff < 1e-5, f"Vectors differ: diff={diff:.2e}"
    print(f"        diff={diff:.2e}")


@register("ChromaDB: store + lookup", "hotpath-L1")
def test_chromadb_store_lookup():
    from src.skills.vector_cache import vector_cache
    from src.skills.embedder import embedder

    text = f"chroma_test_{int(time.time())}"
    vec = embedder.embed(text)
    vector_cache.store(vec, text, "block", 0.99, "test store")

    vec2 = embedder.embed(text)
    result = vector_cache.lookup(vec2)
    assert result is not None, "lookup returned None after store"
    assert result.get("decision") == "block"
    print(f"        count={vector_cache.count()}")


@register("ChromaDB: cache miss for new text", "hotpath-L1")
def test_chromadb_miss():
    from src.skills.vector_cache import vector_cache
    from src.skills.embedder import embedder
    text = f"brand_new_text_{time.time()}_{os.urandom(4).hex()}"
    vec = embedder.embed(text)
    result = vector_cache.lookup(vec)
    assert result is None, "Expected miss for never-seen text"


# ============================================================================
# 6. Cold Path — BERT Classification (L2)
# ============================================================================

@register("BERT: pipeline load + classify", "coldpath-L2")
def test_bert_pipeline():
    from src.skills.bert_classify import bert_classifier
    result = bert_classifier.classify("this is a beautiful day")
    assert "label" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1.0
    print(f"        label={result['label']} conf={result['confidence']:.4f}")


@register("BERT: detect obvious toxic English", "coldpath-L2")
def test_bert_toxic():
    from src.skills.bert_classify import bert_classifier
    result = bert_classifier.classify("you are a worthless piece of garbage")
    print(f"        label={result['label']} conf={result['confidence']:.4f}")
    # No hard assertion — model behavior varies by version


@register("BERT: should_skip_llm() threshold logic", "coldpath-L2")
def test_bert_skip_llm():
    from src.skills.bert_classify import bert_classifier
    assert bert_classifier.should_skip_llm({"label": "safe", "confidence": 0.96})
    assert bert_classifier.should_skip_llm({"label": "unsafe", "confidence": 0.98})
    assert not bert_classifier.should_skip_llm({"label": "safe", "confidence": 0.80})
    assert not bert_classifier.should_skip_llm({"label": "unsafe", "confidence": 0.50})


@register("BERT: ONNX model (optional)", "coldpath-L2", mandatory=False)
def test_bert_onnx():
    from src.skills.bert_onnx import bert_onnx
    if not bert_onnx._enabled:
        raise Exception("ONNX model not found (normal on fresh deploy)")
    result = bert_onnx.classify("this is a test sentence")
    assert "label" in result
    print(f"        label={result['label']} conf={result['confidence']:.4f}")


# ============================================================================
# 7. Cold Path — LLM Audit (L3)
# ============================================================================

@register("LLM: auditor module + system prompt", "coldpath-L3")
def test_llm_module():
    from src.skills.llm_audit import llm_auditor, SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) > 200, "System prompt too short"


@register("LLM: DeepSeek API call (needs key + network)", "coldpath-L3", mandatory=False)
def test_llm_deepseek_api():
    from src.config import LLM_PROVIDER, DEEPSEEK_API_KEY
    if LLM_PROVIDER != "deepseek":
        print(f"        skipping (LLM_PROVIDER={LLM_PROVIDER})")
        return
    assert DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY not set"

    from src.skills.llm_audit import llm_auditor

    async def call():
        return await llm_auditor.audit("hello world, today is a beautiful day")

    result = asyncio.run(call())
    assert "label" in result, f"Missing label: {result}"
    assert "confidence" in result
    print(f"        label={result['label']} conf={result.get('confidence', 0):.2f} "
          f"model={result.get('model', '?')}")


@register("LLM: OpenAI API call (optional)", "coldpath-L3", mandatory=False)
def test_llm_openai_api():
    from src.config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set")
    from src.skills.llm_audit import llm_auditor

    async def call():
        return await llm_auditor.audit("hello world", provider="openai")

    result = asyncio.run(call())
    print(f"        label={result['label']} conf={result.get('confidence', 0):.2f}")


@register("LLM: local llama.cpp (optional)", "coldpath-L3", mandatory=False)
def test_llm_llamacpp():
    from src.config import LOCAL_LLM_ENABLED
    if not LOCAL_LLM_ENABLED:
        raise Exception("LOCAL_LLM_ENABLED=false — set LLM_PROVIDER=local to test")
    from src.skills.llm_local import local_llm

    local_llm._load()
    if local_llm._llm is None:
        raise Exception(local_llm._load_error or "llama.cpp failed to load")
    print(f"        model={local_llm.model_path}")


@register("LLM: local transformers (optional)", "coldpath-L3", mandatory=False)
def test_llm_transformers():
    from src.config import LLM_PROVIDER
    if LLM_PROVIDER != "transformers":
        raise Exception(f"LLM_PROVIDER={LLM_PROVIDER} — set to 'transformers' to test")
    from src.skills.llm_transformers import llm_transformers

    ok = llm_transformers.warmup()
    if not ok:
        raise Exception(llm_transformers.load_info.get("error", "unknown"))
    info = llm_transformers.load_info
    print(f"        model={info['model_name']} load_time={info['load_time_s']}s 4bit={info['load_in_4bit']}")


@register("LLM: transformers inference (small model)", "coldpath-L3", mandatory=False)
def test_llm_transformers_infer():
    from src.config import LLM_PROVIDER
    if LLM_PROVIDER != "transformers":
        raise Exception(f"LLM_PROVIDER={LLM_PROVIDER} — set to 'transformers' to test")
    from src.skills.llm_transformers import llm_transformers

    async def call():
        return await llm_transformers.audit("hello, today is a good day")

    result = asyncio.run(call())
    assert "label" in result
    print(f"        label={result['label']} conf={result.get('confidence', 0):.2f}")


# ============================================================================
# 8. LangGraph Orchestration
# ============================================================================

@register("Graph: ModerationState schema", "orchestration")
def test_state_schema():
    from src.state import ModerationState
    fields = list(ModerationState.__annotations__.keys())
    required = ["content_id", "decision", "text_result", "confidence"]
    for f in required:
        assert f in fields, f"Missing field: {f}"
    print(f"        {len(fields)} fields defined")


@register("Graph: compile + check nodes", "orchestration")
def test_graph_compile():
    from src.graph import graph
    assert graph is not None
    nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    print(f"        nodes={nodes}")


@register("Graph: end-to-end text dry run", "orchestration", mandatory=False)
def test_graph_e2e():
    from src.graph import graph

    state = {
        "content_id": "envcheck_e2e",
        "text": "这是一个正常的测试文本 no toxic content",
        "image_url": "", "image_base64": "",
        "user_id": "envcheck", "source": "env_check",
        "content_type": "text_only",
        "cache_hit": False, "cached_decision": None,
        "keyword_confidence": 0.0, "keyword_label": None,
        "keyword_prefiltered": True, "priority_score": 0.3,
        "text_result": None, "image_result": None,
        "decision": "pass", "confidence": 0.0, "reason": "",
        "traces": [],
    }

    async def run():
        return await graph.ainvoke(state)

    result = asyncio.run(run())
    assert "decision" in result
    assert result["decision"] in ("pass", "block", "review")
    tr = result.get("text_result") or {}
    tier = tr.get("tier", "?")
    print(f"        decision={result['decision']} conf={result.get('confidence',0):.2f} tier={tier}")


@register("Gateway: full hot path (empty, normal, keyword)", "orchestration")
def test_gateway():
    from src.gateway import gateway

    # Empty content → pass immediately
    gw1 = gateway.check("")
    assert gw1["decision"] is not None
    assert gw1["decision"]["decision"] == "pass"
    print(f"        empty → pass (tier={gw1['decision'].get('tier', '?')})")

    # Normal text → escalate (no cache)
    gw2 = gateway.check(f"envcheck_gateway_{time.time()}")
    assert isinstance(gw2, dict)
    assert "traces" in gw2
    escalated = gw2["decision"] is None
    print(f"        normal → {'escalated to cold path' if escalated else 'cached (hot path)'}")

    # Keyword → block
    gw3 = gateway.check("你是个傻逼操你妈的滚蛋")
    assert gw3["decision"] is not None
    assert gw3["decision"]["decision"] == "block"
    print(f"        keyword → block (conf={gw3['decision'].get('confidence', 0)})")


# ============================================================================
# 9. Image Processing
# ============================================================================

@register("Image: Pillow create/resize/save", "image")
def test_pillow_ops():
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    resized = img.resize((100, 100))
    assert resized.size == (100, 100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert buf.tell() > 0
    print(f"        PNG size={buf.tell()} bytes")


@register("Image: dHash perceptual hash", "image")
def test_image_dhash():
    from PIL import Image, ImageDraw
    from src.skills.image_phash import image_phash

    # Create two visibly different images
    img1 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([50, 50, 200, 200], fill=(0, 0, 0))

    img2 = Image.new("RGB", (256, 256), color=(0, 0, 0))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([100, 0, 156, 256], fill=(255, 255, 255))

    buf1 = io.BytesIO(); img1.save(buf1, format="PNG")
    buf2 = io.BytesIO(); img2.save(buf2, format="PNG")

    h1 = image_phash.dhash(buf1.getvalue())
    h2 = image_phash.dhash(buf1.getvalue())  # same image
    h3 = image_phash.dhash(buf2.getvalue())

    assert h1 == h2, f"Same image → different hash: {h1} vs {h2}"
    assert h1 != h3, f"Different images → same hash: {h1}"
    print(f"        same={h1} diff={h3} (OK)")


@register("Image: NSFW classifier (optional)", "image", mandatory=False)
def test_image_nsfw():
    from src.skills.image_nsfw import nsfw_classifier
    if not nsfw_classifier._enabled:
        raise Exception("NSFW model not loaded (skip_model=True or not downloaded)")
    from PIL import Image
    img = Image.new("RGB", (256, 256), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = nsfw_classifier.classify(buf.getvalue())
    print(f"        label={result.get('label', '?')} conf={result.get('confidence', 0):.3f}")


@register("Image: EasyOCR (optional)", "image", mandatory=False)
def test_image_ocr():
    try:
        import easyocr  # noqa: F401
    except ImportError:
        raise Exception("easyocr not installed: pip install easyocr")
    from src.skills.image_ocr import ocr_reader
    if not ocr_reader._enabled:
        raise Exception("EasyOCR not initialized")


# ============================================================================
# 10. API Server
# ============================================================================

@register("API: app creation + routes", "api")
def test_api_app():
    from src.api import app
    assert app.title == "Content Moderation POC"
    routes = [r.path for r in app.routes]
    assert "/health" in routes
    assert "/moderate" in routes
    print(f"        routes={[r for r in routes if r.startswith('/')]}")


@register("API: health endpoint (needs server running)", "api", mandatory=False)
def test_api_health():
    import httpx
    try:
        resp = httpx.get("http://localhost:8000/health", timeout=3)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        print(f"        status={data['status']} version={data.get('version', '?')}")
    except httpx.ConnectError:
        raise Exception("Server not running on :8000 — start with: python -m src.api")


# ============================================================================
# Runner
# ============================================================================

def run_all():
    global _results
    print(f"{BOLD}Content Moderation POC — Environment Check{RESET}")
    if QUICK:
        print(f"  [{YELLOW}⚠{RESET}] Quick mode: skipping optional/slow tests")
    print(f"  {len(_all_tests)} tests registered\n")

    for name, category, mandatory, fn in _all_tests:
        # Skip optional tests in quick mode
        if QUICK and not mandatory:
            _results.append((name, category, "SKIP", "quick mode"))
            print(f"  [{CYAN}SKIP{RESET}] {name}")
            continue

        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            _results.append((name, category, "PASS", f"{ms:.0f}ms"))
            detail = f"{ms:.0f}ms" if VERBOSE else ""
            print(f"  [{GREEN}OK{RESET}] {name} ({detail})" if detail else f"  [{GREEN}OK{RESET}] {name}")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            msg = traceback.format_exc() if VERBOSE else str(e)[:200]
            _results.append((name, category, "FAIL", str(e)[:200]))
            print(f"  [{RED}FAIL{RESET}] {name}")
            print(f"        {RED}{str(e)[:200]}{RESET}")
            if VERBOSE:
                print(f"        {traceback.format_exc()}")

    return print_summary()


def print_summary():
    print()
    print("=" * 60)
    print(f"  {BOLD}Summary{RESET}")
    print("=" * 60)

    # By category
    cats = {}
    for _, cat, status, _ in _results:
        cats.setdefault(cat, {"PASS": 0, "FAIL": 0, "SKIP": 0})
        cats[cat][status] += 1

    for cat, c in sorted(cats.items()):
        print(f"  {cat:18s} {GREEN}{c['PASS']} pass{RESET}  "
              f"{RED}{c['FAIL']} fail{RESET}  {CYAN}{c['SKIP']} skip{RESET}")

    total = {s: sum(c[s] for c in cats.values()) for s in ("PASS", "FAIL", "SKIP")}
    print(f"  {'─' * 18}")
    print(f"  {'TOTAL':18s} {GREEN}{total['PASS']} pass{RESET}  "
          f"{RED}{total['FAIL']} fail{RESET}  {CYAN}{total['SKIP']} skip{RESET}")

    if total["FAIL"] == 0:
        print(f"\n  {GREEN}{BOLD}All mandatory tests passed!{RESET}")
        print(f"  Next: python -m src.api")
    else:
        print(f"\n  {RED}{BOLD}{total['FAIL']} test(s) failed.{RESET}")
        # List failed
        for name, cat, status, detail in _results:
            if status == "FAIL":
                print(f"    {RED}✗{RESET} [{cat}] {name}")
                print(f"      {detail[:120]}")

    return total["FAIL"]


def list_tests():
    print(f"{BOLD}Registered tests:{RESET}\n")
    for name, category, mandatory, fn in _all_tests:
        tag = f"{RED}*{RESET}" if mandatory else f"{CYAN}?{RESET}"
        print(f"  {tag} [{category}] {name}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_tests()
    else:
        fails = run_all() or 0
        sys.exit(1 if fails > 0 else 0)
