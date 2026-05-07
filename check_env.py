#!/usr/bin/env python3
"""Environment check script — verify all components work before running POC."""

import sys, os


def check_python():
    assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"
    print(f"[OK] Python {sys.version} ({sys.executable})")


def check_langgraph():
    import langgraph
    ver = getattr(langgraph, "__version__", "installed")
    print(f"[OK] LangGraph {ver}")


def check_chromadb():
    import chromadb
    print(f"[OK] ChromaDB {chromadb.__version__}")


def check_transformers():
    import transformers
    print(f"[OK] Transformers {transformers.__version__}")


def check_sentence_transformers():
    import sentence_transformers
    print(f"[OK] sentence-transformers {sentence_transformers.__version__}")


def check_fastapi():
    import fastapi
    print(f"[OK] FastAPI {fastapi.__version__}")


def check_dotenv():
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] python-dotenv loaded .env")


def check_deepseek_key():
    key = os.getenv("DEEPSEEK_API_KEY")
    placeholder_values = {
        "sk-your-deepseek-key-here",
        "your-deepseek-key",
        "your-api-key",
    }
    if not key or key.strip() in placeholder_values:
        print("[WARN] DEEPSEEK_API_KEY not set — LLM audit will fail")
    else:
        print(f"[OK] DeepSeek key set ({key[:8]}...)")


def check_keywords():
    from src.skills.keyword_filter import keyword_filter
    result = keyword_filter.match("这是一个测试文本，包含傻逼这个词")
    assert result["label"] is not None
    assert result["confidence"] > 0.9
    print(f"[OK] Keyword filter working (matched: {result['label']})")


def check_embedding():
    """Test local embedding with cached model."""
    from src.skills.embedder import embedder
    vec = embedder.embed("测试文本")
    assert len(vec) > 0, "Vector should not be empty"
    assert any(v != 0.0 for v in vec), "Vector should not be all zeros"
    print(f"[OK] Local embedding working (dim={len(vec)})")


def main():
    print("=== POC Environment Check ===\n")
    checks = [
        ("Python version", check_python),
        ("LangGraph", check_langgraph),
        ("ChromaDB", check_chromadb),
        ("Transformers", check_transformers),
        ("sentence-transformers", check_sentence_transformers),
        ("FastAPI", check_fastapi),
        ("dotenv", check_dotenv),
        ("DeepSeek API key", check_deepseek_key),
        ("Keyword filter", check_keywords),
        ("Local embedding", check_embedding),
    ]

    for name, fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
