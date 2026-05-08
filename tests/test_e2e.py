"""End-to-end smoke tests for the moderation graph.

Run with:  python tests/test_e2e.py
Or:        python -m pytest tests/test_e2e.py -v  (if pytest installed)
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import graph
from src.state import ModerationState


def _make_state(text: str) -> ModerationState:
    return {
        "content_id": "test_001",
        "text": text,
        "user_id": "test_user",
        "source": "test",
        "content_type": "text_only",
        "cache_hit": False,
        "cached_decision": None,
        "keyword_confidence": 0.0,
        "keyword_label": None,
        "keyword_prefiltered": False,
        "priority_score": 0.3,
        "text_result": None,
        "decision": "pass",
        "confidence": 0.0,
        "reason": "",
    }


async def test_safe_content_passes():
    state = _make_state("今天天气真好，适合出去玩")
    result = await graph.ainvoke(state)
    assert result["decision"] in ("pass", "review")


async def test_toxic_keyword_blocks():
    state = _make_state("你真是个傻逼，什么都不懂")
    result = await graph.ainvoke(state)
    # Should be caught by L1 keyword or L2 BERT or L3 LLM
    assert result["decision"] in ("block", "review")


async def test_empty_content():
    state = _make_state("")
    result = await graph.ainvoke(state)
    assert result["decision"] == "pass"


async def test_graph_returns_all_fields():
    state = _make_state("测试内容 moderation test")
    result = await graph.ainvoke(state)
    for field in ["decision", "confidence", "reason"]:
        assert field in result, f"Missing field: {field}"


async def test_parallel_requests():
    """Verify graph can handle concurrent requests."""
    texts = [
        "正常内容测试",
        "你这个傻逼",
        "今天天气不错",
        "我要杀了你",
        "推荐一个好用的产品",
    ]
    tasks = [graph.ainvoke(_make_state(t)) for t in texts]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    for r in results:
        assert "decision" in r


if __name__ == "__main__":
    async def _run():
        await test_safe_content_passes()
        await test_toxic_keyword_blocks()
        await test_empty_content()
        await test_graph_returns_all_fields()
        await test_parallel_requests()
        print("All tests passed!")

    asyncio.run(_run())
