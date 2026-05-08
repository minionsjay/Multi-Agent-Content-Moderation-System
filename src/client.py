"""
Moderation API Client — lightweight SDK for backend services.

Usage (in any service):

  from moderation_client import ModerationClient

  client = ModerationClient("http://moderation-api:8000")

  # Single text
  result = await client.moderate_text("你真是个傻逼")
  if result["decision"] == "block":
      delete_post()

  # Batch
  results = await client.moderate_batch(["text1", "text2", ...])
  for r in results:
      if r["decision"] == "review":
          queue_for_human(r)

Install: pip install httpx (already in requirements.txt)
Zero extra dependencies beyond httpx which the project already uses.
"""

import time
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger("moderation_client")

# Default timeout: 30s (covers cold-path LLM latency)
DEFAULT_TIMEOUT = 30.0


class ModerationClient:
    """Async HTTP client for the Multi-Agent Content Moderation API.

    Each backend service creates ONE client instance and reuses it.
    The client maintains a connection pool internally via httpx.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        service_name: str = "unknown",
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.service_name = service_name
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-Service-Name"] = self.service_name

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._timeout),
                headers=headers,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ---- Core API ----

    async def moderate_text(
        self,
        text: str,
        user_id: str = "anonymous",
        source: str = "api",
    ) -> dict:
        """Moderate a single text. Returns {decision, confidence, reason, tier, ...}.

        decision is one of: "pass" | "block" | "review"
        """
        return await self._post("/moderate", {
            "text": text,
            "user_id": user_id,
            "source": source or self.service_name,
        })

    async def moderate_image(
        self,
        image_url: str = "",
        image_base64: str = "",
        text: str = "",
        user_id: str = "anonymous",
    ) -> dict:
        """Moderate an image (with optional accompanying text)."""
        return await self._post("/moderate", {
            "text": text,
            "image_url": image_url,
            "image_base64": image_base64,
            "user_id": user_id,
            "source": self.service_name,
        })

    async def moderate_batch(
        self,
        texts: list[str],
        user_id: str = "anonymous",
    ) -> list[dict]:
        """Moderate multiple texts in one request. Returns list of results."""
        # Use the batch upload endpoint with JSONL in memory
        import io, json
        buf = io.StringIO()
        for i, t in enumerate(texts):
            buf.write(json.dumps({"id": f"b{i}", "text": t}, ensure_ascii=False) + "\n")

        client = await self._get_client()
        files = {"file": ("batch.jsonl", buf.getvalue().encode(), "application/x-ndjson")}
        resp = await client.post("/moderate/batch", files=files)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    # ---- Convenience methods ----

    async def is_safe(self, text: str) -> bool:
        """Quick check: is this text safe to post? Returns True if pass."""
        result = await self.moderate_text(text)
        return result["decision"] == "pass"

    async def moderate_with_fallback(
        self,
        text: str,
        on_block=None,
        on_review=None,
        on_pass=None,
    ) -> dict:
        """Moderate and call the appropriate callback based on decision."""
        result = await self.moderate_text(text)
        decision = result["decision"]

        if decision == "block" and on_block:
            await on_block(result)
        elif decision == "review" and on_review:
            await on_review(result)
        elif decision == "pass" and on_pass:
            await on_pass(result)

        return result

    # ---- Internal ----

    async def _post(self, path: str, json_data: dict) -> dict:
        """POST with retry logic."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                if attempt > 0:
                    await asyncio.sleep(0.5 * attempt)  # exponential-ish backoff

                t0 = time.perf_counter()
                resp = await client.post(path, json=json_data)
                elapsed = (time.perf_counter() - t0) * 1000

                resp.raise_for_status()
                result = resp.json()

                logger.debug(
                    "moderation | path=%s | decision=%s | %dms",
                    path, result.get("decision", "?"), int(elapsed),
                )
                return result

            except httpx.TimeoutException:
                last_error = f"Timeout after {self._timeout}s"
                logger.warning("moderation timeout (attempt %d/%d)", attempt + 1, self.max_retries + 1)
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                if e.response.status_code < 500:
                    raise  # 4xx: don't retry
                logger.warning("moderation server error (attempt %d/%d)", attempt + 1, self.max_retries + 1)
            except Exception as e:
                last_error = str(e)
                logger.warning("moderation error (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, e)

        raise RuntimeError(f"Moderation failed after {self.max_retries + 1} attempts: {last_error}")


# ============================================================
# Usage examples for different service types
# ============================================================

async def example_social_media():
    """Example: Social media comment service."""
    client = ModerationClient("http://moderation:8000", service_name="social_comments")

    # User posts a comment
    comment = "这个产品真的太垃圾了，谁买谁后悔"

    result = await client.moderate_with_fallback(
        comment,
        on_block=lambda r: print(f"  BLOCKED: {r['reason']}"),
        on_review=lambda r: print(f"  QUEUED for human review: {r['content_id']}"),
        on_pass=lambda r: print(f"  POSTED: conf={r['confidence']}"),
    )

    await client.close()


async def example_live_chat():
    """Example: Live streaming chat — needs fast response."""
    client = ModerationClient(
        "http://moderation:8000",
        service_name="live_chat",
        timeout=5.0,  # shorter timeout for live chat
    )

    messages = [
        "主播唱得真好听",
        "你这个傻逼会不会玩",
        "666666",
        "大家点个关注谢谢",
    ]

    # Check all messages
    for msg in messages:
        try:
            is_ok = await client.is_safe(msg)
            if is_ok:
                print(f"  SHOW: {msg}")
            else:
                print(f"  HIDE: {msg}")
        except RuntimeError:
            # Moderation service down — show anyway (fail open)
            # or hide everything (fail closed)
            print(f"  MODERATION DOWN — showing: {msg}")

    await client.close()


async def example_batch_audit():
    """Example: Nightly batch audit of all posts from the past day."""
    client = ModerationClient("http://moderation:8000", service_name="nightly_audit")

    # Fetch posts from database (simulated)
    posts = [f"post content {i}" for i in range(100)]

    # Batch moderate 50 at a time
    batch_size = 50
    all_results = []
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        results = await client.moderate_batch(batch)
        all_results.extend(results)

    # Analyze
    blocked = [r for r in all_results if r["decision"] == "block"]
    reviews = [r for r in all_results if r["decision"] == "review"]
    print(f"Total: {len(all_results)}, Block: {len(blocked)}, Review: {len(reviews)}")

    await client.close()


async def example_multi_service():
    """Example: Multiple services hitting moderation concurrently."""
    services = {
        "comments": ModerationClient("http://moderation:8000", service_name="comments"),
        "chat": ModerationClient("http://moderation:8000", service_name="chat"),
        "posts": ModerationClient("http://moderation:8000", service_name="posts"),
        "reviews": ModerationClient("http://moderation:8000", service_name="reviews"),
        "profile": ModerationClient("http://moderation:8000", service_name="profile"),
    }

    # All services send concurrently
    tasks = [
        services["comments"].moderate_text("垃圾产品骗人的"),
        services["chat"].moderate_text("你他妈会不会玩"),
        services["posts"].moderate_text("今天天气真好适合出去玩"),
        services["reviews"].moderate_text("质量很好推荐购买"),
        services["profile"].moderate_text("加微信看更多精彩内容"),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for svc_name, result in zip(services.keys(), results):
        if isinstance(result, Exception):
            print(f"  {svc_name}: ERROR — {result}")
        else:
            print(f"  {svc_name}: {result['decision']} (conf={result['confidence']:.2f})")

    # Cleanup
    for c in services.values():
        await c.close()


if __name__ == "__main__":
    print("=== Example: Social Media ===")
    asyncio.run(example_social_media())
    print("\n=== Example: Live Chat ===")
    asyncio.run(example_live_chat())
    print("\n=== Example: Multi-Service ===")
    asyncio.run(example_multi_service())
