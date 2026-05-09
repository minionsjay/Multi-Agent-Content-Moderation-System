#!/usr/bin/env python3
"""
LLM API 连通性测试

用法:
  # 命令行参数
  python test_llm_api.py \
    --provider deepseek \
    --api-key sk-xxx \
    --model deepseek-chat

  # 或读取 .env
  python test_llm_api.py --use-env

  # 自定义 base_url + 关闭代理
  python test_llm_api.py \
    --provider openai \
    --api-key sk-xxx \
    --base-url https://your-endpoint.com/v1 \
    --model qwen2.5-7b \
    --no-proxy
"""

import argparse
import asyncio
import json
import os
import sys
import time

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def test_openai_compatible(
    api_key: str, base_url: str, model: str, no_proxy: bool
) -> dict:
    """Test OpenAI / DeepSeek / compatible API."""
    import httpx
    from openai import AsyncOpenAI

    kwargs = {"api_key": api_key, "base_url": base_url}
    if no_proxy:
        kwargs["http_client"] = httpx.AsyncClient(proxy=None)

    client = AsyncOpenAI(**kwargs)

    test_text = "今天天气真好，适合出去玩"

    t0 = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'请对以下文本回复 JSON: {{"label": "safe", "confidence": 0.99}}\n\n文本: "{test_text}"'},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        content = response.choices[0].message.content or "(empty)"
        return {
            "ok": True,
            "latency_ms": round(elapsed, 1),
            "model": response.model,
            "response": content[:300],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else "?",
                "completion_tokens": response.usage.completion_tokens if response.usage else "?",
            },
        }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "latency_ms": round(elapsed, 1),
            "error": str(e),
        }


async def test_anthropic(
    api_key: str, model: str, no_proxy: bool
) -> dict:
    """Test Anthropic API."""
    import httpx
    from anthropic import AsyncAnthropic

    kwargs = {"api_key": api_key}
    if no_proxy:
        kwargs["http_client"] = httpx.AsyncClient(proxy=None)

    client = AsyncAnthropic(**kwargs)

    t0 = time.perf_counter()
    try:
        response = await client.messages.create(
            model=model,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": 'Reply with JSON only: {"label":"safe","confidence":0.99} for the text: "hello world"'}],
            temperature=0.0,
            max_tokens=128,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        content = response.content[0].text if response.content else "(empty)"
        return {
            "ok": True,
            "latency_ms": round(elapsed, 1),
            "model": response.model,
            "response": content[:300],
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else "?",
                "output_tokens": response.usage.output_tokens if response.usage else "?",
            },
        }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "latency_ms": round(elapsed, 1),
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="LLM API 连通性测试")
    parser.add_argument("--use-env", action="store_true",
                        help="从 .env 文件读取配置")
    parser.add_argument("--provider", type=str, default="deepseek",
                        choices=["deepseek", "openai", "anthropic"],
                        help="LLM 提供商类型")
    parser.add_argument("--api-key", type=str, default="",
                        help="API Key")
    parser.add_argument("--base-url", type=str, default="",
                        help="API Base URL (DeepSeek/OpenAI)")
    parser.add_argument("--model", type=str, default="",
                        help="模型名称")
    parser.add_argument("--no-proxy", action="store_true",
                        help="绕过 HTTP_PROXY/HTTPS_PROXY 环境变量")

    args = parser.parse_args()

    # Resolve config
    api_key = args.api_key
    base_url = args.base_url
    model = args.model
    provider = args.provider

    if args.use_env:
        from dotenv import load_dotenv
        load_dotenv()

        if not api_key:
            if provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY", "")
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if not base_url:
            if provider == "deepseek":
                base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            elif provider == "openai":
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not model:
            if provider == "deepseek":
                model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            elif provider == "openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            elif provider == "anthropic":
                model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

        no_proxy_flag = os.getenv("LLM_NO_PROXY", "false").lower() != "false"
    else:
        # Default base URLs
        if not base_url:
            if provider == "deepseek":
                base_url = "https://api.deepseek.com"
            elif provider == "openai":
                base_url = "https://api.openai.com/v1"
        if not model:
            if provider == "deepseek":
                model = "deepseek-chat"
            elif provider == "openai":
                model = "gpt-4o-mini"
            elif provider == "anthropic":
                model = "claude-3-5-haiku-latest"
        no_proxy_flag = args.no_proxy

    no_proxy_flag = no_proxy_flag or args.no_proxy

    # Print test config
    print(f"\n{BOLD}LLM API 连通性测试{RESET}\n")
    print(f"  Provider:   {GREEN}{provider}{RESET}")
    print(f"  Base URL:   {base_url}")
    print(f"  Model:      {model}")
    print(f"  API Key:    {api_key[:12]}...{api_key[-4:] if len(api_key) > 16 else '(未设置)'}" if api_key else f"  API Key:    {RED}(未设置){RESET}")
    print(f"  No Proxy:   {no_proxy_flag}")

    if not api_key:
        print(f"\n  {RED}错误: API Key 未设置{RESET}")
        print(f"  用法: --api-key sk-xxx  或  --use-env")
        sys.exit(1)

    # Run test
    print(f"\n  {YELLOW}正在连接...{RESET}")

    if provider in ("deepseek", "openai"):
        result = await test_openai_compatible(api_key, base_url, model, no_proxy_flag)
    else:
        result = await test_anthropic(api_key, model, no_proxy_flag)

    # Print result
    print()
    if result["ok"]:
        print(f"  {GREEN}{BOLD}✓ 连接成功{RESET}")
        print(f"  延迟:       {result['latency_ms']}ms")
        print(f"  模型:       {result['model']}")
        print(f"  响应:       {result['response']}")
        if "usage" in result:
            print(f"  Token 用量: {json.dumps(result['usage'])}")
    else:
        print(f"  {RED}{BOLD}✗ 连接失败{RESET}")
        print(f"  延迟:       {result['latency_ms']}ms")
        print(f"  错误信息:   {RED}{result['error']}{RESET}")
        print()
        print(f"  {YELLOW}常见排查:{RESET}")
        print(f"    1. API Key 是否正确？")
        print(f"    2. Base URL 是否可访问？(curl {base_url})")
        print(f"    3. 模型名称是否正确？({model})")
        print(f"    4. 是否需要配置代理或关闭代理？")
        print(f"    5. 网络防火墙是否放行？")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
