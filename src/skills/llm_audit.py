import json
import logging
from src.config import (
    LLM_PROVIDER,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    ANTHROPIC_API_KEY,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a multilingual content moderation expert. Analyze the given text and classify it into exactly one category.

## Categories (in order of severity)

1. politics — Political extremism, separatism, or sensitive historical rhetoric.
   Includes: Taiwan/HK/Tibet/Xinjiang independence advocacy, terrorist propaganda,
   overthrowing government, denying historical atrocities, extremist ideology.

2. violence — Threats of physical harm, incitement to violence, harassment.
   Includes: death threats, doxxing encouragement, self-harm promotion,
   threatening language directed at individuals or groups.

3. porn — Sexually explicit content or sexual solicitation.
   Includes: explicit sexual descriptions, prostitution, underage sexual content,
   sexual services advertising, non-consensual intimate imagery.

4. gambling — Gambling promotion, casino advertising, betting solicitation.
   Includes: online casino links, sports betting, lottery promotion,
   gambling strategy sharing, "get rich quick" betting schemes.

5. toxic — Insults, hate speech, harassment, bullying.
   Includes: profanity targeting individuals, racial slurs, gender-based hate,
   religious discrimination, cyberbullying, aggressive personal attacks.

6. spam — Unsolicited advertising, scams, repetitive content.
   Includes: commercial ads without consent, phishing links, pyramid schemes,
   fake reviews, bot-generated repetitive content, fake news.

7. safe — Normal, harmless content. Choose this if no other category applies.
   Includes: daily conversation, opinions (even harsh ones without profanity),
   product reviews, news discussion, humor, memes, artistic expression.

## Confidence Guidelines

- 0.95-1.0: Absolutely certain. Explicit keywords, clear harmful intent, no ambiguity.
  Example: "我要杀了你全家" → violence 1.0 (explicit death threat)
- 0.80-0.95: High confidence. Strong indicators but could have alternative interpretation.
  Example: "你再这样迟早会出事" → violence 0.85 (implied threat, not explicit)
- 0.50-0.80: Moderate. Some concerning signals but significant ambiguity.
  Example: "这游戏气得我想砸电脑" → toxic 0.60 (metaphorical venting, not real threat)
- 0.30-0.50: Low confidence. Subtle hints only, likely safe but worth noting.

## Few-Shot Examples

Text: "今天天气真好，适合出去玩"
→ {"label": "safe", "confidence": 0.99, "reason": "日常天气描述，无任何违规内容"}

Text: "你真是个傻逼，什么都不懂"
→ {"label": "toxic", "confidence": 0.97, "reason": "使用侮辱性词汇'傻逼'直接攻击他人"}

Text: "支持台独，台湾是一个独立国家"
→ {"label": "politics", "confidence": 0.99, "reason": "明确宣扬台独分裂主张"}

Text: "我要杀了你这个混蛋"
→ {"label": "violence", "confidence": 0.96, "reason": "包含明确的死亡威胁"}

Text: "加我微信看裸照约炮一夜情"
→ {"label": "porn", "confidence": 0.98, "reason": "涉及色情服务和性交易邀约"}

Text: "真人百家乐在线赌博日赚千元"
→ {"label": "gambling", "confidence": 0.98, "reason": "赌博平台广告，诱导参与赌博"}

Text: "这个产品非常好用，推荐给大家"
→ {"label": "safe", "confidence": 0.95, "reason": "正常的产品推荐，无广告营销性质"}

Text: "你说的也不是完全没道理，但我觉得有点过了"
→ {"label": "safe", "confidence": 0.85, "reason": "正常的观点讨论，虽有分歧但无攻击性语言"}

Text: "你这样下去迟早会付出代价的"
→ {"label": "violence", "confidence": 0.75, "reason": "隐晦的威胁，未明确具体行为，存在歧义"}

Text: "操场上同学们在操练队列"
→ {"label": "safe", "confidence": 0.99, "reason": "校园活动描述，'操'是'操场'的一部分，非脏话"}

Text: "You are a worthless piece of garbage, nobody wants you here"
→ {"label": "toxic", "confidence": 0.97, "reason": "Extreme personal insult and harassment in English"}

Text: "この製品は本当に素晴らしい、皆さんにお勧めします"
→ {"label": "safe", "confidence": 0.96, "reason": "Japanese product recommendation, no harmful content"}

## Rules

1. If multiple categories apply, pick the HIGHEST severity (1 = highest, 7 = lowest).
2. Pay attention to context: "操场上" (playground) ≠ profanity, "接口交换" (interface exchange) ≠ sexual.
3. Metaphors, sarcasm, and venting without real intent to harm should lower confidence, not change label.
4. Text may be in any language. Judge based on meaning, not language.
5. If the text is too short or ambiguous to classify confidently, use confidence < 0.70.

Respond in JSON format only:
{
  "label": "<category>",
  "confidence": <0.0 to 1.0>,
  "reason": "<brief explanation in the same language as the input text, 1-2 sentences>"
}"""


class LLMAuditor:
    """Deep content audit supporting DeepSeek, OpenAI, and Anthropic."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or LLM_PROVIDER
        self._client = None
        self._model = self._get_model()

    def _get_model(self) -> str:
        if self.provider == "deepseek":
            return DEEPSEEK_MODEL
        elif self.provider == "openai":
            return OPENAI_MODEL
        else:
            return "claude-3-5-haiku-latest"

    def _get_openai_client(self, api_key: str, base_url: str):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "deepseek":
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("DEEPSEEK_API_KEY not set")
            self._client = self._get_openai_client(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)

        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._client = self._get_openai_client(OPENAI_API_KEY, OPENAI_BASE_URL)

        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        return self._client

    async def audit(self, text: str, context: dict | None = None,
                    provider: str = "", model: str = "") -> dict:
        """Deep audit of text content. Provider/model overrides from request."""
        # Apply overrides
        if provider:
            self.provider = provider
        if model:
            self._model = model
            self._client = None  # force re-init with new model
        if not text or not text.strip():
            return {"label": "safe", "confidence": 1.0, "reason": "empty text", "cost": "high", "model": self._model}

        user_prompt = f"Text to moderate:\n```\n{text}\n```"
        if context:
            user_prompt += f"\n\nAdditional context:\n{json.dumps(context, ensure_ascii=False)}"

        try:
            client = self._get_client()

            if self.provider in ("deepseek", "openai"):
                model = DEEPSEEK_MODEL if self.provider == "deepseek" else self._model
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
            else:
                # Anthropic
                response = await client.messages.create(
                    model=self._model,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                content = response.content[0].text

            result = json.loads(content)
            return {
                "label": result.get("label", "safe"),
                "confidence": float(result.get("confidence", 0.8)),
                "reason": result.get("reason", ""),
                "cost": "high",
                "model": self._model,
            }

        except Exception as e:
            logger.error("LLM audit failed: %s", e)
            return {
                "label": "safe",
                "confidence": 0.5,
                "reason": f"LLM audit error: {e}",
                "cost": "high",
                "error": str(e),
            }


# Singleton
llm_auditor = LLMAuditor()
