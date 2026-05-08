# 02 LLM Audit Latency & Accuracy

## 测试什么

Text Agent L3 层 —— 调用外部 LLM API 做深度审核。这是整个系统中最昂贵、延迟最高的一层。

当前后端: **DeepSeek Chat** (`deepseek-chat`)

## 测试配置

14 条标注文本（9 easy + 5 hard），覆盖 safe/toxic/politics/violence/gambling/porn。

## 测试结果

### Test 1: API 延迟（dry-run 估算）

未执行实际 API 调用。基于 DeepSeek API 文档和社区数据估算：

| 指标 | 估算值 |
|------|------|
| P50 | ~800ms |
| P95 | ~1500ms |
| P99 | ~2000ms |

### 成本估算

```
Provider: DeepSeek
Model: deepseek-chat
Per request tokens: ~300 (input + output)
Cost per request: ~$0.0006
Cost per 1M requests: ~$600
```

对比：如果 20% 的流量进入 L3（日均 1000 万条），每天 LLM 成本约 $1,200。

### 系统提示词

```
You are a content moderation expert. Analyze the given text and classify it.

Categories to check:
- toxic: insults, hate speech, harassment
- politics: politically sensitive content
- violence: threats, incitement to violence
- porn: sexually explicit content, solicitation
- gambling: gambling promotion, casino ads
- spam: advertisements, repetitive content, scams
- safe: normal, harmless content

Respond in JSON format only:
{"label": "<category>", "confidence": <0.0 to 1.0>, "reason": "<brief explanation>"}
```

### 超时处理

Text Agent 设置了 8 秒超时 + 回退到 BERT 结果：

```python
llm_result = await asyncio.wait_for(
    llm_auditor.audit(text, context),
    timeout=8.0)
# On timeout: fall back to BERT result
```

### 三层漏斗效果

| 层级 | 预估流量占比 | 单次成本 | 延迟 |
|------|------|------|------|
| L1 关键词 | 20-25% | $0 | <1ms |
| L2 BERT | 40-50% | $0.0001 | ~50ms |
| L3 LLM | 5-15% | **$0.002** | **500-2000ms** |

L3 是成本控制的终极目标 —— 每降低 1% 的 L3 调用率，日均 5000 万条就节省约 $1,000。
