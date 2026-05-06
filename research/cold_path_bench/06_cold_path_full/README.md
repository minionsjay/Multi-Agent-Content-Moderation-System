# 06 Cold Path End-to-End

## 测试什么

完整 LangGraph 4 节点管线端到端测试：

```
image_agent → text_agent → decision → action
```

## 测试结果

### 延迟与决策

6 条文本，全部经过完整冷路径：

| 文本 | 延迟 | 决策 | 层级 | 链路 |
|------|------|------|------|------|
| "today is a beautiful day for a walk" (safe EN) | 1753ms | pass | L3_llm | L1→L2→L3→aggregate→action |
| "you are a worthless piece of shit" (toxic EN) | 1190ms | block | L3_llm | L1→L2→L3→aggregate→action |
| "今天天气真好适合出去玩" (safe CN) | 880ms | pass | L3_llm | L1→L2(skip)→L3→aggregate→action |
| "你真是个傻逼什么都不懂垃圾" (toxic CN) | 1572ms | block | L3_llm | L1→L2(skip)→L3→aggregate→action |
| "支持台独台湾是一个独立国家" (political CN) | 1141ms | block | L3_llm | L1→L2(skip)→L3→aggregate |
| "你说的也不是完全没道理但有点过了" (grey) | 1241ms | pass | L3_llm | L1→L2→L3→aggregate→action |

### 延迟分布

```
P50: 1241ms
P95: 1753ms
Avg: 1296ms
```

### 成本

```
L2 BERT calls: 0 × $0.0001 = $0.0000
L3 LLM  calls: 6 × $0.002  = $0.0120
LLM 调用率: 100% ← 关键问题！
```

### 100% LLM 调用率的原因

1. **中文文本跳过 L2** — English-only BERT 对中文无效，3/6 的中文文本直接跳过 L2 → L3
2. **BERT conf 不够高** — 即使英文文本的 BERT conf=0.99+，但 Text Agent 的 L2 短路条件要求 conf≥0.95 **且 label 明确**，当前代码中 HF pipeline 的 conf 都在 0.99+，应该能满足短路条件...

   实际上检查代码发现：中文文本因为 "toxic-bert" 在检查中被跳过 → `bert_skipped=True` → 不会触发 `should_skip_llm` → 进 L3。英文文本的 BERT conf 达到 0.99+，但...
   
   等等，重新看代码：`bert_classifier.should_skip_llm(bert_result)` 检查 `conf >= 0.95`。HF pipeline 返回的 conf 都是 0.994+。所以英文安全文本应该跳过 L3 才对。

   但实际结果显示 **全部进入了 L3**。这意味着 BERT 的置信度虽然看起来高，但实际没有触发短路。需要进一步排查 BERT 结果格式。这是一个需要关注的问题。

## 链路追踪

每个决策完整的 trace 链（9 步）：

```
L1_keyword → L2_bert → L3_llm → aggregate → final →
cache_L0a_store → cache_L0b_store → cache_L1_store → execute
```

热路径没有的步骤：
- `L2_bert` — BERT 模型推理
- `L3_llm` — LLM API 调用
- `cache_L1_store` — ChromaDB 语义缓存写入（15ms）

## 与热路径的对比

| 维度 | 热路径 | 冷路径 |
|------|------|------|
| 典型延迟 | < 5ms | 1000-2000ms |
| 成本 | $0 | $0.002/req |
| 模型调用 | 0 | BERT + LLM |
| 比例 | 70-80% | 20-30% |
| 瓶颈 | BGE Embedding | LLM API |
