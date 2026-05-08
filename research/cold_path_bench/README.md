# Cold Path Technology Benchmarks

> 逐层测试冷路径每一项技术：从 BERT 双后端、LLM 延迟，到 NSFW ViT、EasyOCR、Decision 规则、全流程端到端。

## 冷路径架构

```
Gateway 升级 → LangGraph (4 节点)

image_agent          text_agent              decision           action
┌──────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────────┐
│ 1. 图片获取   │    │ L1: AC自动机+分词 │    │ 1. 缓存复用 │    │ L0a 本地缓存  │
│ 2. NSFW ViT  │ →  │ L2: BERT/ONNX   │ → │ 2. 零容忍   │ → │ L0b Redis     │
│ 3. EasyOCR   │    │ L3: DeepSeek LLM │    │ 3. 聚合     │    │ L1  ChromaDB  │
└──────────────┘    └──────────────────┘    │ 4. 灰度区   │    │ 日志输出      │
                                            └────────────┘    └──────────────┘
```

## 测试结果总览

| # | 测试项 | 关键指标 | 结论 |
|------|------|------|------|
| 01 | BERT ONNX vs HF | ONNX 2.9x faster (59 vs 20 QPS) | ONNX 有精度问题，POC 用 HF |
| 02 | LLM Audit | ~$0.002/req, 500-2000ms | 最昂贵的一层，必须控制调用量 |
| 03 | NSFW ViT | POC 跳过 | 模型未下载，需 350MB |
| 04 | EasyOCR | 308ms/图，中文 100% | 对模糊极其敏感 |
| 05 | Decision Rules | 21/21 ✅ (Bug 已修复) | 4 个 Bug 已修复 |
| 06 | Cold Path Full | P50=1241ms, 100% L3 | 中文全走 L3，需多语言 BERT |

## 成本对比

| 路径 | 延迟 | 成本/条 | 占比 |
|------|------|------|------|
| 热路径 (Cache) | < 5ms | $0 | 35-50% |
| 热路径 (Keyword) | < 1ms | $0 | 15-20% |
| 温路径 (BERT L2) | ~50ms | $0.0001 | 30-40% |
| **冷路径 (LLM L3)** | **1000-2000ms** | **$0.002** | **5-15%** |
| 人工复核 | 分钟级 | $0.05 | 1-3% |

## 发现的问题

### 已修复 (POC)

1. **Decision Agent 灰度区 Bug** — 4 个规则逻辑错误，已全部修复（21/21 通过）
2. **Decision Agent toxic 关键词漏判** — Path 2 只处理零容忍类别，toxic 高置信被放行

### 待修复 (Phase 2)

1. **ONNX 精度问题** — 1/6 误分类 + 置信度偏低，需排查 softmax/标签映射
2. **中文全走 L3** — English-only BERT 对中文无效，需换多语言模型
3. **NSFW 模型未下载** — POC 无法验证图片审核准确率
4. **EasyOCR 中文模糊失效** — 2px 高斯模糊即无法识别
5. **100% LLM 调用率** — BERT 高置信短路条件未正常触发（待排查）
