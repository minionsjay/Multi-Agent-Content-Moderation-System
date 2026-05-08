# 01 BERT ONNX vs HuggingFace Pipeline

## 测试什么

Text Agent L2 层在 CPU 上的两种推理后端：
- **ONNX Runtime**：静态计算图，图优化全开，4 线程并行
- **HuggingFace Pipeline**：动态图，PyTorch 原生推理

两者使用同一个模型 `unitary/toxic-bert`。

## 测试结果

### Test 1-2: 推理延迟对比

6 条英文文本，每条推理一次的延迟：

| 文本 | ONNX 延迟 | HF 延迟 | ONNX 标签 | HF 标签 |
|------|------|------|------|------|
| "you are a worthless piece of garbage" | 24ms | 55ms | unsafe (0.79) | unsafe (0.99) |
| "what a beautiful day to go outside" | 18ms | 51ms | **unsafe (0.85)** | **safe (0.99)** |
| "i will fucking kill you and your family" | 22ms | 50ms | unsafe (0.81) | unsafe (0.99) |
| "this tutorial is really helpful thanks" | 20ms | 48ms | safe (0.53) | safe (1.00) |
| "shut up you stupid moron nobody likes you" | 19ms | 51ms | unsafe (0.85) | unsafe (1.00) |
| "the sunset looks amazing" | 18ms | 48ms | safe (0.59) | safe (1.00) |

### Test 3: 准确率

```
ONNX vs HF 标签一致: 5/6 (83%)
不一致: "what a beautiful day" — ONNX判unsafe, HF判safe
```

**ONNX 问题分析**：
- ONNX 置信度整体偏低（0.53-0.85），HF 偏高（0.99-1.00）
- ONNX 的 softmax 计算可能存在问题（标签映射或数值精度）
- 这会导致更多文本进入 L3 LLM（因为 conf < 0.95 不满足短路条件）

### Test 4-5: 吞吐量

| 后端 | 单次延迟 | QPS | vs HF |
|------|------|------|------|
| HF Pipeline | 49ms | **20** | 1x |
| ONNX Runtime | 17ms | **59** | **2.9x** |

### Test 6: 中文跳过

3/9 的中文文本被识别为 CJK → 跳过 BERT L2 → 直接进 L3 LLM。这是 POC 阶段 English-only BERT 的已知局限。

## 结论

ONNX 在 CPU 上快 2.9x，但有一例误分类（safe → unsafe）。**在修复 ONNX 的 softmax/标签映射问题之前，POC 应使用 HF pipeline**（牺牲速度换精度）。

需要排查 `bert_onnx.py` 中的：
1. `_load_labels()` 是否正确加载了 `id2label` 映射
2. softmax 计算是否与 HF pipeline 一致
3. tokenizer 配置是否完全一致
