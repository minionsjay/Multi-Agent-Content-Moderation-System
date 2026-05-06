# 方案 03: ONNX 导出 BGE 模型

## 问题

PyTorch 模型推理有固定开销（Python 解释器开销、动态计算图检查、内存分配模式）。对于 batch_size=1 的 embedding 推理，这个开销占比很大。

```
PyTorch BGE: ~5ms/条
  ├── Python overhead:     ~1ms (20%)
  ├── 计算图检查:          ~0.5ms (10%)  
  └── 实际矩阵运算:        ~3.5ms (70%)
```

## 原理

ONNX Runtime 将模型导出为静态计算图，消除了 PyTorch 的动态图开销：

```
ONNX BGE: ~1.5ms/条 (3x 加速)
  ├── Python overhead:     ~0.1ms (7%)
  ├── 静态图执行:          ~0ms (0%)
  └── 实际矩阵运算:        ~1.4ms (93%)
```

ONNX Runtime 还可以：
- 图优化（算子融合、常量折叠）
- 多线程并行（`intra_op_num_threads`）
- 量化推理（FP16/INT8，进一步加速但精度略降）

## 导出步骤

```bash
# 1. 安装 onnxruntime
pip install onnxruntime onnx

# 2. 导出 BGE 模型为 ONNX
cd research/04_onnx_export
python export_bge_onnx.py

# 3. 验证 ONNX 模型
python export_bge_onnx.py --verify
```

导出后的模型文件：
```
poc/onnx_models/bge-small-zh-v1.5/
├── model.onnx      (~95 MB, ONNX 模型)
├── config.json     (tokenizer 配置)
├── tokenizer.json  (tokenizer 数据)
└── vocab.txt       (词汇表)
```

## 代码集成

```python
# embedder_onnx.py
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

class ONNXEmbedder:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=self._make_options()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _make_options(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4    # 4 线程并行
        opts.inter_op_num_threads = 2
        return opts

    def embed(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, return_tensors='np',
            truncation=True, max_length=512, padding=True
        )
        # BGE models: use [CLS] token or mean pooling
        outputs = self.session.run(None, dict(inputs))
        # Mean pooling over last hidden state
        hidden = outputs[0]  # [1, seq_len, 768]
        mask = inputs['attention_mask'][0]  # [seq_len]
        pooled = (hidden[0] * mask[:, None]).sum(axis=0) / mask.sum()
        # L2 normalize
        norm = np.linalg.norm(pooled)
        return (pooled / norm).tolist()
```

## 与现有 ONNX BERT 的复刻

项目中已有 `poc/src/skills/bert_onnx.py` (BERT 的 ONNX 加速)，BGE 的 ONNX 化可以完全复用同样的模式：

```
bert_onnx.py (已存在):
  - unitary/toxic-bert → ONNX → 2-3x CPU 加速
  - 出口: pytorch → onnx → onnxruntime

embedder_onnx.py (需新增):
  - BAAI/bge-small-zh-v1.5 → ONNX → 3x CPU 加速
  - 出口: same pattern
```

## 对比

| 指标 | PyTorch BGE | ONNX BGE | 提升 |
|------|------|------|------|
| 延迟 (单条) | ~5ms | ~1.5ms | 3.3x |
| 延迟 (批量 32) | ~50ms | ~20ms | 2.5x |
| 内存占用 | ~200 MB | ~150 MB | 25% 节省 |
| 首次加载 | ~500ms | ~200ms | 2.5x 快 |
| 多线程友好 | GIL 限制 | 释放 GIL | - |

## 局限性

1. ONNX 模型是静态的，不支持动态序列长度（需 padding 到固定长度）
2. 导出操作一次性但需要额外维护（模型更新时需重新导出）
3. 量化后精度可能略降（但 embedding 场景下影响很小）
