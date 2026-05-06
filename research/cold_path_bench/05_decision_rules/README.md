# 05 Decision Agent Rule Engine

## 测试什么

Decision Agent 的规则引擎 —— 聚合 Text Agent + Image Agent 结果后，应用策略规则做最终决策。纯规则，无模型调用。

## 决策树

```
Path 1: Cache hit → 复用缓存结果

Path 2: keyword_confidence > 0.99
  ├── politics/violence → block (零容忍，硬覆盖)
  └── 其他 → block (高置信关键词)

Path 3: 聚合 text_result + image_result
  ├── 无 text → 纯图片判断 / fallback
  ├── label = safe → pass (永远不过灰度区)
  ├── image NSFW → label 升级为 unsafe
  ├── label = unsafe + conf < 0.3 → pass (低置信放过)
  ├── label = unsafe + conf [0.3, 0.7] → review (灰度区)
  └── label = unsafe + conf > 0.7 → block
```

## 测试结果：21/21 全部通过（修复后）

### 修复前（17/21）发现的 4 个 Bug

| Bug | 症状 | 修复 |
|------|------|------|
| 灰度区以下不处理 | conf=0.29 unsafe → block（应 pass） | 新增 `< GREY_ZONE_LOW` 分支 → pass |
| safe 被送灰度区 | label=safe conf=0.50 → review（应 pass） | safe 检查移到灰度区之前 |
| toxic 关键词被放行 | keyword=toxic conf=1.0 → pass（应 block） | Path 2 改为处理所有高置信关键词 |
| ONNX 误分类 | "beautiful day" → unsafe (ONNX) vs safe (HF) | 待修复 ONNX softmax |

### 灰度区设计

```
confidence ──────────────────────────────────────
  0.0      0.3              0.7           1.0
   │  pass  │    review      │    block    │
   │ (放过) │   (人工复核)    │   (拦截)    │
   └────────┴────────────────┴─────────────┘
      宁可放过                      宁可误杀
```

`[0.3, 0.7]` 区间是故意留的模糊地带 — 系统不确定的 case 交给人工，不自动决策。
