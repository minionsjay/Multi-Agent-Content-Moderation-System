# 内容安全风控 Agent Prototype · SDD 开发规划

文档版本：V0.2  
开发范式：SDD（Spec-Driven Development）  
直接执行入口：[PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md)  
产品边界：[PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md](PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md)

> 本文档用于解释架构、数据契约和开发规划。Coding Agent 派任务时必须优先使用 `PROTOTYPE_SPEC_INDEX.md` 中的单个 Spec ID 或单个连调点。

## 1. 统一原则

1. 原 POC 已跑通的能力保留，包括 FastAPI、Gateway、LangGraph、图文审核、缓存、人审队列和基础反馈链路。
2. 新增 Prototype 不改变技术入口：仍使用 `python -m src.api`。
3. 图片审核按“图片转文本后进入文本识别”处理，不引入多模态大模型作为本阶段核心能力。
4. 决策枚举沿用原 POC：`pass / block / review`。
5. 风险类型先以 `fraud` 涉诈为示例，框架支持业务后续扩展。
6. 业务已有关键词和规则资产，不在 Prototype 中重新设计规则优化能力；只做规则加载、命中、Trace、调试、回放接口。
7. 模型训练不开发真实训练流程，只保留人工或定时触发接口。

## 2. 总体架构

```text
Content Input
  ↓
Gateway
  ├── text cache
  ├── image hash / url cache
  ├── keyword / rule gate
  ├── whitelist / exclusion
  └── semantic cache
       ↓ miss
Online Agent Graph
  ├── image_text_node
  ├── language_node
  ├── rule_node
  ├── risk_preclassify_node
  ├── algorithm_score_node
  ├── llm_judge_node
  └── decision_node
       ↓
Action
  ├── cache writeback
  ├── trace write
  ├── human review queue
  └── event record
       ↓
Sidecar Debug
  ├── review result
  ├── error analysis
  ├── replay evaluation
  └── training trigger hook
```

## 3. 建议目录

当前 `src/` 保留原 POC 能力。Prototype 新增能力建议放在独立目录，避免影响主链路：

```text
prototype/
├── graph/
│   ├── online_graph.py
│   ├── nodes.py
│   └── state.py
├── offline/
│   ├── error_analyzer.py
│   ├── replay_evaluator.py
│   └── training_trigger.py
├── ui/
│   ├── online_detection_page.py
│   ├── human_review_page.py
│   ├── trace_debug_page.py
│   └── replay_report_page.py
└── utils/
    ├── jsonl_store.py
    ├── schema_validator.py
    └── cost_estimator.py

rules/
├── rules_v0.json
└── candidate_rules.json

prompts/
├── llm_judge_prompt.txt
├── error_analysis_prompt.txt
└── evaluation_summary_prompt.txt

data/
├── sample_cases.jsonl
├── traces.jsonl
├── review_results.jsonl
├── error_analysis_results.jsonl
└── evaluation_report.json
```

如果直接复用 `src/` 中的已实现模块，必须保持本文档数据契约和接口兼容。

## 4. 模块划分

| 模块 | 目标 | 主要输出 |
|---|---|---|
| Module A：在线检测与人审回流 | 跑通输入、检测、Trace、人审 | `DetectionResult`、`TraceRecord`、`ReviewResult` |
| Module B：旁路调试与回放接口 | 跑通误判读取、分析、回放、训练触发接口 | `ErrorAnalysisResult`、`EvaluationReport`、`TrainingTriggerResult` |

模块间只通过 JSON / JSONL 契约协作。

## 5. 核心数据契约

### 5.1 CaseInput

```json
{
  "case_id": "case_001",
  "content": "加我电报，低价办证，包过",
  "image_url": "",
  "image_base64": "",
  "source": "manual_test",
  "country": "CN",
  "language": "zh",
  "domain": "example.com",
  "customer_type": "normal",
  "risk_type": "fraud",
  "algorithm_score": 0.76
}
```

| 字段 | 必填 | 说明 |
|---|---:|---|
| `case_id` | 否 | 不填自动生成 |
| `content` | 否 | 文本内容；纯图片 case 可为空 |
| `image_url` / `image_base64` | 否 | 图片输入 |
| `language` | 否 | 缺省自动识别 |
| `risk_type` | 否 | 示例为 `fraud` |
| `algorithm_score` | 否 | 业务算法分数或模拟分数 |

### 5.2 DetectionResult

```json
{
  "case_id": "case_001",
  "decision": "review",
  "risk_type": "fraud",
  "confidence": 0.86,
  "reason": "疑似涉诈引流，需要人工确认",
  "evidence": ["加我电报", "低价办证", "包过"],
  "matched_rules": ["fraud_001"],
  "llm_reason": "内容包含外部联系方式和非法服务承诺。",
  "fraud_features": {
    "external_contact": true,
    "contact_channel": ["Telegram"],
    "illegal_service": true,
    "service_type": "fake_document",
    "guarantee_terms": ["包过"],
    "evasion_terms": [],
    "payment_signal": false
  },
  "latency_ms": 1280,
  "llm_called": true,
  "cost_estimate": 0.002,
  "trace_id": "trace_001"
}
```

约束：

| 字段 | 取值 |
|---|---|
| `decision` | `pass` / `block` / `review` |
| `risk_type` | 先以 `fraud` 为示例，可扩展 |

### 5.3 FraudFeatures

```json
{
  "external_contact": false,
  "contact_channel": [],
  "illegal_service": false,
  "service_type": "unknown",
  "guarantee_terms": [],
  "evasion_terms": [],
  "payment_signal": false
}
```

`fraud` 是本 Prototype 的示例风险类型；非 fraud case 可以输出默认空对象。

### 5.4 TraceRecord

```json
{
  "case_id": "case_001",
  "trace_id": "trace_001",
  "input": {},
  "gateway_node": {},
  "image_text_node": {},
  "language_node": {},
  "rule_node": {},
  "risk_preclassify_node": {},
  "algorithm_score_node": {},
  "llm_judge_node": {},
  "decision_node": {},
  "runtime": {
    "latency_ms": 1280,
    "llm_called": true,
    "cost_estimate": 0.002
  },
  "version": {
    "rule_version": "v0.1",
    "prompt_version": "judge_v0.1"
  },
  "created_at": "2026-05-06 12:00:00"
}
```

节点失败时也必须写入节点输出，并标记 `error` 字段。

### 5.5 ReviewResult

```json
{
  "case_id": "case_001",
  "trace_id": "trace_001",
  "system_decision": "review",
  "system_risk_type": "fraud",
  "human_decision": "block",
  "human_risk_type": "fraud",
  "is_correct": true,
  "error_type": "true_positive",
  "human_reason": "明显引流办理非法证件",
  "reviewer": "auditor_001",
  "review_time": "2026-05-06 12:10:00"
}
```

`error_type` 计算：

| system_decision | human_decision | error_type |
|---|---|---|
| `pass` | `block` | `false_negative` |
| `block` / `review` | `pass` | `false_positive` |
| `block` / `review` | `block` | `true_positive` |
| `pass` | `pass` | `true_negative` |
| 风险类型不一致 | `block` | `category_error` |

### 5.6 RuleConfig

业务规则资产可以通过适配层进入以下结构：

```json
{
  "rule_id": "fraud_001",
  "risk_type": "fraud",
  "language": "zh",
  "keywords": ["加电报", "低价办证", "包过", "私聊"],
  "pattern": "",
  "exclude_keywords": ["提醒大家", "不要相信", "骗局"],
  "feature_logic": {},
  "action": "review",
  "severity": "high",
  "version": "v0.1",
  "source": "business_rules"
}
```

要求：

- `rules_v0.json` 和 `candidate_rules.json` 使用同一结构。
- `candidate_rules.json` 是调试候选配置，不代表自动上线。
- Prototype 不要求生成高质量业务规则，只要求可加载、可命中、可回放。

### 5.7 ErrorAnalysisResult

```json
{
  "case_id": "case_010",
  "trace_id": "trace_010",
  "error_type": "false_negative",
  "root_cause": "rule_missing",
  "affected_risk_type": "fraud",
  "analysis": "现有规则未覆盖该表达。",
  "suggested_debug_action": "check_rule_or_threshold",
  "candidate_terms": ["特殊证件", "渠道稳"],
  "created_at": "2026-05-06 12:30:00"
}
```

### 5.8 EvaluationReport

```json
{
  "generated_at": "2026-05-06 13:00:00",
  "old_rules_file": "rules/rules_v0.json",
  "candidate_rules_file": "rules/candidate_rules.json",
  "metrics": {
    "current": {},
    "candidate": {},
    "delta": {}
  },
  "summary": "候选配置仅用于调试回放，需人工确认后才能进入业务规则。"
}
```

必须至少包含：

- `total_cases`
- `violation_cases`
- `normal_cases`
- `false_negative_count`
- `false_positive_count`
- `recall`
- `precision`
- `estimated_human_review_count`
- `llm_call_count`

### 5.9 TrainingTriggerResult

```json
{
  "trigger_id": "train_trigger_001",
  "trigger_type": "manual",
  "status": "queued",
  "dataset_path": "data/review_results.jsonl",
  "note": "Prototype only reserves trigger interface; real training is not implemented."
}
```

训练接口只负责记录触发请求或回调占位，不执行真实训练。

## 6. 节点规格

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `gateway_node` | text / image | hot decision or miss | 可复用现有 Gateway |
| `image_text_node` | image_url / image_base64 | OCR text / image signal | 图片转文本信号 |
| `language_node` | content | language | 缺省 fallback `other` |
| `rule_node` | content, language, rules | matched rules | 复用业务规则资产 |
| `risk_preclassify_node` | content, language | risk candidates | `fraud` 为示例 |
| `algorithm_score_node` | algorithm_score | risk level | 业务算法分数 |
| `llm_judge_node` | case + context | label, reason, features | 疑难 case 调用 |
| `decision_node` | all nodes | pass/block/review | 沿用原 POC |
| `action_node` | result | cache/review/event/trace | 写回与记录 |

## 7. LLM 调用条件

建议调用：

- 规则未命中但算法分数中高。
- 规则命中但出现反诈提醒、上下文排除等冲突。
- `fraud` 风险证据不足。
- 多节点判断冲突。
- 图片 OCR 文本存在风险信号但规则无法判断。

可不调用：

- 高置信规则直接 `block`。
- 明确正常且低算法分数。
- 缓存命中。

## 8. 人审与旁路

`decision = review` 时进入人审队列。人审结果写入 `data/review_results.jsonl`，旁路模块读取 Trace + ReviewResult 做调试分析。

旁路当前目标：

- 识别误判类型。
- 给出基础归因和调试建议。
- 支持规则文件或阈值配置回放。
- 保留训练触发接口。

旁路当前不做：

- 自动上线规则。
- 真实模型训练。
- 复杂策略优化平台。

## 9. 最小交付链路

```text
S0.1 - S0.6
→ A1 JSONL 工具
→ A2 规则加载与检测
→ A5 算法分数节点
→ A7 决策聚合
→ A8 在线检测 Pipeline
→ A10 人审提交
→ B1 读取 Trace / Review
→ B2 误判类型识别
→ B8 回放评测接口
→ B9 训练触发接口
```

## 10. 演示脚本

1. 文本涉诈样本命中规则，输出 `review` 或 `block`。
2. 图片样本经 OCR 提取文本后进入文本检测。
3. 正常文本输出 `pass`。
4. 疑难文本调用 LLM，输出证据和 `fraud_features`。
5. `review` 样本进入人审，人工提交 `pass` 或 `block`。
6. 旁路读取 Trace + ReviewResult，输出误判类型。
7. 运行回放接口，生成 `evaluation_report.json`。
8. 调用训练触发接口，记录一次 `TrainingTriggerResult`。
