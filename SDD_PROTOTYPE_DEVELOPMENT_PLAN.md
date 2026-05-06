# 内容安全风控 Agent Prototype · SDD 产品开发文档

文档版本：V0.1  
适用阶段：Prototype / 2 人 / 3 天  
开发范式：SDD（Spec-Driven Development，规格驱动开发）  
目标系统：文本内容安全风控 Agent Prototype  
当前定位：证明型原型，不做生产级系统

> **Coding Agent 阅读规则**
>
> 本文档不是 Coding Agent 的直接开发入口，不应用来自动拆任务或直接生成实现计划。
>
> 开发人员和 Coding Agent 必须优先阅读 [PROTOTYPE_SPEC_INDEX.md](./PROTOTYPE_SPEC_INDEX.md)。该文档按 2 个开发模块列出了 Spec 清单、开发顺序和连调点，是唯一的执行入口。
>
> 本文档的用途是提供完整 SDD 背景、业务解释、详细字段说明和验收依据。只有当某个 Spec 明确要求“参考 SDD 详细说明”时，Coding Agent 才应按需查阅本文档对应章节。

---

## 1. SDD 开发原则

本 Prototype 的开发不以“先写代码再补文档”为路径，而是先固化可执行规格，再按规格实现、验收和演示。

### 1.1 规格分层

| 层级 | 产物 | 作用 |
|---|---|---|
| Product Spec | 本文档 | 固化产品边界、能力模块、双人分工、开发顺序 |
| Data Contract Spec | JSON / JSONL Schema | 固化模块之间的输入输出，降低并行协作耦合 |
| Node Spec | Agent 节点规格 | 固化每个节点职责、输入、输出、失败兜底 |
| UI Spec | Streamlit 页面规格 | 固化演示页面、用户操作和验收路径 |
| Evaluation Spec | 离线回放规格 | 固化新旧规则评测指标和报告格式 |

### 1.2 Prototype 取舍

必须做：

- 文本输入到最终判断的 Agent Graph 可运行。
- 每个节点输出结构化 JSON。
- 每条 case 保存完整 Trace。
- 人工审核结果可回流。
- 旁路可识别漏检、误杀、类别错误。
- 旁路可生成候选规则 proposal。
- 候选规则经人工接受后才进入 `candidate_rules.json`。
- 离线回放可对比当前规则与候选规则。

明确不做：

- 多模态、OCR、ASR、相似图片检索。
- 真实小模型训练与真实线上灰度。
- 复杂权限、复杂前端、生产级数据库。
- 规则自动上线。

---

## 2. 总体交付目标

3 天结束时需要能演示一条完整闭环：

```text
文本 case
→ 在线检测 Agent Graph
→ 结构化系统判断 + Trace
→ 人工审核确认
→ 误判识别与归因
→ 规则 / 特征 proposal
→ 人工接受候选规则
→ 离线回放评测
→ 新旧规则效果对比报告
```

核心证明点：

- 在线链路从散乱 if-else 变成可编排、可追踪、可回放的 Agent Graph。
- 大模型只处理疑难 case，不作为全量实时层。
- 大模型能输出涉诈结构化特征。
- 人工审核从逐条判断逐步转向策略确认。
- 规则优化可以被离线数据验证。

---

## 3. 建议项目结构

在当前仓库已有 `src/`、`data/`、`tests/` 的基础上，Prototype 建议新增或调整如下目录。若复用现有模块，保持接口兼容即可。

```text
.
├── app.py
├── SDD_PROTOTYPE_DEVELOPMENT_PLAN.md
├── prototype/
│   ├── graph/
│   │   ├── online_graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── offline/
│   │   ├── error_analyzer.py
│   │   ├── rule_generator.py
│   │   └── replay_evaluator.py
│   ├── ui/
│   │   ├── online_detection_page.py
│   │   ├── human_review_page.py
│   │   ├── error_analysis_page.py
│   │   ├── rule_proposal_page.py
│   │   └── replay_report_page.py
│   └── utils/
│       ├── jsonl_store.py
│       ├── cost_estimator.py
│       └── schema_validator.py
├── rules/
│   ├── rules_v0.json
│   └── candidate_rules.json
├── prompts/
│   ├── llm_judge_prompt.txt
│   ├── error_analysis_prompt.txt
│   ├── rule_proposal_prompt.txt
│   └── evaluation_summary_prompt.txt
└── data/
    ├── sample_cases.jsonl
    ├── traces.jsonl
    ├── review_results.jsonl
    ├── error_analysis_results.jsonl
    └── evaluation_report.json
```

说明：

- 当前仓库已有 `src/graph.py`、`src/agents/`、`src/skills/`，可复用其思想，但 Prototype 建议放在 `prototype/` 下，避免影响现有 POC 主链路。
- `rules/` 和 `prompts/` 独立出来，便于演示规则版本与 Prompt 版本可追踪。
- `data/` 使用 JSONL / JSON，降低部署成本。

---

## 4. 双人分工总览

按业务能力模块分工，而不是按“前端/后端”切分。两个人各负责一条可独立验收的闭环，中间通过 JSON/JSONL 契约协作。

### 4.1 人员 A：在线检测与人审回流闭环

负责能力：

- 在线检测 Agent Graph。
- 语言识别、风险初判、规则检测、算法分数、大模型 Judge、决策聚合。
- Trace 生成与落盘。
- 在线检测页。
- 人工审核页。
- `review_results.jsonl` 写入。

核心交付：

- `run_online_detection(case_input) -> DetectionResult`
- `traces.jsonl`
- `submit_review(review_input) -> ReviewResult`
- 在线检测页、人工审核页可演示。

### 4.2 人员 B：旁路分析与规则回放闭环

负责能力：

- 读取 `review_results.jsonl` 和 `traces.jsonl`。
- 误判类型识别。
- 大模型归因分析。
- 候选规则 / 特征规则生成。
- 候选规则人工确认机制。
- 离线回放评测。
- 误判分析页、规则建议页、回放报告页。

核心交付：

- `analyze_errors(review_results, traces) -> ErrorAnalysisResult[]`
- `generate_rule_proposals(error_analysis) -> RuleProposal[]`
- `accept_rule_proposal(proposal) -> candidate_rules.json`
- `run_replay_evaluation(samples, labels, old_rules, candidate_rules) -> EvaluationReport`

### 4.3 双方共享契约

两人必须优先共同确认以下文件结构：

| 契约 | Owner | Consumer |
|---|---|---|
| `CaseInput` | A | A / B |
| `DetectionResult` | A | A / B |
| `TraceRecord` | A | A / B |
| `ReviewResult` | A | B |
| `ErrorAnalysisResult` | B | B / UI |
| `RuleProposal` | B | B / 回放 |
| `RuleConfig` | A+B | A / B |
| `EvaluationReport` | B | B / 负责人演示 |

---

## 5. Spec 开发顺序

### 5.1 Day 0 / 开发前 2 小时：契约冻结

目标：先冻结数据契约和文件路径，保证两人可以并行。

| Spec | Owner | 输出 |
|---|---|---|
| S0-01 数据契约 | A+B | Python dataclass / Pydantic model 或 schema 文档 |
| S0-02 文件存储契约 | A+B | JSONL 读写工具、固定文件路径 |
| S0-03 初始规则与样本 | A+B | `rules_v0.json`、至少 20 条启动样本 |

验收：

- 两人可以在本地读取同一份 `sample_cases.jsonl`。
- A 写出的 `traces.jsonl` 和 `review_results.jsonl`，B 可以无转换读取。
- 所有 ID 命名规则统一：`case_xxx`、`trace_xxx`、`proposal_xxx`。

### 5.2 Day 1：在线检测链路优先

A 主线：

1. S1-01 `CaseState` 与 `CaseInput`。
2. S1-02 语言识别节点。
3. S1-03 风险类型初判节点。
4. S1-04 规则检测节点。
5. S1-05 算法分数节点。
6. S1-06 大模型 Judge 节点，允许先用 mock LLM。
7. S1-07 决策聚合节点。
8. S1-08 Trace 落盘。
9. S1-09 在线检测页面。

B 并行：

1. S3-01 误判识别纯函数，先基于 mock `review_results.jsonl` 开发。
2. S3-02 回放评测指标计算纯函数，先只跑规则命中，不依赖 A 的 Graph。
3. S5-01 准备 60 条演示样本初版。

Day 1 共同验收：

- 输入“加我电报，低价办证，包过”可以输出 `need_human_review`。
- 生成 `trace_id` 并写入 `data/traces.jsonl`。
- 页面可展开每个节点输出。

### 5.3 Day 2：人工审核 + 误判分析闭环

A 主线：

1. S2-01 待审核 case 列表。
2. S2-02 人工审核提交。
3. S2-03 `review_results.jsonl` 写入。
4. S2-04 人工审核页。

B 主线：

1. S3-03 读取真实 `review_results.jsonl`。
2. S3-04 漏检 / 误杀 / 类别错误识别。
3. S3-05 大模型归因分析，允许 mock fallback。
4. S3-06 误判分析页。

Day 2 共同验收：

- 样本 2 可人工确认违规，进入 `review_results.jsonl`。
- 样本 3 可人工确认正常，并被识别为 `false_positive`。
- 误判分析页展示 root_cause、suggested_fix、candidate_terms。

### 5.4 Day 3：规则建议 + 离线回放 + 演示

A 主线：

1. S2-05 补齐在线页和人审页演示体验。
2. S1-10 修复 Trace、成本、耗时展示。
3. S5-02 支持从样本集中选择 case 运行。

B 主线：

1. S4-01 规则 proposal 生成。
2. S4-02 规则建议页。
3. S4-03 人工接受 / 拒绝 / 编辑后接受。
4. S4-04 写入 `candidate_rules.json`。
5. S5-03 离线回放评测。
6. S5-04 回放报告页。

Day 3 共同验收：

- 接受候选规则后，`candidate_rules.json` 有新增 proposal。
- 回放报告展示当前规则 vs 候选规则指标变化。
- 完成 5 步领导演示脚本。

---

## 6. 核心数据契约

### 6.1 CaseInput

```json
{
  "case_id": "case_001",
  "content": "加我电报，低价办证，包过",
  "source": "service_log_sample",
  "country": "CN",
  "language": "zh",
  "domain": "abc.com",
  "customer_type": "normal",
  "algorithm_score": 0.76,
  "sample_reason": "random_sample"
}
```

字段要求：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| case_id | string | 否 | 自动生成 | case 唯一 ID |
| content | string | 是 | - | 待检测文本 |
| source | string | 否 | manual_input | 来源 |
| country | string | 否 | unknown | 国家 / 站点 |
| language | string | 否 | 自动识别 | zh / en / id / other |
| domain | string | 否 | empty | 抽样优化预留 |
| customer_type | string | 否 | normal | normal / key_account |
| algorithm_score | float | 否 | 0.0 | 模拟小模型分数 |
| sample_reason | string | 否 | manual_test | 抽样原因 |

### 6.2 DetectionResult

```json
{
  "case_id": "case_001",
  "final_decision": "need_human_review",
  "risk_type": "fraud",
  "confidence": 0.86,
  "evidence": ["加我电报", "低价办证", "包过"],
  "matched_rules": ["fraud_001"],
  "llm_reason": "内容疑似引导用户通过外部联系方式办理非法证件。",
  "fraud_features": {
    "external_contact": true,
    "contact_channel": ["Telegram"],
    "illegal_service": true,
    "service_type": "fake_document",
    "guarantee_terms": ["包过"],
    "evasion_terms": [],
    "payment_signal": false
  },
  "need_human_review": true,
  "latency_ms": 1280,
  "llm_called": true,
  "cost_estimate": 0.002,
  "trace_id": "trace_001"
}
```

枚举约束：

| 字段 | 可选值 |
|---|---|
| final_decision | pass / suspected_violation / need_human_review |
| risk_type | fraud / porn / hate_or_abuse / unknown |

### 6.3 FraudFeatures

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

约束：

- `risk_type = fraud` 时必须输出完整对象，字段不能缺失。
- 非 fraud case 也可输出默认空对象，方便 UI 和回放复用。

### 6.4 TraceRecord

```json
{
  "case_id": "case_001",
  "trace_id": "trace_001",
  "input": {},
  "language_node": {},
  "risk_preclassify_node": {},
  "rule_node": {},
  "algorithm_score_node": {},
  "llm_judge_node": {},
  "final_decision": {},
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

验收：

- 每次检测必须写入一行 JSONL。
- 节点失败时也必须写入节点输出，并标记 `error` 字段。
- `trace_id` 必须能从 UI 关联到完整 Trace。

### 6.5 ReviewResult

```json
{
  "case_id": "case_001",
  "trace_id": "trace_001",
  "system_decision": "need_human_review",
  "system_risk_type": "fraud",
  "human_decision": "violation",
  "human_risk_type": "fraud",
  "is_correct": true,
  "error_type": "true_positive",
  "human_reason": "明显引流办理非法证件",
  "reviewer": "auditor_001",
  "review_time": "2026-05-06 12:10:00"
}
```

`error_type` 计算规则：

| system_decision | human_decision | error_type |
|---|---|---|
| pass | violation | false_negative |
| suspected_violation / need_human_review | normal | false_positive |
| suspected_violation / need_human_review | violation | true_positive |
| pass | normal | true_negative |
| risk_type 不一致 | violation | category_error |

### 6.6 ErrorAnalysisResult

```json
{
  "case_id": "case_010",
  "trace_id": "trace_010",
  "error_type": "false_negative",
  "root_cause": "rule_missing",
  "affected_risk_type": "fraud",
  "analysis": "该内容使用了规避表达，现有规则未覆盖。",
  "suggested_fix": "add_feature_rule",
  "candidate_terms": ["特殊证件", "渠道稳", "速度快", "私我"],
  "candidate_features": {
    "external_contact": true,
    "illegal_service_hint": true,
    "evasion_expression": true
  },
  "problematic_rule_id": null,
  "created_at": "2026-05-06 12:30:00"
}
```

枚举约束：

| 字段 | 可选值 |
|---|---|
| root_cause | rule_missing / keyword_too_broad / threshold_too_high / threshold_too_low / llm_judge_error / context_missing / language_mismatch / category_confusion / policy_gap |
| suggested_fix | add_keyword_rule / modify_keyword_rule / add_context_exclusion / add_fraud_feature_rule / adjust_severity / add_review_condition |

### 6.7 RuleConfig

```json
{
  "rule_id": "fraud_001",
  "risk_type": "fraud",
  "language": "zh",
  "keywords": ["加电报", "低价办证", "包过", "私聊"],
  "pattern": "",
  "exclude_keywords": ["提醒大家", "不要相信", "骗局"],
  "feature_logic": {},
  "action": "flag",
  "severity": "high",
  "version": "v0.1",
  "source": "rules_v0"
}
```

Prototype 要求：

- `rules_v0.json` 和 `candidate_rules.json` 使用同一结构。
- 规则检测逻辑必须能接受任一规则文件路径。
- `exclude_keywords` 可以先实现为简单上下文排除。
- `feature_logic` 可先保存，不强制完整执行；离线回放至少支持关键词和排除词。

### 6.8 RuleProposal

```json
{
  "proposal_id": "proposal_001",
  "source_case_ids": ["case_010"],
  "proposal_type": "add_fraud_feature_rule",
  "risk_type": "fraud",
  "language": "zh",
  "candidate_keywords": ["特殊证件", "渠道稳", "速度快", "私我"],
  "candidate_features": {
    "external_contact": true,
    "illegal_service": true,
    "evasion_expression": true
  },
  "suggested_logic": "当内容同时出现外部联系意图、非法服务暗示和规避表达时，进入人工复核。",
  "expected_effect": "提升涉诈引流类召回率",
  "potential_risk": "可能误伤普通办事咨询或反诈提醒内容",
  "recommended_action": "进入离线回放验证",
  "need_human_approval": true,
  "status": "pending"
}
```

状态流转：

```text
pending → accepted / rejected / edited_accepted
```

接受后写入 `rules/candidate_rules.json`，但不得修改 `rules/rules_v0.json`。

### 6.9 EvaluationReport

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
  "new_hit_cases": ["case_010"],
  "new_false_positive_cases": ["case_015"],
  "summary": "候选规则提升欺诈召回，但带来少量误杀，建议命中后进入人工复核。"
}
```

必须包含指标：

- `total_cases`
- `violation_cases`
- `normal_cases`
- `hit_violation_count`
- `false_negative_count`
- `false_positive_count`
- `recall`
- `precision`
- `estimated_human_review_count`
- `llm_call_count`
- `new_hit_cases`
- `new_false_positive_cases`

---

## 7. 能力模块 Spec

## 7.1 M1 在线检测模块

Owner：人员 A  
核心目标：跑通文本检测 Agent Graph，输出系统判断和完整 Trace。

### 7.1.1 接口

```python
def run_online_detection(case_input: dict) -> dict:
    """输入 CaseInput，返回 DetectionResult，并写入 TraceRecord。"""
```

输入：`CaseInput`  
输出：`DetectionResult`  
副作用：append `data/traces.jsonl`

### 7.1.2 子节点

| 节点 | 输入 | 输出 | 失败兜底 |
|---|---|---|---|
| language_detection | content, optional language | language, confidence | language=other |
| risk_preclassify | content, language | risk_candidates, reason | unknown |
| rule_check | content, language, rules file | hit, matched_rules, matched_terms, severity | hit=false |
| algorithm_score_check | algorithm_score | risk_level, suggested_next_step | score=0.0, low |
| llm_judge | content, context | judge_result, risk_type, evidence, fraud_features | uncertain + need_human_review |
| decision_aggregator | all node outputs | final_decision, confidence, reason | need_human_review |

### 7.1.3 LLM 调用条件

必须调用：

- 规则未命中且 `algorithm_score >= 0.5`。
- 多语种文本：`language not in ["zh", "en"]` 且非低风险。
- 规则命中但内容包含反诈提醒上下文。
- fraud 风险证据不足。

可不调用：

- 规则强命中高危且证据充分，直接进入人工确认。
- 规则未命中且 `algorithm_score < 0.5` 且无风险迹象。

### 7.1.4 验收标准

- 输入样本 1 返回 `need_human_review`，命中 `fraud_001`。
- 输入样本 2 规则未命中但调用 LLM，并提取 fraud_features。
- 输入样本 3 命中规则，但 LLM 或决策理由能表达反诈提醒上下文，最终仍进入人工确认。
- 输入样本 4 识别英文并提取 Telegram、fake documents。
- 输入样本 5 不应判为真实暴力风险，可 pass 或低风险。
- 每条执行都写入 `traces.jsonl`。

---

## 7.2 M2 Trace 追踪模块

Owner：人员 A  
核心目标：每条 case 的每个节点可查看、可回放、可用于旁路分析。

### 7.2.1 接口

```python
def append_trace(trace: dict) -> None:
    """追加一条 TraceRecord 到 data/traces.jsonl。"""

def get_trace(trace_id: str) -> dict | None:
    """按 trace_id 查询 TraceRecord。"""

def list_traces(limit: int = 100) -> list[dict]:
    """返回最近 trace 列表，供 UI 和人审页使用。"""
```

### 7.2.2 验收标准

- 每条 case 有 `trace_id`。
- Trace 包含 input、language、risk_preclassify、rule、algorithm_score、llm_judge、final_decision、runtime、version。
- UI 可展开查看完整 JSON。
- B 的旁路分析可以通过 `case_id` 或 `trace_id` 关联系统判断。

---

## 7.3 M3 人工审核回流模块

Owner：人员 A  
核心目标：让审核员确认或修正系统判断，写入旁路可消费的人工标签。

### 7.3.1 接口

```python
def submit_review(review_input: dict) -> dict:
    """输入人工审核表单，补齐 error_type/is_correct/review_time 后写入 review_results.jsonl。"""
```

输入字段：

- `case_id`
- `trace_id`
- `system_decision`
- `system_risk_type`
- `human_decision`
- `human_risk_type`
- `human_reason`
- `reviewer`

输出：`ReviewResult`

### 7.3.2 页面

人工审核页必须展示：

- 原始内容。
- 系统最终判断。
- 风险类型。
- 置信度。
- 证据片段。
- 命中规则。
- 大模型理由。
- 涉诈特征。
- 完整 Trace。

人工操作：

- 确认违规。
- 确认正常。
- 修正风险类别。
- 填写人工原因。
- 提交审核结果。

### 7.3.3 验收标准

- 提交后 `review_results.jsonl` 新增一行。
- 自动计算 `error_type`。
- `risk_type` 不一致时能标记 `category_error`。
- B 的误判分析模块无需额外清洗即可读取。

---

## 7.4 M4 旁路误判分析模块

Owner：人员 B  
核心目标：从人工反馈中识别漏检、误杀、类别错误，并归因。

### 7.4.1 接口

```python
def analyze_errors(
    review_results_path: str = "data/review_results.jsonl",
    traces_path: str = "data/traces.jsonl",
    output_path: str = "data/error_analysis_results.jsonl",
) -> list[dict]:
    """读取人工审核与 Trace，输出误判分析并落盘。"""
```

输入：

- `ReviewResult[]`
- `TraceRecord[]`

输出：

- `ErrorAnalysisResult[]`

副作用：

- append 或 rewrite `data/error_analysis_results.jsonl`

### 7.4.2 触发规则

| 条件 | 分析类型 |
|---|---|
| `error_type=false_negative` | 漏检分析 |
| `error_type=false_positive` | 误杀分析 |
| `error_type=category_error` | 类别错误分析 |
| `true_positive/true_negative` | 可跳过或用于统计 |

### 7.4.3 LLM 归因输出约束

必须输出 JSON：

```json
{
  "root_cause": "rule_missing",
  "analysis": "现有规则没有覆盖特殊证件、渠道稳等表达。",
  "suggested_fix": "add_fraud_feature_rule",
  "candidate_terms": ["特殊证件", "渠道稳"],
  "candidate_features": {
    "external_contact": true,
    "illegal_service_hint": true,
    "evasion_expression": true
  }
}
```

解析失败兜底：

```json
{
  "root_cause": "policy_gap",
  "analysis": "LLM output parse failed",
  "suggested_fix": "add_review_condition",
  "candidate_terms": [],
  "candidate_features": {}
}
```

### 7.4.4 验收标准

- 样本 2 如果系统 pass、人工 violation，应识别为 `false_negative`。
- 样本 3 系统风险、人工 normal，应识别为 `false_positive`。
- 输出包含 `root_cause`、`suggested_fix`、`candidate_terms`。
- 误判分析页可展示漏检列表、误杀列表、类别错误列表。

---

## 7.5 M5 规则建议模块

Owner：人员 B  
核心目标：把误判归因转为候选规则或特征规则 proposal，人工审批后才进入候选规则集。

### 7.5.1 接口

```python
def generate_rule_proposals(
    error_analysis_path: str = "data/error_analysis_results.jsonl",
) -> list[dict]:
    """基于误判分析生成 RuleProposal。"""

def accept_rule_proposal(proposal: dict, edited_rule: dict | None = None) -> dict:
    """接受或编辑后接受 proposal，并写入 rules/candidate_rules.json。"""

def reject_rule_proposal(proposal_id: str, reason: str = "") -> dict:
    """拒绝 proposal，更新 proposal 状态。"""
```

### 7.5.2 proposal 到规则的转换

| proposal_type | candidate rule 生成方式 |
|---|---|
| add_keyword_rule | 新增 `keywords` |
| modify_keyword_rule | 复制原 rule 并修改 keywords / exclude_keywords |
| add_context_exclusion | 新增或更新 `exclude_keywords` |
| add_fraud_feature_rule | 写入 `feature_logic`，同时可加入关键词组合 |
| adjust_severity | 修改 severity |
| add_review_condition | action=review |

### 7.5.3 验收标准

- proposal 列表可在页面展示。
- 人工可接受、拒绝、编辑后接受。
- 接受后只写 `candidate_rules.json`。
- `rules_v0.json` 不被修改。
- 每条 candidate rule 保留来源 proposal 和 source_case_ids。

---

## 7.6 M6 离线回放评测模块

Owner：人员 B  
核心目标：用同一批样本对比当前规则和候选规则效果。

### 7.6.1 接口

```python
def run_replay_evaluation(
    sample_cases_path: str = "data/sample_cases.jsonl",
    old_rules_path: str = "rules/rules_v0.json",
    candidate_rules_path: str = "rules/candidate_rules.json",
    output_path: str = "data/evaluation_report.json",
) -> dict:
    """运行新旧规则离线回放，输出 EvaluationReport。"""
```

### 7.6.2 回放范围

Prototype 回放以规则逻辑为主：

- 读取样本真实标签：`label=violation/normal`。
- 分别运行 `rules_v0.json` 和 `candidate_rules.json`。
- 计算命中违规、漏检、误杀、召回、精准、预计人审量。
- `llm_call_count` 可按规则未定且 `algorithm_score>=0.5` 的样本估算。

### 7.6.3 指标定义

```text
recall = hit_violation_count / violation_cases
precision = hit_violation_count / (hit_violation_count + false_positive_count)
false_negative_count = violation_cases - hit_violation_count
estimated_human_review_count = 命中规则或需要 LLM / 人审的样本数
```

### 7.6.4 验收标准

- 回放报告页能显示当前规则、候选规则、变化量。
- 能列出 `new_hit_cases`。
- 能列出 `new_false_positive_cases`。
- 生成 `data/evaluation_report.json`。
- 总结建议不鼓励自动上线，建议人工复核或小范围验证。

---

## 8. UI Spec

使用 Streamlit。页面可以放在单个 `app.py` 里用 tabs，也可以拆成 `prototype/ui/` 模块。

### 8.1 在线检测页

Owner：A

控件：

- 文本输入框。
- 国家选择。
- 语言选择，可空。
- domain 输入。
- customer_type 选择。
- algorithm_score 输入。
- 运行检测按钮。

展示：

- 最终判断。
- 风险类型。
- 置信度。
- 证据片段。
- 命中规则。
- 大模型理由。
- 涉诈特征。
- 耗时 / 成本 / 是否调用 LLM。
- Trace 展开区。

### 8.2 人工审核页

Owner：A

控件：

- 待审核 case 列表。
- 人工判断：违规 / 正常。
- 风险类别修正。
- 人工原因输入。
- 提交按钮。

展示：

- 原始内容。
- 系统判断。
- 证据、规则、LLM 理由、fraud_features。
- 完整 Trace。

### 8.3 误判分析页

Owner：B

控件：

- 分析人工反馈按钮。
- error_type 筛选。

展示：

- 漏检列表。
- 误杀列表。
- 类别错误列表。
- root_cause。
- suggested_fix。
- candidate_terms。

### 8.4 规则建议页

Owner：B

控件：

- 生成候选规则按钮。
- 接受 / 拒绝 / 编辑后接受。

展示：

- proposal_type。
- risk_type。
- candidate_keywords。
- candidate_features。
- suggested_logic。
- expected_effect。
- potential_risk。

### 8.5 离线回放报告页

Owner：B

控件：

- 运行回放按钮。

展示：

- 当前规则 vs 候选规则指标表。
- 指标变化。
- 新增命中样本。
- 新增误伤样本。
- 预计人工审核量变化。
- 大模型调用次数变化。
- Agent 总结建议。

---

## 9. Prompt Spec

### 9.1 LLM Judge Prompt

文件：`prompts/llm_judge_prompt.txt`  
Owner：A

要求：

- 输入 content、language、risk_candidates、matched_rules、algorithm_score。
- 输出 JSON，不允许散文。
- fraud case 必须输出 fraud_features。
- 解析失败必须兜底为 uncertain。

输出 Schema：

```json
{
  "judge_result": "pass",
  "risk_type": "unknown",
  "confidence": 0.0,
  "evidence": [],
  "reason": "",
  "need_human_review": true,
  "fraud_features": {
    "external_contact": false,
    "contact_channel": [],
    "illegal_service": false,
    "service_type": "unknown",
    "guarantee_terms": [],
    "evasion_terms": [],
    "payment_signal": false
  }
}
```

### 9.2 Error Analysis Prompt

文件：`prompts/error_analysis_prompt.txt`  
Owner：B

要求：

- 输入原文、系统判断、Trace 摘要、人工判断。
- 输出 root_cause、analysis、suggested_fix、candidate_terms、candidate_features。
- 区分漏检与误杀。

### 9.3 Rule Proposal Prompt

文件：`prompts/rule_proposal_prompt.txt`  
Owner：B

要求：

- 输入多条 ErrorAnalysisResult。
- 输出 RuleProposal JSON。
- 必须声明 potential_risk。
- 必须声明 need_human_approval=true。

### 9.4 Evaluation Summary Prompt

文件：`prompts/evaluation_summary_prompt.txt`  
Owner：B

要求：

- 输入新旧指标变化。
- 输出业务可读总结。
- 禁止给出“自动上线”建议，只能建议人工复核、灰度验证或继续观察。

---

## 10. 测试与验收 Spec

### 10.1 单元测试建议

| 测试 | Owner | 验收点 |
|---|---|---|
| `test_rule_match.py` | A | 关键词命中、排除词、版本输出 |
| `test_language_detection.py` | A | zh/en/id/other |
| `test_decision_aggregator.py` | A | 冲突进入人审、低风险 pass |
| `test_review_error_type.py` | A | false_positive/false_negative/category_error |
| `test_error_analyzer.py` | B | 误判归因结构完整 |
| `test_rule_generator.py` | B | proposal 转 candidate rule |
| `test_replay_evaluator.py` | B | 指标计算正确 |

### 10.2 端到端验收样本

| 样本 | 文本 | 预期 |
|---|---|---|
| D1 | 加我电报，低价办证，包过 | 命中 fraud_001，need_human_review，提取涉诈特征 |
| D2 | 有需要特殊证件的可以私我，渠道稳，速度快。 | 规则未命中，算法中高，LLM 判 fraud |
| D3 | 这篇新闻提醒大家不要相信低价办证骗局。 | 规则命中但上下文为反诈提醒，人工 normal，false_positive |
| D4 | Contact me on Telegram for fake documents. | en，LLM fraud，提取 Telegram/fake documents |
| D5 | 游戏里这个角色杀疯了，太爽了。 | pass 或低风险，不判真实暴力 |

### 10.3 业务验收

必须向负责人证明：

- 可以看到为什么判风险：规则、证据、LLM 理由、fraud_features。
- 可以看到成本和性能：latency、llm_called、cost_estimate。
- 可以看到人工反馈如何变成规则建议。
- 可以看到候选规则在离线回放中的收益与副作用。
- 可以看到系统没有自动上线规则，仍保留人工审批。

---

## 11. 演示脚本

### Step 1：规则直接命中

输入：

```text
加我电报，低价办证，包过
```

展示：

- 命中 `fraud_001`。
- 输出 `need_human_review`。
- 展开 Trace。
- 展示 fraud_features、耗时、成本。

讲解点：

- 明确风险由前两层快速处理，不需要所有内容全量走大模型。

### Step 2：规则漏检，大模型补充

输入：

```text
有需要特殊证件的可以私我，渠道稳，速度快。
```

展示：

- 规则未命中。
- `algorithm_score` 中高。
- 调用 LLM。
- 提取“特殊证件、私我、渠道稳、速度快”等特征。
- 人工确认违规。

讲解点：

- 大模型层提升疑难 case 的召回和结构化分析质量。

### Step 3：规则误杀，大模型纠偏

输入：

```text
这篇新闻提醒大家不要相信低价办证骗局。
```

展示：

- 规则命中“低价办证”。
- 系统保守进入人工确认。
- 人工确认正常。
- 旁路识别 `false_positive`。

讲解点：

- 规则不是简单加关键词，需要上下文排除和人审反馈。

### Step 4：旁路规则优化

展示：

- `false_negative` 生成新增涉诈特征规则。
- `false_positive` 生成上下文排除建议。
- 人工接受 proposal。
- 写入 `candidate_rules.json`。

讲解点：

- 旁路 Agent 把人工经验沉淀成可评测的策略资产。

### Step 5：离线回放报告

展示：

- 当前规则 vs 候选规则。
- 召回、精准、漏检、误杀、人审量、LLM 调用量变化。
- Agent 总结建议。

讲解点：

- 规则优化不靠拍脑袋，通过回放数据验证。

---

## 12. 里程碑检查清单

### Day 1 结束

- [ ] 在线检测页可运行。
- [ ] 规则库可读取。
- [ ] 至少 5 条核心样本可跑通。
- [ ] 每条 case 写入 Trace。
- [ ] LLM 可 mock 或真实调用。
- [ ] B 可以读取 A 产出的 Trace 格式。

### Day 2 结束

- [ ] 人工审核页可提交。
- [ ] `review_results.jsonl` 可写入。
- [ ] false_positive / false_negative / category_error 可识别。
- [ ] 误判分析页可展示归因结果。
- [ ] 至少跑通样本 2 和样本 3 的回流。

### Day 3 结束

- [ ] 规则建议页可生成 proposal。
- [ ] proposal 可接受 / 拒绝 / 编辑后接受。
- [ ] 接受后写入 `candidate_rules.json`。
- [ ] 离线回放报告可生成。
- [ ] 60 条样本准备完成。
- [ ] 5 步演示脚本可连续演示。

---

## 13. 风险与降级方案

| 风险 | 影响 | 降级方案 |
|---|---|---|
| LLM API 不稳定 | 在线 Judge / 归因 / 总结不可用 | 提供 deterministic mock LLM，按关键词返回固定 JSON |
| LangGraph 接入耗时 | Day 1 延误 | 先用顺序函数模拟 Graph，接口保持一致，Day 2 再替换 |
| Streamlit 页面耗时 | 演示体验不足 | 单 `app.py` + tabs，优先展示数据闭环 |
| 样本不足 | 回放报告不可信 | 先保证 60 条演示样本覆盖误杀、漏检、多语种 |
| candidate rule 太复杂 | 回放难实现 | Day 3 只执行 keywords + exclude_keywords，feature_logic 先保存和展示 |
| 两人接口不一致 | 集成阻塞 | Day 0 冻结 JSON 契约，所有模块只通过文件和函数接口协作 |

---

## 14. 最终交付清单

- 可运行本地 Prototype。
- `app.py` 或 Streamlit UI 入口。
- 在线检测页。
- 人工审核页。
- 误判分析页。
- 规则建议页。
- 离线回放报告页。
- `data/sample_cases.jsonl`，至少 60 条。
- `rules/rules_v0.json`。
- `rules/candidate_rules.json`。
- `data/traces.jsonl`。
- `data/review_results.jsonl`。
- `data/error_analysis_results.jsonl`。
- `data/evaluation_report.json`。
- `prompts/*.txt`。
- README 使用说明。
- 领导演示脚本。

---

## 15. 两人并行协作边界总结

人员 A 负责“在线判断是否可信、是否可解释、是否可被人审回流”。

人员 B 负责“人工反馈如何变成策略资产、候选规则是否真的有效”。

双方只通过以下稳定接口协作：

```text
CaseInput
DetectionResult
TraceRecord
ReviewResult
ErrorAnalysisResult
RuleProposal
RuleConfig
EvaluationReport
```

最重要的集成点：

- A 必须稳定写出 `traces.jsonl` 和 `review_results.jsonl`。
- B 必须只读取这些契约，不依赖 A 的内部实现。
- B 写出的 `candidate_rules.json` 必须可被 A 的规则检测逻辑加载。
- 回放评测必须复用在线规则检测逻辑，避免“线上一套、离线一套”。

Prototype 成功标准不是完整替代人审，而是证明这条链路具备继续投入的工程基础：

```text
可追踪在线链路
+ 疑难 case 大模型结构化分析
+ 人工结果回流
+ 误判归因
+ 候选规则生成
+ 离线回放验证
= 面向 0 人审目标的策略优化闭环
```
