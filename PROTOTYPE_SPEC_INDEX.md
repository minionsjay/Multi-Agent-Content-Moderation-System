# 内容安全风控 Agent Prototype · Spec 开发索引

版本：V0.2  
用途：Coding Agent 和开发人员的直接执行入口  
关联文档：[SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](SDD_PROTOTYPE_DEVELOPMENT_PLAN.md)

> 本文档是唯一任务入口。每次只实现一个 Spec ID 或一个连调点。不要直接按 README、PRD 或 SDD 自动拆大任务。

## 1. 统一开发口径

- 保留原 POC 已跑通能力，技术入口仍是 `python -m src.api`。
- 图片审核按图片转文本后进入文本识别链路处理，不引入多模态大模型作为当前核心依赖。
- 在线决策枚举为 `pass / block / review`。
- `fraud` 涉诈是 Prototype 示例风险类型，框架支持扩展。
- 业务已有规则资产，Prototype 不重新设计规则优化能力，只提供加载、调试和回放接口。
- 模型训练不开发真实训练流程，只保留手动或定时触发接口。
- 模块通过 JSON / JSONL 契约协作，不引入内部隐藏耦合。

## 2. 模块划分

| 模块 | Owner | 目标 | 主要产物 |
|---|---|---|---|
| Module A：在线检测与人审回流 | 开发者 A | 跑通内容检测、Trace、人审 | `DetectionResult`、`TraceRecord`、`ReviewResult` |
| Module B：旁路调试与回放接口 | 开发者 B | 跑通误判读取、调试分析、回放、训练触发接口，并预留可插拔 Agent backend | `ErrorAnalysisResult`、`EvaluationReport`、`TrainingTriggerResult` |

## 3. 共享数据契约 Specs

共享 Spec 必须最先完成。

| Spec ID | 名称 | Owner | 输出 | 连调 |
|---|---|---|---|---|
| S0.1 | 目录与文件路径契约 | A+B | 固定 `data/`、`rules/`、`prompts/`、可选 `prototype/` | G0 |
| S0.2 | CaseInput 契约 | A+B | 输入 case JSON Schema | G0 |
| S0.3 | DetectionResult 契约 | A+B | `decision=pass/block/review` 输出 Schema | G0 |
| S0.4 | TraceRecord 契约 | A+B | `data/traces.jsonl` 行结构 | G0 |
| S0.5 | ReviewResult 契约 | A+B | `data/review_results.jsonl` 行结构 | G0 |
| S0.6 | RuleConfig 契约 | A+B | `rules_v0.json` / `candidate_rules.json` 适配结构 | G0 |
| S0.7 | TrainingTrigger 契约 | A+B | 训练触发接口输出结构 | G0 |

### G0：契约连调

A 提供：

- mock `data/traces.jsonl`
- mock `data/review_results.jsonl`
- mock `rules/rules_v0.json`

B 验证：

- 能读取 Trace 和 ReviewResult。
- 能按 `case_id` / `trace_id` join。
- 能加载规则配置。
- 字段无需二次转换。

## 4. Module A Specs

### 4.1 职责

```text
CaseInput
→ Gateway / Online Detection Graph
→ DetectionResult
→ TraceRecord
→ Human Review
→ ReviewResult
```

### 4.2 Spec 清单

| Spec ID | 名称 | 输入 | 输出 | 是否阻塞 B | 连调 |
|---|---|---|---|---|---|
| A1 | JSONL 存储工具 | dict / list[dict] | append/read JSONL | 是 | G0 |
| A2 | 规则加载与检测 | content、language、rules file | rule_node result | 是 | G1 |
| A3 | 语言识别节点 | content、optional language | language_node result | 否 | - |
| A4 | 风险类型初判节点 | content、language | risk_preclassify_node result | 否 | - |
| A5 | 算法分数节点 | algorithm_score | algorithm_score_node result | 否 | - |
| A6 | LLM Judge 节点 | case + node context | llm_judge_node result | 否 | G2 |
| A7 | 决策聚合节点 | 所有节点输出 | decision result: pass/block/review | 是 | G2 |
| A8 | 在线检测 Graph / Pipeline | CaseInput | DetectionResult + TraceRecord | 是 | G2 |
| A9 | 在线检测页 / 调试页 | UI 表单 | DetectionResult + Trace 展示 | 否 | Demo |
| A10 | 人工审核提交逻辑 | 人审表单 | ReviewResult | 是 | G3 |
| A11 | 人工审核页 | Trace / DetectionResult | review_results 写入 | 是 | G3 |
| A12 | 图片转文本接入 | image_url / image_base64 | image_text_node result | 否 | G2 |

### 4.3 对外接口

```python
def run_online_detection(case_input: dict) -> dict:
    """返回 DetectionResult，并写入 data/traces.jsonl。"""

def submit_review(review_input: dict) -> dict:
    """返回 ReviewResult，并写入 data/review_results.jsonl。"""

def list_traces(limit: int = 100) -> list[dict]:
    """供人审页、调试页和 Module B 查询 Trace。"""

def get_trace(trace_id: str) -> dict | None:
    """按 trace_id 查询单条 Trace。"""
```

### 4.4 完成标准

- 文本输入能返回 `pass / block / review`。
- 图片输入能转成文本信号并纳入文本审核。
- 每条检测写入 `data/traces.jsonl`。
- Trace 中包含各节点输出。
- 人审能提交 `pass / block`、风险类型修正和人工原因。
- 每条审核写入 `data/review_results.jsonl`。
- `ReviewResult.error_type` 自动计算。

## 5. Module B Specs

### 5.1 职责

```text
TraceRecord + ReviewResult
→ Sidecar Agent Backend / 默认本地实现
→ ErrorAnalysisResult
→ Replay Evaluation
→ Training Trigger Hook
```

Module B 当前以“调试与接口预留”为主，不要求真实规则优化或模型训练。实现时应保留可插拔 Agent backend 边界：当前默认使用本地确定性函数，后续可接入 OpenClaw、LangGraph 或其他 multi-agent backend，但不得改变共享 JSON / JSONL 契约。

### 5.2 Spec 清单

| Spec ID | 名称 | 输入 | 输出 | 依赖 | 连调 |
|---|---|---|---|---|---|
| B1 | 读取 Trace / Review 数据 | traces.jsonl、review_results.jsonl | joined records | A1/S0 | G0 |
| B2 | 误判类型识别 | ReviewResult | error_type 分类 | S0.5 | G3 |
| B3 | 误判归因分析接口 | joined records | ErrorAnalysisResult | B1/B2 | G3 |
| B4 | Trace / 误判调试页 | ErrorAnalysisResult | 漏检 / 误杀 / 类别错误展示 | B3 | Demo |
| B5 | 规则调试接口 | content + rules file | 命中、排除、证据 | A2 | G4 |
| B6 | candidate_rules 写入接口 | 调试配置 | `rules/candidate_rules.json` | S0.6 | G4 |
| B7 | 规则回放评测器 | sample cases + old/candidate rules | EvaluationReport | A2/B5 | G5 |
| B8 | 回放报告页 | EvaluationReport | 指标表 + 总结 | B7 | Demo |
| B9 | 训练触发接口 | trigger request | TrainingTriggerResult | S0.7 | G6 |

### 5.3 对外接口

Module B 对外接口必须保持稳定。内部可以先调用 `DefaultSidecarBackend` 一类本地实现，未来再替换或扩展为 OpenClaw 等 Agent backend；调用方不应感知具体 backend。

```python
def analyze_errors(
    review_results_path: str = "data/review_results.jsonl",
    traces_path: str = "data/traces.jsonl",
    output_path: str = "data/error_analysis_results.jsonl",
) -> list[dict]:
    """输出 ErrorAnalysisResult，并写入 data/error_analysis_results.jsonl。"""

def debug_rule_match(
    content: str,
    language: str = "unknown",
    rules_path: str = "rules/rules_v0.json",
) -> dict:
    """返回规则命中、排除、证据和调试信息。"""

def accept_debug_rule_config(config: dict, output_path: str = "rules/candidate_rules.json") -> dict:
    """写入候选调试规则配置，不代表自动上线。"""

def run_replay_evaluation(
    sample_cases_path: str = "data/sample_cases.jsonl",
    old_rules_path: str = "rules/rules_v0.json",
    candidate_rules_path: str = "rules/candidate_rules.json",
    output_path: str = "data/evaluation_report.json",
) -> dict:
    """输出 EvaluationReport，并写入 data/evaluation_report.json。"""

def trigger_training_job(trigger_input: dict) -> dict:
    """记录训练触发请求；Prototype 不执行真实训练。"""
```

### 5.4 完成标准

- 能读取 A 写出的 Trace 和 ReviewResult。
- 能识别 `false_negative`、`false_positive`、`category_error`。
- 能输出基础归因和调试建议。
- 能复用规则检测逻辑进行回放。
- 能生成 `data/evaluation_report.json`。
- 能记录一次人工或定时训练触发请求。
- 对外函数和文件契约不绑定具体 Agent 框架；默认backend可运行，后续backend可替换。

## 6. 开发顺序

### 阶段 0：契约冻结

| 顺序 | Spec | 完成后动作 |
|---:|---|---|
| 1 | S0.1 | 创建目录和空文件 |
| 2 | S0.2 | 准备 sample cases |
| 3 | S0.3 | 确认 `pass/block/review` 输出 |
| 4 | S0.4 | mock Trace |
| 5 | S0.5 | mock ReviewResult |
| 6 | S0.6 | 规则配置可加载 |
| 7 | S0.7 | 训练触发结构可记录 |
| 8 | G0 | 契约连调 |

### 阶段 1：核心在线链路

| 顺序 | A | B | 连调 |
|---:|---|---|---|
| 1 | A1 JSONL 工具 | B1 读取数据 | G0 |
| 2 | A2 规则检测 | B5 规则调试接口 | G1/G4 |
| 3 | A3 语言识别 | B2 误判类型识别 | - |
| 4 | A4 风险初判 | B3 mock 归因 | - |
| 5 | A5 算法分数 | B7 mock 回放 | - |

### 阶段 2：Graph、人审、图片转文本

| 顺序 | A | B | 连调 |
|---:|---|---|---|
| 1 | A6 LLM Judge | B3 归因接口 | - |
| 2 | A7 决策聚合 | B4 调试页数据 | - |
| 3 | A12 图片转文本接入 | B5 调试规则 | G2 |
| 4 | A8 在线检测 Pipeline | B1/B2 接真实 Trace | G2 |
| 5 | A10 人审提交 | B3 接真实 ReviewResult | G3 |
| 6 | A9/A11 页面 | B4 页面 | Demo |

### 阶段 3：回放与接口预留

| 顺序 | A | B | 连调 |
|---:|---|---|---|
| 1 | 配合规则检测复用 | B6 candidate_rules 写入接口 | G4 |
| 2 | 提供样本 | B7 回放评测器 | G5 |
| 3 | 配合展示 Trace | B8 回放报告页 | Demo |
| 4 | 提供数据集路径 | B9 训练触发接口 | G6 |

## 7. 连调点

| 连调点 | 目标 | 通过标准 |
|---|---|---|
| G0 | 契约连调 | B 能读取 mock Trace/Review/Rules |
| G1 | 规则逻辑连调 | 在线检测和规则调试命中一致 |
| G2 | 在线检测连调 | 真实 DetectionResult + Trace 可被 B 读取 |
| G3 | 人审回流连调 | ReviewResult 可生成 error_type |
| G4 | 候选配置连调 | candidate_rules 可被检测和回放加载 |
| G5 | 离线回放连调 | 生成 evaluation_report.json |
| G6 | 训练触发连调 | 记录 TrainingTriggerResult，不执行真实训练 |
| Demo | 最终演示 | 文本、图片转文本、人审、回放、训练触发接口连续跑通 |

## 8. 派任务示例

```text
请实现 A2：规则加载与检测。
要求读取 rules/rules_v0.json，输入 content/language，输出 rule_node result。
完成后保证 B5 debug_rule_match 可以复用同一套检测函数。
```

```text
请实现 B9：训练触发接口。
要求接收 trigger_type=manual/scheduled，写入触发记录并返回 TrainingTriggerResult。
Prototype 不执行真实训练。
```
