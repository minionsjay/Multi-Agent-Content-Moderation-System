# 内容安全风控 Agent Prototype · Spec 开发索引

版本：V0.1  
用途：给开发人员和 Coding Agent 作为直接开发入口  
关联文档：[SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](./SDD_PROTOTYPE_DEVELOPMENT_PLAN.md)

> Scope：本文档只适用于“内容安全风控 Agent Prototype”这条新增开发线。
>
> 仓库原有 README / ARCHITECTURE / OVERVIEW 描述的是已存在的 Multi-Agent Content Moderation POC，包含 FastAPI、Gateway、图文审核、缓存和离线反馈飞轮。不要用原有 POC 文档替代本文档的 Spec 拆分；也不要用本文档改写原有 POC 的启动方式和架构说明。

---

## 1. 开发模块划分

本 Prototype 只分为 2 个可独立交付的业务模块。

| 模块 | Owner | 目标 | 主要产物 |
|---|---|---|---|
| Module A：在线检测与人审回流 | 开发者 A | 跑通文本检测、Trace、人审回流 | 在线检测页、人审页、`traces.jsonl`、`review_results.jsonl` |
| Module B：旁路分析与规则回放 | 开发者 B | 跑通误判分析、规则建议、离线回放 | 误判分析页、规则建议页、回放报告页、`candidate_rules.json`、`evaluation_report.json` |

两者通过 JSON / JSONL 文件和固定接口协作，不依赖彼此内部实现。

---

## 2. 共享数据契约 Spec

共享 Spec 必须最先完成。两位开发者都要读取并遵守。

| Spec ID | Spec 名称 | Owner | 开发顺序 | 输出 | 连调要求 |
|---|---|---|---:|---|---|
| S0.1 | 目录与文件路径契约 | A+B | 1 | 固定 `data/`、`rules/`、`prompts/` 路径 | 必须连调 |
| S0.2 | CaseInput 契约 | A+B | 2 | 输入 case JSON Schema | 必须连调 |
| S0.3 | DetectionResult 契约 | A+B | 3 | 在线检测结果 JSON Schema | 必须连调 |
| S0.4 | TraceRecord 契约 | A+B | 4 | `data/traces.jsonl` 行结构 | 必须连调 |
| S0.5 | ReviewResult 契约 | A+B | 5 | `data/review_results.jsonl` 行结构 | 必须连调 |
| S0.6 | RuleConfig 契约 | A+B | 6 | `rules_v0.json` / `candidate_rules.json` 结构 | 必须连调 |

完成 S0.1 - S0.6 后，需要进行第一次连调：

```text
连调点 G0：契约连调

A 产出：
- 一条 mock TraceRecord
- 一条 mock ReviewResult

B 验证：
- 能读取 traces.jsonl
- 能读取 review_results.jsonl
- 能读取 rules_v0.json
- 不需要转换字段即可进入后续分析
```

---

## 3. Module A：在线检测与人审回流 Specs

### 3.1 Module A 职责

开发者 A 负责从文本输入到人工审核回流：

```text
CaseInput
→ 在线检测 Agent Graph
→ DetectionResult
→ TraceRecord
→ 人工审核
→ ReviewResult
```

### 3.2 Module A Spec 清单

| Spec ID | Spec 名称 | 开发顺序 | 输入 | 输出 | 是否阻塞 B | 连调点 |
|---|---|---:|---|---|---|---|
| A1 | JSONL 存储工具 | 1 | dict / list[dict] | append/read JSONL | 是 | G0 |
| A2 | 规则加载与规则检测 | 2 | `content`、`language`、`rules_v0.json` | rule_node result | 是 | G1 |
| A3 | 语言识别节点 | 3 | `content`、optional `language` | language_node result | 否 | - |
| A4 | 风险类型初判节点 | 4 | `content`、`language` | risk_preclassify_node result | 否 | - |
| A5 | 算法分数节点 | 5 | `algorithm_score` | algorithm_score_node result | 否 | - |
| A6 | LLM Judge 节点 | 6 | case + node context | llm_judge_node result | 否 | G2 |
| A7 | 决策聚合节点 | 7 | 所有节点输出 | final_decision result | 是 | G2 |
| A8 | 在线检测 Graph / Pipeline | 8 | CaseInput | DetectionResult + TraceRecord | 是 | G2 |
| A9 | 在线检测页 | 9 | UI 表单 | DetectionResult 展示 | 否 | Demo |
| A10 | 人工审核提交逻辑 | 10 | 人工审核表单 | ReviewResult | 是 | G3 |
| A11 | 人工审核页 | 11 | Trace / DetectionResult | review_results 写入 | 是 | G3 |

### 3.3 Module A 对外接口

```python
def run_online_detection(case_input: dict) -> dict:
    """返回 DetectionResult，并写入 data/traces.jsonl。"""

def submit_review(review_input: dict) -> dict:
    """返回 ReviewResult，并写入 data/review_results.jsonl。"""

def list_traces(limit: int = 100) -> list[dict]:
    """供人审页和 Module B 查询 Trace。"""

def get_trace(trace_id: str) -> dict | None:
    """按 trace_id 查询单条 Trace。"""
```

### 3.4 Module A 完成标准

- 输入文本后能返回结构化 `DetectionResult`。
- 每条检测写入 `data/traces.jsonl`。
- Trace 中包含所有节点输出。
- 人工审核页能提交违规 / 正常 / 风险类型修正 / 人工原因。
- 每条审核写入 `data/review_results.jsonl`。
- `ReviewResult.error_type` 自动计算。

---

## 4. Module B：旁路分析与规则回放 Specs

### 4.1 Module B 职责

开发者 B 负责从人工反馈到规则优化验证：

```text
ReviewResult + TraceRecord
→ ErrorAnalysisResult
→ RuleProposal
→ candidate_rules.json
→ EvaluationReport
```

### 4.2 Module B Spec 清单

| Spec ID | Spec 名称 | 开发顺序 | 输入 | 输出 | 依赖 A | 连调点 |
|---|---|---:|---|---|---|---|
| B1 | 读取 Trace / Review 数据 | 1 | `traces.jsonl`、`review_results.jsonl` | joined case records | 是 | G0 |
| B2 | 误判类型识别 | 2 | ReviewResult | error_type 分类 | 部分依赖 | G3 |
| B3 | 误判归因分析 | 3 | joined case records | ErrorAnalysisResult | 是 | G3 |
| B4 | 误判分析页 | 4 | ErrorAnalysisResult | 漏检 / 误杀 / 类别错误展示 | 是 | G3 |
| B5 | 规则 Proposal 生成 | 5 | ErrorAnalysisResult | RuleProposal | 是 | G4 |
| B6 | 规则建议页 | 6 | RuleProposal | 接受 / 拒绝 / 编辑后接受 | 否 | G4 |
| B7 | candidate_rules 写入 | 7 | accepted proposal | `candidate_rules.json` | 否 | G4 |
| B8 | 离线回放评测器 | 8 | sample cases + old/candidate rules | EvaluationReport | 部分依赖 A2 | G5 |
| B9 | 回放报告页 | 9 | EvaluationReport | 指标表 + 总结 | 否 | Demo |

### 4.3 Module B 对外接口

```python
def analyze_errors(
    review_results_path: str = "data/review_results.jsonl",
    traces_path: str = "data/traces.jsonl",
    output_path: str = "data/error_analysis_results.jsonl",
) -> list[dict]:
    """输出 ErrorAnalysisResult，并写入 data/error_analysis_results.jsonl。"""

def generate_rule_proposals(
    error_analysis_path: str = "data/error_analysis_results.jsonl",
) -> list[dict]:
    """基于误判分析生成 RuleProposal。"""

def accept_rule_proposal(proposal: dict, edited_rule: dict | None = None) -> dict:
    """接受 proposal，并写入 rules/candidate_rules.json。"""

def run_replay_evaluation(
    sample_cases_path: str = "data/sample_cases.jsonl",
    old_rules_path: str = "rules/rules_v0.json",
    candidate_rules_path: str = "rules/candidate_rules.json",
    output_path: str = "data/evaluation_report.json",
) -> dict:
    """输出 EvaluationReport，并写入 data/evaluation_report.json。"""
```

### 4.4 Module B 完成标准

- 能读取 A 写出的 `traces.jsonl` 和 `review_results.jsonl`。
- 能识别 `false_negative`、`false_positive`、`category_error`。
- 能输出 `root_cause`、`suggested_fix`、`candidate_terms`。
- 能生成候选规则 proposal。
- 人工接受后只写入 `rules/candidate_rules.json`。
- 能用同一批样本对比 `rules_v0.json` 和 `candidate_rules.json`。
- 能生成 `data/evaluation_report.json`。

---

## 5. Spec 开发顺序总表

### 5.1 第 0 阶段：先冻结共享契约

| 顺序 | Spec | Owner | 完成后动作 |
|---:|---|---|---|
| 1 | S0.1 目录与文件路径契约 | A+B | 创建空文件和目录 |
| 2 | S0.2 CaseInput 契约 | A+B | 准备 5 条 sample case |
| 3 | S0.3 DetectionResult 契约 | A+B | 确认在线输出字段 |
| 4 | S0.4 TraceRecord 契约 | A+B | A 写 mock trace，B 读取 |
| 5 | S0.5 ReviewResult 契约 | A+B | A 写 mock review，B 读取 |
| 6 | S0.6 RuleConfig 契约 | A+B | A/B 均能加载规则 |
| 7 | G0 契约连调 | A+B | 通过后两人并行 |

### 5.2 第 1 阶段：并行开发核心能力

| 顺序 | 开发者 A | 开发者 B | 连调 |
|---:|---|---|---|
| 1 | A1 JSONL 存储工具 | B1 读取 Trace / Review 数据 | G0 |
| 2 | A2 规则加载与检测 | B2 误判类型识别 | G1 |
| 3 | A3 语言识别节点 | B3 mock 误判归因分析 | - |
| 4 | A4 风险类型初判节点 | B8 mock 离线回放评测器 | - |
| 5 | A5 算法分数节点 | B5 mock 规则 Proposal 生成 | - |

### 5.3 第 2 阶段：在线链路与人审回流

| 顺序 | 开发者 A | 开发者 B | 连调 |
|---:|---|---|---|
| 1 | A6 LLM Judge 节点 | 继续完善 B3 归因输出 | - |
| 2 | A7 决策聚合节点 | 准备误判分析页数据结构 | - |
| 3 | A8 在线检测 Graph / Pipeline | B1/B2 接入真实 Trace | G2 |
| 4 | A9 在线检测页 | B4 误判分析页 | - |
| 5 | A10 人工审核提交逻辑 | B3 接入真实 ReviewResult | G3 |
| 6 | A11 人工审核页 | B4 接入真实误判数据 | G3 |

### 5.4 第 3 阶段：规则建议与离线回放

| 顺序 | 开发者 A | 开发者 B | 连调 |
|---:|---|---|---|
| 1 | 修复 Trace / UI 展示问题 | B5 规则 Proposal 生成 | - |
| 2 | 确认规则检测可加载 candidate rules | B6 规则建议页 | G4 |
| 3 | 配合验证 candidate_rules 格式 | B7 candidate_rules 写入 | G4 |
| 4 | 提供规则检测复用函数 | B8 离线回放评测器 | G5 |
| 5 | 准备演示输入样本 | B9 回放报告页 | Demo |

---

## 6. 连调点清单

### G0：契约连调

触发时机：完成 S0.1 - S0.6。

A 提供：

- mock `data/traces.jsonl`
- mock `data/review_results.jsonl`
- `rules/rules_v0.json`

B 验证：

- 能读取所有文件。
- 能按 `case_id` / `trace_id` join 数据。
- 字段无需二次转换。

通过标准：

- B 可以输出 joined record。

### G1：规则逻辑连调

触发时机：A2 完成，B8 开始前。

A 提供：

- `match_rules(content, language, rules_path)` 或等价规则检测函数。

B 验证：

- 离线回放可复用同一套规则检测逻辑。
- `rules_v0.json` 和 `candidate_rules.json` 都能加载。

通过标准：

- 同一条 case 在线检测和离线回放的规则命中结果一致。

### G2：在线检测结果连调

触发时机：A8 完成。

A 提供：

- 真实 `DetectionResult`
- 真实 `TraceRecord`

B 验证：

- 能从 Trace 中读取系统判断、命中规则、LLM 判断、fraud_features。
- 能基于 Trace 做误判归因上下文。

通过标准：

- B 可以用真实 Trace 生成一条 mock `ErrorAnalysisResult`。

### G3：人审回流连调

触发时机：A10 / A11 完成。

A 提供：

- 真实 `review_results.jsonl`
- 至少包含 `false_positive`、`false_negative`、`true_positive` 示例。

B 验证：

- 能自动识别漏检、误杀、类别错误。
- 能输出 `error_analysis_results.jsonl`。

通过标准：

- 样本 3 能被识别为 `false_positive`。

### G4：候选规则连调

触发时机：B5 - B7 完成。

B 提供：

- `rules/candidate_rules.json`
- 至少一条新增关键词规则或上下文排除规则。

A 验证：

- 在线规则检测逻辑可以加载 candidate rules。
- candidate rule 不破坏原始 `rules_v0.json`。

通过标准：

- A 用 candidate rules 跑样本 2 能新增命中，跑样本 3 能体现排除或降风险。

### G5：离线回放连调

触发时机：B8 完成。

A 提供：

- 可复用规则检测函数。
- 至少 20 条样本，最终补齐 60 条。

B 验证：

- 当前规则和候选规则指标可对比。
- 能生成 `evaluation_report.json`。

通过标准：

- 回放报告能显示召回、精准、漏检、误杀、人审量、LLM 调用量变化。

### Demo：最终演示连调

触发时机：全部 Spec 完成。

必须连续跑通：

1. 规则直接命中。
2. 规则漏检，大模型识别。
3. 规则误杀，人工确认正常。
4. 旁路生成候选规则。
5. 离线回放展示新旧规则对比。

---

## 7. 最小可交付顺序

如果时间紧，按以下最小链路优先完成：

```text
S0.1 - S0.6
→ A1
→ A2
→ A5
→ A7
→ A8
→ A10
→ B1
→ B2
→ B3
→ B5
→ B7
→ B8
```

最小演示允许降级：

- LLM Judge 可以先 mock。
- Error Analysis 可以先 mock。
- Rule Proposal 可以先基于模板生成。
- UI 可以先用 Streamlit tabs 简单展示。
- 离线回放先只支持 keywords + exclude_keywords。

---

## 8. Coding Agent 使用说明

给 Coding Agent 派任务时，优先使用 Spec ID。

示例：

```text
请实现 Module A 的 A2：规则加载与规则检测。
要求读取 rules/rules_v0.json，输入 content/language，输出 rule_node result。
完成后确保 G1 中 B 可以复用同一个规则检测函数。
```

```text
请实现 Module B 的 B8：离线回放评测器。
要求读取 sample_cases.jsonl、rules_v0.json、candidate_rules.json，
复用 A2 的规则检测逻辑，输出 evaluation_report.json。
这是 G5 连调前置 spec。
```

不要直接下达“实现整个系统”这类任务。每次只派一个 Spec 或一个连调点。
