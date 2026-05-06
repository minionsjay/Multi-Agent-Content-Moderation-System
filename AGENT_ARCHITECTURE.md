# 内容安全风控 Agent Prototype · Agent 架构说明

版本：V0.1  
定位：说明多 Agent 分工、协作链路和边界约束  
执行入口：[PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md)

> 本文档不是新的开发任务入口，不替代 SDD 或 Spec。Coding Agent 派任务时仍必须使用 `PROTOTYPE_SPEC_INDEX.md` 中的单个 Spec ID 或单个连调点。

## 1. 文档边界

本文档只回答三个问题：

1. Prototype 中有哪些 Agent 或节点角色。
2. 这些 Agent 如何通过在线链路、人工审核和离线旁路协作。
3. Agent 之间通过哪些 JSON / JSONL 契约解耦。

字段级 Schema、接口参数和开发顺序以以下文档为准：

| 文档 | 职责 |
|---|---|
| [README.md](README.md) | 项目总入口和统一架构口径 |
| [PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md](PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md) | 产品目标、范围和验收 |
| [SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](SDD_PROTOTYPE_DEVELOPMENT_PLAN.md) | 数据契约、接口和实现细节 |
| [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) | Coding Agent 直接执行入口 |
| [AGENTS.md](AGENTS.md) | Coding Agent 项目规则 |

## 2. 总体 Agent 结构

```text
CaseInput
  ↓
Gateway Agent / 热路径预过滤
  ↓
Online Detection Agents
  ├── Image Text Agent
  ├── Language Agent
  ├── Rule Detection Agent
  ├── Risk Preclassify Agent
  ├── Algorithm Score Agent
  ├── LLM Judge Agent
  └── Decision Agent
       ↓
DetectionResult + TraceRecord
       ↓
Human Review Agent
       ↓
ReviewResult
       ↓
Sidecar Debug Agents
  ├── Error Analysis Agent
  ├── Rule Debug Agent
  ├── Replay Evaluation Agent
  └── Training Trigger Agent
```

Agent 在实现上可以是 LangGraph 节点、普通 Python 函数、API Handler 或页面动作；“Agent”表示职责边界，不要求每个角色都独立成服务。

## 3. Agent 职责

| Agent | 所属模块 | 主要输入 | 主要输出 | 职责边界 |
|---|---|---|---|---|
| Gateway Agent | A | `CaseInput` | gateway 节点结果 | 做缓存、规则快速命中、白名单和上下文排除；未命中进入在线检测链路 |
| Image Text Agent | A | `image_url` / `image_base64` | `image_text_node` | 将图片内容转成文本信号；不引入多模态 LLM 作为当前核心依赖 |
| Language Agent | A | content、可选 language | `language_node` | 识别或规范化语言信息 |
| Rule Detection Agent | A | content、language、rules | `rule_node` | 加载业务规则并输出命中、排除、证据 |
| Risk Preclassify Agent | A | content、language | `risk_preclassify_node` | 做风险类型初判；`fraud` 是首个示例，不硬编码成唯一类型 |
| Algorithm Score Agent | A | `algorithm_score` 或算法上下文 | `algorithm_score_node` | 接入业务算法分数或模拟分数 |
| LLM Judge Agent | A | case + 节点上下文 | `llm_judge_node` | 在需要时做补充判断和原因生成 |
| Decision Agent | A | 各节点输出 | `decision_node`、`DetectionResult` | 聚合为 `pass / block / review` |
| Human Review Agent | A | Trace / DetectionResult / 人审输入 | `ReviewResult` | 提交人工结论、修正风险类型、计算或写入误判类型 |
| Error Analysis Agent | B | `TraceRecord + ReviewResult` | `ErrorAnalysisResult` | 分析漏检、误杀、类别错误并输出基础归因 |
| Rule Debug Agent | B | content、language、rules | 调试结果、`candidate_rules.json` | 复用规则检测逻辑，支持候选规则调试；不代表自动上线 |
| Replay Evaluation Agent | B | sample cases、old/candidate rules | `EvaluationReport` | 离线回放候选规则或配置，输出指标报告 |
| Training Trigger Agent | B | manual / scheduled trigger | `TrainingTriggerResult` | 只记录训练触发请求；Prototype 不执行真实训练 |

## 4. 在线审核链路

```text
CaseInput
  ↓
Gateway Agent
  ↓ miss or need deeper check
Image Text Agent
  ↓
Language Agent
  ↓
Rule Detection Agent
  ↓
Risk Preclassify Agent
  ↓
Algorithm Score Agent
  ↓
LLM Judge Agent
  ↓
Decision Agent
  ↓
DetectionResult
  ↓
TraceRecord 写入 data/traces.jsonl
```

在线链路约束：

- 技术入口保持 `python -m src.api`。
- 在线决策只使用 `pass / block / review`。
- 图片先提取 OCR 等文本信号，再进入文本识别和规则判断链路。
- 每次检测必须能够落 Trace，便于人审、误判分析和回放复用。
- 节点失败时也应在 Trace 中写入节点输出，并标记 `error` 字段。

## 5. 人审反馈链路

```text
TraceRecord + DetectionResult
  ↓
Human Review Agent
  ↓
ReviewResult 写入 data/review_results.jsonl
  ↓
Error Analysis Agent
```

人审链路约束：

- 人审结论使用 `human_decision=pass/block`。
- 系统结论和人审结论共同生成 `error_type`。
- `ReviewResult` 必须保留 `case_id` 和 `trace_id`，保证 Module B 可以与 Trace join。
- 人审反馈只通过 JSONL 契约传递给旁路分析，不直接读取 Module A 内部状态。

## 6. 旁路调试与回放链路

```text
data/traces.jsonl + data/review_results.jsonl
  ↓
Error Analysis Agent
  ↓
ErrorAnalysisResult

rules/rules_v0.json + content
  ↓
Rule Debug Agent
  ↓
rules/candidate_rules.json

data/sample_cases.jsonl + old/candidate rules
  ↓
Replay Evaluation Agent
  ↓
data/evaluation_report.json

manual / scheduled request
  ↓
Training Trigger Agent
  ↓
TrainingTriggerResult
```

旁路链路约束：

- Module B 只能依赖共享 JSON / JSONL 契约和可复用的公共规则检测函数。
- 规则调试可以产出候选配置，但不自动上线。
- 回放评测输出报告，不直接改变在线检测结果。
- 训练触发只保留接口和记录，不执行真实训练流程。

## 7. 共享数据契约

Agent 间协作必须使用 `PROTOTYPE_SPEC_INDEX.md` 和 `SDD_PROTOTYPE_DEVELOPMENT_PLAN.md` 定义的共享契约：

| 契约 | 生产方 | 消费方 | 默认位置 |
|---|---|---|---|
| `CaseInput` | API / UI / sample cases | Module A | `data/sample_cases.jsonl` |
| `DetectionResult` | Online Detection Agents | API / UI / Human Review | API 返回和 Trace 引用 |
| `TraceRecord` | Module A | Human Review / Module B | `data/traces.jsonl` |
| `ReviewResult` | Human Review Agent | Module B | `data/review_results.jsonl` |
| `RuleConfig` | 业务规则资产 / Rule Debug Agent | Module A / Module B | `rules/rules_v0.json`、`rules/candidate_rules.json` |
| `ErrorAnalysisResult` | Error Analysis Agent | 调试页 / 后续分析 | `data/error_analysis_results.jsonl` |
| `EvaluationReport` | Replay Evaluation Agent | 回放报告页 / Demo | `data/evaluation_report.json` |
| `TrainingTriggerResult` | Training Trigger Agent | 训练触发记录 / Demo | SDD 指定记录位置 |

禁止通过以下方式绕过契约：

- Module B 直接读取 Module A 内部运行状态。
- 页面逻辑直接依赖未落盘的临时对象。
- 用私有字段替代 `case_id` / `trace_id` 关联。
- 新增与 `pass / block / review` 冲突的在线决策枚举。

## 8. 与 Spec 的映射

| 架构能力 | 主要 Spec / 连调点 |
|---|---|
| 目录、JSON / JSONL、规则和训练触发契约 | `S0.1` - `S0.7`、`G0` |
| Gateway 和在线检测基础链路 | `A1` - `A8`、`G1`、`G2` |
| 图片转文本信号 | `A12`、`G2` |
| 在线检测页和人审页 | `A9`、`A10`、`A11`、`G3`、`Demo` |
| 误判分析和 Trace 调试 | `B1` - `B4`、`G3` |
| 规则调试和候选规则 | `B5`、`B6`、`G4` |
| 离线回放评估 | `B7`、`B8`、`G5` |
| 训练触发接口 | `B9`、`G6` |

## 9. 实现原则

- 每次开发只实现一个 Spec ID 或一个连调点。
- 优先复用原 POC 已跑通的 FastAPI、Gateway、LangGraph、图文审核、缓存、人审队列和离线反馈能力。
- 新增 Prototype 能力可以放在 `prototype/`，但必须兼容现有 `src/` 技术入口。
- `fraud` 是当前示例风险类型，风险类型字段必须保留扩展空间。
- 业务规则资产已经存在，当前只做加载、命中、Trace、调试和回放。
- Agent 边界服务于解耦和调试，不要求为每个 Agent 创建独立进程或服务。
