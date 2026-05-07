# 内容安全风控 Agent Prototype PRD

文档版本：V0.2  
文档类型：PRD / 产品需求文档  
适用对象：内容风控策略、审核、开发、技术负责人、Coding Agent  
当前定位：在现有 Multi-Agent Content Moderation POC 上，补齐可追踪、可调试、可回放的业务框架

## 1. 背景

当前仓库已有一条可运行的内容审核 POC：

```text
FastAPI
→ Gateway 热路径
→ LangGraph 冷路径
→ 图文审核
→ 缓存
→ 人审队列
→ 离线反馈基础能力
```

新需求不是推翻这条链路，而是将其整理成更贴近业务调试和后续演进的 Agent Prototype：

```text
规则 / 关键词 / 缓存 / 小模型
→ LLM 处理疑难 case
→ pass / block / review
→ Trace
→ 人审回流
→ 旁路调试与回放
→ 规则和模型训练接口预留
→ 可插拔 Agent backend 预留
```

## 2. 产品目标

### 2.1 核心目标

1. 保留已跑通的原 POC 能力，包括 `python -m src.api` 技术入口、FastAPI、Gateway、LangGraph、图文审核、缓存、人审队列和事件记录。
2. 形成一套清晰的业务框架，使每次审核可追踪、可解释、可调试、可回放。
3. 图片审核按照当前项目思路处理：图片转文本后进入文本识别链路，不引入多模态大模型作为当前核心依赖。
4. 在线决策沿用 `pass / block / review`。
5. 涉诈 `fraud` 作为 Prototype 的示例风险类型，框架支持接入业务已有规则资产和更多风险类型。
6. 规则优化、模型微调、训练闭环不作为当前开发重点，只预留人工或定时触发接口。
7. 旁路优化预留可插拔 Agent backend，当前默认使用本地确定性实现，后续可接入 OpenClaw 等旁路智能分析框架。

### 2.2 一句话定义

本 Prototype 是一个内容安全风控 Agent 业务框架：保留原 POC 的在线审核能力，将文本和图片转文本信号统一纳入可追踪 Agent Graph，输出 `pass / block / review`，并通过 Trace、人审回流、旁路调试和回放接口支撑后续策略与模型迭代。

## 3. 用户角色

| 角色 | 主要诉求 |
|---|---|
| 审核员 | 快速看到原文、系统判断、证据、命中规则、LLM 理由，提交人工结果 |
| 风控策略人员 | 调试规则命中、查看误判、验证规则或阈值调整影响 |
| 算法 / 模型人员 | 获取 Trace、人工标签、训练触发入口和调试样本 |
| 技术负责人 | 判断链路是否稳定、接口是否清晰、是否可扩展 |

## 4. 范围

### 4.1 必须保留

- FastAPI 服务入口：`python -m src.api`。
- Gateway 热路径预过滤。
- LangGraph 在线检测链路。
- 文本检测：规则 / 关键词、小模型或算法分数、LLM Judge。
- 图片检测：图片指纹、OCR 或等价方式提取文本信号，再进入文本链路。
- 决策枚举：`pass / block / review`。
- 人审队列和人审提交。
- Trace、事件、JSON / JSONL 存储。
- 旁路调试接口：误判分析、规则回放、模型训练触发接口。
- 旁路可插拔 Agent backend 边界：对外函数和数据契约稳定，内部实现可替换。

### 4.2 当前不重点开发

| 能力 | 当前处理方式 |
|---|---|
| 多模态大模型 | 不作为当前核心依赖 |
| 相似图片知识库检索 | 预留，不开发完整能力 |
| 业务规则自动生成 | 不重新发明规则资产，仅支持加载、调试和回放 |
| 规则自动上线 | 不做，风控规则必须保留人工确认 |
| 模型真实训练 | 不做训练流程，只留手动或定时触发接口 |
| 旁路Agent框架重度集成 | 不作为当前强依赖，仅保留backend适配层 |
| 复杂权限 / 复杂前端 | Prototype 阶段不做 |
| 生产级数据库改造 | 继续使用 JSON / JSONL 或现有轻量存储 |

## 5. 总体流程

```text
输入文本或图片
  ↓
Gateway 热路径
  ↓ 未命中
LangGraph 在线检测
  ├── 图片转文本信号
  ├── 文本规则 / 关键词检测
  ├── 小模型或算法分数节点
  ├── LLM Judge
  └── Decision 聚合
  ↓
DetectionResult: pass / block / review
  ↓
TraceRecord 落盘
  ↓
review 时进入人审
  ↓
ReviewResult 落盘
  ↓
旁路调试 / 误判分析 / 回放 / 训练触发接口
  ↓
可插拔 Agent backend 预留
```

## 6. 功能需求

### 6.1 在线检测

输入：

| 字段 | 类型 | 说明 |
|---|---|---|
| `case_id` / `content_id` | string | 可自动生成 |
| `content` / `text` | string | 待审核文本 |
| `image_url` | string | 可选图片 URL |
| `image_base64` | string | 可选图片内容 |
| `language` | string | 可选，缺省自动识别 |
| `risk_type` | string | 可选，默认由系统判断 |
| `algorithm_score` | float | 可选，业务已有小模型或模拟分数 |
| `source` | string | 来源 |
| `user_id` / `domain` / `customer_type` | string | 业务上下文预留 |

输出：

| 字段 | 类型 | 说明 |
|---|---|---|
| `case_id` / `content_id` | string | 输入 ID |
| `decision` | string | `pass` / `block` / `review` |
| `risk_type` | string | 先以 `fraud` 为示例，支持扩展 |
| `confidence` | float | 置信度 |
| `reason` | string | 系统理由 |
| `evidence` | list | 证据片段 |
| `matched_rules` | list | 命中规则 |
| `llm_reason` | string | LLM 解释 |
| `fraud_features` | object | 涉诈示例结构化特征 |
| `trace_id` | string | Trace 关联 ID |
| `latency_ms` | float | 耗时 |
| `llm_called` | bool | 是否调用 LLM |

### 6.2 图片转文本审核

图片链路按当前 POC 继续保留：

```text
图片 URL / Base64
→ 图片缓存或指纹
→ OCR / 文本提取
→ 将图片文本追加为审核文本信号
→ Text Agent / Decision
```

这属于文本识别链路的输入扩展，不等同于引入多模态大模型。

### 6.3 人审回流

人审输入：

| 字段 | 类型 | 说明 |
|---|---|---|
| `trace_id` | string | 关联 Trace |
| `case_id` / `content_id` | string | 关联内容 |
| `system_decision` | string | `pass` / `block` / `review` |
| `system_risk_type` | string | 系统风险类型 |
| `human_decision` | string | `pass` / `block` |
| `human_risk_type` | string | 人工修正风险类型 |
| `human_reason` | string | 人工原因 |
| `reviewer` | string | 审核员 |

系统自动补齐：

| 字段 | 说明 |
|---|---|
| `is_correct` | 系统判断是否正确 |
| `error_type` | `false_negative` / `false_positive` / `true_positive` / `true_negative` / `category_error` |
| `review_time` | 审核时间 |

### 6.4 旁路调试与回放

旁路模块当前目标不是自动优化规则或真实训练模型，而是留出稳定接口：

| 能力 | 要求 |
|---|---|
| 误判分析 | 能读取 Trace + ReviewResult，输出 error_type 和基础归因 |
| 规则调试 | 能加载业务规则文件，查看命中、排除、证据 |
| 离线回放 | 能用样本对比不同规则文件或阈值配置 |
| 训练触发 | 保留手动或定时触发接口，允许后续接入真实训练 |
| 可插拔 Agent backend | 当前默认本地函数实现；后续可接入 OpenClaw / LangGraph / multi-agent 分析框架，但不得改变共享数据契约 |

旁路backend只负责增强分析、归因、建议和报告生成，不直接改变在线检测结果，不自动上线规则，不绕过人工审批和离线回放。

## 7. 数据文件

| 文件 | 说明 |
|---|---|
| `data/traces.jsonl` | 每次检测 Trace |
| `data/review_results.jsonl` | 人审结果 |
| `data/error_analysis_results.jsonl` | 旁路误判分析结果 |
| `data/evaluation_report.json` | 回放报告 |
| `rules/rules_v0.json` | 当前规则配置或业务规则适配层 |
| `rules/candidate_rules.json` | 调试候选配置，非自动上线 |
| `prompts/*.txt` | LLM Judge / 分析 / 总结 Prompt |

## 8. 验收标准

- 服务仍可通过 `python -m src.api` 启动。
- 文本输入可返回 `pass / block / review`。
- 图片输入可通过 OCR 等方式提取文本信号并纳入审核。
- 每次检测有 Trace，可关联人审结果。
- 人审提交能写入结构化结果。
- 涉诈示例 case 可输出 `fraud_features`。
- 规则和模型训练不要求真实优化能力，但必须有调试或触发接口。
- 旁路调试接口具备默认backend实现，并保留未来替换为Agent框架backend的适配边界。
- Coding Agent 后续开发必须按 [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) 的 Spec ID 执行。
