# Multi-Agent Content Moderation System

本文档是当前仓库唯一总入口。项目在原有 Multi-Agent Content Moderation POC 基础上，统一收敛为“内容安全风控 Agent Prototype”开发线。

## 1. 当前定位

本项目不是重新推翻原 POC，而是在已跑通能力上补齐一套可执行、可调试、可回放的业务框架：

```text
内容输入
→ Gateway 热路径预过滤
→ LangGraph 在线检测链路
→ pass / block / review 决策
→ Trace 记录
→ 人工审核回流
→ 旁路调试、误判分析、规则/模型接口预留
→ 可插拔 Agent backend 预留
```

原 POC 已跑通的 FastAPI、Gateway、LangGraph、图文审核、缓存、人审队列、离线反馈基础能力继续保留。新增 SDD / Spec 文档用于规范后续开发入口、数据契约和模块分工。

当前MVP不把OpenClaw或复杂multi-agent框架作为旁路运行时强依赖。旁路优化先通过稳定函数和JSON / JSONL契约交付，内部预留可插拔Agent backend，后续可替换为OpenClaw、LangGraph或其他智能分析编排实现。

## 2. 启动方式

技术入口保持不变：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
cp .env.example .env
pip install -r requirements.txt
python check_env.py
python -m src.api
```

运行环境要求：

- Python `3.10+`
- 推荐使用 Python `3.11` 创建虚拟环境，避免系统自带 Python `3.9` 带来的类型语法兼容问题

服务启动后访问：

```text
http://localhost:8000
```

常用接口：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/moderate` | POST | 单条审核 |
| `/moderate/stream` | POST | 流式审核 |
| `/moderate/batch` | POST | 批量审核 |
| `/gateway/stats` | GET | Gateway 统计 |
| `/review/pending` | GET | 待人审列表 |
| `/review/resolve` | POST | 提交人审结果 |
| `/health` | GET | 健康检查 |

## 3. 统一架构口径

```text
请求
  ↓
Gateway 热路径
  ├── L0a 本地缓存
  ├── L0b Redis 共享缓存
  ├── 关键词 / 规则检测
  ├── 白名单与上下文排除
  ├── 图片指纹缓存
  └── 语义缓存
       ↓ 未命中
LangGraph 在线检测
  ├── Image Agent：图片下载、指纹、OCR，将图片文字转成文本信号
  ├── Text Agent：关键词 / 小模型 / LLM 分层判断
  ├── Decision：聚合为 pass / block / review
  └── Action：缓存写回、人审入队、事件记录
       ↓
旁路能力
  ├── Trace 调试
  ├── 人审回流
  ├── 误判分析接口
  ├── 规则调试 / 回放接口
  ├── 模型训练触发接口预留
  └── 可插拔 Agent backend：默认本地函数，后续可接 OpenClaw 等旁路智能分析框架
```

图片审核按当前项目思路开发：图片先通过 OCR 等方式转为文本，再进入文本识别和内容风控链路；不引入多模态大模型作为本阶段核心依赖。

## 4. 决策与风险类型

在线决策沿用原 POC：

| 字段 | 取值 |
|---|---|
| `decision` | `pass` / `block` / `review` |

Prototype 的风险类型先以涉诈为例，但框架必须支持扩展：

| 风险类型 | 当前要求 |
|---|---|
| `fraud` | 本阶段示例和重点调试场景 |
| `porn` / `hate_or_abuse` / `violence` / `politics` / `other` | 预留或沿用现有 POC 能力 |

关键词规则资产来自业务沉淀，不需要在 Prototype 中重新发明规则优化能力。当前目标是把规则加载、命中、Trace、调试和回放接口打通。

## 5. 技术栈

| 层级 | 技术 |
|---|---|
| API | FastAPI + Uvicorn |
| Agent 编排 | LangGraph StateGraph |
| 热路径 | Gateway、TTLCache、Redis、AC 自动机、ChromaDB |
| 文本检测 | 关键词 / 规则、小模型或算法分数、LLM Judge |
| 图片文本化 | dHash / URL 缓存、EasyOCR 等 OCR 信号 |
| LLM | DeepSeek / OpenAI / Anthropic / 本地 llama.cpp 路径 |
| 人审与 Trace | JSON / JSONL |
| 旁路调试 | 误判分析、规则回放、训练触发接口预留、可插拔 Agent backend |

## 6. 项目结构

```text
src/
├── api.py                  # FastAPI 入口
├── gateway.py              # 热路径预过滤
├── graph.py                # LangGraph 在线检测图
├── state.py                # 在线检测状态
├── agents/
│   ├── image_agent.py      # 图片转文本信号 + 图片基础检测
│   ├── text_agent.py       # 文本分层审核
│   ├── decision.py         # pass/block/review 聚合
│   └── action.py           # 缓存、人审、事件
├── skills/
│   ├── keyword_filter.py
│   ├── bert_classify.py
│   ├── bert_onnx.py
│   ├── llm_audit.py
│   ├── llm_local.py
│   ├── embedder.py
│   ├── memory_cache.py
│   ├── redis_cache.py
│   ├── vector_cache.py
│   ├── image_phash.py
│   ├── image_nsfw.py
│   ├── image_ocr.py
│   └── review_queue.py
└── feedback/
    ├── event_collector.py
    ├── dataset_builder.py
    ├── finetune_trigger.py     # 保留训练触发接口
    ├── train_lora.py           # 不作为当前开发重点
    └── pipeline.py
```

建议新增 Prototype 相关目录时遵守 [SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](SDD_PROTOTYPE_DEVELOPMENT_PLAN.md)：

```text
prototype/
rules/
prompts/
data/
```

## 7. 权威文档集

根目录只保留这一套权威 Markdown 文档：

| 文档 | 用途 |
|---|---|
| [README.md](README.md) | 项目总入口、统一架构口径 |
| [PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md](PRD_CONTENT_SAFETY_RISK_CONTROL_AGENT_PROTOTYPE.md) | 产品目标、范围、非目标、角色和验收 |
| [SDD_PROTOTYPE_DEVELOPMENT_PLAN.md](SDD_PROTOTYPE_DEVELOPMENT_PLAN.md) | SDD 开发规划、数据契约、详细接口 |
| [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md) | Agent 分工、协作链路和边界约束 |
| [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) | Coding Agent 直接执行入口、Spec 顺序和连调点 |
| [AGENTS.md](AGENTS.md) | Coding Agent 项目规则 |

旧的 `ARCHITECTURE.md`、`OVERVIEW.md`、`ANALYSIS.md`、`OFFLINE_DEPLOY.md` 内容已整合到上述文档，不再作为执行或架构判断来源。

## 8. 开发规则

- 以后派发 Coding Agent 任务时，只使用 [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) 中的单个 Spec ID 或单个连调点。
- 原 POC 功能已跑通的部分优先保留，不因 Prototype 文档重构而回退。
- 训练、规则自动优化、线上灰度不是当前开发重点；只保留调试、触发和扩展接口。
- 模块之间通过 JSON / JSONL 契约协作，不引入隐藏耦合。
- 旁路优化的Agent框架接入采用backend适配层方式推进：当前默认本地实现，后续按单个Spec接入OpenClaw等框架，不改Module A内部状态和共享数据契约。

## 9. 开发 Roadmap（A / B 协作）

本 Roadmap 用于协作者在 GitHub 上快速理解开发顺序和联调闸门。具体派发任务时仍以 [PROTOTYPE_SPEC_INDEX.md](PROTOTYPE_SPEC_INDEX.md) 的单个 Spec ID 或单个连调点为准。

### 阶段 0：共享契约冻结，完成 G0

目标：先让 A 生产的 JSON / JSONL 文件，B 可以不做字段转换直接读取。

1. A 牵头完成 `S0.1-S0.6`：
   - 创建 `rules/`、`prompts/`、可选 `prototype/`。
   - 准备 `data/sample_cases.jsonl`。
   - 准备 mock `data/traces.jsonl`。
   - 准备 mock `data/review_results.jsonl`。
   - 准备 `rules/rules_v0.json` 和 `rules/candidate_rules.json`。
2. A+B 共同确认 `S0.7`：
   - 固定 `TrainingTriggerResult` 字段。
   - 确认 `manual/scheduled` 触发结构。
3. B 并行实现 `B1` 读取 / join 骨架：
   - 读取 Trace、ReviewResult、RuleConfig。
   - 按 `case_id` / `trace_id` join。
   - 不读取 Module A 内部运行状态。
4. B 同步预留旁路 backend 适配层：
   - Module B public function → Sidecar backend interface → 默认本地 backend。
   - 当前不引入 OpenClaw 或复杂 multi-agent runtime 强依赖。
5. A+B 执行 `G0`：
   - B 能读取 mock Trace / Review / Rules。
   - B 能 join。
   - 字段无需二次转换。

`G0` 不通过，不进入正式 `A2` / `B5`。

### 阶段 1：规则检测与基础旁路，完成 G1

目标：规则检测只实现一套，A 在线检测和 B 调试复用同一函数。

1. A 实现 `A1` JSONL 存储工具。
2. A 实现 `A2` 规则加载与检测：
   - 读取 `rules/rules_v0.json`。
   - 输出标准 `rule_node result`。
   - 现有 `KeywordFilter` 可以复用，但必须适配 RuleConfig。
3. B 在 `A2` 后实现 `B5` 第一版：
   - `debug_rule_match(content, language, rules_path)`。
   - 只读 `rules_v0.json`。
   - 必须复用 A2 的规则检测函数。
4. B 并行实现 `B2` 误判类型识别。
5. A+B 执行 `G1`：
   - 同一条内容在 A2 和 B5 的命中规则、排除词、证据必须一致。

`G1` 通过后，规则函数才能被在线检测和回放评测稳定复用。

### 阶段 2：在线检测真实链路，完成 G2

目标：真实 DetectionResult + TraceRecord 能被 B 读取和分析。

1. A 实现 `A3/A4/A5`：
   - 语言识别或 fallback。
   - 风险类型初判，`fraud` 是示例但不能写死为唯一类型。
   - 算法分数节点。
2. A 实现 `A6/A7/A12/A8`：
   - LLM Judge 节点。
   - 决策聚合节点。
   - 图片 OCR 文本信号接入。
   - 在线检测 Pipeline，写正式 TraceRecord。
3. B 并行实现 `B3` mock 版：
   - 基于 G0 mock joined records 输出 `ErrorAnalysisResult`。
   - 内部走默认本地 backend。
4. A+B 执行 `G2`：
   - A 跑真实 `/moderate` 或 `run_online_detection`。
   - 生成真实 DetectionResult + TraceRecord。
   - B1 能读取真实 Trace。
   - B3 能基于真实 Trace 和 mock Review 输出分析。

`G2` 不通过，不进入真实人审回流分析。

### 阶段 3：人审回流与误判分析，完成 G3

目标：ReviewResult 能驱动真实误判分析。

1. A 实现 `A10` 人审提交逻辑：
   - 返回正式 ReviewResult。
   - 自动计算 `error_type`。
   - 写入 `data/review_results.jsonl`。
2. A 实现 `A11` 人审页或最小人审操作界面。
3. B 完成 `B3` 正式版：
   - 输入真实 Trace + ReviewResult。
   - 输出 `data/error_analysis_results.jsonl`。
   - 给出 `root_cause`、`analysis`、`suggested_debug_action` 和 `candidate_terms`。
4. B 实现 `B4` 数据层或简版页面。
5. A+B 执行 `G3`：
   - 至少覆盖 `false_negative`、`false_positive`、`category_error` 三类样例。

### 阶段 4：候选规则与回放，完成 G4 / G5

目标：candidate rules 可写入、可检测、可回放。

1. B 实现 `B6`：
   - 写入 `rules/candidate_rules.json`。
   - 明确候选规则不自动上线。
2. B 完善 `B5` 第二版：
   - 支持 `rules_path=rules/candidate_rules.json`。
3. A 配合确认 A2 规则加载函数同时支持 `rules_v0.json` 和 `candidate_rules.json`。
4. A+B 执行 `G4`：
   - B 写 candidate rule。
   - B5 能调试 candidate rule。
   - A2 能加载同一 candidate rule。
5. B 实现 `B7` 回放评测器：
   - 读取 `data/sample_cases.jsonl`。
   - 对比 old rules 和 candidate rules。
   - 生成 `data/evaluation_report.json`。
6. B 实现 `B8` 回放报告页或最小报告 API。
7. A+B 执行 `G5`：
   - 共同确认 `evaluation_report.json` 指标和候选规则影响。

### 阶段 5：训练触发与最终 Demo，完成 G6 / Demo

目标：训练触发只是接口和记录，不执行真实训练。

1. B 实现 `B9`：
   - 支持 `manual/scheduled`。
   - 返回 `TrainingTriggerResult`。
   - 不执行真实训练。
2. A 配合提供默认 dataset path，例如 `data/review_results.jsonl`。
3. A+B 执行 `G6`：
   - 触发一次 manual。
   - 记录 `trigger_id`、`dataset_path`、`status` 和 `note`。
4. A+B 做最终 Demo：
   - 文本涉诈命中规则。
   - 正常文本输出 `pass`。
   - 图片 OCR 后进入文本检测。
   - review 样本进入人审并落盘。
   - B 输出误判分析。
   - B 写 candidate rules。
   - B 回放生成 evaluation report。
   - B 训练触发接口返回 queued。

### 关键依赖闸门

```text
S0.1-S0.7 + G0
→ A1/B1

A2
→ B5 第一版
→ G1

A8 + A12
→ B1/B3 接真实 Trace
→ G2

A10/A11
→ B3 接真实 ReviewResult
→ G3

B6 + B5 第二版
→ B7 真实 candidate 回放
→ G4/G5

B9
→ G6
```
