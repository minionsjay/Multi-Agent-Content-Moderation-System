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
```

原 POC 已跑通的 FastAPI、Gateway、LangGraph、图文审核、缓存、人审队列、离线反馈基础能力继续保留。新增 SDD / Spec 文档用于规范后续开发入口、数据契约和模块分工。

## 2. 启动方式

技术入口保持不变：

```bash
cp .env.example .env
pip install -r requirements.txt
python check_env.py
python -m src.api
```

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
  └── 模型训练触发接口预留
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
| 旁路调试 | 误判分析、规则回放、训练触发接口预留 |

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
