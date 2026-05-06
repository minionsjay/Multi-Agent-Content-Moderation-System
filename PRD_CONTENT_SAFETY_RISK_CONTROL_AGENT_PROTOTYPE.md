# 内容安全风控 Agent Prototype PRD

文档版本：V0.1  
文档类型：Prototype PRD / 产品开发需求文档  
适用对象：Coding Agent、开发人员、内容风控策略人员、技术负责人  
开发周期目标：2 人力 / 3 天内跑通  
产品定位：证明型 Prototype，不是生产级系统  
核心目标：证明 Agent 化链路对内容安全风控判断、人工提效、规则优化闭环有实际价值

---

## 1. 项目背景

当前内容安全风控系统已有基础检测链路：

```text
关键词 / 规则 → 小模型 / 路由模型 / MOE → 大模型分析 → 人工确认
```

其中：

1. **关键词 + 小模型层**承担高速检测能力，要求低延迟、低成本、高吞吐，适合处理明确违规、高频模式、规则可覆盖的内容。
2. **大模型层**不作为全量实时在线检测层，而是处理前两层无法判断、低置信度、多语种、边界语义、疑难风险 case，输出结构化分析、风险证据、置信度和涉诈特征，最终交由人工确认。
3. **人工审核结果**当前尚未充分沉淀为规则优化、评测样本、特征抽取和策略迭代资产。
4. 业务长期目标是向 **0 人审** 演进。虽然短期内无法完全实现 0 人审，但系统建设方向应从“人工逐条判断”逐步转向“系统自动判断 + 人工处理疑难 + 人工确认策略优化”。

当前主要痛点包括：

### 1.1 抽样侧痛点

当前从服务日志中抽样检查并产生告警，告警量约 5 万，原始日志数据量更大。现有抽样规则较粗糙，存在：

```text
同一域名重复抽样过多；
正规大客户抽样过多，容易引发客诉；
抽样覆盖不够广；
缺少基于算法识别结果的动态抽样优化。
```

本 Prototype 暂不完整解决抽样策略优化，但会预留相关字段，为后续 Sampling Strategy Agent 扩展做准备。

### 1.2 算法侧痛点

当前希望提升大模型审核效率和准确率，在保持检出率的前提下减少人工确认压力，并逐步向 0 人审靠近。

其中，涉诈场景需要大模型具备：

```text
涉诈特征提取；
外部联系方式识别；
非法服务暗示识别；
规避表达识别；
承诺性表达识别；
可沉淀为规则 / 特征 / 训练数据的结构化输出。
```

本 Prototype 将重点覆盖涉诈特征提取。

### 1.3 审核侧痛点

监管部门下发的违规站点存在大量相似图片，希望能从知识库中关联历史违规站点，提高人审准确率。

本 Prototype 暂不覆盖相似图片检索、图片知识库、向量检索等能力。该能力进入后续 MVP 规划。

---

## 2. 产品目标

### 2.1 Prototype 核心目标

本 Prototype 只验证两个核心闭环：

#### 目标一：在线检测链路可追踪

通过一个轻量化 Agent Graph 跑通：

```text
文本输入
→ 语言识别
→ 规则检测
→ 模拟小模型 / 算法分数判断
→ 大模型结构化分析
→ 决策聚合
→ 输出检测结果与 Trace
```

证明在线检测链路可以从散乱 if-else 升级为：

```text
流程可编排；
状态可追踪；
节点可替换；
结果可解释；
case 可回放；
成本可统计。
```

#### 目标二：旁路评测可基于人工判定优化规则

通过人工确认结果回流，跑通：

```text
人工判定
→ 漏检 / 误杀识别
→ 大模型归因分析
→ 涉诈特征提取
→ 候选规则 / 特征规则生成
→ 离线回放评测
→ 新旧规则效果对比
```

证明旁路 Agent 可以辅助规则运营和策略优化。

---

## 3. 非目标范围

本 Prototype 明确不做以下内容：

| 不做内容 | 原因 |
| --- | --- |
| 图片、视频、音频、多模态检测 | 3 天内复杂度过高 |
| OCR / ASR | 会引入额外工程链路 |
| 相似图片知识库检索 | 需要图片 embedding、向量库、历史违规库 |
| 9 个小语种完整覆盖 | Prototype 只证明多语种可路由 |
| 小模型 / MOE 真实训练 | 本阶段证明闭环，不证明训练能力 |
| 完整抽样策略优化 | 依赖真实日志分布和业务策略 |
| 真实线上灰度 | 本阶段只做离线回放 |
| 完整 A/B test 平台 | 本阶段只做新旧规则对比 |
| Agent 自动上线规则 | 风控场景必须保留人工审批 |
| 复杂权限系统 | Prototype 不需要 |
| 复杂前端工程 | Streamlit / Gradio 即可 |
| 生产级数据库 | JSONL / SQLite 即可 |

---

## 4. Prototype 一句话定义

本 Prototype 是一个：

> **面向 0 人审目标的文本内容安全风控 Agent 原型：用规则和模拟小模型分数承担前两层高速漏斗，对无法判断的疑难 case 调用大模型进行结构化风险分析和涉诈特征提取，并将人工确认结果回流到旁路评测模块，生成候选规则和特征优化建议，再通过离线回放验证效果。**

---

## 5. 用户角色

### 5.1 内容审核员

主要使用人工审核页面。

核心需求：

```text
快速看到系统判断结果；
快速理解为什么被判风险；
看到命中规则、证据片段、大模型理由；
对系统判断进行确认或修正；
补充人工判断原因。
```

### 5.2 风控策略运营人员

主要使用误判分析、规则建议、回放评测页面。

核心需求：

```text
看到哪些 case 被漏检；
看到哪些 case 被误杀；
看到 Agent 给出的归因；
看到候选规则和特征建议；
判断是否接受候选规则；
查看新旧规则效果对比。
```

### 5.3 技术负责人 / 业务负责人

主要关注 Prototype 是否值得继续投入。

核心需求：

```text
判断 Agent 是否能提升审核效率；
判断是否不影响在线检测性能；
判断成本是否可控；
判断是否能形成持续优化闭环；
判断是否具备向 0 人审演进的可能性。
```

---

## 6. 总体业务流程

```text
输入文本 case
  ↓
语言识别节点
  ↓
规则检测节点
  ↓
模拟小模型 / 算法分数节点
  ↓
大模型 Judge 节点
  ↓
决策聚合节点
  ↓
输出系统判断 + Trace
  ↓
人工审核确认
  ↓
人工结果回流
  ↓
旁路识别漏检 / 误杀 / 类别错误
  ↓
大模型归因分析
  ↓
候选规则 / 涉诈特征规则生成
  ↓
人工接受候选规则
  ↓
离线回放评测
  ↓
输出新旧规则对比报告
```

---

## 7. 系统模块

Prototype 包含 5 个模块：

| 模块 | 说明 |
| --- | --- |
| 在线检测模块 | 跑通文本内容检测链路 |
| Trace 追踪模块 | 保存每个节点执行结果 |
| 人工审核回流模块 | 采集人工确认结果 |
| 旁路误判分析模块 | 识别漏检、误杀并做归因 |
| 规则建议与离线回放模块 | 生成候选规则并验证效果 |

---

## 8. 功能需求

### 8.1 功能模块一：在线检测模块

#### 8.1.1 功能说明

用户输入一条文本内容，系统执行内容风控检测链路，输出风险判断、证据、置信度、大模型理由、涉诈特征和完整 Trace。

#### 8.1.2 输入字段

| 字段 | 类型 | 是否必填 | 说明 |
| --- | --- | ---: | --- |
| case_id | string | 否 | 不填则系统自动生成 |
| content | string | 是 | 待检测文本 |
| source | string | 否 | 来源，如 service_log_sample、comment、profile |
| country | string | 否 | 国家 / 站点，如 CN、ID、US |
| language | string | 否 | 可不填，由系统识别 |
| domain | string | 否 | 域名，用于后续抽样优化预留 |
| customer_type | string | 否 | normal / key_account |
| algorithm_score | float | 否 | 模拟小模型 / 算法分数 |
| sample_reason | string | 否 | 抽样原因，如 random_sample、high_score_sample |

#### 8.1.3 输入示例

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

#### 8.1.4 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| final_decision | string | pass / suspected_violation / need_human_review |
| risk_type | string | fraud / porn / hate_or_abuse / unknown |
| confidence | float | 综合置信度 |
| evidence | list[string] | 证据片段 |
| matched_rules | list[string] | 命中规则 ID |
| llm_reason | string | 大模型判断理由 |
| fraud_features | object | 涉诈特征结构化输出 |
| need_human_review | bool | 是否需要人工确认 |
| latency_ms | int | 总耗时 |
| llm_called | bool | 是否调用大模型 |
| cost_estimate | float | 成本估算 |
| trace_id | string | Trace ID |

#### 8.1.5 输出示例

```json
{
  "final_decision": "need_human_review",
  "risk_type": "fraud",
  "confidence": 0.86,
  "evidence": ["加我电报", "低价办证", "包过"],
  "matched_rules": ["fraud_001"],
  "llm_reason": "内容疑似引导用户通过外部联系方式办理非法证件，存在欺诈或违法交易风险。",
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

### 8.2 功能模块二：语言识别节点

#### 8.2.1 功能说明

识别输入文本语言，用于后续选择中文策略或多语种策略。

#### 8.2.2 Prototype 覆盖语言

| 语言 | 是否必须 |
| --- | ---: |
| 中文 zh | 必须 |
| 英文 en | 必须 |
| 1 个小语种，例如 id / es / ar | 建议 |
| other | 必须兜底 |

#### 8.2.3 输出示例

```json
{
  "node": "language_detection",
  "language": "zh",
  "confidence": 0.96
}
```

#### 8.2.4 验收标准

| 验收项 | 标准 |
| --- | --- |
| 中文识别 | 可稳定识别 |
| 英文识别 | 可稳定识别 |
| 小语种识别 | 能初步识别 |
| 无法识别 | 返回 other，不中断流程 |

### 8.3 功能模块三：风险类型初判节点

#### 8.3.1 功能说明

对内容做轻量风险类型初判，为后续规则、大模型判断和人工审核提供上下文。

#### 8.3.2 风险类型范围

| 风险类型 | 说明 |
| --- | --- |
| fraud | 涉诈、引流、非法交易 |
| porn | 色情、低俗 |
| hate_or_abuse | 仇恨、辱骂、攻击性表达 |
| unknown | 无法判断或其他 |

#### 8.3.3 输出示例

```json
{
  "node": "risk_preclassify",
  "risk_candidates": ["fraud"],
  "reason": "内容包含办证、加电报、包过等欺诈引流特征"
}
```

### 8.4 功能模块四：规则检测节点

#### 8.4.1 功能说明

基于 JSON 规则库进行关键词 / 正则匹配，输出命中规则、命中词、风险类型和严重程度。

#### 8.4.2 规则配置结构

```json
{
  "rule_id": "fraud_001",
  "risk_type": "fraud",
  "language": "zh",
  "keywords": ["加电报", "低价办证", "包过", "私聊"],
  "pattern": "",
  "action": "flag",
  "severity": "high",
  "version": "v0.1"
}
```

#### 8.4.3 输出示例

```json
{
  "node": "rule_check",
  "hit": true,
  "matched_rule_id": "fraud_001",
  "matched_terms": ["加电报", "低价办证"],
  "risk_type": "fraud",
  "severity": "high",
  "decision": "suspected_violation"
}
```

#### 8.4.4 验收标准

| 验收项 | 标准 |
| --- | --- |
| 规则可配置 | 从 JSON 文件读取 |
| 命中可解释 | 输出 rule_id 和 matched_terms |
| 版本可追踪 | 输出规则版本 |
| 可替换 | 支持加载 rules_v0 和 candidate_rules |
| 可回放 | 离线评测复用同一套规则逻辑 |

### 8.5 功能模块五：模拟小模型 / 算法分数节点

#### 8.5.1 功能说明

Prototype 阶段不接入真实小模型，用输入字段 `algorithm_score` 模拟小模型或算法识别结果。

该节点用于表达三层漏斗中的第二层能力：

```text
前两层承担高速度、低成本、批量初筛；
当规则未命中但算法分数偏高时，进入大模型分析；
当规则未命中且算法分数低时，可直接 pass 或低优先级人工抽检。
```

#### 8.5.2 输入

```json
{
  "algorithm_score": 0.76
}
```

#### 8.5.3 输出示例

```json
{
  "node": "algorithm_score_check",
  "score": 0.76,
  "risk_level": "medium",
  "suggested_next_step": "llm_judge"
}
```

#### 8.5.4 阈值建议

| algorithm_score | risk_level | suggested_next_step |
| ---: | --- | --- |
| >= 0.8 | high | need_human_review 或 llm_judge |
| 0.5 - 0.8 | medium | llm_judge |
| < 0.5 | low | pass 或抽检池 |

### 8.6 功能模块六：大模型 Judge 节点

#### 8.6.1 功能说明

大模型 Judge 节点处理前两层无法明确判断的疑难 case，输出结构化风险分析、证据、置信度和涉诈特征。

大模型层不是全量在线实时层，不处理所有 case。

#### 8.6.2 调用条件

| 场景 | 是否调用大模型 |
| --- | ---: |
| 规则强命中且高危 | 可不调用，直接进入人工 |
| 规则未命中但 algorithm_score 中高 | 调用 |
| 多语种文本 | 调用 |
| 规则命中但疑似上下文误杀 | 调用 |
| 风险类型为 fraud 且证据不足 | 调用 |
| 无风险迹象且 algorithm_score 低 | 不调用 |

#### 8.6.3 大模型输出 Schema

```json
{
  "judge_result": "suspected_violation",
  "risk_type": "fraud",
  "confidence": 0.86,
  "evidence": ["加我电报", "低价办证", "包过"],
  "reason": "内容疑似引导用户通过外部联系方式办理非法证件，存在欺诈或违法交易风险。",
  "need_human_review": true,
  "fraud_features": {
    "external_contact": true,
    "contact_channel": ["Telegram"],
    "illegal_service": true,
    "service_type": "fake_document",
    "guarantee_terms": ["包过"],
    "evasion_terms": [],
    "payment_signal": false
  }
}
```

#### 8.6.4 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| judge_result | string | pass / suspected_violation / uncertain |
| risk_type | string | fraud / porn / hate_or_abuse / unknown |
| confidence | float | 大模型置信度 |
| evidence | list[string] | 原文证据片段 |
| reason | string | 简短判断理由 |
| need_human_review | bool | 是否建议人工确认 |
| fraud_features | object | 涉诈特征，仅 fraud 相关 case 强制输出 |

#### 8.6.5 涉诈特征字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| external_contact | bool | 是否引导外部联系 |
| contact_channel | list[string] | Telegram、WhatsApp、微信、私聊等 |
| illegal_service | bool | 是否涉及非法服务 |
| service_type | string | fake_document、loan、rebate、gambling、unknown |
| guarantee_terms | list[string] | 包过、稳赚、秒到账、保下款等 |
| evasion_terms | list[string] | 特殊渠道、懂的来、私我、暗号等 |
| payment_signal | bool | 是否出现付款、收益、返利等信号 |

#### 8.6.6 约束

大模型必须输出 JSON，不允许自然语言散文作为主输出。

如果解析失败，系统应返回：

```json
{
  "judge_result": "uncertain",
  "risk_type": "unknown",
  "confidence": 0.0,
  "evidence": [],
  "reason": "LLM output parse failed",
  "need_human_review": true
}
```

### 8.7 功能模块七：决策聚合节点

#### 8.7.1 功能说明

综合规则检测、算法分数和大模型判断结果，输出最终系统判断。

#### 8.7.2 决策规则

| 条件 | final_decision |
| --- | --- |
| 规则强命中高危 | need_human_review |
| 规则命中 + 大模型确认风险 | need_human_review |
| 规则未命中 + algorithm_score 中高 + 大模型判断风险 | need_human_review |
| 规则未命中 + algorithm_score 低 + 无风险迹象 | pass |
| 规则命中但大模型判断正常 | need_human_review |
| 大模型 uncertain | need_human_review |
| 多节点冲突 | need_human_review |

Prototype 阶段不做自动处置，只输出判断与人工确认建议。

#### 8.7.3 输出示例

```json
{
  "node": "decision_aggregator",
  "final_decision": "need_human_review",
  "risk_type": "fraud",
  "confidence": 0.78,
  "reason": "规则命中欺诈引流关键词，大模型也提取到外部联系和非法办证特征，建议人工确认。",
  "need_human_review": true
}
```

---

## 9. Trace 追踪需求

### 9.1 功能说明

每条 case 的完整执行过程必须记录，用于回溯、展示、回放和后续评测。

### 9.2 Trace 结构

```json
{
  "case_id": "case_001",
  "trace_id": "trace_001",
  "input": {
    "content": "加我电报，低价办证，包过",
    "source": "service_log_sample",
    "domain": "abc.com",
    "customer_type": "normal",
    "algorithm_score": 0.76
  },
  "language_node": {
    "language": "zh",
    "confidence": 0.96
  },
  "risk_preclassify_node": {
    "risk_candidates": ["fraud"],
    "reason": "内容包含欺诈引流特征"
  },
  "rule_node": {
    "hit": true,
    "matched_rule_id": "fraud_001",
    "matched_terms": ["加电报", "低价办证"]
  },
  "algorithm_score_node": {
    "score": 0.76,
    "risk_level": "medium",
    "suggested_next_step": "llm_judge"
  },
  "llm_judge_node": {
    "called": true,
    "judge_result": "suspected_violation",
    "confidence": 0.86,
    "fraud_features": {
      "external_contact": true,
      "contact_channel": ["Telegram"],
      "illegal_service": true
    }
  },
  "final_decision": {
    "decision": "need_human_review",
    "risk_type": "fraud",
    "confidence": 0.78
  },
  "runtime": {
    "latency_ms": 1280,
    "llm_called": true,
    "cost_estimate": 0.002
  },
  "version": {
    "rule_version": "v0.1",
    "prompt_version": "judge_v0.1"
  }
}
```

### 9.3 验收标准

| 验收项 | 标准 |
| --- | --- |
| 每条 case 有 trace_id | 必须 |
| 每个节点输出可查看 | 必须 |
| 是否调用大模型可查看 | 必须 |
| 耗时可查看 | 必须 |
| 规则版本可查看 | 必须 |
| Prompt 版本可查看 | 必须 |
| 页面可展开 Trace | 必须 |

---

## 10. 人工审核回流需求

### 10.1 功能说明

人工审核员对系统判断进行确认或修正，审核结果写入 `review_results.jsonl`，用于旁路分析。

### 10.2 页面展示字段

人工审核页面展示：

```text
原始内容
系统最终判断
风险类型
置信度
证据片段
命中规则
大模型判断理由
涉诈特征
完整 Trace
```

### 10.3 人工操作项

| 操作 | 说明 |
| --- | --- |
| 确认违规 | 人工认为该内容违规 |
| 确认正常 | 人工认为该内容正常 |
| 修正风险类别 | 修正 fraud / porn / hate_or_abuse |
| 填写人工原因 | 记录人工判断依据 |
| 提交审核结果 | 写入 review_results |

### 10.4 人工反馈数据结构

```json
{
  "case_id": "case_001",
  "system_decision": "suspected_violation",
  "system_risk_type": "fraud",
  "human_decision": "violation",
  "human_risk_type": "fraud",
  "is_correct": true,
  "human_reason": "明显引流办理非法证件",
  "reviewer": "auditor_001",
  "review_time": "2026-04-30 15:30:00"
}
```

### 10.5 误判类型自动识别

| system_decision | human_decision | error_type |
| --- | --- | --- |
| pass | violation | false_negative |
| suspected_violation / need_human_review | normal | false_positive |
| suspected_violation / need_human_review | violation | true_positive |
| pass | normal | true_negative |
| risk_type 不一致 | category_error | |

---

## 11. 旁路误判分析需求

### 11.1 功能说明

系统读取人工审核结果，自动识别漏检、误杀、类别错误，并调用大模型进行归因分析。

### 11.2 漏检分析

触发条件：

```text
system_decision = pass
human_decision = violation
```

输出示例：

```json
{
  "case_id": "case_010",
  "error_type": "false_negative",
  "root_cause": "rule_missing",
  "affected_risk_type": "fraud",
  "analysis": "该内容使用了'特殊证件'、'渠道稳'等规避表达，现有规则未覆盖。",
  "suggested_fix": "add_feature_rule",
  "candidate_terms": ["特殊证件", "渠道稳", "速度快", "私我"],
  "candidate_features": {
    "external_contact": true,
    "illegal_service_hint": true,
    "evasion_expression": true
  }
}
```

### 11.3 误杀分析

触发条件：

```text
system_decision = suspected_violation / need_human_review
human_decision = normal
```

输出示例：

```json
{
  "case_id": "case_015",
  "error_type": "false_positive",
  "root_cause": "keyword_too_broad",
  "problematic_rule_id": "fraud_001",
  "analysis": "内容为反诈提醒，并非真实引流。",
  "suggested_fix": "add_context_exclusion",
  "candidate_adjustment": "若上下文包含'提醒大家'、'不要相信'、'骗局'，降低风险分或进入人工确认。"
}
```

### 11.4 归因类型枚举

| root_cause | 说明 |
| --- | --- |
| rule_missing | 规则缺失 |
| keyword_too_broad | 关键词过宽 |
| threshold_too_high | 阈值过高导致漏检 |
| threshold_too_low | 阈值过低导致误杀 |
| llm_judge_error | 大模型判断错误 |
| context_missing | 缺少上下文 |
| language_mismatch | 语言或策略路由错误 |
| category_confusion | 风险类别混淆 |
| policy_gap | 策略定义缺失 |

---

## 12. 候选规则 / 特征规则生成需求

### 12.1 功能说明

旁路 Agent 根据漏检、误杀和人工反馈，生成候选规则或特征化规则建议。

Agent 不能直接上线规则，只能生成 proposal。

### 12.2 规则建议类型

| proposal_type | 说明 |
| --- | --- |
| add_keyword_rule | 新增关键词规则 |
| modify_keyword_rule | 修改已有关键词规则 |
| add_context_exclusion | 增加上下文排除条件 |
| add_fraud_feature_rule | 新增涉诈特征组合规则 |
| adjust_severity | 调整风险等级 |
| add_review_condition | 增加人工复核条件 |

### 12.3 候选规则结构

```json
{
  "proposal_id": "proposal_001",
  "proposal_type": "add_fraud_feature_rule",
  "risk_type": "fraud",
  "language": "zh",
  "candidate_keywords": ["特殊证件", "渠道稳", "速度快", "私我"],
  "candidate_features": {
    "external_contact": true,
    "illegal_service": true,
    "evasion_expression": true
  },
  "suggested_logic": "当内容同时出现外部联系意图、非法服务暗示和规避表达时，进入大模型分析或人工复核。",
  "expected_effect": "提升涉诈引流类召回率",
  "potential_risk": "可能误伤普通办事咨询或反诈提醒内容",
  "recommended_action": "进入离线回放验证",
  "need_human_approval": true
}
```

### 12.4 人工确认候选规则

人工可执行：

| 操作 | 结果 |
| --- | --- |
| 接受 | 写入 candidate_rules.json |
| 拒绝 | 标记 rejected |
| 编辑后接受 | 修改后写入 candidate_rules.json |

Prototype 阶段不允许自动上线。

---

## 13. 离线回放评测需求

### 13.1 功能说明

系统使用同一批样本，分别运行：

```text
当前规则集 rules_v0.json
候选规则集 candidate_rules.json
```

输出新旧规则效果对比。

### 13.2 输入数据

| 数据文件 | 说明 |
| --- | --- |
| sample_cases.jsonl | 历史样本 |
| review_results.jsonl | 人工标签 |
| rules_v0.json | 当前规则 |
| candidate_rules.json | 候选规则 |
| traces.jsonl | 历史执行 Trace，可选 |

### 13.3 输出指标

| 指标 | 说明 |
| --- | --- |
| total_cases | 样本总数 |
| violation_cases | 人工标注违规样本数 |
| normal_cases | 人工标注正常样本数 |
| hit_violation_count | 命中违规样本数 |
| false_negative_count | 漏检数 |
| false_positive_count | 误杀数 |
| recall | 召回率 |
| precision | 精准率 |
| estimated_human_review_count | 预计进入人工数量 |
| llm_call_count | 大模型调用次数 |
| new_hit_cases | 新规则新增命中的违规样本 |
| new_false_positive_cases | 新规则新增误伤样本 |

### 13.4 报告示例

| 指标 | 当前规则 | 候选规则 | 变化 |
| --- | ---: | ---: | ---: |
| 命中违规样本数 | 18 | 23 | +5 |
| 漏检数 | 7 | 2 | -5 |
| 误杀数 | 4 | 6 | +2 |
| 召回率 | 72% | 84% | +12% |
| 精准率 | 81% | 78% | -3% |
| 预计人工审核量 | 22 | 29 | +7 |
| 大模型调用次数 | 12 | 14 | +2 |

系统生成总结：

```text
候选规则提升了欺诈引流类召回率，但新增 2 条误杀样本。
建议该规则不要直接全量上线，可先用于高风险欺诈引流场景，或设置为“命中后进入人工复核”，而不是自动判定违规。
```

---

## 14. 页面需求

Prototype 建议使用 Streamlit 或 Gradio 实现 5 个页面。

### 14.1 页面一：在线检测页

页面目标：展示在线检测 Agent Graph。

页面元素：

```text
文本输入框
国家选择
语言选择，可选
domain 输入，可选
customer_type 选择，可选
algorithm_score 输入，可选
运行检测按钮
最终判断卡片
风险类型
置信度
证据片段
命中规则
大模型理由
涉诈特征
耗时 / 成本
Trace 展开区
```

### 14.2 页面二：人工审核页

页面目标：人工确认系统判断，并生成回流数据。

页面元素：

```text
待审核 case 列表
原始内容
系统判断
风险类型
置信度
证据片段
命中规则
大模型理由
涉诈特征
人工判断按钮：违规 / 正常
风险类别修正
人工原因输入框
提交按钮
```

### 14.3 页面三：误判分析页

页面目标：展示系统如何根据人工反馈识别漏检和误杀。

页面元素：

```text
分析人工反馈按钮
漏检列表
误杀列表
类别错误列表
每条 case 的归因结果
候选修复方向
```

### 14.4 页面四：规则建议页

页面目标：展示 Agent 生成的候选规则 / 特征规则。

页面元素：

```text
候选规则列表
proposal_type
risk_type
candidate_keywords
candidate_features
suggested_logic
expected_effect
potential_risk
接受 / 拒绝 / 编辑后接受按钮
```

### 14.5 页面五：离线回放报告页

页面目标：展示新旧规则对比，让负责人判断策略是否值得继续投入。

页面元素：

```text
运行回放按钮
当前规则 vs 候选规则指标表
新增命中样本列表
新增误伤样本列表
预计人工审核量变化
大模型调用次数变化
Agent 总结建议
```

---

## 15. 数据文件设计

### 15.1 sample_cases.jsonl

```json
{"case_id":"case_001","content":"加我电报，低价办证，包过","label":"violation","risk_type":"fraud","language":"zh","domain":"a.com","customer_type":"normal","algorithm_score":0.82}
{"case_id":"case_002","content":"这篇新闻提醒大家不要相信低价办证骗局","label":"normal","risk_type":"none","language":"zh","domain":"b.com","customer_type":"key_account","algorithm_score":0.64}
```

### 15.2 rules_v0.json

```json
[
  {
    "rule_id": "fraud_001",
    "risk_type": "fraud",
    "language": "zh",
    "keywords": ["加电报", "低价办证", "包过"],
    "pattern": "",
    "severity": "high",
    "version": "v0.1"
  }
]
```

### 15.3 candidate_rules.json

保存人工接受后的候选规则。

### 15.4 traces.jsonl

保存每次在线检测的完整 Trace。

### 15.5 review_results.jsonl

保存人工审核结果。

### 15.6 error_analysis_results.jsonl

保存旁路误判分析结果。

### 15.7 evaluation_report.json

保存新旧规则回放对比结果。

---

## 16. 样本准备要求

Prototype 演示前至少准备 60 条文本样本。

### 16.1 样本分布

| 样本类型 | 数量 |
| --- | -: |
| 中文正常内容 | 15 |
| 中文欺诈 / 引流 | 10 |
| 中文色情 / 低俗 | 10 |
| 中文辱骂 / 仇恨 | 10 |
| 英文风险内容 | 8 |
| 小语种风险内容 | 5 |
| 容易误杀内容 | 7 |
| 模拟漏检内容 | 5 |

### 16.2 必须包含的演示样本

#### 样本 1：规则直接命中

```text
加我电报，低价办证，包过。
```

预期：

```text
规则命中 fraud_001；
系统判定 need_human_review；
提取涉诈特征 external_contact、illegal_service、guarantee_terms。
```

#### 样本 2：规则漏检，大模型识别

```text
有需要特殊证件的可以私我，渠道稳，速度快。
```

预期：

```text
规则未命中；
algorithm_score 中高；
进入大模型 Judge；
大模型判断疑似涉诈；
提取“特殊证件、私我、渠道稳、速度快”等特征。
```

#### 样本 3：规则误杀，大模型纠偏

```text
这篇新闻提醒大家不要相信低价办证骗局。
```

预期：

```text
规则命中“低价办证”；
大模型识别为反诈提醒上下文；
系统输出 need_human_review；
人工确认正常；
旁路记录为 false_positive。
```

#### 样本 4：英文涉诈

```text
Contact me on Telegram for fake documents.
```

预期：

```text
语言识别 en；
进入多语种大模型判断；
提取 Telegram、fake documents 等涉诈特征。
```

#### 样本 5：边界正常内容

```text
游戏里这个角色杀疯了，太爽了。
```

预期：

```text
不应判定为真实暴力风险；
可 pass 或低风险。
```

---

## 17. 技术建议

| 模块 | 推荐 |
| --- | --- |
| 编排框架 | LangGraph |
| 后端语言 | Python |
| Demo UI | Streamlit |
| 数据存储 | JSONL / SQLite |
| 规则配置 | JSON |
| LLM 接入 | 内部大模型 API / 可用 LLM API |
| 语言识别 | langdetect / fastText / LLM |
| 回放评测 | Python 脚本 |
| 报告展示 | Streamlit 表格 + 文本总结 |

开发原则：

```text
低依赖；
低部署成本；
优先跑通闭环；
优先可演示；
不追求生产级架构；
每个节点输出结构化 JSON；
所有结果可落盘。
```

---

## 18. 建议项目目录

```text
content-moderation-agent-prototype/
  app.py
  README.md

  graph/
    online_graph.py
    nodes.py
    state.py

  rules/
    rules_v0.json
    candidate_rules.json

  prompts/
    llm_judge_prompt.txt
    error_analysis_prompt.txt
    rule_proposal_prompt.txt
    evaluation_summary_prompt.txt

  data/
    sample_cases.jsonl
    traces.jsonl
    review_results.jsonl
    error_analysis_results.jsonl
    evaluation_report.json

  offline/
    error_analyzer.py
    rule_generator.py
    replay_evaluator.py

  ui/
    online_detection_page.py
    human_review_page.py
    error_analysis_page.py
    rule_proposal_page.py
    replay_report_page.py

  utils/
    jsonl_store.py
    cost_estimator.py
    schema_validator.py
```

---

## 19. 开发排期

### Day 1：在线检测链路

目标：

```text
跑通文本输入 → 语言识别 → 规则检测 → 模拟算法分数 → 大模型 Judge → 决策聚合 → Trace 展示
```

交付：

```text
项目骨架；
CaseState；
规则库；
在线检测 Graph；
LLM Judge Prompt；
Trace 存储；
在线检测页面。
```

### Day 2：人工审核与误判分析

目标：

```text
跑通系统判断 → 人工确认 → 回流数据 → 漏检 / 误杀识别 → 归因分析
```

交付：

```text
人工审核页面；
review_results 存储；
误判识别逻辑；
Error Analysis Agent；
误判分析页面。
```

### Day 3：规则建议与离线回放

目标：

```text
跑通误判样本 → 候选规则 → 人工接受 → 离线回放 → 新旧规则对比报告
```

交付：

```text
Rule Proposal Agent；
候选规则确认机制；
candidate_rules 存储；
Replay Evaluator；
评测报告页面；
演示样本集；
演示脚本。
```

---

## 20. 验收标准

### 20.1 功能验收

| 模块 | 验收标准 |
| --- | --- |
| 在线检测 | 输入文本后能输出结构化判断 |
| 语言识别 | 能识别中文、英文和至少 1 个小语种或 other |
| 规则检测 | 能读取 JSON 规则并返回命中结果 |
| 模拟算法分数 | 能根据 score 决定是否进入大模型 |
| 大模型 Judge | 能输出固定 JSON Schema |
| 涉诈特征提取 | fraud case 能输出 fraud_features |
| 决策聚合 | 能输出 final_decision |
| Trace | 能展示每个节点执行结果 |
| 人工审核 | 能提交人工判断 |
| 回流数据 | 人工判断能写入 review_results |
| 误判识别 | 能自动区分漏检和误杀 |
| 归因分析 | 能输出 root_cause 和 suggested_fix |
| 规则建议 | 能生成候选规则 / 特征规则 |
| 规则确认 | 人工可接受候选规则 |
| 离线回放 | 能对比当前规则和候选规则 |
| 报告展示 | 能展示指标变化和 Agent 总结 |

### 20.2 业务验收

Prototype 需要证明：

#### 1）对告警判断有帮助

系统能展示：

```text
为什么判定风险；
命中了什么规则；
证据在哪里；
大模型为什么这么判断；
涉诈特征是什么；
是否需要人工确认。
```

#### 2）不明显影响在线性能

系统能展示：

```text
规则和算法分数优先执行；
只有疑难 case 调用大模型；
每条 case 有耗时统计；
每条 case 有成本估算；
旁路分析不阻塞在线链路。
```

#### 3）成本可控

系统能展示：

```text
大模型调用次数；
大模型调用比例；
单 case 成本估算；
规则命中可减少大模型调用。
```

#### 4）具备向 0 人审演进的潜力

系统能展示：

```text
人工判断结果可回流；
漏检 / 误杀可归因；
大模型能提取可复用涉诈特征；
候选规则可自动生成；
新旧规则可离线评测；
人工角色可从逐条判断转向策略确认。
```

---

## 21. 演示脚本

### Step 1：展示规则直接命中

输入：

```text
加我电报，低价办证，包过
```

展示：

```text
规则命中 fraud_001；
大模型提取涉诈特征；
系统输出 need_human_review；
Trace 可展开；
耗时和成本可见。
```

说明：

```text
明确风险可以由前两层快速处理，不需要所有内容全量走大模型。
```

### Step 2：展示规则漏检，大模型补充分析

输入：

```text
有需要特殊证件的可以私我，渠道稳，速度快。
```

展示：

```text
规则未命中；
algorithm_score 中高；
进入大模型；
大模型判断疑似涉诈；
提取特殊证件、私我、渠道稳、速度快等特征；
人工确认违规。
```

说明：

```text
大模型层用于提升疑难 case 的召回率和分析质量。
```

### Step 3：展示规则误杀，大模型纠偏

输入：

```text
这篇新闻提醒大家不要相信低价办证骗局。
```

展示：

```text
规则命中“低价办证”；
大模型识别为反诈提醒上下文；
人工确认正常；
旁路记录为 false_positive。
```

说明：

```text
大模型可以辅助降低粗规则带来的误杀。
```

### Step 4：展示旁路规则优化

系统基于人工反馈生成：

```text
新增涉诈特征规则：
特殊证件 + 私我 + 渠道稳 / 速度快 → 提升欺诈风险；
提醒大家 / 不要相信 / 骗局 → 降低自动风险或进入人工确认。
```

说明：

```text
旁路 Agent 不是简单加关键词，而是提炼可复用风控特征。
```

### Step 5：展示离线回放报告

展示：

```text
当前规则 vs 候选规则；
召回率变化；
精准率变化；
漏检变化；
误杀变化；
预计人工审核量变化；
大模型调用量变化。
```

说明：

```text
策略优化可以被数据验证，不再只靠经验判断。
```

---

## 22. 后续 MVP 扩展方向

Prototype 成功后，下一阶段 MVP 可扩展：

| 方向 | 说明 |
| --- | --- |
| 接入真实告警流 | 从真实服务日志 / 告警系统读取 case |
| 接入真实小模型 | 替换 algorithm_score 模拟节点 |
| 增加多语种覆盖 | 从 1 个小语种扩展到 9 个 |
| 增加抽样策略 Agent | 根据域名、客户类型、历史违规率、算法分数优化抽样 |
| 增加策略版本中心 | 管理规则、Prompt、模型版本 |
| 增加灰度系统 | 小范围验证候选策略 |
| 增加数据资产池 | hard case、漏检、误杀、回归集 |
| 接入人工审核平台 | 与现有审核系统打通 |
| 增加相似图片检索 | 关联监管下发违规站点和历史图片库 |
| 增加成本监控 | 按模型、风险类型、国家统计成本 |

---

## 23. 给 Coding Agent 的开发指令摘要

```text
请开发一个 Python + LangGraph + Streamlit 的内容安全风控 Agent Prototype。

目标是在 3 天内跑通一个证明型原型，不做生产级系统。

系统需要包含：
1. 在线检测页：输入文本 case，执行语言识别、规则检测、模拟算法分数、大模型 Judge、决策聚合，输出结构化判断和完整 Trace。
2. 人工审核页：展示系统判断、命中规则、大模型理由、涉诈特征，允许人工确认违规/正常、修正风险类别、填写原因，并写入 review_results.jsonl。
3. 误判分析页：读取人工审核结果，自动识别 false_negative、false_positive、category_error，并调用大模型输出 root_cause 和 suggested_fix。
4. 规则建议页：基于误判分析生成候选关键词规则或涉诈特征规则，人工可接受/拒绝/编辑后接受，接受后写入 candidate_rules.json。
5. 离线回放页：使用 sample_cases.jsonl、rules_v0.json、candidate_rules.json 做新旧规则对比，输出召回率、精准率、漏检数、误杀数、预计人工审核量、大模型调用次数和总结建议。

关键要求：
- 所有节点输出结构化 JSON。
- 每条 case 必须保存完整 trace。
- 大模型只处理前两层无法判断的疑难 case，不作为全量实时在线检测层。
- fraud 风险必须输出 fraud_features，包括 external_contact、contact_channel、illegal_service、service_type、guarantee_terms、evasion_terms、payment_signal。
- Agent 不允许自动上线规则，只能生成候选规则，由人工接受后进入 candidate_rules.json。
- Prototype 使用 JSON / JSONL / SQLite 均可，优先简单可跑。
- UI 用 Streamlit 或 Gradio，重点是可演示，不追求复杂前端。
```

---

## 24. 最终交付物清单

3 天结束时，开发应交付：

```text
可运行的本地 Prototype；
在线检测页面；
人工审核页面；
误判分析页面；
规则建议页面；
离线回放报告页面；
sample_cases.jsonl；
rules_v0.json；
candidate_rules.json；
traces.jsonl；
review_results.jsonl；
evaluation_report.json；
README 使用说明；
领导演示样本和演示路径。
```

---

## 25. 最终判断

本 Prototype 的成功标准不是“实现完整内容安全风控系统”，而是证明：

```text
Agent Graph 能让在线检测链路可追踪；
大模型能提升疑难 case 的结构化分析质量；
涉诈特征可以从大模型分析中沉淀出来；
人工确认结果可以回流；
旁路 Agent 能识别漏检和误杀；
候选规则可以被生成；
规则优化可以通过离线回放验证；
整个链路具备向 0 人审演进的工程基础。
```

