# 方案 05: 全链路水平扩展

## 问题

前面的方案都是单机优化，最终都会遇到物理极限（CPU 核数、内存带宽、网卡吞吐）。当日均请求量达到亿级别时，必须多机水平扩展。

## 目标架构

```
                    ┌──────────────────────────────┐
                    │     Nginx / Envoy (L7 LB)    │
                    │     rate_limit · health_check │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   API Instance 1 │  │   API Instance 2 │  │   API Instance N │
    │   (FastAPI × M   │  │   (FastAPI × M   │  │   (FastAPI × M   │
    │    workers)      │  │    workers)      │  │    workers)      │
    │                 │  │                 │  │                 │
    │  Gateway:       │  │  Gateway:       │  │  Gateway:       │
    │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
    │  │ L0 内存   │  │  │  │ L0 内存   │  │  │  │ L0 内存   │  │
    │  │ (本地)    │  │  │  │ (本地)    │  │  │  │ (本地)    │  │
    │  │ AC 自动机 │  │  │  │ AC 自动机 │  │  │  │ AC 自动机 │  │
    │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  Redis Cluster  │ │ ChromaDB Server │ │ Embedding Pool  │
    │  (共享 L0 缓存) │ │ (共享 L1 缓存) │ │ (BGE × K GPU)  │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                                         │
              ▼                                         ▼
    ┌─────────────────┐                   ┌─────────────────────────┐
    │  Celery / Ray   │ ←── async ─────── │  LLM Service (vLLM)     │
    │  Worker Pool    │                   │  (Qwen2.5-7B × N GPU)  │
    └────────┬────────┘                   └─────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  PostgreSQL     │
    │  (审计日志 ·     │
    │   用户画像)     │
    └─────────────────┘
```

## 各层扩容方案

### 1. API 层（无状态，水平扩容）

```
扩容方式: 增加 API 实例数
瓶颈: CPU (BGE Embedding)
触发条件: CPU > 70% 或 P99 latency > 100ms
```

Kubernetes HPA 配置：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: moderation-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: moderation-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: moderation_requests_per_second
      target:
        type: AverageValue
        averageValue: "500"
```

### 2. Embedding 层（有状态，按需扩容）

```
扩容方式: 增加 BGE GPU 副本数
瓶颈: GPU 吞吐
触发条件: 请求排队 > 100 或 GPU 利用率 > 80%
```

### 3. 冷路径（异步，削峰填谷）

实时审核（直播）和近实时审核（评论）走同步路径，批量审核（帖子、历史回溯）走异步：

```python
# 同步路径：直播/评论 → 必须 < 1s 返回
@app.post("/moderate")
async def moderate(req: ModerationRequest):
    return await process_sync(req)

# 异步路径：帖子/回溯 → 可以等几分钟
@app.post("/moderate/async")
async def moderate_async(req: ModerationRequest):
    task_id = await celery.send_task("moderate", args=(req.dict(),))
    return {"task_id": task_id, "status": "queued"}

@app.get("/moderate/result/{task_id}")
async def get_result(task_id: str):
    return await celery.get_result(task_id)
```

## QPS 容量规划

| 组件 | 单实例 QPS | 实例数 (10K QPS) | 实例数 (100K QPS) |
|------|------|------|------|
| API (含 Gateway) | 500 | 20 | 200 |
| BGE Embedding (CPU) | 200 | 50 | 500 |
| BGE Embedding (GPU) | 5,000 | 2 | 20 |
| ChromaDB | 1,000 | 10 | 100 |
| LangGraph (冷路径) | 50 | 200 | 2,000 |
| LLM (vLLM GPU) | 200 | 50 | 500 |

## 部署拓扑（100K QPS 目标）

```yaml
# docker-compose.yml (production overlay)
services:
  # API: 200 instances (50 per machine × 4 machines)
  api:
    deploy:
      replicas: 200
      resources:
        limits: { memory: "4Gi", cpu: "2000m" }

  # Embedding: 20 GPU instances
  embedding:
    deploy:
      replicas: 20
      resources:
        limits: { memory: "8Gi", "nvidia.com/gpu": 1 }

  # ChromaDB: 100 shards
  chromadb:
    deploy:
      replicas: 100

  # LangGraph workers: 2000 async workers
  langgraph-worker:
    deploy:
      replicas: 2000
```

## Nginx 配置

```nginx
upstream moderation_api {
    least_conn;
    keepalive 64;
    # Dynamic upstream via DNS / K8s service discovery
    server api.moderation.svc.cluster.local:8000;
}

server {
    listen 80;

    # Rate limiting: 500 req/s per IP
    limit_req_zone $binary_remote_addr zone=perip:10m rate=500r/s;

    location /moderate {
        limit_req zone=perip burst=100 nodelay;
        proxy_pass http://moderation_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # Health check endpoint
    location /health {
        proxy_pass http://moderation_api/health;
    }
}
```

## 容量演进路线

```
Phase 1 (POC):                Phase 2 (灰度):              Phase 3 (全量):
  1 台机器                      2-4 台机器                   N 台机器 (K8s)
  ┌──────────┐                 ┌──────────────────┐         ┌──────────────────────────┐
  │ 单进程    │                 │ API × 4 workers   │         │ API × 200 (K8s HPA)      │
  │ ChromaDB  │                 │ ChromaDB (独立)   │         │ ChromaDB Cluster          │
  │ Embedded  │                 │ Redis (共享缓存)  │         │ Embedding Pool (GPU)      │
  │ LLM API   │                 │ ONNX BGE          │         │ Ray Workers              │
  └──────────┘                 └──────────────────┘         │ vLLM (自建)               │
   500 QPS                       5,000 QPS                  └──────────────────────────┘
                                                             100,000 QPS
```
