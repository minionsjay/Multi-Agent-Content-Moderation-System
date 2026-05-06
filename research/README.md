# 热路径性能优化 · 技术深入研究

> 针对多服务高并发场景下热路径瓶颈的解决方案，包含原理说明、benchmark 脚本和可执行示例。

## 场景假设

10 个服务 × 500 万条/天 = 50,000,000 条/天
- 平均 QPS: 578
- 峰值 QPS: 5,780 (10x)
- 突发 QPS: 11,560 (20x)

## 瓶颈定位

热路径唯一瓶颈: **BGE Embedding（单核 200 QPS）**

| 层 | 单核 QPS | 578 QPS 均值 | 5780 QPS 峰值 |
|------|------|------|------|
| L0 内存缓存 | 100,000 | OK | OK |
| AC 自动机 | 3,000 | OK | OK |
| BGE Embedding | **200** | **超了 (272)** | **严重不足** |
| ChromaDB | 1,000 | OK | OK |

## 解决方案索引

| 编号 | 方案 | 提升倍数 | 改动量 | Phase |
|------|------|------|------|------|
| 00 | Embedding 缓存 | 1.5x | 10 行 | POC |
| 01 | 多 Worker | N 倍 | 1 行命令 | POC |
| 02 | 批量推理 | 2-3x | 中等 | 灰度 |
| 03 | ONNX 导出 | 3x | 中等 | 灰度 |
| 04 | Redis 共享缓存 | 集群化 | 中等 | 灰度 |
| 05 | 全链路水平扩展 | 无限 | 大 | 全量 |

## 快速开始

```bash
# 方案 00: Embedding 缓存基准测试
cd 01_embedding_cache
python bench_embed_cache.py

# 方案 02: 批量推理基准测试
cd ../03_bge_batch
python bench_batch.py

# 方案 03: 导出 ONNX 模型
cd ../04_onnx_export
python export_bge_onnx.py
```
