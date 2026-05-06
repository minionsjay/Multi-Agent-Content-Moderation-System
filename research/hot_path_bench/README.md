# Hot Path Technology Benchmarks

> 逐层测试热路径每一项技术：从 L0 内存缓存到完整 Gateway 流程，最后到多服务负载测试。

## 测试结果总览

| # | 测试项 | 关键指标 | 结论 |
|------|------|------|------|
| 01 | L0 Memory Cache | **739K-957K QPS** | 纯内存操作，不是瓶颈 |
| 02 | AC Automaton | **O(1) vs 词库大小, 25x 快于暴力匹配** | 关键词扫描不是瓶颈 |
| 03 | jieba Context | **20/22 正确 (91%)** | 白名单和分词验证有效预防误杀 |
| 04 | ChromaDB Cache | **Read 956 QPS, Write 36 QPS** | 读够快，写是潜在瓶颈 |
| 05 | BGE Embedding | **单条 1637 QPS, 批量 21167 QPS** | 批量推理 13x 加速 |
| 06 | Gateway Full | **968 QPS, 1.03ms/req** | 单核近千 QPS |
| 07 | Multi-Service Load | **227 QPS, P99=16ms** | 单核刚好覆盖 5 服务场景 |

## 瓶颈链

```
当前硬件 (6.7 GB RAM, 无 GPU):

L0 内存缓存:    957,000 QPS  ████████████████████████████ 无限
AC 自动机:      300,000 QPS  ████████████████████████████ 无限
ChromaDB 读:       956 QPS  ████                        够用
BGE Embedding:     200 QPS  █                            瓶颈！
  + 缓存 (91%命中)   1,800 QPS  ███████                      解除
  + 批量推理    21,000 QPS  ██████████████████████████   解除
  + ONNX 加速    5,000 QPS  ██████████████████           解除
ChromaDB 写:        36 QPS  ▏                            潜在瓶颈
```

## 运行方式

```bash
# 逐项运行
cd research/hot_path_bench
python 01_l0_memory_cache/bench.py
python 02_ac_automaton/bench.py
python 03_jieba_context/test_cases.py
python 04_chromadb_cache/bench.py
python 05_bge_embedding/bench.py
python 06_gateway_full/bench.py

# 多服务负载测试
python 07_multi_service_load/load_test.py
python 07_multi_service_load/load_test.py --services 10 --per-service 5000
```
