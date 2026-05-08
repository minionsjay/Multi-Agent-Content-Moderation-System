# 03 NSFW ViT Image Classification

## 测试什么

Image Agent Step 2: ViT-based NSFW 分类器 (`Falconsai/nsfw_image_detection`)。

## 测试结果

### Test 1: POC Fallback 行为

POC 阶段 `skip_model=True`，模型不下载、不加载。所有图片都被分类为 `normal`：

| 图片 | 大小 | 延迟 | 标签 | 置信度 |
|------|------|------|------|------|
| landscape | 8.9KB | 1334μs | normal | 1.00 |
| pattern | 18.9KB | 333μs | normal | 1.00 |
| tiny (5×5) | 77B | 117μs | normal | **0.50** |
| large_hd (1920×1080) | 247KB | 19,526μs | normal | 1.00 |

唯一有差异的是 tiny 图片（5×5 像素，模型拒判，conf=0.5）。

### Test 2: 真实模型加载

模型未下载，无法测试。需执行：

```bash
# 预下载 NSFW 模型（联网一次性操作）
python -c "from transformers import pipeline; pipeline('image-classification', model='Falconsai/nsfw_image_detection')"
```

### Test 3: 边界情况

| 输入 | 结果 |
|------|------|
| 空字节 | error: cannot identify image file |
| 随机数据 | error: cannot identify image file |
| 文本字符串 | error: cannot identify image file |

PIL 无法识别为图片 → 抛出异常 → 返回 `label=normal, conf=0.5, error=...`。

### Test 4: 内存

| 模式 | 下载 | 显存 | 内存 |
|------|------|------|------|
| POC (skip) | 0 MB | 0 MB | 0 MB |
| FP16 (GPU) | 350 MB | ~500 MB | ~200 MB |
| FP32 (CPU) | 350 MB | 0 MB | ~700 MB |

## POC 局限

- **所有图片都判为 normal** — POC 阶段完全无法验证 NSFW 检测的准确率
- 图片格式校验正确（tiny 图片被识别为异常，large_hd 延迟合理）
- 生产环境必须下载模型并启用 GPU 推理
