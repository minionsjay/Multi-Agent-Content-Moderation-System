#!/usr/bin/env python3
"""
小模型 (BERT) 本地加载测试

用于测试本地下载的 BERT 分类模型是否能正常加载和推理。
支持 HuggingFace 模型 ID 或本地路径。

使用方法:
    # 测试 HF 模型 (自动下载到 ~/.cache/huggingface/)
    python tests/test_bert_local.py --model KoalaAI/Text-Moderation

    # 测试本地路径
    python tests/test_bert_local.py --model /home/ninini/models/my-bert-model

    # 测试默认模型 (读取 BERT_MODEL 环境变量)
    python tests/test_bert_local.py

模型文件要求 (本地路径需包含):
    config.json           # 模型配置
    pytorch_model.bin 或 model.safetensors  # 权重
    tokenizer_config.json # tokenizer 配置
    vocab.txt 或 tokenizer.json  # 词表

运行前先设置环境变量:
    cp .env.example .env
    # 编辑 .env, 设置 BERT_MODEL=/your/local/path
"""

import argparse
import os
import sys
import time
import json


def test_model(model_path: str):
    """加载并测试一个 BERT 分类模型。"""
    print(f"\n{'='*60}")
    print(f"BERT 模型加载测试")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")

    # -- 1. 检查路径是否存在 (仅本地路径) --
    if os.path.exists(model_path):
        required = ["config.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
        # 检查权重文件 (pytorch_model.bin 或 model.safetensors 或 tf_model.h5)
        weight_files = [
            f for f in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
            if os.path.exists(os.path.join(model_path, f))
        ]
        if missing:
            print(f"\n[WARN] 缺少文件: {missing}")
            print(f"  HuggingFace 模型目录通常需要: config.json, tokenizer_config.json, vocab.txt, 权重文件")
            if not weight_files:
                print(f"  [ERROR] 未找到权重文件 (pytorch_model.bin / model.safetensors / tf_model.h5)")
        if not weight_files:
            print(f"  将尝试从 HuggingFace Hub 加载...")
        else:
            print(f"  找到权重: {weight_files[0]}")
            print(f"  模型文件检查通过 ✓")
    else:
        print(f"  本地路径不存在，将作为 HuggingFace 模型 ID 处理")
        print(f"  (模型将下载到 ~/.cache/huggingface/hub/)")

    # -- 2. 加载模型 --
    print(f"\n[1/3] 加载模型...")
    t0 = time.perf_counter()

    from transformers import pipeline, AutoConfig
    import torch

    # 先看配置
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  模型架构: {config.model_type}")
        if hasattr(config, "num_labels"):
            print(f"  标签数:   {config.num_labels}")
        if hasattr(config, "id2label") and config.id2label:
            labels = list(config.id2label.values())[:10]
            print(f"  标签示例: {labels}")
    except Exception as e:
        print(f"  [WARN] 无法读取配置: {e}")

    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU (cuda:0)" if device == 0 else "CPU"
    print(f"  推理设备: {device_name}")

    try:
        pipe = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=device,
            truncation=True,
            max_length=512,
        )
        load_time = time.perf_counter() - t0
        print(f"  加载耗时: {load_time:.1f}s ✓")
    except Exception as e:
        print(f"\n[ERROR] 模型加载失败: {e}")
        print(f"\n可能的原因:")
        print(f"  1. 路径不存在或拼写错误")
        print(f"  2. 模型文件不完整 (缺少 config.json 或权重)")
        print(f"  3. 磁盘空间不足")
        print(f"  4. transformers 版本不兼容 (pip install --upgrade transformers)")
        print(f"\n如果模型在 HuggingFace 上, 可以先用 python 下载:")
        print(f"  from transformers import AutoModel")
        print(f"  AutoModel.from_pretrained('{model_path}').save_pretrained('./models/my-model')")
        sys.exit(1)

    # -- 3. 推理测试 --
    print(f"\n[2/3] 推理测试...")

    test_cases = [
        # (文本, 预期类别, 语言)
        ("今天天气真好，适合出去郊游野餐", "safe", "中文"),
        ("你真是个傻逼，什么都不懂还在这瞎说", "unsafe", "中文辱骂"),
        ("支持台独，台湾是一个独立的国家", "unsafe", "政治敏感"),
        ("我要杀了你这个混蛋，你等着瞧吧", "unsafe", "暴力威胁"),
        ("这个产品非常好用，强烈推荐给大家", "safe", "中文正常"),
        ("you are a worthless piece of garbage", "unsafe", "英文辱骂"),
        ("the sunset looks absolutely stunning", "safe", "英文正常"),
        ("操场上同学们在操练队列，体育老师很严格", "safe", "白名单测试"),
    ]

    total = 0
    correct = 0
    total_latency = 0.0
    results = []

    for text, expected, desc in test_cases:
        t1 = time.perf_counter()
        raw = pipe(text)
        latency = (time.perf_counter() - t1) * 1000
        total_latency += latency

        # 解析输出
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], list):
                items = raw[0]  # top_k 返回格式
            else:
                items = raw  # 单标签格式
        else:
            items = []

        top_label = items[0]["label"] if items else "?"
        top_score = items[0]["score"] if items else 0.0

        # 判断 safe/unsafe
        # 这个映射需要根据具体模型调整
        UNSAFE_LABELS = {
            "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
            "unsafe", "hate_speech", "violence", "porn", "nsfw",
            "H", "H2", "SH", "HR", "V", "V2", "S", "S3",  # KoalaAI
        }
        is_safe = top_label not in UNSAFE_LABELS and "unsafe" not in str(top_label).lower()
        predicted = "safe" if is_safe else "unsafe"
        match = "✓" if predicted == expected else f"✗ (预期:{expected})"

        total += 1
        if predicted == expected:
            correct += 1

        text_short = text[:40]
        print(f"  {match} [{desc}] {text_short}... → {top_label} ({top_score:.4f}) | {latency:.0f}ms")
        results.append({
            "text": text, "desc": desc, "expected": expected,
            "predicted": predicted, "top_label": top_label,
            "top_score": round(top_score, 4), "latency_ms": round(latency, 1),
        })

    # -- 4. 汇总 --
    accuracy = correct / total if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0

    print(f"\n[3/3] 测试结果汇总")
    print(f"  准确率: {correct}/{total} = {accuracy:.1%}")
    print(f"  平均延迟: {avg_latency:.1f}ms")
    print(f"  加载耗时: {load_time:.1f}s")
    print(f"  推理设备: {device_name}")

    # 模型大小估算
    try:
        total_params = sum(p.numel() for p in pipe.model.parameters())
        print(f"  参数量:   {total_params/1e6:.0f}M")
    except Exception:
        pass

    print(f"\n{'='*60}")
    if accuracy >= 0.75:
        print("测试通过 ✓ — 模型可以正常使用")
        print(f"在 .env 中设置: BERT_MODEL={model_path}")
    else:
        print("测试未通过 ✗ — 准确率过低，请检查模型是否匹配任务")
        print("如果是非分类模型，可能不适用于内容审核")

    return accuracy, load_time, avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT 小模型本地加载测试")
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("BERT_MODEL", "KoalaAI/Text-Moderation"),
        help="模型路径或 HuggingFace 模型 ID (默认: 读取 $BERT_MODEL 或 KoalaAI/Text-Moderation)",
    )
    args = parser.parse_args()

    # 如果设置了 BERT_MODEL 但没有传 --model，打印提示
    if "--model" not in sys.argv and "BERT_MODEL" in os.environ:
        print(f"使用环境变量 BERT_MODEL={os.environ['BERT_MODEL']}")

    test_model(args.model)
