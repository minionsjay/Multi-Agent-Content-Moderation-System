#!/usr/bin/env python3
"""
大模型 (LLM) 本地加载测试

用于测试本地下载的大语言模型是否能正常加载和推理。
支持 HuggingFace transformers 格式的模型 (CausalLM)。

使用方法:
    # 测试 HuggingFace 模型 (自动下载)
    python tests/test_llm_local.py --model Qwen/Qwen2.5-1.5B-Instruct

    # 测试本地路径
    python tests/test_llm_local.py --model /home/ninini/models/my-qwen-model

    # 指定量化方式
    python tests/test_llm_local.py --model Qwen/Qwen2.5-1.5B-Instruct --4bit
    python tests/test_llm_local.py --model Qwen/Qwen2.5-1.5B-Instruct --fp16

    # 使用 CPU 加载
    python tests/test_llm_local.py --model /path/to/model --device cpu

本地路径需包含:
    config.json           # 模型配置
    pytorch_model.bin 或 model.safetensors  # 权重
    tokenizer_config.json # tokenizer 配置
    tokenizer.json 或 vocab.txt  # 词表

配置到 .env 中:
    LLM_PROVIDER=transformers
    TRANSFORMERS_LLM_MODEL=/your/local/path
    TRANSFORMERS_LLM_LOAD_IN_4BIT=true   # 可选，4-bit 量化省显存
    TRANSFORMERS_LLM_DEVICE_MAP=auto     # auto / cpu / cuda:0
"""

import argparse
import os
import sys
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("test_llm")


def test_model(model_path: str, load_in_4bit: bool = False, device_map: str = "auto"):
    """加载并测试一个本地 LLM 模型。"""
    print(f"\n{'='*60}")
    print(f"LLM 大模型加载测试")
    print(f"{'='*60}")
    print(f"模型路径:    {model_path}")
    print(f"4-bit 量化:  {load_in_4bit}")
    print(f"设备映射:    {device_map}")

    # -- 1. 检查依赖 --
    print(f"\n[1/4] 检查依赖...")
    deps_ok = True
    try:
        import torch
        print(f"  torch:      {torch.__version__} ✓")
        print(f"  CUDA:       {'可用' if torch.cuda.is_available() else '不可用 (将使用CPU)'}")
        if torch.cuda.is_available():
            print(f"  GPU:        {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
    except ImportError:
        print(f"  [ERROR] torch 未安装 — pip install torch")
        deps_ok = False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  transformers: ✓")
    except ImportError:
        print(f"  [ERROR] transformers 未安装 — pip install transformers")
        deps_ok = False

    if load_in_4bit:
        try:
            import bitsandbytes
            print(f"  bitsandbytes: ✓")
        except ImportError:
            print(f"  [WARN] bitsandbytes 未安装 — 将使用全精度加载")
            print(f"         安装: pip install bitsandbytes")
            load_in_4bit = False

    if not deps_ok:
        sys.exit(1)

    # -- 2. 检查模型文件 --
    print(f"\n[2/4] 检查模型文件...")
    if os.path.exists(model_path):
        required = ["config.json", "tokenizer_config.json"]
        missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
        weight_files = [
            f for f in ["pytorch_model.bin", "model.safetensors",
                        "pytorch_model-00001-of-00002.bin"]  # sharded
            if os.path.exists(os.path.join(model_path, f))
        ]
        if missing:
            print(f"  [WARN] 缺少: {missing}")
        if weight_files:
            total_gb = sum(
                os.path.getsize(os.path.join(model_path, f))
                for f in weight_files
            ) / 1024**3
            print(f"  权重文件:  {', '.join(weight_files)}")
            print(f"  模型大小:  {total_gb:.2f} GB")
            print(f"  模型文件检查通过 ✓")
    else:
        print(f"  本地路径不存在，将作为 HuggingFace 模型 ID 处理")
        print(f"  (模型将从 HuggingFace Hub 下载)")

    # -- 3. 加载模型 --
    print(f"\n[3/4] 加载模型 (这可能需要几分钟)...")
    t0 = time.perf_counter()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Tokenizer
        print(f"  加载 tokenizer...")
        t_tok = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print(f"  tokenizer 加载完成 ({time.perf_counter() - t_tok:.1f}s)")
        print(f"  vocab size: {tokenizer.vocab_size}")

        # Config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"  模型架构:   {config.model_type}")
        if hasattr(config, "num_hidden_layers"):
            print(f"  隐藏层数:   {config.num_hidden_layers}")
        if hasattr(config, "hidden_size"):
            print(f"  隐藏维度:   {config.hidden_size}")

        # Model
        print(f"  加载模型权重...")
        t_model = time.perf_counter()

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": device_map if device_map != "cpu" else None,
        }

        if load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            load_kwargs["bnb_4bit_use_double_quant"] = True

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if device_map == "cpu":
            model = model.cpu()

        load_time = time.perf_counter() - t0
        model_time = time.perf_counter() - t_model
        print(f"  权重加载耗时: {model_time:.1f}s")
        print(f"  总加载耗时:   {load_time:.1f}s ✓")

        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量:       {total_params/1e9:.2f}B")

        # 显存使用
        if torch.cuda.is_available() and device_map != "cpu":
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU 显存占用: {vram_used:.2f} GB")

    except Exception as e:
        print(f"\n[ERROR] 模型加载失败: {e}")
        print(f"\n可能的原因:")
        print(f"  1. 显存不足 — 尝试 --4bit 量化或 --device cpu")
        print(f"  2. 模型文件不完整")
        print(f"  3. 磁盘空间不足")
        print(f"  4. transformers 版本不兼容")
        print(f"\n提示: 小显存 GPU 可以尝试:")
        print(f"  python tests/test_llm_local.py --model {model_path} --4bit")
        print(f"  或者纯 CPU 推理 (较慢但不需要 GPU):")
        print(f"  python tests/test_llm_local.py --model {model_path} --device cpu")
        sys.exit(1)

    # -- 4. 推理测试 --
    print(f"\n[4/4] 推理测试...")

    # 导入审核 system prompt
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    try:
        from skills.llm_audit import SYSTEM_PROMPT
    except ImportError:
        SYSTEM_PROMPT = "You are a content moderation expert. Classify text into: safe, toxic, violence, politics, porn, gambling, spam. Respond in JSON: {\"label\": \"...\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}"

    test_cases = [
        ("今天天气真好，适合出去郊游野餐", "safe", "中文安全"),
        ("你真是个傻逼，什么都不懂还在这瞎说", "unsafe", "中文辱骂"),
        ("支持台独，台湾是一个独立的国家", "unsafe", "政治敏感"),
        ("我要杀了你这个混蛋", "unsafe", "暴力威胁"),
        ("这个产品非常好用，强烈推荐给大家", "safe", "中文推荐"),
        ("you are a worthless piece of garbage, nobody wants you here", "unsafe", "英文辱骂"),
        ("the sunset looks absolutely stunning in this photograph", "safe", "英文正常"),
    ]

    total = 0
    correct = 0
    total_latency = 0.0
    parse_failures = 0

    for text, expected, desc in test_cases:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text to moderate:\n```\n{text}\n```"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t1 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency = (time.perf_counter() - t1) * 1000
        total_latency += latency

        # 解析 JSON
        label = "safe"
        confidence = 0.5
        reason = ""
        try:
            result = json.loads(raw_output)
            label = result.get("label", "safe")
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            m = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                    label = result.get("label", "safe")
                    confidence = result.get("confidence", 0.5)
                    reason = result.get("reason", "")
                except json.JSONDecodeError:
                    parse_failures += 1
                    label = "parse_error"

        predicted = "safe" if label == "safe" else "unsafe"
        match = "✓" if predicted == expected else f"✗ (预期:{expected})"

        total += 1
        if predicted == expected:
            correct += 1

        text_short = text[:40]
        print(f"  {match} [{desc}] {text_short}... → {label} ({confidence:.2f}) | {latency:.0f}ms")
        if reason:
            print(f"        理由: {reason[:100]}")

    # -- 汇总 --
    accuracy = correct / total if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"测试结果汇总")
    print(f"  准确率:       {correct}/{total} = {accuracy:.1%}")
    print(f"  JSON 解析失败: {parse_failures}/{total}")
    print(f"  平均延迟:     {avg_latency:.0f}ms")
    print(f"  加载耗时:     {load_time:.1f}s")

    if accuracy >= 0.7 and parse_failures < total * 0.5:
        print(f"测试通过 ✓ — 模型可以用于内容审核")
        print(f"\n在 .env 中设置:")
        print(f"  LLM_PROVIDER=transformers")
        print(f"  TRANSFORMERS_LLM_MODEL={model_path}")
        if load_in_4bit:
            print(f"  TRANSFORMERS_LLM_LOAD_IN_4BIT=true")
    elif parse_failures >= total * 0.5:
        print(f"测试未通过 ✗ — JSON 输出格式不兼容，建议使用更强大的模型")
        print(f"  推荐模型: Qwen/Qwen2.5-7B-Instruct 或以上")
    else:
        print(f"测试部分通过 — 准确率偏低但模型可用，建议调优 prompt 或使用更强模型")

    return accuracy, load_time, avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 大模型本地加载测试")
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("TRANSFORMERS_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="模型路径或 HuggingFace 模型 ID (默认: 读取 $TRANSFORMERS_LLM_MODEL 或 Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--4bit", dest="load_in_4bit", action="store_true",
        help="使用 4-bit 量化加载 (节省显存)",
    )
    parser.add_argument(
        "--fp16", dest="load_in_4bit", action="store_false",
        help="使用 FP16 全精度加载 (需要更多显存)",
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cpu", "cuda:0", "cuda:1"],
        help="推理设备 (默认: auto)",
    )
    parser.set_defaults(load_in_4bit=None)

    args = parser.parse_args()

    # 默认: 有 CUDA 就用 4bit，没 CUDA 就用 FP32
    if args.load_in_4bit is None:
        import torch
        args.load_in_4bit = torch.cuda.is_available()

    test_model(args.model, args.load_in_4bit, args.device)
