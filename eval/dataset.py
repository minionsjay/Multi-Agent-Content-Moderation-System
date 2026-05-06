"""Test dataset loader and synthetic data generator for POC benchmarking."""

import json
import random

# Synthetic test cases for quick smoke testing when no labeled dataset exists
SYNTHETIC_CASES = [
    # (text, expected_label)
    # Normal content
    ("今天天气真好，适合出去玩", "safe"),
    ("这个产品非常好用，推荐给大家", "safe"),
    ("谁能告诉我最近的医院在哪", "safe"),
    ("Python 是一门很棒的编程语言", "safe"),
    ("今天午餐吃了麻辣烫，味道不错", "safe"),
    ("请问这个周末有什么好看的电影", "safe"),
    ("谢谢大家的支持，我会继续努力的", "safe"),
    ("这个教程讲得很清楚，学会了", "safe"),
    ("有没有人一起去看演唱会", "safe"),
    ("分享一下我的学习笔记，希望对大家有帮助", "safe"),
    # Toxic / insult
    ("你真是个傻逼，什么都不懂", "unsafe"),
    ("垃圾产品，完全是骗人的", "unsafe"),
    ("脑残才买这个东西", "unsafe"),
    ("你他妈的会不会做事", "unsafe"),
    ("滚蛋吧，这里不欢迎你", "unsafe"),
    ("弱智吧你，这都不会", "unsafe"),
    ("操你妈的，敢惹我", "unsafe"),
    # Politics
    ("支持台独，台湾是一个国家", "unsafe"),
    ("港独是香港唯一的出路", "unsafe"),
    ("天安门事件真相必须揭露", "unsafe"),
    # Violence
    ("我要杀了你这个混蛋", "unsafe"),
    ("谁敢惹我我就砍死谁", "unsafe"),
    ("弄死你信不信", "unsafe"),
    # Gambling
    ("加我微信买六合彩，稳赢", "unsafe"),
    ("真人百家乐在线赌博，日赚千元", "unsafe"),
    # Porn
    ("约炮软件下载，同城交友", "unsafe"),
    ("一夜情交友平台，免费注册", "unsafe"),
    # Grey zone / boundary cases
    ("你说的也不是完全没道理，但我觉得有点过了", "safe"),
    ("傻人有傻福吧算是", "safe"),
    ("这游戏打得我想骂人", "safe"),
    ("我操，这也太厉害了吧（称赞）", "safe"),
    ("有些人真的就是欠骂", "unsafe"),
]


def load_synthetic() -> list[dict]:
    """Load synthetic test cases."""
    return [
        {"text": text, "expected": label}
        for text, label in SYNTHETIC_CASES
    ]


def load_jsonl(path: str) -> list[dict]:
    """Load test cases from a JSONL file.

    Expected format per line:
    {"text": "...", "label": "safe"|"unsafe"}
    """
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_dataset(path: str | None = None) -> list[dict]:
    """Load test dataset. Falls back to synthetic if no path given."""
    if path:
        return load_jsonl(path)
    return load_synthetic()
