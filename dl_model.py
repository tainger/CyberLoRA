#!/usr/bin/env python3
"""dl_model.py —— 从 ModelScope 下载 SDXL 底模（仅 fp16 变体）

HF 被墙时的绕行方案：ModelScope 国内镜像 + 只拉 fp16 变体减半下载量。
用法：python dl_model.py [--cache-dir ./models] [--repo AI-ModelScope/stable-diffusion-xl-base-1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "AI-ModelScope/stable-diffusion-xl-base-1.0"
# 只拉 fp16 变体 + 必要元数据，减少下载量
ALLOW_PATTERNS = [
    ".*fp16.safetensors",
    "*.json",
    "*.txt",
    "tokenizer/*",
    "scheduler/*",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从 ModelScope 下载 SDXL 底模（仅 fp16 变体）")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="ModelScope 模型 ID")
    ap.add_argument("--cache-dir", default="./models", help="缓存目录（默认 ./models）")
    args = ap.parse_args(argv)

    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "缺少 modelscope：请先运行 ./setup_infer.sh，或手动 `pip install modelscope`",
            file=sys.stderr,
        )
        return 1

    cache_dir = str(Path(args.cache_dir).resolve())
    print(f"从 ModelScope 下载 {args.repo} → {cache_dir}（仅 fp16 变体）...")
    local = snapshot_download(
        args.repo,
        cache_dir=cache_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    print(f"完成。本地路径（inference.py --model 使用）：{local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
