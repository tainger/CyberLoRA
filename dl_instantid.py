#!/usr/bin/env python3
"""dl_instantid.py —— 从 ModelScope 下载 InstantID + antelopev2（免训练路线）

- InstantX/InstantID：仅 ControlNetModel/* + ip-adapter.bin（约 2.5G + 1.7G）
- AI-ModelScope/antelopev2 → ./models/antelopev2（5 个 onnx）
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

INSTANTID_REPO = "InstantX/InstantID"
ANTELOPE_REPO = "AI-ModelScope/antelopev2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从 ModelScope 下载 InstantID + antelopev2")
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

    print(f"[1/2] 下载 {INSTANTID_REPO}（仅 ControlNetModel/* + ip-adapter.bin）...")
    local = snapshot_download(
        INSTANTID_REPO,
        cache_dir=cache_dir,
        allow_patterns=["ControlNetModel/*", "ip-adapter.bin"],
    )
    print(f"  → {local}")

    print(f"[2/2] 下载 {ANTELOPE_REPO} → ./models/antelopev2 ...")
    local_ant = snapshot_download(ANTELOPE_REPO, cache_dir=cache_dir)
    target = Path(cache_dir) / "antelopev2"
    if target.exists():
        print(f"  {target} 已存在，跳过拷贝")
    else:
        shutil.copytree(local_ant, target)
        print(f"  → {target}")

    print("完成。instantid_infer.py 会自动定位本地资源。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
