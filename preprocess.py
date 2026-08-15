#!/usr/bin/env python3
"""preprocess.py —— 图片标准化为 1024×1024

规则：
- 读取图片，保持 RGB。
- 计算短边，等比例缩放使短边 = 1024。
- 从中心裁剪 1024×1024。
- 保存为 PNG（无损，便于训练）。
- 若图片已为 1024×1024 且无裁剪偏移，则原样复制（避免重新编码损失）。

CLI：-i INPUT_DIR -o OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

TARGET = 1024
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def standardize(src: Path, dst: Path) -> bool:
    """标准化单张图；已达标则原样复制。返回是否原样复制。"""
    img = Image.open(src)
    img = img.convert("RGB")
    w, h = img.size

    if w == TARGET and h == TARGET:
        # 已达标：原样复制，避免重新编码损失
        shutil.copy2(src, dst)
        return True

    # 等比例缩放使短边 = 1024
    if w <= h:
        new_w, new_h = TARGET, round(h * TARGET / w)
    else:
        new_w, new_h = round(w * TARGET / h), TARGET
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 中心裁剪 1024×1024
    left = (new_w - TARGET) // 2
    top = (new_h - TARGET) // 2
    img = img.crop((left, top, left + TARGET, top + TARGET))

    img.save(dst, "PNG")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="图片标准化为 1024×1024（等比缩放+居中裁剪）")
    ap.add_argument("-i", "--input", required=True, help="输入目录")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    args = ap.parse_args(argv)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"错误：输入目录不存在 {in_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in in_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTS
    )
    if not files:
        print(f"警告：{in_dir} 下没有找到有效图片（{sorted(VALID_EXTS)}）", file=sys.stderr)
        return 1

    copied = 0
    converted = 0
    for f in files:
        try:
            dst = out_dir / f"{f.stem}.png"
            if standardize(f, dst):
                copied += 1
            else:
                converted += 1
        except Exception as e:  # noqa: BLE001 —— 坏图不阻断批处理
            print(f"跳过 {f}：{e}", file=sys.stderr)

    print(f"完成：转换 {converted} 张，原样复制 {copied} 张 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
