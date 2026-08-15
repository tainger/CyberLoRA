#!/usr/bin/env python3
"""validate_dataset.py —— 数据集校验（数量/分辨率/重复/宽高比/黑白），输出 md 报告

阈值：
- MIN_COUNT=20 / MAX_COUNT=30：训练集理想数量区间
- MIN_SIDE=512：短边下限（SDXL 训练质量要求）
- MAX_ASPECT=2.0：宽高比上限

去重：8×8 均值哈希（转灰度 resize 8×8，逐像素 > 均值 ? '1' : '0'）。
黑白判定：resize 64×64 RGB，逐像素 |r-g|>10 或 |g-b|>10 即判彩色，否则黑白。

analyze() 返回 (issues, notes)：
- issues（硬性，必须修复）：数量 < 20；短边 < 512 的图；重复图。
- notes（提示，需人工确认）：数量 > 30；宽高比 > 2.0；疑似黑白；
  「侧脸/表情覆盖」无法自动校验（建议正脸 > 10，侧脸 > 5，表情 > 3）。

有 issues 时 SystemExit(1)（便于 run_demo.sh / CI 感知）。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

MIN_COUNT = 20
MAX_COUNT = 30
MIN_SIDE = 512
MAX_ASPECT = 2.0
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 无法自动校验、需人工确认的选图规范提示
MANUAL_NOTE = (
    "侧脸/表情覆盖无法自动校验，请人工确认（建议正脸 > 10 张，侧脸 > 5 张，表情 > 3 张）"
)


def _hash_image(img: Image.Image) -> str:
    """8×8 均值哈希：转灰度 resize 8×8，逐像素 > 均值 ? '1' : '0'。"""
    small = img.convert("L").resize((8, 8), Image.LANCZOS)
    px = list(small.getdata())
    mean = sum(px) / len(px)
    return "".join("1" if v > mean else "0" for v in px)


def _is_grayscale(img: Image.Image) -> bool:
    """resize 64×64 RGB，逐像素 |r-g|>10 或 |g-b|>10 即判彩色，否则黑白。"""
    small = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    for r, g, b in small.getdata():
        if abs(r - g) > 10 or abs(g - b) > 10:
            return False
    return True


def analyze(input_dir: str | Path) -> tuple[list[str], list[str]]:
    """逐图采集信息并返回 (issues, notes)。"""
    in_dir = Path(input_dir)
    files = sorted(
        f for f in in_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTS
    )
    if not files:
        return [f"输入目录 {in_dir} 下没有找到有效图片（{sorted(VALID_EXTS)}）"], []

    issues: list[str] = []
    notes: list[str] = []

    if len(files) < MIN_COUNT:
        issues.append(f"图片数量 {len(files)} < {MIN_COUNT}，训练集数量不足")

    hash_groups: dict[str, list[str]] = defaultdict(list)
    small_imgs: list[str] = []
    wide_imgs: list[str] = []
    gray_imgs: list[str] = []
    side_bins: dict[str, int] = defaultdict(int)

    for f in files:
        try:
            img = Image.open(f)
            w, h = img.size
            rel = str(f.relative_to(in_dir))
            min_side = min(w, h)
            aspect = round(max(w, h) / max(min_side, 1), 2)

            if min_side < MIN_SIDE:
                small_imgs.append(rel)
            if aspect > MAX_ASPECT:
                wide_imgs.append(rel)
            if _is_grayscale(img):
                gray_imgs.append(rel)
            hash_groups[_hash_image(img)].append(rel)

            # 分辨率分布（按短边分桶）
            for lo in range(512, 2049, 512):
                if min_side <= lo:
                    side_bins[f"<={lo}"] += 1
                    break
            else:
                side_bins[f">{2048}"] += 1
        except Exception as e:  # noqa: BLE001
            issues.append(f"无法读取 {f}：{e}")

    if small_imgs:
        issues.append(f"短边 < {MIN_SIDE} 的图片 {len(small_imgs)} 张：{', '.join(small_imgs[:5])}{'...' if len(small_imgs) > 5 else ''}")
    for rels in hash_groups.values():
        if len(rels) > 1:
            issues.append(f"重复/近似重复图片：{', '.join(rels)}")
    if len(files) > MAX_COUNT:
        notes.append(f"图片数量 {len(files)} > {MAX_COUNT}，建议精简到 20~30 张")
    if wide_imgs:
        notes.append(f"宽高比 > {MAX_ASPECT} 的图片 {len(wide_imgs)} 张：{', '.join(wide_imgs[:5])}{'...' if len(wide_imgs) > 5 else ''}")
    if gray_imgs:
        notes.append(f"疑似黑白图 {len(gray_imgs)} 张：{', '.join(gray_imgs[:5])}{'...' if len(gray_imgs) > 5 else ''}")
    notes.append(MANUAL_NOTE)

    return issues, notes


def build_report(input_dir: str | Path, issues: list[str], notes: list[str]) -> str:
    """md 报告：结论 + 分辨率分布 + 短边范围 + 明细表。"""
    in_dir = Path(input_dir)
    files = sorted(
        f for f in in_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTS
    )

    if issues:
        conclusion = "✗ 不通过"
    elif notes and len(notes) > 1:  # 除 MANUAL_NOTE 外还有提示
        conclusion = "⚠ 通过（有提示）"
    else:
        conclusion = "✅ 通过"

    # 明细行
    rows = []
    side_bins: dict[str, int] = defaultdict(int)
    min_sides = []
    for f in files:
        rel = str(f.relative_to(in_dir))
        try:
            img = Image.open(f)
            w, h = img.size
            min_side = min(w, h)
            aspect = round(max(w, h) / max(min_side, 1), 2)
            size_kb = round(f.stat().st_size / 1024, 1)
            gray = "是" if _is_grayscale(img) else "否"
            rows.append(f"| {rel} | {w}×{h} | {min_side} | {aspect} | {size_kb} | {img.mode} | {_hash_image(img)[:16]}… | {gray} |")
            min_sides.append(min_side)
            for lo in range(512, 2049, 512):
                if min_side <= lo:
                    side_bins[f"≤ {lo}"] += 1
                    break
            else:
                side_bins[f"> {2048}"] += 1
        except Exception as e:  # noqa: BLE001
            rows.append(f"| {rel} | 读取失败：{e} | - | - | - | - | - | - |")

    lines = [
        f"# 数据集校验报告：{in_dir}",
        "",
        f"**结论：{conclusion}**",
        "",
        f"- 图片总数：{len(files)}（要求 {MIN_COUNT}~{MAX_COUNT}）",
    ]
    if min_sides:
        lines.append(f"- 短边范围：{min(min_sides)} ~ {max(min_sides)}（要求 ≥ {MIN_SIDE}）")
    lines += ["", "## 分辨率分布", "", "| 短边范围 | 数量 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(side_bins.items(), key=lambda kv: kv[0])]
    lines += ["", "## 明细", "", "| 文件 | 尺寸 | 短边 | 宽高比 | KB | 模式 | 哈希 | 黑白 |", "|---|---|---|---|---|---|---|---|"]
    lines += rows

    if issues:
        lines += ["", "## ✗ 硬性问题（必须修复）", ""]
        lines += [f"- {i}" for i in issues]
    if notes:
        lines += ["", "## 提示（需人工确认）", ""]
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据集校验（数量/分辨率/重复/宽高比/黑白），输出 md 报告")
    ap.add_argument("-i", "--input", required=True, help="数据集目录")
    ap.add_argument("-o", "--output", default="dataset_report.md", help="报告输出路径")
    args = ap.parse_args(argv)

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        print(f"错误：输入目录不存在 {in_dir}", file=sys.stderr)
        return 2

    issues, notes = analyze(in_dir)
    report = build_report(in_dir, issues, notes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"报告已写入 {out}")
    print(report)

    if issues:
        print("结论：✗ 不通过（存在硬性问题）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
