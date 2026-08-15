#!/usr/bin/env python3
"""eval_report.py —— 多 seed 稳定性评测：均值/标准差/区间 + md 报告

同一参考、同一 prompt，多个 seed 重复出图，统计均值 ± 标准差 ± 区间。
用意：单张分数含运气成分，报告才能说明方案稳定（实测 0.792 ± 0.018）。

CLI：--ref/-r 参考集目录，--gen/-g 多 seed 生成图目录，--report/-o 报告路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from face_similarity import build_analyzer, evaluate


def build_report(rows: list, scores: list, ref_count: int, gen_dir, ref_dir) -> str:
    lines = [
        "# 多 seed 稳定性评测报告",
        "",
        f"- 参考集：{ref_dir}（{ref_count} 张，取最像的作为上限）",
        f"- 生成图：{gen_dir}（{len(rows)} 张，视为同 prompt 不同 seed 的重复出图）",
    ]
    if len(scores) >= 2:
        arr = np.asarray(scores)
        lines += [
            "",
            "## 汇总",
            "",
            f"- 均值：{arr.mean():.3f}",
            f"- 标准差：{arr.std(ddof=1):.3f}",
            f"- 区间：[{arr.min():.3f}, {arr.max():.3f}]",
            f"- 结论：**{arr.mean():.3f} ± {arr.std(ddof=1):.3f}**（区间 [{arr.min():.3f}, {arr.max():.3f}]）",
        ]
    elif len(scores) == 1:
        lines += ["", f"注意：仅 1 张有效图，无法计算标准差（需 ≥ 2 个 seed）。单张分数：{scores[0]:.3f}"]
    else:
        lines += ["", "注意：没有可用分数（全部 no_face / no_ref）。"]
    lines += [
        "",
        "## 逐图明细",
        "",
        "| 图片 | 相似度 | 评级 | 备注 |",
        "|---|---|---|---|",
    ]
    for name, sim, band, err in rows:
        lines.append(f"| {name} | {sim if sim is not None else '-'} | {band or '-'} | {err or ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多 seed 稳定性评测（均值/标准差/区间 + md 报告）")
    ap.add_argument("--ref", "-r", required=True, help="本人照片（参考集）目录")
    ap.add_argument("--gen", "-g", required=True, help="多 seed 生成图目录")
    ap.add_argument("--report", "-o", default="eval_report.md", help="报告输出路径")
    args = ap.parse_args(argv)

    if not Path(args.ref).is_dir() or not Path(args.gen).is_dir():
        print("错误：参考集或生成图目录不存在", file=sys.stderr)
        return 2

    try:
        app = build_analyzer()
    except Exception as e:  # noqa: BLE001
        print(f"错误：InsightFace 初始化失败（请先运行 ./setup_infer.sh）：{e}", file=sys.stderr)
        return 1

    rows, scores, ref_files = evaluate(app, args.ref, args.gen)
    report = build_report(rows, scores, len(ref_files), args.gen, args.ref)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"报告已写入 {args.report}")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
