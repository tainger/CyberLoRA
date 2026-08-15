#!/usr/bin/env python3
"""select_photos.py —— 自动选片（把选图规范自动化）

流程：
1. 硬过滤：短边 ≥ 512；InsightFace 至少检测一张人脸；det_score ≥ 0.55；
   人脸面积占比 ≥ 0.04；拉普拉斯方差 ≥ 60。
2. 质量分排序：quality = blur * 0.4 + det_score * 0.3 + (1 - abs(yaw)/90) * 0.3。
3. Yaw 分桶轮转：[-90,-54), [-54,-18), [-18,18), [18,54), [54,90] 共 5 桶，
   每桶最多取 3 张，确保角度多样性。
4. 去重：对已选图的 embedding 计算余弦相似度，≥ 0.92 则跳过。
5. 补充：若不足 20 张，从高质量候补中继续选，直到满 20 张或耗尽。

输出：每张选中图的 path/yaw/blur/score/face_ratio + yaw_distribution 饼图数据。
CLI：-i INPUT -o OUTPUT -r REPORT
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from face_similarity import VALID_EXTS, cosine, embed_array

MIN_SIDE = 512
DET_SCORE_THR = 0.55
FACE_MIN_RATIO = 0.04
BLUR_THR = 60.0
DEDUP_COS = 0.92
YAW_BUCKETS = 5
TARGET_PER_BUCKET = 3
TARGET_TOTAL = 20

# yaw 分桶边界（左闭右开）
YAW_BOUNDS = [(-90.0, -54.0), (-54.0, -18.0), (-18.0, 18.0), (18.0, 54.0), (54.0, 90.0)]


def blur_score(bgr: np.ndarray) -> float:
    """拉普拉斯方差（越高越清晰）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bucket_index(yaw: float) -> int:
    for i, (lo, hi) in enumerate(YAW_BOUNDS):
        if lo <= yaw < hi:
            return i
    return 2  # 兜底归入正脸桶


def select(app, input_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """主流程：返回 (selected, candidates)。"""
    files = sorted(
        f for f in Path(input_dir).rglob("*") if f.suffix.lower() in VALID_EXTS
    )

    candidates: list[dict] = []
    rejected: list[dict] = []

    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            rejected.append({"path": str(f), "reason": "无法读取"})
            continue
        h, w = bgr.shape[:2]
        if min(h, w) < MIN_SIDE:
            rejected.append({"path": str(f), "reason": f"短边 < {MIN_SIDE}"})
            continue

        faces = app.get(bgr)
        if not faces:
            rejected.append({"path": str(f), "reason": "未检测到人脸"})
            continue
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        det_score = float(face.det_score)
        if det_score < DET_SCORE_THR:
            rejected.append({"path": str(f), "reason": f"det_score {det_score:.2f} < {DET_SCORE_THR}"})
            continue
        face_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        face_ratio = face_area / (w * h)
        if face_ratio < FACE_MIN_RATIO:
            rejected.append({"path": str(f), "reason": f"人脸占比 {face_ratio:.3f} < {FACE_MIN_RATIO}"})
            continue
        blur = blur_score(bgr)
        if blur < BLUR_THR:
            rejected.append({"path": str(f), "reason": f"模糊 {blur:.1f} < {BLUR_THR}"})
            continue

        pose = getattr(face, "pose", None)
        yaw = float(pose[1]) if pose is not None and len(pose) > 1 else 0.0
        emb, err = embed_array(app, bgr)
        quality = blur * 0.4 + det_score * 0.3 + (1 - abs(yaw) / 90) * 0.3
        candidates.append({
            "path": str(f),
            "yaw": round(yaw, 2),
            "blur": round(blur, 1),
            "det_score": round(det_score, 3),
            "face_ratio": round(face_ratio, 3),
            "quality": round(quality, 2),
            "embedding": emb,
        })

    # 质量分降序
    candidates.sort(key=lambda c: c["quality"], reverse=True)

    # yaw 分桶轮转 + embedding 去重
    selected: list[dict] = []
    bucket_counts = [0] * YAW_BUCKETS
    selected_embs: list[np.ndarray] = []

    def try_add(c: dict) -> bool:
        b = _bucket_index(c["yaw"])
        if bucket_counts[b] >= TARGET_PER_BUCKET:
            return False
        if c["embedding"] is not None:
            for e in selected_embs:
                if cosine(c["embedding"], e) >= DEDUP_COS:
                    return False
        selected.append(c)
        selected_embs.append(c["embedding"])
        bucket_counts[b] += 1
        return True

    # 第一轮：轮转每桶挑质量最高者
    changed = True
    while changed and len(selected) < TARGET_TOTAL:
        changed = False
        for b in range(YAW_BUCKETS):
            if bucket_counts[b] >= TARGET_PER_BUCKET:
                continue
            for c in candidates:
                if c in selected or _bucket_index(c["yaw"]) != b:
                    continue
                if try_add(c):
                    changed = True
                    break

    # 补充：不足 20 张时从候补继续选（不再受每桶上限约束）
    if len(selected) < TARGET_TOTAL:
        for c in candidates:
            if len(selected) >= TARGET_TOTAL:
                break
            if c in selected:
                continue
            if c["embedding"] is not None and any(
                cosine(c["embedding"], e) >= DEDUP_COS for e in selected_embs
            ):
                continue
            selected.append(c)
            selected_embs.append(c["embedding"])
            bucket_counts[_bucket_index(c["yaw"])] += 1

    return selected, rejected


def build_report(selected: list[dict], rejected: list[dict], input_dir, output_dir) -> str:
    dist = [0] * YAW_BUCKETS
    for c in selected:
        dist[_bucket_index(c["yaw"])] += 1
    bucket_labels = ["[-90,-54)", "[-54,-18)", "[-18,18)", "[18,54)", "[54,90]"]

    lines = [
        f"# 自动选片报告",
        "",
        f"- 输入：{input_dir}",
        f"- 输出：{output_dir}",
        f"- 选中：{len(selected)} 张（目标 {TARGET_TOTAL} 张）",
        f"- 过滤掉：{len(rejected)} 张",
        "",
        "## Yaw 分布（饼图数据）",
        "",
        "| 角度桶 | 数量 |",
        "|---|---|",
    ]
    for label, n in zip(bucket_labels, dist):
        lines.append(f"| {label} | {n} |")
    lines += [
        "",
        "## 选中明细",
        "",
        "| 图片 | yaw | blur | 质量分 | 人脸占比 |",
        "|---|---|---|---|---|",
    ]
    for c in selected:
        lines.append(
            f"| {Path(c['path']).name} | {c['yaw']} | {c['blur']} | {c['quality']} | {c['face_ratio']} |"
        )
    if rejected:
        lines += ["", "## 被过滤", "", "| 图片 | 原因 |", "|---|---|"]
        for r in rejected[:50]:
            lines.append(f"| {Path(r['path']).name} | {r['reason']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="自动选片：单人/清晰/多角度/去重")
    ap.add_argument("-i", "--input", required=True, help="原始照片目录")
    ap.add_argument("-o", "--output", required=True, help="选中图输出目录")
    ap.add_argument("-r", "--report", default="select_report.md", help="报告路径")
    args = ap.parse_args(argv)

    if not Path(args.input).is_dir():
        print(f"错误：输入目录不存在 {args.input}", file=sys.stderr)
        return 2

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=1, det_size=(640, 640))
    except Exception as e:  # noqa: BLE001
        print(f"错误：InsightFace 初始化失败（请先运行 ./setup_infer.sh）：{e}", file=sys.stderr)
        return 1

    selected, rejected = select(app, args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in selected:
        shutil.copy2(c["path"], out_dir / Path(c["path"]).name)

    report = build_report(selected, rejected, args.input, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"选中 {len(selected)} 张 → {out_dir}")
    print(f"报告已写入 {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
