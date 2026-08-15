#!/usr/bin/env python3
"""face_similarity.py —— ArcFace 相似度评测 + 4 档评级 + md 报告

- build_analyzer：InsightFace buffalo_l（评测用）+ CPUExecutionProvider。
- embed_array：bgr → 面积最大人脸的 512 维 normed_embedding；无脸返回 (None, 'no_face')。
- embed_image：cv2.imread（BGR）→ embed_array。
- cosine：dot / (|a||b|)。
- 评级 BANDS：≥0.60 高度相似 / ≥0.45 相似 / ≥0.30 中等 / <0.30 偏低。
- 生成图 vs 参考集取最大相似度（参考集多张，取最像的作为上限）。

CLI：--ref/-r 参考集目录，--gen/-g 生成图目录，--report/-o 报告路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 4 档评级（与 face_similarity.band_of 供 instantid_infer / inference 复用）
BANDS: list = [
    (0.60, "高度相似"),
    (0.45, "相似"),
    (0.30, "中等"),
    (float("-inf"), "偏低"),
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def band_of(sim: float) -> str:
    """按阈值返回评级中文名。"""
    for thr, name in BANDS:
        if sim >= thr:
            return name
    return "偏低"


def build_analyzer():
    """InsightFace buffalo_l 分析器（评测用，CPU 后端）。"""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=1, det_size=(640, 640))
    return app


def embed_array(app, bgr: np.ndarray):
    """bgr → 面积最大人脸的 normed_embedding（512 维）。

    返回 (embedding, None)；无脸返回 (None, 'no_face')。
    """
    faces = app.get(bgr)
    if not faces:
        return None, "no_face"
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding
    if emb is None:  # 个别版本无 normed 字段，手动归一化兜底
        emb = np.asarray(face.embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
    return np.asarray(emb, dtype=np.float32), None


def embed_image(app, path: str | Path):
    """cv2.imread（BGR）→ embed_array。"""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"无法读取图片：{path}")
    return embed_array(app, bgr)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：dot / (|a||b|)。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_ref_embeddings(app, ref_dir: str | Path) -> list:
    """加载参考集全部 embedding。"""
    embs = []
    files = sorted(
        f for f in Path(ref_dir).rglob("*") if f.suffix.lower() in VALID_EXTS
    )
    for f in files:
        emb, err = embed_image(app, f)
        if emb is not None:
            embs.append((f, emb))
    return embs


def score_face(app, ref_embs: list, pil_or_bgr) -> tuple[float | None, str | None]:
    """对单张图（路径 / PIL / BGR）打分：与参考集取最大余弦。

    返回 (similarity, error)。无参考集 → (None, 'no_ref')；无脸 → (None, 'no_face')。
    设计为纯函数，供 inference.py 在内存中直接调用（PIL → BGR → embed）。
    """
    if not ref_embs:
        return None, "no_ref"
    if isinstance(pil_or_bgr, (str, Path)):  # 文件路径 → cv2.imread
        bgr = cv2.imread(str(pil_or_bgr))
        if bgr is None:
            return None, "unreadable"
    elif hasattr(pil_or_bgr, "convert"):  # PIL Image → BGR
        rgb = pil_or_bgr.convert("RGB")
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    else:
        bgr = pil_or_bgr
    emb, err = embed_array(app, bgr)
    if emb is None:
        return None, err
    best = max(cosine(emb, r) for _, r in ref_embs)
    return best, None


def evaluate(app, ref_dir: str | Path, gen_dir: str | Path) -> tuple[list, list, list]:
    """评测生成图目录，返回 (rows, scores, ref_files)。"""
    ref_embs = load_ref_embeddings(app, ref_dir)
    files = sorted(
        f for f in Path(gen_dir).rglob("*") if f.suffix.lower() in VALID_EXTS
    )
    rows, scores = [], []
    for f in files:
        sim, err = score_face(app, ref_embs, f)
        if sim is None:
            rows.append((f.name, None, None, err or "?"))
        else:
            scores.append(sim)
            rows.append((f.name, round(sim, 4), band_of(sim), None))
    return rows, scores, [f for f, _ in ref_embs]


def build_report(rows: list, scores: list, ref_count: int, gen_dir, ref_dir) -> str:
    lines = [
        f"# 人脸相似度评测报告",
        "",
        f"- 参考集：{ref_dir}（{ref_count} 张，取最像的作为上限）",
        f"- 生成图：{gen_dir}（{len(rows)} 张）",
    ]
    if scores:
        arr = np.asarray(scores)
        lines += [
            f"- 均值：{arr.mean():.4f}",
            f"- 最高：{arr.max():.4f}",
            f"- 最低：{arr.min():.4f}",
        ]
    lines += [
        "",
        "## 分档标准",
        "",
        "| 相似度 | 评级 |",
        "|---|---|",
        "| ≥ 0.60 | 高度相似 |",
        "| ≥ 0.45 | 相似 |",
        "| ≥ 0.30 | 中等 |",
        "| < 0.30 | 偏低 |",
        "",
        "## 逐图评级",
        "",
        "| 图片 | 相似度 | 评级 | 备注 |",
        "|---|---|---|---|",
    ]
    for name, sim, band, err in rows:
        lines.append(f"| {name} | {sim if sim is not None else '-'} | {band or '-'} | {err or ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ArcFace 相似度评测 + 4 档评级 + md 报告")
    ap.add_argument("--ref", "-r", required=True, help="本人照片（参考集）目录")
    ap.add_argument("--gen", "-g", required=True, help="生成图目录")
    ap.add_argument("--report", "-o", default="sim_report.md", help="报告输出路径")
    args = ap.parse_args(argv)

    if not Path(args.ref).is_dir():
        print(f"错误：参考集目录不存在 {args.ref}", file=sys.stderr)
        return 2
    if not Path(args.gen).is_dir():
        print(f"错误：生成图目录不存在 {args.gen}", file=sys.stderr)
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
