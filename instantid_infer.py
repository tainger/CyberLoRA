#!/usr/bin/env python3
"""instantid_infer.py —— InstantID 免训练单张出图 + 可选评分

原理：人脸 embedding 注入 IP-Adapter（身份）+ 关键点经 ControlNet 约束布局。
与 LoRA 路线的区别：身份来自 embedding，不需要触发词（反而要去掉 `cyberboy`）。

流程：
- find_local_sdxl：找 models/**/model_index.json（优先 snapshots/ 布局，兼容 ModelScope 布局）。
- find_instantid：找 models/**/ip-adapter.bin 及同级 ControlNetModel/。
- 人脸分析器用 antelopev2：FaceAnalysis(name='antelopev2', root='.') → 读 ./models/antelopev2。
- 参考照 resize 1024 → 取最大脸 → face['embedding']（身份）+ draw_kps（ControlNet 条件图）。
- 管线：ControlNetModel.from_pretrained(cn_dir)
  + StableDiffusionXLInstantIDPipeline.from_pretrained(model, controlnet=..., variant='fp16')（失败回退）
  → load_ip_adapter_instantid(adapter) → set_ip_adapter_scale(ip_scale)。
- 出图：image_embeds=face_emb，image=face_kps，controlnet_conditioning_scale=cn_scale，
  ip_adapter_scale=ip_scale。
- --score：用 antelopev2 对生成图再 embed，与参考照 embedding 算余弦，套 band_of 评级。

CLI：--ref/-r，--scene/-s（默认 business），--model/-m，--steps（28），--seed（42），
     --ip-scale（0.8），--cn-scale（0.8），--output/-o，--score。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from prompt_gen import INFER_PARAMS, build_prompt

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SCENE = "business"
GEN_SIZE = 640  # InstantID 官方默认输出尺寸


# ---------------------------------------------------------------------------
# 本地资源定位
# ---------------------------------------------------------------------------

def find_local_sdxl(models_dir: str | Path = "./models") -> str | None:
    """找 models/**/model_index.json（snapshots/ 布局优先，兼容 ModelScope 布局）。"""
    hits = list(Path(models_dir).rglob("model_index.json"))
    if not hits:
        return None
    hits.sort(key=lambda p: 0 if "snapshots" in p.parts else 1)
    return str(hits[0].parent)


def find_instantid(models_dir: str | Path = "./models") -> tuple[str | None, str | None]:
    """找 models/**/ip-adapter.bin 及同级 ControlNetModel/。"""
    for adapter in Path(models_dir).rglob("ip-adapter.bin"):
        cn = adapter.parent / "ControlNetModel"
        if cn.is_dir():
            return str(adapter), str(cn)
    return None, None


# ---------------------------------------------------------------------------
# 人脸分析（antelopev2）
# ---------------------------------------------------------------------------

def build_antelopev2():
    """FaceAnalysis(name='antelopev2', root='.') → 读 ./models/antelopev2。"""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="antelopev2", root=".")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_largest_face(app, bgr: np.ndarray):
    faces = app.get(bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def draw_kps(kps, size=GEN_SIZE):
    """关键点画成 ControlNet 条件图（参考 InstantID 官方实现）。"""
    kps = np.asarray(kps, dtype=np.float32)
    # 归一化到 [0,1] 再放大到 size（参考照已 resize 1024，此处按比例换算）
    stickwidth = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    h, w = size, size
    out_img = np.zeros([h, w, 3], dtype=np.uint8)

    for i in range(len(limb_seq)):
        index = limb_seq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    return out_img


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------

def load_instantid_pipeline(model_id: str, cn_dir: str, adapter_path: str, ip_scale: float, device, dtype):
    """vendored InstantID 管线（Apache-2.0）+ IP-Adapter。"""
    from diffusers import ControlNetModel

    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

    controlnet = ControlNetModel.from_pretrained(cn_dir, torch_dtype=dtype)
    try:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )
    except Exception:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
        )
    pipe.load_ip_adapter_instantid(adapter_path)
    pipe.set_ip_adapter_scale(ip_scale)
    return pipe.to(device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="InstantID 免训练单张出图 + 可选评分")
    ap.add_argument("--ref", "-r", required=True, help="本人参考照路径")
    ap.add_argument("--scene", "-s", default=DEFAULT_SCENE, help="场景 key（默认 business）")
    ap.add_argument("--model", "-m", default=None, help="SDXL 底模 ID 或本地路径（默认自动找本地）")
    ap.add_argument("--steps", type=int, default=28, help="采样步数（默认 28）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--ip-scale", type=float, default=0.8, help="IP-Adapter 强度")
    ap.add_argument("--cn-scale", type=float, default=0.8, help="ControlNet 强度")
    ap.add_argument("--output", "-o", default="demo_out/instantid", help="输出目录")
    ap.add_argument("--score", action="store_true", help="生成后计算与参考照的相似度")
    args = ap.parse_args(argv)

    if not Path(args.ref).is_file():
        print(f"错误：参考照不存在 {args.ref}", file=sys.stderr)
        return 2
    try:
        scene = build_prompt(args.scene)
    except KeyError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    import torch

    # ---- 设备 ----
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    # ---- 本地资源定位 ----
    model_id = args.model or find_local_sdxl() or DEFAULT_MODEL
    adapter_path, cn_dir = find_instantid()
    if adapter_path is None:
        print("错误：未找到本地 InstantID 资源。请先运行 dl_instantid.py", file=sys.stderr)
        return 1
    print(f"底模：{model_id}")
    print(f"InstantID：{adapter_path}\nControlNet：{cn_dir}")

    # ---- 人脸分析（antelopev2）----
    print("初始化 antelopev2 ...")
    app = build_antelopev2()
    bgr = cv2.imread(str(Path(args.ref).resolve()))
    if bgr is None:
        print(f"错误：无法读取参考照 {args.ref}", file=sys.stderr)
        return 1
    # 参考照 resize 1024 后取最大脸
    bgr = cv2.resize(bgr, (1024, 1024))
    face = get_largest_face(app, bgr)
    if face is None:
        print("错误：参考照未检测到人脸", file=sys.stderr)
        return 1
    face_emb = face["embedding"]
    kps = face["kps"]
    # 关键点按 1024 → GEN_SIZE 比例缩放
    kps_scaled = kps * (GEN_SIZE / 1024.0)
    face_kps = draw_kps(kps_scaled, size=GEN_SIZE)
    print("参考照人脸提取完成")

    # ---- 管线 ----
    print("加载 InstantID 管线 ...")
    pipe = load_instantid_pipeline(model_id, cn_dir, adapter_path, args.ip_scale, device, dtype)

    # 去掉触发词：身份来自 embedding，不需要 cyberboy
    positive = ", ".join(
        [scene["subject"], scene["scene"], scene["style"], scene["quality"]]
    )

    print(f"出图：{scene['name']}（steps={args.steps}, seed={args.seed}）...")
    image = pipe(
        image=face_kps,
        prompt=positive,
        negative_prompt=scene["negative"],
        image_embeds=face_emb,
        controlnet_conditioning_scale=args.cn_scale,
        ip_adapter_scale=args.ip_scale,
        num_inference_steps=args.steps,
        guidance_scale=INFER_PARAMS["cfg_scale"],
        generator=torch.Generator("cpu").manual_seed(args.seed),
        width=GEN_SIZE,
        height=GEN_SIZE,
    ).images[0]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"instantid_{args.scene}.png"
    image.save(out_path)
    print(f"已保存 {out_path}")

    # ---- 可选评分 ----
    if args.score:
        gen_bgr = cv2.resize(
            cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR), (1024, 1024)
        )
        gen_face = get_largest_face(app, gen_bgr)
        if gen_face is None:
            print("评分：生成图未检测到人脸（no_face）")
        else:
            from face_similarity import band_of, cosine

            sim = cosine(
                np.asarray(face_emb, dtype=np.float32),
                np.asarray(gen_face["embedding"], dtype=np.float32),
            )
            print(f"相似度：{sim:.4f}（{band_of(sim)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
