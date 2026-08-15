#!/usr/bin/env python3
"""inference.py —— 本地 MPS 出图 + 权重对比网格 + 相似度标注 + 曲线（路线 B）

要点（照抄即可，勿改）：
- 惰性导入 torch/diffusers：--list / --dry-run 无需安装即可运行。
- pick_device：MPS → float16，CUDA → float16，CPU → float32。
- load_pipeline：from_pretrained(variant='fp16') 失败回退无 variant；
  采样器固化 DPM++ 2M Karras；enable_attention_slicing + vae.enable_tiling 防 OOM。
- LoRA：pipe.load_lora_weights(lora_dir, weight_name=lora_file)；
  权重通过 cross_attention_kwargs={'scale': w} 传入（diffusers 不解析 <lora:name> 语法）。
- --compare：COMPARE_WEIGHTS=[0.5,0.75,0.95] 各出一张，横向拼接网格（顶部留白多行标签）。
- --ref：加载参考集 embedding，每档出图后打分标注在网格，并输出「权重 vs 相似度」折线图
  （纯 PIL，英文标签避免乱码，no_face 跳过不画）。

CLI：--scene/-s（单场景或 all），--lora，--model/-m，--seed，--output/-o，
     --compare，--ref，--list，--dry-run，--weight/-w。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prompt_gen import (
    COMPARE_WEIGHTS,
    INFER_PARAMS,
    SCENES,
    build_prompt,
)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SEED = 42
SIZE = 1024


# ---------------------------------------------------------------------------
# 设备与管线（惰性导入 torch/diffusers）
# ---------------------------------------------------------------------------

def pick_device():
    """MPS → float16，CUDA → float16，CPU → float32。"""
    import torch

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_pipeline(model_id: str, device: str, dtype):
    """SDXL 管线：fp16 variant 失败回退；DPM++ 2M Karras；切片 + tiling 防 OOM。

    VAE 坑（实测）：VAE 保持 fp16 加载由管线自动 upcast，手动改 float32 反而 MPS NaN 黑图。
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )
    # 固化采样器：DPM++ 2M Karras
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.vae.enable_tiling()
    return pipe.to(device)


def load_lora(pipe, lora_path: str):
    """--lora 支持 .safetensors 文件或目录（目录取第一个 safetensors）。"""
    p = Path(lora_path)
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"{p} 下没有 .safetensors")
        lora_dir, lora_file = str(p), files[0].name
    else:
        lora_dir, lora_file = str(p.parent), p.name
    pipe.load_lora_weights(lora_dir, weight_name=lora_file)
    return lora_file


def generate(pipe, prompt: str, negative: str, weight: float | None, seed: int):
    """出图：LoRA 权重走 cross_attention_kwargs={'scale': w}（不是 <lora> 语法）。"""
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=INFER_PARAMS["steps"],
        guidance_scale=INFER_PARAMS["cfg_scale"],
        width=SIZE,
        height=SIZE,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    if weight is not None:
        kwargs["cross_attention_kwargs"] = {"scale": weight}
    return pipe(**kwargs).images[0]


# ---------------------------------------------------------------------------
# 对比网格 + 相似度曲线（纯 PIL，无 torchvision 依赖）
# ---------------------------------------------------------------------------

def make_compare_grid(images: list, labels: list[str], cell=SIZE, margin_top=120) -> "object":
    """横向拼接网格：顶部留白画多行标签。"""
    from PIL import Image, ImageDraw

    n = len(images)
    canvas = Image.new("RGB", (cell * n, cell + margin_top), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * cell, margin_top))
        for j, line in enumerate(label.split("\n")):
            draw.text((i * cell + 8, 8 + j * 22), line, fill=(0, 0, 0))
    return canvas


def draw_sim_curve(points: list[tuple[float, float]], out_path: Path, size=640):
    """「权重 vs 相似度」折线图（纯 PIL，英文标签避免乱码）。

    points: [(weight, similarity), ...]；no_face 的点由调用方跳过不画。
    """
    from PIL import Image, ImageDraw

    W, H = size, size
    pad = 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # 坐标映射
    def px(w: float) -> float:
        return pad + (w - 0.4) / (1.0 - 0.4) * (W - 2 * pad)

    def py(s: float) -> float:
        return H - pad - s / 1.0 * (H - 2 * pad)

    # 轴
    d.line([(pad, H - pad), (W - pad, H - pad)], fill=(0, 0, 0), width=2)
    d.line([(pad, pad), (pad, H - pad)], fill=(0, 0, 0), width=2)
    d.text((10, H - pad + 6), "0.0", fill=(0, 0, 0))
    d.text((10, py(1.0)), "1.0", fill=(0, 0, 0))
    d.text((pad, H - 40), "Similarity vs LoRA Weight", fill=(0, 0, 0))
    for w, s in points:
        d.text((px(w) - 12, H - pad + 20), f"{w:g}", fill=(0, 0, 0))

    # 折线 + 点
    if points:
        pts = [(px(w), py(s)) for w, s in points]
        d.line(pts, fill=(30, 100, 200), width=3)
        for (x, y), (w, s) in zip(pts, points):
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(200, 60, 40))
            d.text((x + 8, y - 14), f"{s:.3f}", fill=(0, 0, 0))
    img.save(out_path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _scene_keys(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(SCENES)
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENES]
    if unknown:
        print(f"错误：未知场景 {unknown}。可用场景：{', '.join(sorted(SCENES))}", file=sys.stderr)
        sys.exit(2)
    return keys


def print_task_list(scenes: list[str], compare: bool, ref: str | None) -> None:
    print("任务清单：")
    for k in scenes:
        s = SCENES[k]
        line = f"  - {k}（{s['name']}，{s['shot']}，默认权重 {build_prompt(k)['weight']:g}）"
        if compare:
            line += f" → 三档对比 {COMPARE_WEIGHTS}"
        print(line)
    print(f"参数：{INFER_PARAMS}")
    if ref:
        print(f"参考集：{ref}（每档出图后标注相似度 + 输出曲线）")
    print("模型默认：stabilityai/stable-diffusion-xl-base-1.0（--model 可指定本地路径）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本地 MPS 出图 + 权重对比网格 + 相似度标注（路线 B）")
    ap.add_argument("--scene", "-s", default="all", help="场景 key（逗号分隔或 all）")
    ap.add_argument("--lora", default=None, help="LoRA 权重文件或目录（.safetensors）")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help="SDXL 底模 ID 或本地路径")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    ap.add_argument("--weight", "-w", type=float, default=None, help="强制 LoRA 权重（默认按景别档）")
    ap.add_argument("--compare", action="store_true", help="三档权重对比网格")
    ap.add_argument("--ref", default=None, help="本人照片目录（标注相似度 + 曲线）")
    ap.add_argument("--output", "-o", default="demo_out/infer", help="输出目录")
    ap.add_argument("--list", action="store_true", help="列出全部场景后退出（无需安装依赖）")
    ap.add_argument("--dry-run", action="store_true", help="打印任务清单，不加载模型（无需安装依赖）")
    args = ap.parse_args(argv)

    scenes = _scene_keys(args.scene)

    if args.list:
        for k in scenes:
            p = build_prompt(k)
            print(f"{k:10s} {p['name']}（{p['shot']}，权重 {p['weight']:g}）")
        return 0
    if args.dry_run:
        print_task_list(scenes, args.compare, args.ref)
        return 0

    # ---- 重依赖从这里才开始加载 ----
    print(f"设备：{pick_device()}")
    device, dtype = pick_device()
    print(f"加载管线：{args.model} ...")
    pipe = load_pipeline(args.model, device, dtype)

    lora_file = None
    if args.lora:
        lora_file = load_lora(pipe, args.lora)
        print(f"已挂载 LoRA：{lora_file}")

    # 参考集（可选）
    analyzer, ref_embs = None, []
    if args.ref:
        from face_similarity import build_analyzer, load_ref_embeddings
        analyzer = build_analyzer()
        ref_embs = load_ref_embeddings(analyzer, args.ref)
        print(f"参考集 embedding：{len(ref_embs)} 张")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in scenes:
        base = build_prompt(k)
        # diffusers 不解析 <lora:name:weight> 前缀；去掉触发词外的 lora 语法
        prompt = base["positive"]
        if args.lora:
            prompt = prompt.split(">", 1)[-1].lstrip() if prompt.startswith("<lora:") else prompt

        if args.compare:
            images, labels, points = [], [], []
            for w in COMPARE_WEIGHTS:
                print(f"[{k}] weight={w:g} 出图中 ...")
                img = generate(pipe, prompt, base["negative"], w, args.seed)
                label = f"weight={w:g}"
                if analyzer is not None:
                    from face_similarity import score_face
                    sim, err = score_face(analyzer, ref_embs, img)
                    if sim is not None:
                        label += f"\nsim={sim:.3f}"
                        points.append((w, sim))  # no_face 跳过不画
                    else:
                        label += f"\n({err})"
                images.append(img)
                labels.append(label)
            grid = make_compare_grid(images, labels)
            grid.save(out_dir / f"{k}_compare.png")
            print(f"  已保存 {out_dir / f'{k}_compare.png'}")
            if len(points) >= 2:
                draw_sim_curve(points, out_dir / f"{k}_sim_curve.png")
                print(f"  已保存 {out_dir / f'{k}_sim_curve.png'}")
            elif points:
                print("  注意：有效相似度点 < 2，不画曲线（no_face 被跳过）")
        else:
            w = args.weight if args.weight is not None else base["weight"]
            print(f"[{k}] weight={w:g} 出图中 ...")
            img = generate(pipe, prompt, base["negative"], w, args.seed)
            img.save(out_dir / f"{k}.png")
            print(f"  已保存 {out_dir / f'{k}.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
