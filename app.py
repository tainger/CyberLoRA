#!/usr/bin/env python3
"""app.py —— Gradio Web Demo（数据准备 / 本地出图 / 相似度评测 三 Tab）

要点：
- 管线单例缓存 _PIPE_CACHE：不重复加载 6.5GB 底模。
- 检测本地模型是否就绪并给出提示（未就绪提示先跑 dl_model.py）。
- 上传走 tempfile。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from prompt_gen import INFER_PARAMS, SCENES, build_prompt

# 管线单例缓存：{model_id: pipe}
_PIPE_CACHE: dict = {}

SCENE_CHOICES = [(f"{k}（{v['name']}）", k) for k, v in SCENES.items()]
SCENE_DEFAULT = "business"


def model_ready_hint() -> str:
    """检测本地模型是否就绪。"""
    from instantid_infer import find_local_sdxl

    if find_local_sdxl():
        return "✅ 本地检测到 SDXL 底模"
    return "⚠ 未检测到本地模型：请先运行 `dl_model.py`（或 `--model` 使用 HF 模型 ID）"


# ---------------------------------------------------------------------------
# Tab 1：数据准备
# ---------------------------------------------------------------------------

def data_prepare(files, progress=gr.Progress()) -> str:
    """上传 → 校验 → 预处理，返回 md 摘要。"""
    if not files:
        return "请先上传图片。"
    from validate_dataset import analyze, build_report
    from preprocess import standardize

    tmp = Path(tempfile.mkdtemp(prefix="cyberlora_prep_"))
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    for f in files:
        shutil.copy(f.name, raw_dir / Path(f.name).name)

    progress(0.3, desc="校验中 ...")
    issues, notes = analyze(raw_dir)
    report = build_report(raw_dir, issues, notes)

    progress(0.7, desc="预处理中 ...")
    out_dir = tmp / "train_data" / "100_cyberboy"
    out_dir.mkdir(parents=True)
    ok = 0
    for f in sorted(raw_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            standardize(f, out_dir / f"{f.stem}.png")
            ok += 1
        except Exception:
            pass

    lines = [report, "", f"## 预处理结果", "", f"- 已标准化 {ok} 张 → 1024×1024 PNG（临时目录：{out_dir}）"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 2：本地出图
# ---------------------------------------------------------------------------

def _get_pipe(model_id: str):
    """管线单例缓存（不重复加载底模）。"""
    from inference import load_pipeline, pick_device

    key = model_id
    if key not in _PIPE_CACHE:
        device, dtype = pick_device()
        pipe = load_pipeline(model_id, device, dtype)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        _PIPE_CACHE[key] = pipe
    return _PIPE_CACHE[key]


def generate_tab(scene_key: str, weight: float, seed: int, model_id: str, ref_photo, lora_path: str):
    """选场景/权重，可传本人照片自动标注相似度。"""
    from inference import generate, load_lora

    scene = build_prompt(scene_key)
    prompt = scene["positive"]
    pipe = _get_pipe(model_id or "stabilityai/stable-diffusion-xl-base-1.0")

    lora_file = None
    if lora_path and lora_path.strip():
        lora_file = load_lora(pipe, lora_path.strip())
    # diffusers 不解析 <lora:name:weight> 前缀
    if lora_file:
        prompt = prompt.split(">", 1)[-1].lstrip() if prompt.startswith("<lora:") else prompt

    img = generate(pipe, prompt, scene["negative"], weight, int(seed))

    sim_text = ""
    if ref_photo is not None:
        from face_similarity import band_of, build_analyzer, load_ref_embeddings, score_face

        try:
            analyzer = build_analyzer()
            with tempfile.TemporaryDirectory() as td:
                ref_dir = Path(td)
                shutil.copy(ref_photo.name, ref_dir / "ref.jpg")
                ref_embs = load_ref_embeddings(analyzer, ref_dir)
            sim, err = score_face(analyzer, ref_embs, img)
            sim_text = f"相似度：{sim:.4f}（{band_of(sim)}）" if sim is not None else f"相似度：无法计算（{err}）"
        except Exception as e:  # noqa: BLE001
            sim_text = f"相似度：计算失败（{e}）"

    out = f"场景：{scene['name']} | 权重：{weight:g} | seed：{seed}\n{sim_text}"
    return img, out


# ---------------------------------------------------------------------------
# Tab 3：相似度评测
# ---------------------------------------------------------------------------

def similarity_tab(ref_files, gen_files) -> str:
    if not ref_files or not gen_files:
        return "请分别上传参考集与生成图。"
    from face_similarity import build_analyzer, build_report, evaluate

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ref_dir, gen_dir = base / "ref", base / "gen"
        ref_dir.mkdir()
        gen_dir.mkdir()
        for f in ref_files:
            shutil.copy(f.name, ref_dir / Path(f.name).name)
        for f in gen_files:
            shutil.copy(f.name, gen_dir / Path(f.name).name)
        analyzer = build_analyzer()
        rows, scores, ref_files_ok = evaluate(analyzer, ref_dir, gen_dir)
        return build_report(rows, scores, len(ref_files_ok), gen_dir, ref_dir)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CyberLoRA 写真工具链") as demo:
        gr.Markdown("# CyberLoRA AI 写真工具链\n" + model_ready_hint())

        with gr.Tab("数据准备"):
            gr.Markdown("上传 20~30 张生活照 → 校验 → 标准化为 1024×1024（`100_cyberboy` 可直接上传 Colab 训练）")
            files_in = gr.File(file_count="multiple", label="照片（多选）")
            prep_btn = gr.Button("校验 + 预处理")
            prep_out = gr.Markdown()

        with gr.Tab("本地出图"):
            gr.Markdown(f"参数：{INFER_PARAMS}")
            with gr.Row():
                scene_dd = gr.Dropdown(SCENE_CHOICES, value=SCENE_DEFAULT, label="场景")
                weight_sl = gr.Slider(0.4, 0.95, value=0.75, step=0.05, label="LoRA 权重")
                seed_nb = gr.Number(value=42, precision=0, label="seed")
            model_tb = gr.Textbox(value="", placeholder="留空 = stabilityai/stable-diffusion-xl-base-1.0（或本地路径）", label="SDXL 底模")
            lora_tb = gr.Textbox(value="", placeholder="LoRA 权重路径（可选）", label="LoRA")
            ref_img = gr.Image(type="filepath", label="本人照片（可选，出图后自动标注相似度）")
            gen_btn = gr.Button("出图")
            gen_img = gr.Image(label="生成结果")
            gen_info = gr.Textbox(label="信息")

        with gr.Tab("相似度评测"):
            gr.Markdown("参考集（本人照片） vs 生成图，ArcFace 余弦 + 4 档评级")
            with gr.Row():
                ref_files_in = gr.File(file_count="multiple", label="参考集（本人照片）")
                gen_files_in = gr.File(file_count="multiple", label="生成图")
            sim_btn = gr.Button("评测")
            sim_out = gr.Markdown()

        prep_btn.click(data_prepare, inputs=[files_in], outputs=[prep_out])
        gen_btn.click(
            generate_tab,
            inputs=[scene_dd, weight_sl, seed_nb, model_tb, ref_img, lora_tb],
            outputs=[gen_img, gen_info],
        )
        sim_btn.click(similarity_tab, inputs=[ref_files_in, gen_files_in], outputs=[sim_out])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
