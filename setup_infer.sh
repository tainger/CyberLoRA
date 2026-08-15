#!/usr/bin/env bash
# setup_infer.sh —— 搭建本地 MPS 推理环境（.venv-infer）
# 安装：torch(MPS) / diffusers / transformers / insightface / onnxruntime /
#       opencv-python / gradio / modelscope
# 说明：Apple Silicon 推理走 MPS；训练（bitsandbytes/xformers）CUDA-only，本地不可用，
#       训练走 Colab（见 cyberlora_colab.ipynb）。
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-infer"

# 优先 Python 3.12（torch 对 3.14 支持不稳，脚本亦兼容 3.11）
PYTHON=""
for CAND in python3.12 python3.11 python3; do
  if command -v "$CAND" >/dev/null 2>&1; then
    PYTHON="$CAND"
    break
  fi
done
echo "使用 $PYTHON（$($PYTHON --version 2>&1)）"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip

echo "[1/2] 安装 torch + 扩散模型栈（MPS 可用）..."
"$PY" -m pip install torch torchvision

echo "[2/2] 安装 diffusers / 人脸分析 / Web Demo 依赖..."
"$PY" -m pip install \
  diffusers transformers accelerate safetensors \
  insightface onnxruntime \
  opencv-python \
  gradio \
  modelscope \
  pillow

echo "完成。环境：$VENV"
echo "下一步："
echo "  $VENV/bin/python inference.py --dry-run -s all   # 先看任务清单，不下模型"
echo "  $VENV/bin/python dl_model.py                    # HF 被墙时从 ModelScope 拉 SDXL（~6.5GB fp16）"
