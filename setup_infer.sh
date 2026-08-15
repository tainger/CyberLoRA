#!/usr/bin/env bash
# setup_infer.sh —— 搭建本地 MPS 推理环境（.venv-infer）
# 安装：torch / diffusers / transformers / insightface / onnxruntime /
#       opencv-python / gradio / modelscope
# 说明：Apple Silicon 推理走 MPS；训练（bitsandbytes/xformers）CUDA-only，本地不可用，
#       训练走 Colab（见 cyberlora_colab.ipynb）。
# 用法：
#   ./setup_infer.sh                     # 默认用清华 tuna 镜像（国内直连）
#   ./setup_infer.sh ""                  # 走官方 PyPI
#   ./setup_infer.sh https://mirrors.aliyun.com/pypi/simple/
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-infer"
MIRROR="${1:-https://pypi.tuna.tsinghua.edu.cn/simple}"
if [ -n "$MIRROR" ]; then
  PIP_ARGS=(-i "$MIRROR")
else
  PIP_ARGS=()
fi

# 优先 Python 3.12（torch 对 3.14 支持不稳，脚本亦兼容 3.11）
PYTHON=""
for CAND in python3.12 python3.11 python3; do
  if command -v "$CAND" >/dev/null 2>&1; then
    PYTHON="$CAND"
    break
  fi
done
echo "使用 ${PYTHON}（$(${PYTHON} --version 2>&1)）"

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"   # 必须用选定的解释器，不能硬编码 python3（否则可能建成 3.14 导致 torch 无轮子）
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip "${PIP_ARGS[@]}"

echo "[1/2] 安装 torch + 扩散模型栈..."
# 架构适配：
# - Apple Silicon (arm64)：最新 torch，MPS 加速
# - Intel macOS (x86_64)：PyTorch 2.2.2 之后不再发布 x86_64 macOS 轮子，固定 2.2.2（无 MPS，走 CPU）
if [ "$(uname -m)" = "arm64" ]; then
  "$PY" -m pip install torch torchvision "${PIP_ARGS[@]}"
else
  "$PY" -m pip install torch==2.2.2 torchvision==0.17.2 "${PIP_ARGS[@]}"
fi

echo "[2/2] 安装 diffusers / 人脸分析 / Web Demo 依赖..."
if [ "$(uname -m)" != "arm64" ]; then
  # Intel macOS 兼容性约束（2026-08 实测验证过的组合）：
  # - torch 2.2.2：PyTorch 2.2.2 之后不再发布 x86_64 macOS 轮子（无 MPS，走 CPU）
  # - numpy<2：torch 2.2.2 与 insightface 均按 NumPy 1.x 编译，numpy 2 会 ABI 报错
  # - opencv-python==4.9.0.80：4.10+ 按 NumPy 2 编译，与 numpy 1.x 冲突
  # - transformers==4.46.3：transformers 5.x 要求 torch>=2.5，会禁用 PyTorch 后端
  # - gradio==5.4.0：gradio 6.x 要求 huggingface-hub>=1.16，与 transformers 4.46（hub<1.0）冲突
  # - diffusers<0.33：与 torch 2.2.2 配套的上限
  # - insightface 不能直接 pip：无 x86_64 macOS 0.7.x 轮子，pip 自动选的 1.0.x
  #   是新模型包格式，不兼容 antelopev2（assert 'detection' 失败），必须源码构建 0.7.3（见下）
  "$PY" -m pip install \
    torch==2.2.2 torchvision==0.17.2 \
    "numpy<2" \
    "diffusers<0.33" "transformers==4.46.3" accelerate safetensors \
    onnxruntime "opencv-python==4.9.0.80" \
    "gradio==5.4.0" modelscope pillow "${PIP_ARGS[@]}"

  # insightface 0.7.3 源码构建 + 运行期依赖（2026-08 实测验证）：
  # - --no-deps 跳过 opencv-python-headless（5.x 无 x86_64 轮子会触发源码编译失败）
  # - albumentations 同样 --no-deps；其依赖 albucore/stringzilla/simsimd 有纯轮子
  # - matplotlib 与 ml_dtypes<0.6 均为 0.7.3/onnx 运行期依赖，缺了 ImportError
  # - 源码构建要写 TMPDIR（macOS 沙盒 /var/folders 不可写，用 /private/tmp）
  echo "  - 源码构建 insightface 0.7.3（antelopev2 必需）..."
  TMPDIR=/private/tmp "$PY" -m pip install --no-deps insightface==0.7.3 "${PIP_ARGS[@]}"
  "$PY" -m pip install --no-deps albumentations "${PIP_ARGS[@]}"
  "$PY" -m pip install albucore stringzilla simsimd matplotlib "ml_dtypes<0.6" "${PIP_ARGS[@]}"
else
  "$PY" -m pip install \
    diffusers transformers accelerate safetensors \
    insightface onnxruntime \
    opencv-python \
    gradio \
    modelscope \
    pillow "${PIP_ARGS[@]}"
fi

echo "完成。环境：$VENV"
echo "下一步："
echo "  $VENV/bin/python inference.py --dry-run -s all   # 先看任务清单，不下模型"
echo "  $VENV/bin/python dl_model.py                    # HF 被墙时从 ModelScope 拉 SDXL（~6.5GB fp16）"
