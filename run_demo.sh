#!/usr/bin/env bash
# run_demo.sh —— 零 GPU 一键入口（建 venv → 校验 → 预处理 → Prompt）
# 用法：
#   ./run_demo.sh                      # 使用仓库自带样例图（自动生成），跑通全流程
#   ./run_demo.sh ~/Pictures/my_name   # 使用自己的照片目录
# 产物：
#   demo_out/train_data/100_cyberboy/  # 标准化训练集
#   demo_out/dataset_report.md         # 校验报告（样例图会触发数量/黑白等提示，正常）
#   demo_out/prompts_generated.md/.json
set -euo pipefail
cd "$(dirname "$0")"

INPUT="${1:-assets/samples}"
OUT="demo_out"
VENV=".venv-demo"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[1/5] 创建 ${VENV}（仅需 Pillow）..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q Pillow
fi

# 未指定输入目录且样例图不存在时，生成样例图（触发数量/黑白/重复提示属预期行为）
if [ "$#" -eq 0 ] && [ ! -d "$INPUT" ]; then
  echo "[1/5] 生成仓库样例图 → $INPUT ..."
  "$PY" - "$INPUT" <<'PYEOF'
import random, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
random.seed(42)

def make(name, size, base, gray=False):
    img = Image.new("RGB", size, base if gray else (base[0] // 3, base[1] // 3, base[2]))
    d = ImageDraw.Draw(img)
    for x in range(0, size[0], 96):  # 条纹背景，模拟真实照片纹理
        d.rectangle([x, 0, x + 48, size[1]], fill=base)
    cx, cy = size[0] // 2, size[1] // 2
    r = min(size) // 4
    face = (180, 180, 180) if gray else (222, 184, 135)   # "人脸"
    hair = (60, 60, 60) if gray else (30, 30, 30)         # 头发
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=face)
    d.ellipse([cx - r // 2, cy - r, cx + r // 2, cy + r // 2], fill=hair)
    img.save(out / name)

sizes = [(1024, 1024), (1200, 900), (900, 1200), (1080, 720), (720, 1080),
         (1024, 768), (768, 1024), (1536, 1024)]
for i, (w, h) in enumerate(sizes):
    make(f"photo_{i + 1:02d}.jpg", (w, h), (30 + i * 25, 90, 160))
make("photo_09.jpg", (512, 768), (120, 60, 30))
make("photo_10.jpg", (1024, 1024), (128, 128, 128), gray=True)   # 触发黑白提示
shutil.copy2(out / "photo_01.jpg", out / "photo_11.jpg")          # 触发重复提示
make("photo_12.jpg", (480, 640), (90, 30, 120))                   # 触发短边提示
print(f"样例图已生成：{out}")
PYEOF
fi

echo "[2/5] 校验数据集 → $OUT/dataset_report.md ..."
RC=0
"$PY" validate_dataset.py -i "$INPUT" -o "$OUT/dataset_report.md" || RC=$?
if [ "$RC" -ne 0 ]; then
  echo "注意：校验存在硬性问题（用样例图跑通时属正常），演示流程继续 ..."
fi

echo "[3/5] 标准化为 1024×1024 → $OUT/train_data/100_cyberboy ..."
"$PY" preprocess.py -i "$INPUT" -o "$OUT/train_data/100_cyberboy"

echo "[4/5] 生成 Prompt 清单 → $OUT/prompts_generated.* ..."
"$PY" prompt_gen.py -t cyberboy -s all -f md -o "$OUT/prompts_generated.md"
"$PY" prompt_gen.py -t cyberboy -s all -f json -o "$OUT/prompts_generated.json"

echo "[5/5] 完成。产物："
ls -1 "$OUT/train_data/100_cyberboy" | head -5
echo "  ... 共 $(ls "$OUT/train_data/100_cyberboy" | wc -l | tr -d ' ') 张训练图"
echo "  $OUT/dataset_report.md"
echo "  $OUT/prompts_generated.md"
echo "  $OUT/prompts_generated.json"
