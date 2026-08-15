# CyberLoRA
当前的AI写真生成处于“黑箱”困境——要么过度美化导致“面目全非”，要么在极端光影下产生五官畸形与油腻塑料感。我们拒绝赌博式的抽卡生成。
# CyberLoRA 复现文档（交给其他大模型的完整还原说明）

目标：只凭本文档，另一个大模型 / 开发者就能从零重建「赛博少年 CyberLoRA」—— 一套面向个人用户的 AI 写真工具链。文档给出定位、技术栈、每个模块的接口与算法规格、关键设计决策与踩坑，以及可直接执行的复现步骤与验收标准。

阅读顺序建议：一（做什么）→ 二（技术栈）→ 五（模块规格，照此重写代码）→ 七（复现命令）→ 八（验收）。


## 一、项目定位与要解决的问题

面向个人用户：用户提供 20~30 张生活照 → 训练专属人物 LoRA → 批量产出「看不出是 AI 生成」的高质量人像。

核心痛点：现成 AI 写真 App 要么不像本人，要么一眼假（五官畸形、塑料感、光影平淡）。

本项目把「像本人 + 像照片 + 有情绪张力」拆成可控工程参数，并沉淀为可复现、有量化指标、有故障排查的流水线。

覆盖场景：棚拍肖像、商务正装、咖啡厅日常、户外运动、旅行街景、赛博朋克、古风侠客、未来战士。

底层思维：不做一次性脚本，做可复现的工程系统——每一步都有量化指标、有文档、有故障排查。

关键成果（真实跑出，用于验证方法有效）：
- InstantID 免训练路径：avatar.jpg → 商务正装，人脸相似度 0.888（高度相似），8 步。
- 多 seed 稳定性：同参考 prompt 只换 seed 三次，相似度 0.792 ± 0.018（区间 [0.773, 0.808]），标准差仅 0.018 → 出图稳定、不靠 seed 运气。
- 对照组：基础 SDXL 无身份注入时相似度 -0.019（完全不像），证明相似度指标不误报。


### 背景与由来（给接手者的上下文）

- 本项目源于一份简历项目描述（resume.md），目标是把「一段简历描述」落地成「clone 下来就能跑、有质量闭环、有技术路线对比」的完整工具链。因此文档化、可量化指标、故障排查是一等公民，与代码同等重要。

- 首次评估环境：Apple M1 / macOS 14.2。评估时发现仓库原始状态有多处阻塞点（notebook 6 个致命 bug、README 空文件、不可执行推理脚本、训练依赖 CUDA-only），所以才拆出第七节的三条路线来分档。

- 开发是按阶段增量推进的（见附录 A 的 Phase 0~8 时间线）：先做零 GPU 工程化 → 修复训练 → 本地推理 → 质量闭环 → 曲线驱动选档 → Web Demo → 自动选片 → InstantID 免训练。复现时建议也按这个顺序，每步都能独立验证。

- 面向读者：本文预设读者（人或大模型）具备基本的扩散模型 / Python 工程能力；不熟悉的术语见附录 B 术语速查。


## 二、技术栈

| 层 | 选型 |
|---|---|
| 底模 | SDXL Base 1.0（stabilityai/stable-diffusion-xl-base-1.0） |
| 训练框架 | kohya-ss/sd-scripts 的 sdxl_train_network.py（tag v0.8.7） |
| 训练环境 | Google Colab T4 16GB（CUDA；Apple Silicon 无法本地训练） |
| 推理（本地） | diffusers + Apple MPS（M1/M2/M3） |
| 推理（免训练） | InstantID（ControlNet + IP-Adapter） |
| 人脸分析 | InsightFace（评测用 buffalo_l，InstantID 用 antelopev2） |
| 自动打标 | wd14-tagger（ONNX） |
| 清晰度评估 | OpenCV 拉普拉斯方差 |
| Web Demo | Gradio |
| 语言 | Python 3.12（torch 对 3.14 支持不稳，脚本回退到 3.11） |
| 模型下载 | ModelScope（国内镜像，HuggingFace 被墙时备用） |


## 三、环境前置

macOS / Linux，Python 3.12+（推理），约 12GB 磁盘。

- Apple Silicon 推理需 MPS 支持；训练需 NVIDIA GPU / Colab（bitsandbytes、xformers 依赖 CUDA）。
- 两个隔离 venv：
  - `.venv-demo`：零 GPU 工程化（只需 Pillow），由 `run_demo.sh` 建。
  - `.venv-infer`：本地推理 + 评测（torch / diffusers / insightface / gradio），由 `setup_infer.sh` 建。


## 四、文件清单与职责

| 文件 | 职责 | 依赖 |
|---|---|---|
| prompt_gen.py | 场景库 + 五段式 Prompt 生成 + 权重分档（单一数据源） | 纯 Python |
| validate_dataset.py | 数据集校验（数量/分辨率/重复/宽高比/黑白），输出 md 报告 | Pillow |
| select_photos.py | 自动选片：单人/清晰/多角度/去重 | insightface + opencv |
| preprocess.py | 图片标准化为 1024×1024（等比缩放+居中裁剪） | Pillow |
| run_demo.sh | 零 GPU 一键入口（建 venv → 校验 → 预处理 → Prompt） | 上述三者 |
| setup_infer.sh | 搭建本地 MPS 推理环境 | 无 |
| inference.py | 本地 MPS 出图 + 权重对比网格 + 相似度标注 + 曲线 | torch/diffusers + prompt_gen + face_similarity |
| face_similarity.py | ArcFace 相似度评测 + 4 档评级 + md 报告 | insightface + opencv |
| eval_report.py | 多 seed 稳定性评测：均值/标准差/区间 + md 报告 | 同上 |
| instantid_infer.py | InstantID 免训练单张出图 + 可选评分 | torch/diffusers + insightface + 自定义 pipeline |
| pipeline_stable_diffusion_xl_instantid.py + ip_adapter/ | vendored InstantID 管线（Apache-2.0） | diffusers |
| dl_model.py | 从 ModelScope 下载 SDXL 底模（仅 fp16 变体） | modelscope |
| dl_instantid.py | 从 ModelScope 下载 InstantID + antelopev2 | modelscope |
| cyberlora_colab.ipynb | Colab 一键训练 Notebook | kohya-ss/sd-scripts |
| kohya_config.json | 训练参数参考（SDXL + T4） | 无 |
| app.py | Gradio Web Demo（数据准备/出图/相似度三 Tab） | 上述全部 |
| 复现文档.md | 手写 Prompt 模板 / 加载说明 / 故障速查 / 路线评估 | 无 |

**架构原则（务必遵守）：** `SCENES` / `WEIGHT_TIERS` / `INFER_PARAMS` 只在 `prompt_gen.py` 定义一次，`inference.py`、`instantid_infer.py`、`app.py` 全部 import 复用，禁止各自重复定义。

**数据流：**

```
原始照片 → validate_dataset.py（校验）/ select_photos.py（自动选片）
    → preprocess.py（1024×1024）→ wd14 打标（.txt）→ kohya sdxl_train_network（Colab）
    → cyberboy_sdxl.safetensors → inference.py 出图 → face_similarity.py 评测闭环

免训练分支：单张照片 → instantid_infer.py（人脸 embedding 注入 IP-Adapter）→ 评测
```

**模块依赖图：**

```
prompt_gen.py → inference.py → app.py
    ↓
    SCENES
    WEIGHT_TIERS
    ↑
    embed_array()

select_photos.py → instantid_infer.py
    ↑
validate_dataset.py ← run_demo.sh
    ↑
preprocess.py
```


## 五、模块实现规格（照此可重写每个脚本）

### 5.1 prompt_gen.py —— 单一数据源

三个模块级常量，供全项目复用：

**SCENES：** 8 个场景 key → dict，字段：`name`（中文名）、`shot`（景别：wide / medium / closeup）、`subject`（默认：`lboy`）、`scene`（构图/场景英文）、`style`（艺术/摄影风格）、`quality`（画质词）、`negative`（负向词）。

8 个 key：`studio` / `cyberpunk` / `astronaut` / `business` / `sports` / `wuxia` / `cafe` / `soldier`。

**WEIGHT_TIERS：** 景别 →（默认权重，中文理由）
- `wide` →（0.60，远景/全景，低权重避免五官畸变）
- `medium` →（0.75，常规场景，最佳平衡区间）
- `closeup` →（0.90，特写/肖像，高权重保留低权重）

**INFER_PARAMS：** `{sampler: 'DPM++ 2M Karras', steps: 28, cfg_scale: 7, size: '1024x1024', clip_skip: 2}`

**五段式拼装（`build_prompt(scene_key, trigger, lora_name=None, weight=None)`）：**

`[trigger, subject, scene, style, quality]` 用 '，' 连接。触发词必须在最前面，否则 LoRA 可能不生效。若使用 `lora_name`，则前缀 `<lora:name:weight>`（供 WebUI；diffusers 不解析此语法，权重走 API）。

输出 dict：`key/name/shot/weight/weight_reason/positive/negative`。

**CLI：** `--trigger/-t`（默认：`cyberboy`），`--lora-name/-l`，`--scene/-s`（逗号分隔或 `all`），`--weight/-w`（强制权重），`--format/-f`（`md/txt/json`），`--output/-o`，`--list`。

未知场景要报错并列出可用场景。md 输出含参数表、场景权重表、逐场景 Prompt 代码块。


### 5.2 validate_dataset.py —— 数据集校验

**阈值：** `MIN_COUNT=20`，`MAX_COUNT=30`，`MIN_SIDE=512`，`MAX_ASPECT=2.0`，有效扩展名 `.jpg/.jpeg/.png/.webp/.bmp`。

逐图采集：`width/height/min_side/aspect（=max/min，2位）/size_kb/mode/hash/grayscale`。

- **去重：** 8×8 均值哈希（转灰度 resize 8×8，逐像素 > 均值 ? '1' : '0'），hash 相同则重复/近似重复。
- **黑白判定：** resize 64×64 RGB，逐像素 `|r-g|>10` 或 `|g-b|>10` 即判彩色，否则黑白。

**`analyze()` 返回（issues, notes）：**
- `issues`（硬性，必须修复）：数量 < 20；短边 < 512 的图；重复图。
- `notes`（提示，需人工确认）：数量 > 30；宽高比 > 2.0；疑似黑白；「侧脸/表情覆盖」无法自动校验（建议正脸 > 10，侧脸 > 5，表情 > 3）。

报告含结论（✗ 不通过 / ⚠ 通过 / ✅ 通过）、分辨率分布、短边范围、明细表。

有 `issues` 时 `SystemExit(1)`（便于 `run_demo.sh` / CI 感知）。


### 5.3 select_photos.py —— 自动选片（把选图规范自动化）

**阈值：** `MIN_SIDE=512`，`DET_SCORE_THR=0.55`，`FACE_MIN_RATIO=0.04`，`BLUR_THR=60.0`，`DEDUP_COS=0.92`，`YAW_BUCKETS=5`，`TARGET_PER_BUCKET=3`，`TARGET_TOTAL=20`。

**流程：**
1. **硬过滤：** 短边 ≥ 512；用 InsightFace 检测至少一张人脸；`det_score ≥ 0.55`；人脸面积占比 ≥ 0.04；拉普拉斯方差 ≥ 60。
2. **质量分排序：** `quality = blur * 0.4 + det_score * 0.3 + (1 - abs(yaw)/90) * 0.3`。
3. **Yaw 分桶轮转：** 将 yaw 按 `[-90, -54), [-54, -18), [-18, 18), [18, 54), [54, 90]` 分 5 桶，每桶最多取 3 张，确保角度多样性。
4. **去重：** 对已选图的 embedding 计算余弦相似度，`≥ 0.92` 则跳过。
5. **补充：** 若不足 20 张，从高质量候补中继续选，直到满 20 张或耗尽。

输出：每张选中图的 `path / yaw / blur / score / face_ratio`，以及 `yaw_distribution` 饼图数据。CLI：`-i INPUT -o OUTPUT -r REPORT`。


### 5.4 preprocess.py —— 图片标准化

- 读取图片，保持 RGB。
- 计算短边，等比例缩放使短边 = 1024。
- 从中心裁剪 1024×1024。
- 保存为 PNG（无损，便于训练）。
- 若图片已为 1024×1024 且无裁剪偏移，则原样复制（避免重新编码损失）。

CLI：`-i INPUT_DIR -o OUTPUT_DIR`。


### 5.5 inference.py —— 本地 MPS 出图（路线 B）

- 惰性导入 `torch/diffusers`：`--list` / `--dry-run` 无需安装即可运行。
- **`pick_device()`：** MPS → float16，CUDA → float16，CPU → float32。
- **`load_pipeline`：**
  - `StableDiffusionXLPipeline.from_pretrained(..., torch_dtype=dtype, use_safetensors=True, variant='fp16')`，失败则回退 `variant`。
  - 采样器固化 DPM++ 2M Karras = `DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True, algorithm_type='dpmsolver++')`。
  - `enable_attention_slicing()` + `vae.enable_tiling()` 防 OOM。
- **LoRA：** `pipe.load_lora_weights(lora_dir, weight_name=lora_file)`。
- **`generate`：** `generator=torch.Generator('cpu').manual_seed(seed)`；有 LoRA 时权重通过 `cross_attention_kwargs={'scale': weight}` 传入（不是 `<lora>` 语法）；`width=height=1024`，`steps`/`cfg` 取 `INFER_PARAMS`。
- **`--compare`：** 对 `COMPARE_WEIGHTS=[0.5, 0.75, 0.95]` 各出一张，`make_grid` 横向拼接（顶部留白写多行标签）。
- **`--ref` 本人照片目录：** 加载参考集 embedding，每档出图后 `score_face`（PIL → BGR → embed → 与参考集取最大余弦）标注在网格，并 `draw_sim_curve` 输出「权重 vs 相似度」折线图（纯 PIL，英文标签避免乱码，no_face 跳过不画）。


### 5.6 face_similarity.py —— 质量闭环

- **`build_analyzer`：** `FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])`，`prepare(ctx_id=1, det_size=(640,640))`。
- **`embed_array(app, bgr)`：** `app.get(bgr)`，取面积最大人脸的 `normed_embedding`（512 维）；无脸返回 `(None, 'no_face')`。
- **`embed_image`：** `cv2.imread`（BGR）→ `embed_array`。
- **`cosine(a, b)`：** `dot / (|a||b|)`。
- **评级 BANDS：** `≥ 0.60 高度相似 / ≥ 0.45 相似 / ≥ 0.30 中等 / < 0.30 偏低`。
- 生成图 vs 参考集取最大相似度（参考集多张，取最像的作为上限）。
- 报告含均值/最高/最低、逐图评级、分档标准表。CLI：`--ref/-r`，`--gen/-g`，`--report/-o`。


### 5.7 eval_report.py —— 多 seed 稳定性

- 同一参考、同一 prompt，多个 seed 重复出图，统计均值 ± 标准差 ± 区间，输出 md。
- 用意：单张分数含运气成分，报告才能说明方案稳定（实测 0.792 ± 0.018）。


### 5.8 instantid_infer.py —— 免训练路线

- 自动定位本地资源：`find_local_sdxl`（找 `models/**/snapshots/*/model_index.json`），`find_instantid`（找 `models/**/ip-adapter.bin` 及同级 `ControlNetModel/`）。
- 人脸分析器用 `antelopev2`：`FaceAnalysis(name='antelopev2', root='.')` → 读 `./models/antelopev2`。
- 参考照 resize 1024 → 取最大脸 → `face['embedding']`（身份）+ `draw_kps(img, face['kps'])`（ControlNet 条件图）。
- 管线：`ControlNetModel.from_pretrained(cn_dir)` + `StableDiffusionXLInstantIDPipeline.from_pretrained(model, controlnet=..., variant='fp16')`（失败回退）→ `load_ip_adapter_instantid(adapter)` → `set_ip_adapter_scale(ip_scale)`。
- 出图：去掉触发词（身份来自 embedding，不需要 `cyberboy`），传 `image_embeds=face_emb`，`image=face_kps`，`controlnet_conditioning_scale=cn_scale`，`ip_adapter_scale=ip_scale`。
- `--score`：用 `antelopev2` 对生成图再 embed，与参考照 embedding 算余弦，套用 `face_similarity.band_of`。

CLI：`--ref/-r`，`--scene/-s`（`business`），`--model/-m`，`--steps`（28），`--seed`（42），`--ip-scale`（0.8），`--cn-scale`（0.8），`--output/-o`，`--score`。


### 5.9 下载脚本（HF 被墙时用）

- **`dl_model.py`：** `snapshot_download('AI-ModelScope/stable-diffusion-xl-base-1.0', cache_dir='./models', allow_patterns=['.*fp16.safetensors', '*.json', '*.txt', 'tokenizer/*', 'scheduler/*'])`——只拉 fp16 变体减少下载量。
- **`dl_instantid.py`：** `InstantX/InstantID`（仅 `ControlNetModel/*` + `ip-adapter.bin`）+ `AI-ModelScope/antelopev2`（→ `./models/antelopev2`）。


### 5.10 app.py —— Gradio Web Demo

三 Tab：
1. **数据准备：** 上传 → 校验 → 预处理
2. **本地出图：** 选场景/权重，可传本人照片自动标注相似度
3. **相似度评测**

要点：管线单例缓存 `_PIPE_CACHE`（不重复加载 6.5GB 底模）；检测本地模型是否就绪并给出提示；上传走 `tempfile`。


## 六、Colab 训练规格（cyberlora_colab.ipynb）

关键（曾修复 6 个致命 bug，务必照做）：

1. 训练器用 `sdxl_train_network.py`（不是 `train_network.py`）。
2. 直接克隆 `kohya-ss/sd-scripts`（`git clone --depth 1 -b v0.8.7`），不要用 `bmaltais/kohya_ss`（子模块无克隆为空）。
3. 用 `accelerate launch` 启动，并先 `write_basic_config()`（单卡非交互）。
4. 必开 `--gradient_checkpointing`（T4 16GB 否则 OOM）。
5. 必开 `--no_half_vae`（避免 SDXL VAE fp16 出 NaN 黑图）。
6. wd14 打标：自动探测 ONNX 输入名 + CPU 回退。


**训练参数（kohya_config.json，把 SDXL 压进 16GB 的组合参）：**

| 参数 | 值 |
|---|---|
| `mixed_precision` | fp16（T4 Turing 仅 fp16 有 Tensor Core 加速，bf16 需 Ampere） |
| `no_half_vae` | true |
| `optimizer_type` | AdamW8bit（优化器状态显存降至 1/4） |
| `gradient_checkpointing` | true（算力换显存） |
| `cache_latents / cache_latents_to_disk` | true（减少 CPU-GPU 传输） |
| `network_dim` | 32 |
| `network_alpha` | 16（表达力与体积平衡，产物 50~100MB） |
| `enable_bucket` | true（min 512 / max 1536，自适应不同尺寸原图） |
| `noise_offset` | 0.1（增强噪点） |
| `train_batch_size` | 1 |
| `max_train_epochs` | 8 |
| `lr` | 1e-4（UNet）/ 5e-5（TE） |
| `lr_scheduler` | cosine |
| `seed` | 42 |
| 数据目录命名 | `train_data/100_cyberboy`（100 是 kohya 的 repeats 前缀） |
| 触发词 | `cyberboy` |
| `keep_tokens` | 1 |


**无 CUDA 时的验证方式：** 正则提取 notebook 里 f-string 生成的训练脚本 → `compile()` 静态语法校验。


## 七、三条实现路线（分档交付策略）

项目不是一次性堆完，而是按成本/说服力分三档增量实现，可按需选跑。三条路线相互独立又能组合（推荐 A + C）。

| | 路线 A：Colab 真训练 | 路线 B：本地 M1 推理 | 路线 C：零 GPU 工程化 |
|---|---|---|---|
| **做什么** | 修好 notebook，上传照片跑真实 LoRA，出多场景成品图 | diffusers 加载 SDXL + 现成/自训 LoRA 出图 | 把仓库补成「clone 下来就能跑」的工具链，不碰模型 |
| **耗时** | 20~45 分钟 | ~1 小时（含 7GB 下载） | 10~15 分钟 |
| **硬件** | Colab T4（免费额度够） | 本机 M1 | 本机，无 GPU |
| **下载量** | 云端，本机 0 | ~7GB | 0 |
| **出人像图** | ✅ 出本人图 | ✅ 出图但非本人 | ❌ 不出图 |
| **产出** | LoRA `cyberboy_sdxl.safetensors` | 用现成 LoRA 验证 | 数据规范、预处理、工程沉淀 |
| **验证的能力点** | 全部：显存压缩/相似度/过拟合平衡/Prompt 模板/多端交付 | Prompt 模板、权重分档、推理参数 | 数据校验、预处理、Prompt 生成 |
| **说服力** | 最高 | 中 | 较低 |
| **对应产物** | 第六节 + `cyberlora_colab.ipynb` | `setup_infer.sh` + `inference.py` | `run_demo.sh` + `validate_dataset.py` + `prompt_gen.py` |

**为什么这样分：** 训练环节 Apple Silicon 无解（AdamW8bit/bitsandbytes、xformers 均 CUDA-only），只能上 Colab/NVIDIA → 独立成路线 A；本地想看出图效果但不训练 → 路线 B（用现成 LoRA 验证 Prompt 与权重分档）；只想让仓库可复现、零成本跑通前置流程 → 路线 C。

**落地顺序建议：** 先 C（成本最低且必要，保证别人 clone 能复现前置流程）→ 再 A（产出模型与成品图，实证核心论述）→ 有时间再补 B。A + C 组合性价比最高。


## 八、复现步骤（可直接执行）

```bash
cd repo/CyberLoRA

# 路线 C：零 GPU 工程化（约 1 分钟，用仓库自带样例图跑通）
./run_demo.sh                    # 或 ./run_demo.sh ~/Pictures/my_name
# 产出 demo_out/train_data/100_cyberboy/ + demo_out/dataset_report.md + demo_out/prompts_generated.md/.json

# 路线 B：本地 MPS 推理
./setup_infer.sh                 # 建 .venv-infer，装 torch/diffusers/insightface/gradio
./venv-infer/bin/python inference.py --dry-run -s all   # 先看任务清单，不下模型
./venv-infer/bin/python dl_model.py                     # HF 被墙时从 ModelScope 拉 SDXL（~6.5GB fp16）
./venv-infer/bin/python inference.py -s studio --lora ./cyberboy_sdxl.safetensors --compare --ref 本人照片目录

# 质量闭环：相似度评测
./venv-infer/bin/python face_similarity.py --ref 本人照片目录 --gen ./demo_out/infer --report ./demo_out/sim_report.md

# 免训练路线：InstantID（额外 ~4.6GB）
./venv-infer/bin/python dl_instantid.py
./venv-infer/bin/python instantid_infer.py --ref ./assets/avatar.jpg --scene business --score

# 自动选片
./venv-infer/bin/python select_photos.py -i ./raw_photos -o ./selected -r ./select_report.md

# Web Demo
./venv-infer/bin/python app.py   # 浏览器打开 http://127.0.0.1:7860
```

**Colab 训练：** 上传 `cyberlora_colab.ipynb` → 运行时选 T4 GPU → 逐格执行 → Step 2 上传 `100_cyberboy` 训练集 → Step 7 下载 `.safetensors`。

**推理图（WebUI/ComfyUI）：** 把 `.safetensors` 放进 `models/Lora/`（或 `models/loras/`），套用 `prompts_generated.md` 的 Prompt。


## 九、验收标准与预期产出

| 步骤 | 预期 |
|---|---|
| `run_demo.sh` | 生成标准化训练集 + 校验报告 + Prompt 清单；样例图会触发数量/黑白等提示（正常） |
| `validate_dataset.py` | 有硬性问题时退出码为 1，报告结论为 ✗ |
| `inference.py --dry-run` | 打印 8 场景任务清单，无需 torch |
| `inference.py --compare --ref` | 每场景一张三档对比网格 `{scene}_compare.png` + 一张 `{scene}_sim_curve.png` |
| `face_similarity.py` | 每张生成图输出相似度与评级；无身份注入时应近 0（对照组实测 -0.019） |
| `instantid_infer.py --score` | 生成图 + 与参考照相似度（avatar → business 实测 0.888） |

**关键量化基线：** InstantID 多 seed 均值 0.792 ± 0.018；相似度评级阈值 0.60 / 0.45 / 0.30。


## 十、关键设计决策与踩坑（复现时务必注意）

1. **16GB 显存压缩组合：** fp16 + AdamW8bit + gradient_checkpointing + cache_latents_to_disk + dim32/alpha16 + enable_bucket。
2. **相似度从数据侧解决：** 不靠调参，靠选图规范 + `select_photos.py` 自动选片（角度多样是关键）。
3. **LoRA 权重按景别划分：** 远景 0.5~0.65 / 常规 0.7~0.85 / 特写 ≤ 0.95；高权重会把远处的脸也拉向训练样本导致畸变。
4. **曲线驱动选档：** 把 ArcFace 相似度嵌入出图流程，取曲线拐点（相似度够高、画面又没崩的那一档），不拍脑袋。
5. **MPS VAE 坑（实测）：** VAE 保持 fp16 加载，由管线自动 upcast；手动改 float32 会因 latents 仍 fp16 在 MPS 上产生 NaN 黑图。
6. **HF 被墙绕行：** ModelScope 镜像 + 只拉 fp16 variant 减半下载量。
7. **diffusers 不解析 `<lora:name>` 语法：** LoRA 权重必须通过 `cross_attention_kwargs={'scale': w}` 传入。
8. **触发词位置：** 必须在 Prompt 最前面，否则 LoRA 可能不生效；InstantID 路线反而要去掉触发词。
9. **免训练 vs LoRA 选型：** C 端即时出图 → InstantID；B 端批量多场景 → LoRA；两条路线共用同一套相似度评测闭环，用数据做判据。


## 十一、已知限制与后续方向

| 限制 | 状态 |
|---|---|
| M1 出图慢（LoRA ~6 分钟/张，InstantID ~24 分钟/张，183s/步） | 正常，生产需 GPU |
| InstantID 画质受步数限制（8 步偏粗，28 步 M1 需 ~85 分钟） | 已知 |
| 相似度评测对插画风格漏检 | InsightFace 只识别写实人脸 |
| Colab notebook 未在 T4 上实测过 | 静态语法校验通过，兼容性需现场验证 |

**后续可加：** LCM-LoRA / SDXL-Turbo 加速（28→4 步）、CLIP 美学评分、证件照/电商模特/换装（IP-Adapter + ControlNet pose）、微信小程序 / Gradio Share 部署。

**复现建议：** 先用 `run_demo.sh`（零 GPU）验证工程骨架，再 `setup_infer.sh` + `dl_model.py` 打通本地出图与相似度闭环。最后按第六节在 Colab 训练得到 `.safetensors`。免训练路线（InstantID）可独立于训练单独验证，最适合快速出成果。


## 附录 A：迭代时间线（Phase 0~8，还原「怎么一步步建起来的」）

| 阶段 | 目标 | 关键产出 / 改动 |
|---|---|---|
| Phase 0 | 原始状态 | `preprocess.py`、含 6 个致命 bug 的 notebook、`kohya_config.json`、`prompts.md` 等；README 为空 |
| Phase 1 | 零 GPU 工程化（路线 C） | 新增 `validate_dataset.py` / `prompt_gen.py` / `run_demo.sh` / `.gitignore`，补齐 README。确立「SCENES/WEIGHT_TIERS 单一数据源」 |
| Phase 2 | Colab 训练修复（路线 A） | 修 6 个致命缺陷（见第六节）。本地无 CUDA，用正则提取 f-string 脚本 + `compile()` 静态校验 |
| Phase 3 | 本地 MPS 推理（路线 B） | 新增 `setup_infer.sh` / `inference.py` / `dl_model.py`。惰性导入 torch、固化 DPM++ 2M Karras、MPS VAE 保 fp16 |
| Phase 4 | 质量闭环 | 新增 `face_similarity.py`：ArcFace embedding + 余弦 + 4 档评级 + md 报告；抽出 `embed_array()` 支持内存图直接打分 |
| Phase 5 | 出图集成相似度 + 权重曲线 | `inference.py --compare --ref`：每档时算相似度标在网格，输出「权重 vs 相似度」折线图，取拐点选档 |
| Phase 6 | Web Demo | 新增 `app.py`：三 Tab；管线单例缓存 `_PIPE_CACHE` 不重复加载底模 |
| Phase 7 | 自动选片 | 新增 `select_photos.py`：硬过滤 + 质量分 + yaw 分桶轮转 + embedding 去重 |
| Phase 8 | InstantID 免训练路径 | 新增 `instantid_infer.py` / `dl_instantid.py` + vendored pipeline；证明「免训练 vs LoRA」取舍，实测 0.888 |


## 附录 B：术语速查

| 术语 | 含义 |
|---|---|
| SDXL Base 1.0 | Stability AI 的高分辨率文生图底模（1024×1024），本项目一切生成的基础 |
| LoRA | 低秩适配微调，只训练少量参数即可让底模学会「某个人的脸」，产物 50~100MB，可挂载/卸载 |
| 触发词（trigger） | LoRA 训练时绑定的关键词（本项目 `cyberboy`），推理时必须放在 Prompt 最前面才生效 |
| kohya-ss/sd-scripts | 社区主流的 LoRA 训练脚本集合；SDXL 用 `sdxl_train_network.py` |
| network_dim / alpha | LoRA 的秩与缩放；32/16 兼顾表达力与体积 |
| DPM++ 2M Karras | 采样器；diffusers 里 = DPMSolverMultistep + karras sigmas + dpmsolver++ |
| CFG scale | 提示词引导强度，本项目固定 7 |
| VAE | 潜空间与像素图互转的编码器；SDXL 的 VAE 在 fp16 下易出 NaN 黑图 |
| ArcFace | 人脸识别模型，输出 512 维 embedding，用余弦相似度衡量「像不像同一个人」 |
| InsightFace | 人脸分析库；评测用 `buffalo_l`，InstantID 用 `antelopev2` |
| InstantID | 免训练人物一致性方案：人脸 embedding 注入 IP-Adapter + 关键点经 ControlNet 约束布局 |
| IP-Adapter | 把图像特征作为条件注入扩散模型的适配器（此处承载人脸身份） |
| ControlNet | 用条件图（此处为人脸关键点）约束生成结构的控制网络 |
| MPS | Apple Silicon 的 GPU 后端（Metal Performance Shaders） |
| wd14-tagger | 自动给训练图打标签（caption）的 ONNX 模型 |
| yaw | 人脸偏航角（左右转头角度），用于判断正脸/侧脸 |


## 附录 C：8 套场景 Prompt 实例（手写参考版，来自 prompts.md）

通用参数：Sampler DPM++ 2M Karras / Steps 25-30 / CFG 7 / LoRA Weight 0.75。触发词 `cyberboy` 必须在最前面。

`prompt_gen.py` 是这些模板的参数化版本；两者结构一致（触发词 → 主体 → 场景 → 风格 → 画质）。

| # | 场景 | Prompt |
|---|---|---|
| 1 | 棚拍肖像 | `cyberboy, 1boy, portrait, studio lighting, bokeh, detailed face, looking at camera` |
| 2 | 赛博朋克 | `cyberboy, 1boy, standing on Tokyo street, neon lights, rain, night, cyberpunk, 8k, cinematic lighting` |
| 3 | 科幻宇航员 | `cyberboy, 1boy, astronaut in spacesuit, walking on Mars surface, red desert, sci-fi, cinematic, hdr, 8k` |
| 4 | 商务正装 | `cyberboy, 1boy, wearing black suit, white shirt, necktie, office background, professional, natural light, corporate headshot` |
| 5 | 户外运动 | `cyberboy, 1boy, surfing on ocean waves, sunset sky, dynamic action pose, splashing water, golden hour` |
| 6 | 古风侠客 | `cyberboy, 1boy, ancient Chinese warrior, traditional armor, holding sword, temple in background, ink painting style, epic` |
| 7 | 咖啡厅日常 | `cyberboy, 1boy, sitting in cozy cafe, holding coffee cup, reading book, warm lighting, window with street view, depth of field, 35mm photography` |
| 8 | 未来战士 | `cyberboy, 1boy, futuristic soldier, mechanical armor, glowing visor, destroyed city background, sci-fi, hdr, 8k, unreal engine 5 render` |

**通用负向词：** `worst quality, low quality, blurry, deformed face, bad anatomy, extra fingers, extra limbs, modern elements`（按场景微调）


## 附录 D：故障速查（复现最易踩的坑）

**训练（Colab）**

| 故障 | 解决方案 |
|---|---|
| OOM | 减 `network_dim` 到 16 或关 `cache_latents`；确保 `train_batch_size=1`；必开 `--gradient_checkpointing` |
| `.safetensors` 0KB | 训练异常终止，确保至少跑完 1 个 epoch |
| xformers 装不上 | 改用 `--sdpa` |
| HF 下载超时 | `export HF_ENDPOINT=https://hf-mirror.com` 或改 ModelScope |

**本地推理（MPS，实机踩坑）**

| 故障 | 解决方案 |
|---|---|
| 出图全黑 + `invalid value encountered in cast` | SDXL VAE fp16 出 NaN。保持 VAE fp16 交管线自动 upcast，**不要**手动改 float32（latents 仍 fp16 反而 NaN） |
| 下载 `connection_reset` | HF 不可达 → 用 `dl_model.py` 从 ModelScope 拉，之后 `--model` 本地路径 |
| 下载量翻倍 | 只拉 `*.fp16.safetensors` + 加载时 `variant='fp16'` |
| `<lora:name>` 写进 prompt 无效 | diffusers 不解析，权重走 `cross_attention_kwargs={'scale': w}`（`--compare` 已处理） |
| 1024×1024 爆内存 | 开 `enable_attention_slicing()` + `vae.enable_tiling()`（已内置） |

**通用**

| 故障 | 解决方案 |
|---|---|
| 生成脸不像 | 照片角度单一/数量不足 → 补到 20~30 张，覆盖侧脸与多表情（或用 `select_photos.py`） |
| 五官变形 | 过拟合 → 降 LoRA 权重到 0.5~0.65 或减 epoch |
| 触发词不生效 | 拼写错误或不在前面 → 放 Prompt 开头 |
| 底模不匹配 | LoRA 必须配 SDXL Base 1.0，不能用 SD1.5 底模 |


## 附录 E：已验证环境版本（首次评估实测，供对齐）

| 项 | 版本 / 值 |
|---|---|
| 评估机器 | Apple M1 / macOS 14.2 |
| 推理环境 Python | 3.12（脚本回退支持 3.11；torch 对 3.14 支持不稳） |
| torch / diffusers | 实测 torch 2.13.0 / diffusers 0.39.0，MPS 可用 = True |
| 零 GPU 环境 Pillow | 12.3.0（`.venv-demo`，仅 `preprocess.py` / `validate_dataset.py` 用） |
| SDXL 底模体积 | 约 6.5~7GB（仅 fp16 变体约减半） |
| InstantID 资源 | ControlNetModel ~2.5G + ip-adapter.bin ~1.7G + antelopev2（5 个 onnx） |
| sd-scripts | tag v0.8.7（若与 Colab 预装 torch 冲突，去掉 `-b v0.8.7` 用最新版） |