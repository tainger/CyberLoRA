#!/usr/bin/env python3
"""prompt_gen.py —— 场景库 + 五段式 Prompt 生成 + 权重分档（单一数据源）

全项目唯一数据源：SCENES / WEIGHT_TIERS / INFER_PARAMS 只在此定义一次，
inference.py / instantid_infer.py / app.py 一律 import 复用，禁止重复定义。

五段式拼装：trigger → subject → scene → style → quality，用 ', ' 连接。
触发词必须在最前面，否则 LoRA 可能不生效。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# 单一数据源：8 个场景
# ---------------------------------------------------------------------------
SCENES: dict = {
    "studio": {
        "name": "棚拍肖像",
        "shot": "closeup",
        "subject": "1boy",
        "scene": "portrait, studio lighting, bokeh, detailed face, looking at camera",
        "style": "professional photography",
        "quality": "high quality, sharp focus",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy, extra fingers",
    },
    "cyberpunk": {
        "name": "赛博朋克",
        "shot": "wide",
        "subject": "1boy",
        "scene": "standing on Tokyo street, neon lights, rain, night",
        "style": "cyberpunk",
        "quality": "8k, cinematic lighting",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy, extra limbs",
    },
    "astronaut": {
        "name": "科幻宇航员",
        "shot": "medium",
        "subject": "1boy",
        "scene": "astronaut in spacesuit, walking on Mars surface, red desert",
        "style": "sci-fi, cinematic",
        "quality": "hdr, 8k",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy",
    },
    "business": {
        "name": "商务正装",
        "shot": "closeup",
        "subject": "1boy",
        "scene": "wearing black suit, white shirt, necktie, office background",
        "style": "professional, corporate headshot",
        "quality": "natural light, 8k",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy",
    },
    "sports": {
        "name": "户外运动",
        "shot": "wide",
        "subject": "1boy",
        "scene": "surfing on ocean waves, sunset sky, dynamic action pose, splashing water",
        "style": "action photography",
        "quality": "golden hour, 8k",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy, extra limbs",
    },
    "wuxia": {
        "name": "古风侠客",
        "shot": "medium",
        "subject": "1boy",
        "scene": "ancient Chinese warrior, traditional armor, holding sword, temple in background",
        "style": "ink painting style",
        "quality": "epic, 8k",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy, modern elements",
    },
    "cafe": {
        "name": "咖啡厅日常",
        "shot": "medium",
        "subject": "1boy",
        "scene": "sitting in cozy cafe, holding coffee cup, reading book, window with street view",
        "style": "35mm photography, depth of field",
        "quality": "warm lighting, high quality",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy",
    },
    "soldier": {
        "name": "未来战士",
        "shot": "wide",
        "subject": "1boy",
        "scene": "futuristic soldier, mechanical armor, glowing visor, destroyed city background",
        "style": "sci-fi",
        "quality": "hdr, 8k, unreal engine 5 render",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy, extra limbs",
    },
    "sabo": {
        "name": "动漫萨博",
        "shot": "medium",
        "subject": "1boy",
        "scene": "cosplay, black top hat with goggles, blonde hair, long blue coat over white shirt, blue flames, confident smile",
        "style": "One Piece anime style, vibrant colors, clean lineart",
        "quality": "masterpiece, 8k",
        "negative": "worst quality, low quality, blurry, deformed face, bad anatomy",
    },
}

# ---------------------------------------------------------------------------
# 单一数据源：景别 → 权重分档
# ---------------------------------------------------------------------------
WEIGHT_TIERS: dict = {
    "wide": (0.60, "远景/全景，低权重避免五官畸变"),
    "medium": (0.75, "常规场景，最佳平衡区间"),
    "closeup": (0.90, "特写/肖像，高权重保留五官细节"),
}

# ---------------------------------------------------------------------------
# 单一数据源：推理参数
# ---------------------------------------------------------------------------
INFER_PARAMS: dict = {
    "sampler": "DPM++ 2M Karras",
    "steps": 28,
    "cfg_scale": 7,
    "size": "1024x1024",
    "clip_skip": 2,
}

# --compare 使用的权重档
COMPARE_WEIGHTS: list = [0.5, 0.75, 0.95]

DEFAULT_TRIGGER = "cyberboy"


def get_weight_for_shot(shot: str, override: Optional[float] = None) -> float:
    """按景别返回默认权重；可用 override 强制覆盖。"""
    if override is not None:
        return float(override)
    if shot not in WEIGHT_TIERS:
        raise KeyError(f"unknown shot '{shot}'（可用：{list(WEIGHT_TIERS)}）")
    return WEIGHT_TIERS[shot][0]


def build_prompt(
    scene_key: str,
    trigger: str = DEFAULT_TRIGGER,
    lora_name: Optional[str] = None,
    weight: Optional[float] = None,
) -> dict:
    """五段式拼装：trigger, subject, scene, style, quality。

    - lora_name 存在时，前缀 `<lora:name:weight>`（供 WebUI 使用；
      diffusers 不解析此语法，权重走 API cross_attention_kwargs）。
    - 返回 dict：key/name/shot/weight/weight_reason/positive/negative。
    """
    if scene_key not in SCENES:
        raise KeyError(
            f"unknown scene '{scene_key}'（可用：{', '.join(sorted(SCENES))}）"
        )

    s = SCENES[scene_key]
    shot = s["shot"]
    w = get_weight_for_shot(shot, override=weight)
    w_reason = WEIGHT_TIERS[shot][1] if weight is None else "用户强制权重"

    body = ", ".join([s["subject"], s["scene"], s["style"], s["quality"]])
    positive = f"{trigger}, {body}"
    if lora_name:
        positive = f"<lora:{lora_name}:{w:g}>{positive}"

    return {
        "key": scene_key,
        "name": s["name"],
        "shot": shot,
        "weight": w,
        "weight_reason": w_reason,
        # 原样透传 SCENES 字段（单一数据源）：InstantID 路线要去掉触发词，
        # 需要自行拼 subject/scene/style/quality
        "subject": s["subject"],
        "scene": s["scene"],
        "style": s["style"],
        "quality": s["quality"],
        "positive": positive,
        "negative": s["negative"],
    }


def build_all(
    trigger: str = DEFAULT_TRIGGER,
    lora_name: Optional[str] = None,
    weight: Optional[float] = None,
    scene_keys: Optional[list] = None,
) -> list:
    """批量生成；scene_keys=None 时生成全部 8 个场景。"""
    keys = scene_keys if scene_keys is not None else list(SCENES)
    return [build_prompt(k, trigger, lora_name, weight) for k in keys]


# ---------------------------------------------------------------------------
# 输出格式
# ---------------------------------------------------------------------------

def _md_report(items: list, trigger: str, lora_name: Optional[str]) -> str:
    """md 输出：参数表 + 场景权重表 + 逐场景 Prompt 代码块。"""
    lines = [
        "# Prompt 生成清单",
        "",
        "## 参数",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 触发词 | `{trigger}` |",
        f"| LoRA 前缀 | {('`<lora:' + lora_name + ':{weight}>`') if lora_name else '无（diffusers 权重走 API）'} |",
        f"| 采样器 | {INFER_PARAMS['sampler']} |",
        f"| Steps | {INFER_PARAMS['steps']} |",
        f"| CFG | {INFER_PARAMS['cfg_scale']} |",
        f"| 尺寸 | {INFER_PARAMS['size']} |",
        f"| Clip Skip | {INFER_PARAMS['clip_skip']} |",
        "",
        "## 场景权重表",
        "",
        "| 景别 | 默认权重 | 说明 |",
        "|---|---|---|",
    ]
    for shot, (w, reason) in WEIGHT_TIERS.items():
        lines.append(f"| {shot} | {w:g} | {reason} |")
    lines += ["", "## 逐场景 Prompt", ""]
    for it in items:
        lines += [
            f"### {it['name']}（{it['key']}，{it['shot']}，权重 {it['weight']:g}）",
            "",
            f"理由：{it['weight_reason']}",
            "",
            "```",
            it["positive"],
            "```",
            "",
            f"负向词：`{it['negative']}`",
            "",
        ]
    return "\n".join(lines)


def _txt_report(items: list) -> str:
    lines = []
    for it in items:
        lines.append(f"[{it['key']}] {it['name']} (weight={it['weight']:g})")
        lines.append(f"  positive: {it['positive']}")
        lines.append(f"  negative: {it['negative']}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="场景库 + 五段式 Prompt 生成（单一数据源）")
    p.add_argument("--trigger", "-t", default=DEFAULT_TRIGGER, help="LoRA 触发词（默认 cyberboy）")
    p.add_argument("--lora-name", "-l", default=None, help="LoRA 名称，生成 <lora:name:weight> 前缀（供 WebUI）")
    p.add_argument("--scene", "-s", default="all", help="场景 key，逗号分隔或 all（默认 all）")
    p.add_argument("--weight", "-w", type=float, default=None, help="强制覆盖所有权重")
    p.add_argument("--format", "-f", choices=["md", "txt", "json"], default="md", help="输出格式（默认 md）")
    p.add_argument("--output", "-o", default=None, help="输出文件路径（默认 stdout）")
    p.add_argument("--list", action="store_true", help="列出全部可用场景后退出")
    args = p.parse_args(argv)

    if args.list:
        for key, s in SCENES.items():
            print(f"{key:10s} {s['name']}（{s['shot']}）")
        return 0

    if args.scene.strip().lower() == "all":
        keys = list(SCENES)
    else:
        keys = [k.strip() for k in args.scene.split(",") if k.strip()]
        unknown = [k for k in keys if k not in SCENES]
        if unknown:
            print(
                f"错误：未知场景 {unknown}。可用场景：{', '.join(sorted(SCENES))}",
                file=sys.stderr,
            )
            return 2

    items = build_all(args.trigger, args.lora_name, args.weight, keys)

    if args.format == "md":
        out = _md_report(items, args.trigger, args.lora_name)
    elif args.format == "txt":
        out = _txt_report(items)
    else:
        out = json.dumps(items, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"已写入 {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
