#!/usr/bin/env python3
"""Build 04_verify_strict.json: single-prompt, two-chain bitwise verification.

Why this is the only strict setup:
  * Both chains live in ONE prompt and share the same RandomNoise, SIGMAS,
    conditioning and latent_image objects, so no run-to-run nondeterminism
    can be blamed for a mismatch.
  * The table is baked in-memory from chain A's own MODEL and wired straight
    into Apply -- no Save/Load nodes, no file lookup, no upload widget.
  * Only three of our nodes are used (Bake / Apply / Compare), all present
    since the first release, so no "missing node" from newer additions.

Run: python tools/build_strict.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_workflows import Graph, templates  # noqa: E402

FULL = "minimax_h3_ref2va_int8_convrot.safetensors"
PRUNED = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
LORA = "dmad_minimax_h3_8step_full_critic_comfyui.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
TABLE_TYPE = "H3_ADALN_TABLE"


def build(tpl):
    g = Graph(tpl)
    g.group("共享条件（同一次 prompt，A/B 完全一致）", [20, 20, 390, 1300])
    g.group("A 完整模型 · 原生 AdaLN", [430, 20, 380, 1300], "#3f789e")
    g.group("B 剪枝模型 · 精确表", [830, 20, 380, 1300], "#389e6b")
    g.group("烘焙 / 比对", [1230, 20, 400, 1300], "#9e6b3f")

    clip = g.add("CLIPLoader", (40, 80), [CLIP, "minimax", "default"], "共享文本编码器")
    vae = g.add("VAELoader", (40, 260), [VIDEO_VAE], "共享视频 VAE")
    cond = g.add("MiniMaxH3ImageToVideo", (40, 440),
                 ["A lone figure walking slowly through a neon-lit street at night, "
                  "steady camera, shallow depth of field", 832, 480, 39],
                 "共享 conditioning + latent")
    noise = g.add("RandomNoise", (40, 800), [1, "fixed"], "共享固定 seed")
    sampler_sel = g.add("KSamplerSelect", (40, 940), ["euler"], "共享 sampler")
    sched = g.add("BasicScheduler", (40, 1080), ["simple", 8, 1.0],
                  "共享 sigma 网格（8 步）")

    unet_a = g.add("UNETLoader", (450, 80), [FULL, "default"], "A: FULL 34GB")
    lora_a = g.add("LoraLoaderModelOnly", (450, 250), [LORA, 1.0], "A: DMAD 8-step")
    shift_a = g.custom(
        "MiniMaxH3SigmaShift", (450, 410),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="A: shift 12 / 2", size=[300, 130])
    guider_a = g.add("BasicGuider", (450, 740))
    samp_a = g.add("SamplerCustomAdvanced", (450, 900), None, "A: latent")

    unet_b = g.add("UNETLoader", (850, 80), [PRUNED, "default"], "B: PRUNED 21GB")
    lora_b = g.add("LoraLoaderModelOnly", (850, 250), [LORA, 1.0], "B: DMAD 8-step")
    shift_b = g.custom(
        "MiniMaxH3SigmaShift", (850, 410),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="B: shift 12 / 2", size=[300, 130])

    bake = g.custom(
        "H3ExactAdaLNBake", (1250, 80),
        inputs=[("model", "MODEL"), ("sigmas", "SIGMAS")],
        outputs=[("table", TABLE_TYPE), ("report", "STRING")],
        widgets=[8, "simple", "bfloat16", "auto", False,
                 "", 12.0, 2.0, 0.999, 1.0, FULL, "dmad8_s1.0", ""],
        title="烘焙（用 A 链自己的 MODEL）", size=[380, 330])
    apply = g.custom(
        "H3ExactAdaLNApply", (850, 570),
        inputs=[("model", "MODEL"), ("table", TABLE_TYPE)],
        outputs=[("model", "MODEL"), ("report", "STRING")],
        widgets=["strict", ""], title="B: 注入精确表", size=[330, 180])
    guider_b = g.add("BasicGuider", (850, 790))
    samp_b = g.add("SamplerCustomAdvanced", (850, 950), None, "B: latent")

    compare = g.custom(
        "H3LatentCompare", (1250, 560),
        inputs=[("latent_a", "LATENT"), ("latent_b", "LATENT")],
        outputs=[("report", "STRING")],
        widgets=["full model (native)", "pruned + exact table"],
        title="逐位比对", size=[340, 160])
    g.note((1250, 780), (
        "离线比对口径（Bake 路径）\n"
        "• 注意：Bake 是离线重算，拿不到采样时的量化 kernel，\n"
        "  结果与原生仍差一个 bf16 ulp，8 步后放大到 ~0.145。\n"
        "  要逐位一致请用 05_record_strict.json（记录式）。\n"
        "• 两条链共享 RandomNoise / SIGMAS /\n"
        "  conditioning / latent_image，不存在跨次运行的非确定性。\n"
        "• 表由 Bake 在本次执行中从 A 链 MODEL 直接算出，\n"
        "  table 输出直连 Apply，不需要 Save / Load / 上传。\n"
        "• 只用 Bake / Apply / Compare 三个节点（首发即有），\n"
        "  不依赖 H3LatentSave 等后加节点。\n"
        "• 通过标准：Compare 报告里 video/audio 各分量\n"
        "  equal=True、max_abs=0、VERDICT: PASS。\n"
        "• 对照实验：把 Apply 的 mode 设为 off 再跑，应为 FAIL。\n"
        "• 若 sigmas 换成自定义表，直接接 Bake 的 sigmas 输入；\n"
        "  shift_v/shift_a 也要与运行时一致。"), size=(430, 400))

    # shared conditioning
    g.link(clip, 0, cond, 0, "CLIP")
    g.link(vae, 0, cond, 1, "VAE")
    g.link(cond, 0, guider_a, 1, "CONDITIONING")
    g.link(cond, 0, guider_b, 1, "CONDITIONING")
    g.link(cond, 1, samp_a, 4, "LATENT")
    g.link(cond, 1, samp_b, 4, "LATENT")
    g.link(noise, 0, samp_a, 0, "NOISE")
    g.link(noise, 0, samp_b, 0, "NOISE")
    g.link(sampler_sel, 0, samp_a, 2, "SAMPLER")
    g.link(sampler_sel, 0, samp_b, 2, "SAMPLER")
    g.link(sched, 0, samp_a, 3, "SIGMAS")
    g.link(sched, 0, samp_b, 3, "SIGMAS")
    g.link(sched, 0, bake, 1, "SIGMAS")

    # A chain
    g.link(unet_a, 0, lora_a, 0, "MODEL")
    g.link(lora_a, 0, shift_a, 0, "MODEL")
    g.link(shift_a, 0, sched, 0, "MODEL")
    g.link(shift_a, 0, guider_a, 0, "MODEL")
    g.link(shift_a, 0, bake, 0, "MODEL")
    g.link(guider_a, 0, samp_a, 1, "GUIDER")

    # B chain + in-memory table
    g.link(unet_b, 0, lora_b, 0, "MODEL")
    g.link(lora_b, 0, shift_b, 0, "MODEL")
    g.link(shift_b, 0, apply, 0, "MODEL")
    g.link(bake, 0, apply, 1, TABLE_TYPE)
    g.link(apply, 0, guider_b, 0, "MODEL")
    g.link(guider_b, 0, samp_b, 1, "GUIDER")

    # compare
    g.link(samp_a, 0, compare, 0, "LATENT")
    g.link(samp_b, 0, compare, 1, "LATENT")
    return g.save("04_verify_strict.json")


if __name__ == "__main__":
    build(templates())
