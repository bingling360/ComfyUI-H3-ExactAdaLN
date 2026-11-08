#!/usr/bin/env python3
"""Generate the bX-49 hybrid workflows.

`06_b25-49_hybrid.json` records the Ref2VA teacher once and merges it with an
existing FL2VA table in the same run. `07_merge_tables.json` is the CPU-only
merge step, for when both tables are already on disk.

Run: python tools/build_hybrid.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    CLIP, FULL, LORA, PROMPT, TABLE_TYPE, VIDEO_VAE, Graph, templates)

# The base half of a bX-49 hybrid comes from the FL2VA full teacher.
FL2VA = "minimax_h3_fl2va_int8_convrot.safetensors"
# The donor must be recorded with the base teacher's embedding, otherwise the
# tail rows carry the donor's own t_emb and drift from the assembled hybrid by
# ~0.1 (max) -- measured; see HANDOFF 11.2.
TE_BASE = FL2VA
# Table recorded on the FL2VA full teacher by 08_record_fl2va.json.
BASE_TABLE = "fl2va_dmad8_00001.safetensors"
BASE_TAG = "fl2va full int8_convrot"
BASE_PREFIX = "h3_adaln/fl2va_dmad8"
DONOR_PREFIX = "h3_adaln/ref2va_dmad8"
HYBRID_PREFIX = "h3_adaln/b25-49_hybrid"
DONOR_TAG = "ref2va full int8_convrot"
LORA_TAG = "dmad8_s1.0"
SPLIT = 25


def _shared(g):
    clip = g.add("CLIPLoader", (40, 80), [CLIP, "minimax", "default"],
                 "共享文本编码器")
    vae = g.add("VAELoader", (40, 240), [VIDEO_VAE], "共享视频 VAE")
    cond = g.add("MiniMaxH3ImageToVideo", (40, 400),
                 [PROMPT, 832, 480, 39], "共享 conditioning + latent")
    noise = g.add("RandomNoise", (40, 700), [1, "fixed"], "共享固定 seed")
    sampler_sel = g.add("KSamplerSelect", (40, 840), ["euler"], "共享 sampler")
    sched = g.add("BasicScheduler", (40, 980), ["simple", 8, 1.0],
                  "共享 sigma 网格（8 步，必须与 A 表一致）")
    return clip, vae, cond, noise, sampler_sel, sched


def build_hybrid(tpl):
    g = Graph(tpl)
    g.group("共享条件 · 必须与录制 A 表时完全一致", [20, 20, 390, 1200])
    g.group("Ref2VA 完整教师 · 现录尾部 25 块的 AdaLN", [430, 20, 400, 1200],
            "#3f789e")
    g.group("合并 + 保存", [850, 20, 780, 1200], "#9e6b3f")

    clip, vae, cond, noise, sampler_sel, sched = _shared(g)

    unet = g.add("UNETLoader", (450, 80), [FULL, "default"],
                 "B 教师：Ref2VA FULL 34GB")
    lora = g.add("LoraLoaderModelOnly", (450, 240), [LORA, 1.0],
                 "与 A 表相同的 LoRA / 强度")
    shift = g.custom(
        "MiniMaxH3SigmaShift", (450, 400),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="shift 12 / 2", size=[300, 130])
    record = g.custom(
        "H3AdaLNRecord", (450, 560),
        inputs=[("model", "MODEL"), ("after", "*")],
        outputs=[("model", "MODEL"), ("table", TABLE_TYPE), ("report", "STRING")],
        widgets=["strict", DONOR_TAG, LORA_TAG, TE_BASE],
        title="记录 B 表（Ref2VA 旋钮 + FL 时间嵌入）", size=[340, 220])
    guider = g.add("BasicGuider", (450, 850))
    samp = g.add("SamplerCustomAdvanced", (450, 990), None,
                 "B: Ref2VA latent（只为填表）")

    load_a = g.custom(
        "H3ExactAdaLNLoad", (870, 80),
        inputs=[],
        outputs=[("table", TABLE_TYPE), ("info", "STRING")],
        widgets=[BASE_TABLE, ""],
        title="A 表：FL2VA full（已有）", size=[340, 170])
    merge = g.custom(
        "H3AdaLNMerge", (870, 300),
        inputs=[("table_a", TABLE_TYPE), ("table_b", TABLE_TYPE),
                ("after", "*")],
        outputs=[("table", TABLE_TYPE), ("report", "STRING")],
        widgets=[SPLIT, "a"],
        title="合并：0-24 取 A，25-49 取 B", size=[340, 190])
    save = g.custom(
        "H3ExactAdaLNSave", (870, 560),
        inputs=[("table", TABLE_TYPE), ("after", "*")],
        outputs=[("file_path", "STRING")],
        widgets=[HYBRID_PREFIX, "output"],
        title="保存 b25-49 混合表（可分享）", size=[340, 190])
    save_b = g.custom(
        "H3ExactAdaLNSave", (1250, 300),
        inputs=[("table", TABLE_TYPE), ("after", "*")],
        outputs=[("file_path", "STRING")],
        widgets=[DONOR_PREFIX, "output"],
        title="顺便保存 B 表（Ref2VA，供 07 复用）", size=[340, 190])

    g.note((870, 800), (
        "b25-49 混合表 = A(0-24) + B(25-49)\n"
        "• b25-49 整份文件都是 8 维 curve 形态（51 个 [96768,8]，\n"
        "  没有 time_embedder），不能直接 Record；它后 25 块的\n"
        "  adaln_proj 来自 Ref2VA。\n"
        "• 这里用 Ref2VA full 现录一张表（B 链），再和已有的\n"
        "  FL2VA 表（A，Load 节点）按块拼接。\n"
        "• Record 的 time_embedder_from 必须填 FL2VA 的 checkpoint：\n"
        "  尾部块在混合模型里吃的是 FL 的 t_emb，不是 Ref 自己的。\n"
        "  留空会让 25-49 偏移 0.09~0.19（实测）。\n"
        "• 两张表必须同 t 集合：同 8 步 / simple / shift 12,2 /\n"
        "  同分辨率 / 同 LoRA 强度。Merge 会校验，缺 t 直接报错。\n"
        "• final_layer 取 A —— b25-49 的 final 来自 FL2VA 骨架。\n"
        "• 产物 b25-49_hybrid_*.safetensors 用 Apply 注入 b25-49：\n"
        "  0-24 块与 FL2VA full 逐位一致，25-49 块与 Ref2VA full 逐位\n"
        "  一致，都是教师原生输出，不经过 8 维截断。\n"
        "• 只有 Ref2VA full 才有完整信息；只有 pruned 版无法还原。\n"
        "• A 表文件名按需改；Merge 的 after 接采样器，保证录完再合并。"),
        size=(430, 480))

    g.link(clip, 0, cond, 0, "CLIP")
    g.link(vae, 0, cond, 1, "VAE")
    g.link(cond, 0, guider, 1, "CONDITIONING")
    g.link(cond, 1, samp, 4, "LATENT")
    g.link(noise, 0, samp, 0, "NOISE")
    g.link(sampler_sel, 0, samp, 2, "SAMPLER")
    g.link(sched, 0, samp, 3, "SIGMAS")

    g.link(unet, 0, lora, 0, "MODEL")
    g.link(lora, 0, shift, 0, "MODEL")
    g.link(shift, 0, record, 0, "MODEL")
    g.link(shift, 0, sched, 0, "MODEL")
    g.link(record, 0, guider, 0, "MODEL")
    g.link(guider, 0, samp, 1, "GUIDER")

    g.link(load_a, 0, merge, 0, TABLE_TYPE)
    g.link(record, 1, merge, 1, TABLE_TYPE)
    g.link(samp, 0, merge, 2, "LATENT")
    g.link(merge, 0, save, 0, TABLE_TYPE)
    g.link(samp, 0, save, 1, "LATENT")
    g.link(record, 1, save_b, 0, TABLE_TYPE)
    g.link(samp, 0, save_b, 1, "LATENT")
    return g.save("06_b25-49_hybrid.json")


def build_record_fl2va(tpl):
    """Record the FL2VA full teacher (the 0..split-1 half of a bX-49 hybrid)."""
    g = Graph(tpl)
    g.group("共享条件 · 后面的 06 会按同一调度录 Ref2VA", [20, 20, 390, 1200])
    g.group("FL2VA 完整教师 · 记录基准表", [430, 20, 400, 1200], "#3f789e")

    clip, vae, cond, noise, sampler_sel, sched = _shared(g)

    unet = g.add("UNETLoader", (450, 80), [FL2VA, "default"],
                 "A 教师：FL2VA FULL 34GB")
    lora = g.add("LoraLoaderModelOnly", (450, 240), [LORA, 1.0],
                 "与 06 相同的 LoRA / 强度")
    shift = g.custom(
        "MiniMaxH3SigmaShift", (450, 400),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="shift 12 / 2", size=[300, 130])
    record = g.custom(
        "H3AdaLNRecord", (450, 560),
        inputs=[("model", "MODEL"), ("after", "*")],
        outputs=[("model", "MODEL"), ("table", TABLE_TYPE), ("report", "STRING")],
        widgets=["strict", BASE_TAG, LORA_TAG],
        title="记录 A 表（FL2VA 原生 AdaLN）", size=[340, 220])
    guider = g.add("BasicGuider", (450, 850))
    samp = g.add("SamplerCustomAdvanced", (450, 990), None,
                 "FL2VA latent（只为填表）")
    save = g.custom(
        "H3ExactAdaLNSave", (870, 300),
        inputs=[("table", TABLE_TYPE), ("after", "*")],
        outputs=[("file_path", "STRING")],
        widgets=[BASE_PREFIX, "output"],
        title="保存 A 表（FL2VA，06 会用）", size=[340, 190])

    g.note((870, 560), (
        "跑完 08 再跑 06。\n"
        "• b25-49 = FL2VA 的 blocks 0-24 + Ref2VA 的 blocks 25-49。\n"
        "• 两张表必须同 8 步 / simple / shift 12,2 / 同分辨率 /\n"
        "  同 LoRA 强度，否则 06 的 Merge 会拒绝。\n"
        "• 这里录到的 fl2va_dmad8_*.safetensors 就是 06 里\n"
        "  `A 表：FL2VA full（已有）` 要 Load 的文件。\n"
        "• 换 LoRA 或强度后，08 和 06 都要重跑。"),
        size=(430, 260))

    g.link(clip, 0, cond, 0, "CLIP")
    g.link(vae, 0, cond, 1, "VAE")
    g.link(cond, 0, guider, 1, "CONDITIONING")
    g.link(cond, 1, samp, 4, "LATENT")
    g.link(noise, 0, samp, 0, "NOISE")
    g.link(sampler_sel, 0, samp, 2, "SAMPLER")
    g.link(sched, 0, samp, 3, "SIGMAS")
    g.link(unet, 0, lora, 0, "MODEL")
    g.link(lora, 0, shift, 0, "MODEL")
    g.link(shift, 0, record, 0, "MODEL")
    g.link(shift, 0, sched, 0, "MODEL")
    g.link(record, 0, guider, 0, "MODEL")
    g.link(guider, 0, samp, 1, "GUIDER")
    g.link(record, 1, save, 0, TABLE_TYPE)
    g.link(samp, 0, save, 1, "LATENT")
    return g.save("08_record_fl2va.json")


def build_merge_only(tpl):
    g = Graph(tpl)
    g.group("纯 CPU 合并 · 两张表都已落盘时用", [20, 20, 900, 700], "#9e6b3f")

    load_a = g.custom(
        "H3ExactAdaLNLoad", (60, 80),
        inputs=[],
        outputs=[("table", TABLE_TYPE), ("info", "STRING")],
        widgets=[BASE_TABLE, ""],
        title="A 表：FL2VA full", size=[340, 170])
    load_b = g.custom(
        "H3ExactAdaLNLoad", (60, 300),
        inputs=[],
        outputs=[("table", TABLE_TYPE), ("info", "STRING")],
        widgets=["ref2va_dmad8", ""],
        title="B 表：Ref2VA full", size=[340, 170])
    merge = g.custom(
        "H3AdaLNMerge", (460, 170),
        inputs=[("table_a", TABLE_TYPE), ("table_b", TABLE_TYPE)],
        outputs=[("table", TABLE_TYPE), ("report", "STRING")],
        widgets=[SPLIT, "a"],
        title="合并：0-24 取 A，25-49 取 B", size=[340, 190])
    save = g.custom(
        "H3ExactAdaLNSave", (460, 420),
        inputs=[("table", TABLE_TYPE)],
        outputs=[("file_path", "STRING")],
        widgets=[HYBRID_PREFIX, "output"],
        title="保存 b25-49 混合表", size=[340, 190])

    g.note((60, 520), (
        "Load 用自由文本定位表：完整文件名 / 唯一子串 / 绝对路径。\n"
        "A 表默认 recorded_00001.safetensors（05 在 FL2VA full 上录的）。\n"
        "B 表默认子串 ref2va_dmad8（06 录出来的）。\n"
        "如果子串命中多个文件，Load 会报错并列出候选，填完整文件名即可。\n"
        "Merge 校验：t 集合、投影集合、out_dtype、LoRA 指纹。"),
        size=(430, 220))

    g.link(load_a, 0, merge, 0, TABLE_TYPE)
    g.link(load_b, 0, merge, 1, TABLE_TYPE)
    g.link(merge, 0, save, 0, TABLE_TYPE)
    return g.save("07_merge_tables.json")


def main():
    tpl = templates()
    build_record_fl2va(tpl)
    build_hybrid(tpl)
    build_merge_only(tpl)


if __name__ == "__main__":
    main()
