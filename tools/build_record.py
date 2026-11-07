#!/usr/bin/env python3
"""Build 05_record_strict.json: single-prompt, two-chain bitwise verification
using the exact (record) path.

Layout
------
  A chain: FULL teacher -> LoRA -> shift -> H3AdaLNRecord -> guider -> sampler
  B chain: PRUNED base  -> LoRA -> shift -> Apply(table, after=A.latent) -> sampler
  Save   : record.table -> .safetensors (after=A.latent)
  Compare: A.latent vs B.latent, must be equal=True / max_abs=0

Why the `after` edges are required
----------------------------------
ComfyUI runs a node as soon as its inputs exist. The table object returned by
Record is *empty* until chain A has actually sampled, so anything that reads it
must be ordered behind chain A. `after` is an ignored wildcard input whose only
job is to create that edge -- Apply also needs it so that chain B's model is not
patched before the rows exist.

Run: python tools/build_record.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_workflows import Graph, templates  # noqa: E402

FULL = "minimax_h3_ref2va_int8_convrot.safetensors"
PRUNED = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
LORA = "dmad_minimax_h3_8step_full_critic_comfyui.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
TABLE_TYPE = "H3_ADALN_TABLE"
PROMPT = ("A lone figure walking slowly through a neon-lit street at night, "
          "steady camera, shallow depth of field")
TEACHER_TAG = "ref2va full int8_convrot"
LORA_TAG = "dmad8_s1.0"


def build(tpl):
    g = Graph(tpl)
    g.group("共享条件（同一次 prompt，A/B 完全一致）", [20, 20, 390, 1420])
    g.group("A 完整模型 · 记录原生 AdaLN", [430, 20, 380, 1420], "#3f789e")
    g.group("B 剪枝模型 · 注入记录表", [830, 20, 380, 1420], "#389e6b")
    g.group("保存 / 比对", [1230, 20, 400, 1420], "#9e6b3f")

    clip = g.add("CLIPLoader", (40, 80), [CLIP, "minimax", "default"], "共享文本编码器")
    vae = g.add("VAELoader", (40, 260), [VIDEO_VAE], "共享视频 VAE")
    cond = g.add("MiniMaxH3ImageToVideo", (40, 440),
                 [PROMPT, 832, 480, 39], "共享 conditioning + latent")
    noise = g.add("RandomNoise", (40, 800), [1, "fixed"], "共享固定 seed")
    sampler_sel = g.add("KSamplerSelect", (40, 940), ["euler"], "共享 sampler")
    sched = g.add("BasicScheduler", (40, 1080), ["simple", 8, 1.0],
                  "共享 sigma 网格（8 步）")

    unet_a = g.add("UNETLoader", (450, 80), [FULL, "default"], "A: FULL 34GB 教师")
    lora_a = g.add("LoraLoaderModelOnly", (450, 250), [LORA, 1.0], "A: DMAD 8-step")
    shift_a = g.custom(
        "MiniMaxH3SigmaShift", (450, 410),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"), ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="A: shift 12 / 2", size=[300, 130])
    record = g.custom(
        "H3AdaLNRecord", (450, 570),
        inputs=[("model", "MODEL"), ("after", "*")],
        outputs=[("model", "MODEL"), ("table", TABLE_TYPE), ("report", "STRING")],
        widgets=["strict", TEACHER_TAG, LORA_TAG],
        title="记录（只读 hook，采样中填表）", size=[340, 220])
    guider_a = g.add("BasicGuider", (450, 860))
    samp_a = g.add("SamplerCustomAdvanced", (450, 1000), None, "A: latent (原生)")

    unet_b = g.add("UNETLoader", (850, 80), [PRUNED, "default"], "B: PRUNED 21GB")
    lora_b = g.add("LoraLoaderModelOnly", (850, 250), [LORA, 1.0], "B: DMAD 8-step")
    shift_b = g.custom(
        "MiniMaxH3SigmaShift", (850, 410),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"), ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")], widgets=[12.0, 2.0],
        title="B: shift 12 / 2", size=[300, 130])
    apply = g.custom(
        "H3ExactAdaLNApply", (850, 570),
        inputs=[("model", "MODEL"), ("table", TABLE_TYPE), ("after", "*")],
        outputs=[("model", "MODEL"), ("report", "STRING")],
        widgets=["strict", ""], title="B: 注入记录表", size=[340, 210])
    guider_b = g.add("BasicGuider", (850, 850))
    samp_b = g.add("SamplerCustomAdvanced", (850, 990), None, "B: latent (表)")

    save = g.custom(
        "H3ExactAdaLNSave", (1250, 80),
        inputs=[("table", TABLE_TYPE), ("after", "*")],
        outputs=[("file_path", "STRING")],
        widgets=["h3_adaln/recorded", "output"],
        title="保存表（可复用）", size=[340, 190])
    compare = g.custom(
        "H3LatentCompare", (1250, 340),
        inputs=[("latent_a", "LATENT"), ("latent_b", "LATENT")],
        outputs=[("report", "STRING")],
        widgets=["full model (native)", "pruned + recorded table"],
        title="逐位比对", size=[340, 190])
    g.note((1250, 580), (
        "严格验证口径（record 路径）\n"
        "• 表不是离线算出来的，而是 A 链采样时用只读 hook\n"
        "  抓到的原生 AdaLN 输出，逐位保存，所以不存在\n"
        "  「量化 backend 不同」这类差异来源。\n"
        "• `after` 不是数据输入，只用来定执行顺序：\n"
        "  Apply / Save 都要等 A 链采样结束才拿得到行，\n"
        "  所以它们必须接 A 链的 output。\n"
        "• 通过标准：Compare 里 video / audio 都\n"
        "  equal=True、max_abs=0、VERDICT: PASS。\n"
        "• 对照实验：把 Apply 的 mode 改成 off 再跑，\n"
        "  必须 FAIL（剪枝 base 的曲线表与完整模型不等）。\n"
        "• 表的 t 行来自真实 schedule；换步数 / shift /\n"
        "  分辨率都必须重新记录，strict 会直接报缺哪些 t。\n"
        "• AdaLN 只依赖 t，所以也可以拆成两步：先\n"
        "  record+save，再 load+apply，结果同样逐位一致。"), size=(430, 460))

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

    # A chain (record sits between shift and guider)
    g.link(unet_a, 0, lora_a, 0, "MODEL")
    g.link(lora_a, 0, shift_a, 0, "MODEL")
    g.link(shift_a, 0, record, 0, "MODEL")
    g.link(record, 0, guider_a, 0, "MODEL")
    g.link(guider_a, 0, samp_a, 1, "GUIDER")
    g.link(shift_a, 0, sched, 0, "MODEL")

    # B chain + recorded table, ordered behind A's sampler
    g.link(unet_b, 0, lora_b, 0, "MODEL")
    g.link(lora_b, 0, shift_b, 0, "MODEL")
    g.link(shift_b, 0, apply, 0, "MODEL")
    g.link(record, 1, apply, 1, TABLE_TYPE)
    g.link(samp_a, 0, apply, 2, "LATENT")
    g.link(apply, 0, guider_b, 0, "MODEL")
    g.link(guider_b, 0, samp_b, 1, "GUIDER")

    # save + compare
    g.link(record, 1, save, 0, TABLE_TYPE)
    g.link(samp_a, 0, save, 1, "LATENT")
    g.link(samp_a, 0, compare, 0, "LATENT")
    g.link(samp_b, 0, compare, 1, "LATENT")
    return g.save("05_record_strict.json")


def validate(path):
    """Structural checks: link endpoints, slot types, duplicate ids."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    nodes = {n["id"]: n for n in d["nodes"]}
    assert len(nodes) == len(d["nodes"]), "duplicate node ids in " + path
    problems = []
    for lid, src, src_slot, dst, dst_slot, typ in d["links"]:
        if src not in nodes or dst not in nodes:
            problems.append("link {} has a dangling endpoint".format(lid))
            continue
        outs = nodes[src].get("outputs") or []
        ins = nodes[dst].get("inputs") or []
        if src_slot >= len(outs):
            problems.append("link {} bad source slot {}".format(lid, src_slot))
            continue
        if dst_slot >= len(ins):
            problems.append("link {} bad target slot {}".format(lid, dst_slot))
            continue
        out_t, in_t = outs[src_slot]["type"], ins[dst_slot]["type"]
        if typ not in (out_t, in_t):
            problems.append("link {} type {} vs {}->{}".format(lid, typ, out_t, in_t))
        if lid not in (outs[src_slot].get("links") or []):
            problems.append("link {} not listed on source output".format(lid))
        if ins[dst_slot].get("link") != lid:
            problems.append("link {} not referenced by target input".format(dst_slot))
    for n in d["nodes"]:
        for i, inp in enumerate(n.get("inputs") or []):
            if inp.get("link") is not None and not any(
                    l[0] == inp["link"] and l[3] == n["id"] and l[4] == i
                    for l in d["links"]):
                problems.append("node {} input {} points at missing link".format(
                    n["id"], inp.get("name")))
        for i, out in enumerate(n.get("outputs") or []):
            for lid in (out.get("links") or []):
                if not any(l[0] == lid and l[1] == n["id"] and l[2] == i
                           for l in d["links"]):
                    problems.append("node {} output {} points at missing link".format(
                        n["id"], out.get("name")))
    if problems:
        raise SystemExit("{}: {} problem(s)\n  {}".format(
            os.path.basename(path), len(problems), "\n  ".join(problems)))
    return len(d["nodes"]), len(d["links"])


if __name__ == "__main__":
    path = build(templates())
    n_nodes, n_links = validate(path)
    print("built {}: {} nodes, {} links".format(os.path.basename(path), n_nodes, n_links))
