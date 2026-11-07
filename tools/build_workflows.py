#!/usr/bin/env python3
"""Generate the two verification workflows from ComfyUI's official H3 blueprint.

Using the shipped blueprint as a node-template source keeps input/output slot
definitions identical to the real ComfyUI nodes instead of hand-guessing them.
Run: python tools/build_workflows.py
"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "workflows")
BLUEPRINT_NAME = "Image to Video (MiniMax H3).json"


def blueprint_path():
    """Locate ComfyUI's shipped H3 blueprint.

    Override with H3_BLUEPRINT=<file>, or point COMFYUI_DIR at the ComfyUI
    root. Only used to copy node slot definitions, never at runtime.
    """
    candidates = []
    if os.environ.get("H3_BLUEPRINT"):
        candidates.append(os.environ["H3_BLUEPRINT"])
    for root in (os.environ.get("COMFYUI_DIR"),
                 "D:/ComfyUI_windows_portable/ComfyUI",
                 os.path.expanduser("~/ComfyUI")):
        if root:
            candidates.append(os.path.join(root, "blueprints", BLUEPRINT_NAME))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(
        "blueprint not found: {}\nTried://n  {}\nSet H3_BLUEPRINT to the file "
        "or COMFYUI_DIR to the ComfyUI root.".format(
            BLUEPRINT_NAME, "\n  ".join(candidates)))

FULL = "minimax_h3_ref2va_int8_convrot.safetensors"
PRUNED = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
LORA = "dmad_minimax_h3_8step_full_critic_comfyui.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
TABLE = "h3_exact_dmad8_v1"
TABLE_TYPE = "H3_ADALN_TABLE"
SAVE_PREFIX = "h3_adaln/" + TABLE
SAVED_NAME = TABLE + "_00001.safetensors"
# From the LoRA's own safetensors metadata:
#   "8 steps, video shift 12, audio shift 2, re-noise step rule,
#    1344x768, 124 frames at 24 fps, no CFG"
STEPS = 8
SHIFT_V, SHIFT_A = 12.0, 2.0
LORA_STRENGTH = 1.0       # alpha = rank = 128, scale 1.0
LORA_TAG = "dmad8_s1.0"
SCHEDULER = "simple"
SAMPLER = "euler"
SEED = 1
PROMPT = ("A lone figure walking slowly through a neon-lit street at night, "
          "steady camera, shallow depth of field")
# Short clip keeps the A/B run cheap. For production use 1344x768 / 124 frames;
# change BOTH chains together and re-bake the table afterwards.
WIDTH, HEIGHT, LENGTH = 832, 480, 39   # 39 = 17*2+5


def templates():
    with open(blueprint_path(), "r", encoding="utf-8") as f:
        d = json.load(f)
    sg = d["definitions"]["subgraphs"][0]
    return {n["type"]: copy.deepcopy(n) for n in sg["nodes"]}


class Graph:
    def __init__(self, tpl):
        self.tpl = tpl
        self.nodes, self.links, self.groups = [], [], []
        self.nid = self.lid = 0

    def _fresh(self, node):
        for i in node.get("inputs") or []:
            i["link"] = None
        for o in node.get("outputs") or []:
            o["links"] = []
        node["flags"] = node.get("flags") or {}
        node["properties"] = node.get("properties") or {}
        node["mode"] = 0
        node["order"] = len(self.nodes)
        return node

    def add(self, type_, pos, widgets=None, title=None):
        self.nid += 1
        n = copy.deepcopy(self.tpl.get(type_))
        if n is None:
            raise RuntimeError("template missing: " + type_)
        n = self._fresh(n)
        n["id"] = self.nid
        n["pos"] = list(pos)
        n["size"] = n.get("size") or [280, 120]
        if widgets is not None:
            n["widgets_values"] = widgets
        if title:
            n["title"] = title
        self.nodes.append(n)
        return self.nid

    def custom(self, type_, pos, inputs, outputs, widgets, title=None, size=None):
        self.nid += 1
        n = {
            "id": self.nid, "type": type_, "pos": list(pos),
            "size": size or [340, 180], "flags": {}, "order": len(self.nodes),
            "mode": 0, "properties": {}, "widgets_values": list(widgets),
            "inputs": [{"name": name, "type": typ, "link": None}
                       for name, typ in inputs],
            "outputs": [{"name": name, "type": typ, "links": []}
                        for name, typ in outputs],
        }
        if title:
            n["title"] = title
        self.nodes.append(n)
        return self.nid

    def note(self, pos, text, size=(430, 300)):
        self.nid += 1
        self.nodes.append({
            "id": self.nid, "type": "Note", "pos": list(pos),
            "size": list(size), "flags": {}, "order": len(self.nodes),
            "mode": 0, "properties": {}, "widgets_values": [text],
            "inputs": [], "outputs": [],
            "color": "#432", "bgcolor": "#653",
        })
        return self.nid

    def group(self, title, bounding, color="#3f789e"):
        self.groups.append({"title": title, "bounding": list(bounding),
                            "color": color, "font_size": 24, "flags": {}})

    def by_id(self, nid):
        return next(n for n in self.nodes if n["id"] == nid)

    def link(self, src, src_slot, dst, dst_slot, typ):
        self.lid += 1
        lid = self.lid
        self.links.append([lid, src, src_slot, dst, dst_slot, typ])
        sn, dn = self.by_id(src), self.by_id(dst)
        if src_slot >= len(sn["outputs"]) or dst_slot >= len(dn["inputs"]):
            raise RuntimeError("bad slot for link {}".format(lid))
        dn["inputs"][dst_slot]["link"] = lid
        sn["outputs"][src_slot]["links"].append(lid)
        return lid

    def validate(self):
        ids = {n["id"] for n in self.nodes}
        for lid, s, ss, d, ds, typ in self.links:
            assert s in ids and d in ids, "dangling node in link {}".format(lid)
            sn, dn = self.by_id(s), self.by_id(d)
            assert ss < len(sn["outputs"]), "bad out slot {}".format(lid)
            assert ds < len(dn["inputs"]), "bad in slot {}".format(lid)
            out_t, in_t = sn["outputs"][ss]["type"], dn["inputs"][ds]["type"]
            # ComfyUI treats "*" as Any: it accepts any producer type. The
            # `after` ordering edges rely on exactly that.
            ok = (typ == out_t == in_t) or "*" in (out_t, in_t, typ)
            assert ok, "type mismatch on link {}: {} ({}) -> {} ({})".format(
                lid, sn["type"], out_t, dn["type"], in_t)

    def save(self, name):
        self.validate()
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, name)
        payload = {
            "last_node_id": self.nid, "last_link_id": self.lid,
            "nodes": self.nodes, "links": self.links, "groups": self.groups,
            "config": {}, "extra": {}, "version": 0.4,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("wrote {} ({} nodes, {} links)".format(
            path, len(self.nodes), len(self.links)))
        return path


def build_bake(tpl):
    g = Graph(tpl)
    g.group("1) 烘焙 + 保存表文件（像 Save Image 一样）", [20, 20, 1360, 900])
    unet = g.add("UNETLoader", (60, 90), [FULL, "default"],
                 "FULL teacher (34GB, 不剪枝)")
    lora = g.add("LoraLoaderModelOnly", (60, 280), [LORA, LORA_STRENGTH],
                 "DMAD 8-step (与验证完全相同)")
    sched = g.add("BasicScheduler", (60, 470), [SCHEDULER, STEPS, 1.0],
                  "与验证相同的 scheduler + 8 步")
    shift = g.custom(
        "MiniMaxH3SigmaShift", (60, 670),
        inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                ("shift_audio", "FLOAT")],
        outputs=[("MODEL", "MODEL")],
        widgets=[SHIFT_V, SHIFT_A],
        title="DMAD 推荐 shift 12 / 2", size=[300, 130])
    bake = g.custom(
        "H3ExactAdaLNBake", (480, 200),
        inputs=[("model", "MODEL"), ("sigmas", "SIGMAS")],
        outputs=[("table", TABLE_TYPE), ("report", "STRING")],
        # auto_shift=False: use the explicit 12 / 2 below. Relying on
        # model_sampling after an object patch could silently fall back to 3.
        widgets=[STEPS, SCHEDULER, "bfloat16", "auto", False,
                 "", SHIFT_V, SHIFT_A, 0.999, 1.0, FULL, LORA_TAG, ""],
        title="烘焙（输出表对象，不写盘）", size=[420, 330])
    save = g.custom(
        "H3ExactAdaLNSave", (960, 240),
        inputs=[("table", TABLE_TYPE)],
        outputs=[("file_path", "STRING")],
        widgets=[SAVE_PREFIX, "output"],
        title="保存表文件（可分享）", size=[340, 150])
    g.note((480, 560), (
        "烘焙 + 保存\n"
        "1. 这里必须加载完整（未剪枝）模型，量化格式与验证用的剪枝模型一致。\n"
        "2. LoRA = dmad_minimax_h3_8step_full_critic_comfyui.safetensors，强度 1.0。\n"
        "   该 LoRA 元数据：8 步 / video shift 12 / audio shift 2 / no CFG。\n"
        "3. shift 节点必须与验证工作流一致（audio shift 是 2，不是默认 3）。\n"
        "4. Bake 只产出内存中的表对象；Save 才写盘，和 Save Image 一样自动编号，\n"
        "   默认落在 ComfyUI/output/h3_adaln/<prefix>_00001.safetensors，\n"
        "   并镜像一份到 input/（Load 和文件浏览器都能直接看到）。\n"
        "5. Save 报告里会打印 /view?... 下载直链，粘到浏览器即可下载；\n"
        "   同时会登记到 ComfyUI 输出面板，不必进文件夹翻。\n"
        "6. 把那个 .safetensors 拷给别人即可分享：文件头里已含 teacher、LoRA 指纹、\n"
        "   sigmas、shift、dtype、ComfyUI 版本，不需要任何附属文件。\n"
        "7. Save 的 location 也可选 models/h3_exact_adaln，方便集中管理。\n"
        "7. 改 LoRA / 强度 / 步数 / sigmas / shift 都必须重新烘焙并另存。\n"
        "8. strict 报「table has no rows for t=...」时，把列出的 t 填进 extra_t 重烘焙。"))
    g.note((960, 460), (
        "分享 / 下载流程\n"
        "Save -> output/h3_adaln/*.safetensors（+ input/ 镜像）\n"
        "-> 用 Save 报告里的 /view?... 直链下载，或复制发送\n"
        "-> 对方把文件放进 ComfyUI/input/，在 Load 的 table_name\n"
        "   里填文件名（支持部分匹配）或绝对路径即可\n"
        "-> Load 的 table 接 Apply 的 table 输入。\n"
        "Load 用自由文本输入，不依赖启动时缓存的下拉列表，\n"
        "也不受图片上传控件只收图片格式的限制。"))
    g.link(unet, 0, lora, 0, "MODEL")
    g.link(lora, 0, shift, 0, "MODEL")
    g.link(shift, 0, sched, 0, "MODEL")
    g.link(shift, 0, bake, 0, "MODEL")
    g.link(sched, 0, bake, 1, "SIGMAS")
    g.link(bake, 0, save, 0, TABLE_TYPE)
    return g.save("01_bake_exact_adaln.json")


def build_verify(tpl):
    g = Graph(tpl)
    g.group("共享条件（两边必须完全一致）", [20, 20, 400, 1360])
    g.group("A 完整模型 · 原生 AdaLN", [440, 20, 380, 1360], "#3f789e")
    g.group("B 剪枝模型 · 加载表 + 注入", [840, 20, 380, 1360], "#389e6b")
    g.group("表文件 / 比对", [1240, 20, 400, 1360], "#9e6b3f")

    clip = g.add("CLIPLoader", (50, 90), [CLIP, "minimax", "default"],
                 "共享文本编码器 (int8, A10 可用)")
    vae = g.add("VAELoader", (50, 270), [VIDEO_VAE], "共享视频 VAE")
    cond = g.add("MiniMaxH3ImageToVideo", (50, 450),
                 [PROMPT, WIDTH, HEIGHT, LENGTH], "共享 conditioning + latent")
    noise = g.add("RandomNoise", (50, 800), [SEED, "fixed"], "共享固定 seed")
    sampler_sel = g.add("KSamplerSelect", (50, 940), [SAMPLER], "共享 sampler")

    def shift_node(pos, title):
        return g.custom(
            "MiniMaxH3SigmaShift", pos,
            inputs=[("model", "MODEL"), ("shift_video", "FLOAT"),
                    ("shift_audio", "FLOAT")],
            outputs=[("MODEL", "MODEL")],
            widgets=[SHIFT_V, SHIFT_A],
            title=title, size=[300, 130])

    # A: full model, native AdaLN path (no table).
    unet_a = g.add("UNETLoader", (470, 80), [FULL, "default"], "A: FULL 34GB")
    lora_a = g.add("LoraLoaderModelOnly", (470, 250), [LORA, LORA_STRENGTH],
                   "A: DMAD 8-step")
    shift_a = shift_node((470, 410), "A: shift 12 / 2")
    sched = g.add("BasicScheduler", (470, 570), [SCHEDULER, STEPS, 1.0],
                  "共享 sigma 网格（8 步）")
    guider_a = g.add("BasicGuider", (470, 740))
    samp_a = g.add("SamplerCustomAdvanced", (470, 900), None, "A: latent")

    # B: pruned model, exact table injected.
    unet_b = g.add("UNETLoader", (870, 80), [PRUNED, "default"], "B: PRUNED 21GB")
    lora_b = g.add("LoraLoaderModelOnly", (870, 250), [LORA, LORA_STRENGTH],
                   "B: DMAD 8-step")
    shift_b = shift_node((870, 410), "B: shift 12 / 2")
    load = g.custom(
        "H3ExactAdaLNLoad", (870, 570),
        inputs=[],
        outputs=[("table", TABLE_TYPE), ("info", "STRING")],
        widgets=[SAVED_NAME, ""],
        title="按名称/路径加载表文件", size=[330, 150])
    apply = g.custom(
        "H3ExactAdaLNApply", (870, 760),
        inputs=[("model", "MODEL"), ("table", TABLE_TYPE)],
        outputs=[("model", "MODEL"), ("report", "STRING")],
        widgets=["strict", ""],
        title="B: 注入精确表", size=[330, 180])
    guider_b = g.add("BasicGuider", (870, 980))
    samp_b = g.add("SamplerCustomAdvanced", (870, 1140), None, "B: latent")

    compare = g.custom(
        "H3LatentCompare", (1270, 900),
        inputs=[("latent_a", "LATENT"), ("latent_b", "LATENT")],
        outputs=[("report", "STRING")],
        widgets=["full model (native)", "pruned + exact table"],
        title="逐位比对", size=[340, 160])
    g.note((1270, 160), (
        "验证口径（DMAD 8-step）\n"
        "• 两条链只有底模不同：A 用完整 34GB，B 用剪枝 21GB；量化同为 int8_convrot。\n"
        "• LoRA = dmad_minimax_h3_8step_full_critic_comfyui.safetensors，强度 1.0。\n"
        "  元数据：8 步 / video shift 12 / audio shift 2 / no CFG / rank 128。\n"
        "• 注意：该 LoRA 是 backbone-only（208 个 attn/mlp 目标，无 AdaLN / time_embedder 键）。\n"
        "  因此本次验证检验的是「精确表能否让剪枝底模 ≡ 完整底模」，\n"
        "  而不是「LoRA 的 AdaLN delta 被烘焙」。要测后者需换含 AdaLN 键的 LoRA\n"
        "  （例如 HyperFlow 的 full 版），其它接线不变。\n"
        "• 表文件用 Load 节点加载：table_name 填文件名/部分名字/相对路径，\n"
        "  或 table_path 填绝对路径。自由文本输入，不受下拉缓存与图片格式限制。\n"
        "  Load 的 table 输出接 Apply 的 table 输入。\n"
        "• A 不接 Apply（原生 AdaLN 路径），B 接 Apply（跳过插值表与剪枝 adaln_proj）。\n"
        "• 共享 BasicScheduler / RandomNoise / Conditioning / shift，保证 t 集合一致。\n"
        "  两个 shift 节点参数必须完全相同（audio shift 2，不是默认 3）。\n"
        "• 通过标准：compare 报告 equal=True 且 max_abs=0。\n"
        "• 对照实验：把 Apply 的 mode 设为 off，应当得到 FAIL。\n"
        "• strict 报缺 t 时会列出数值，填进 Bake 的 extra_t 重烘焙。\n"
        "• 验证档 8 步 / euler / simple / 832x480 / 39 帧。DMAD 推荐 1344x768 / 124 帧\n"
        "  及 re-noise step rule；改尺寸、步数或换成自定义 sigmas 时两边同步改并重新烘焙。\n"
        "• 条件部分用官方 I2V conditioning；换成你自己的 ref2va 条件链也可以，\n"
        "  只要两条链共享同一组条件节点。"), size=(430, 420))

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

    # A chain (shift must come after the LoRA, before scheduler and guider)
    g.link(unet_a, 0, lora_a, 0, "MODEL")
    g.link(lora_a, 0, shift_a, 0, "MODEL")
    g.link(shift_a, 0, sched, 0, "MODEL")
    g.link(shift_a, 0, guider_a, 0, "MODEL")
    g.link(guider_a, 0, samp_a, 1, "GUIDER")
    g.link(sched, 0, samp_a, 3, "SIGMAS")
    g.link(sched, 0, samp_b, 3, "SIGMAS")

    # B chain
    g.link(unet_b, 0, lora_b, 0, "MODEL")
    g.link(lora_b, 0, shift_b, 0, "MODEL")
    g.link(shift_b, 0, apply, 0, "MODEL")
    g.link(load, 0, apply, 1, TABLE_TYPE)
    g.link(apply, 0, guider_b, 0, "MODEL")
    g.link(guider_b, 0, samp_b, 1, "GUIDER")

    # compare
    g.link(samp_a, 0, compare, 0, "LATENT")
    g.link(samp_b, 0, compare, 1, "LATENT")
    return g.save("02_verify_full_vs_pruned.json")


def main():
    tpl = templates()
    build_bake(tpl)
    build_verify(tpl)


if __name__ == "__main__":
    main()
