#!/usr/bin/env python3
"""Build 03_verify_save_latent.json from the shipped 02 verification workflow.

Also writes both latents to disk as an independent cross-check. The writer is
the plugin's own H3LatentSave: ComfyUI's core SaveLatent calls .contiguous()
on the NestedTensor that 0.44 uses for H3 latents and crashes.
Every existing node, id, link and widget is left untouched.
"""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "workflows", "02_verify_full_vs_pruned.json")
DST = os.path.join(REPO, "workflows", "03_verify_save_latent.json")

with open(SRC, "r", encoding="utf-8") as f:
    d = json.load(f)

nodes = d["nodes"]
samps = sorted(
    [n for n in nodes if n["type"] == "SamplerCustomAdvanced"],
    key=lambda n: n["pos"][0])
if len(samps) != 2:
    raise RuntimeError("expected exactly 2 SamplerCustomAdvanced nodes, got {}".format(len(samps)))
samp_a, samp_b = samps

nid = max(n["id"] for n in nodes) + 1
lid = max(l[0] for l in d["links"]) + 1
max_order = max(n.get("order", 0) for n in nodes)


def make_save(node_id, pos, title, prefix, order):
    return {
        "id": node_id,
        "type": "H3LatentSave",
        "pos": list(pos),
        "size": [340, 110],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [{"name": "latent", "type": "LATENT", "link": None}],
        "outputs": [],
        "properties": {"Node name for S&R": "H3LatentSave"},
        "widgets_values": [prefix],
        "title": title,
    }


save_a = make_save(nid, [470, 1340], "A: 保存 latent (verify_a)",
                   "latents/verify_a", max_order + 1)
save_b = make_save(nid + 1, [870, 1370], "B: 保存 latent (verify_b)",
                   "latents/verify_b", max_order + 2)
nodes.extend([save_a, save_b])

for src, dst, link_id in ((samp_a, save_a, lid), (samp_b, save_b, lid + 1)):
    d["links"].append([link_id, src["id"], 0, dst["id"], 0, "LATENT"])
    src["outputs"][0].setdefault("links", []).append(link_id)
    dst["inputs"][0]["link"] = link_id

note = {
    "id": nid + 2,
    "type": "Note",
    "pos": [1270, 1120],
    "size": [430, 330],
    "flags": {},
    "order": max_order + 3,
    "mode": 0,
    "inputs": [],
    "outputs": [],
    "properties": {},
    "widgets_values": [
        "SaveLatent 比对流程（ComfyUI 0.44）\n"
        "• 两个 SaveLatent 分别接 A/B 采样器的 output，其余节点与 02 完全一致。\n"
        "• Queue 一次，产物在 ComfyUI/output/latents/：\n"
        "    verify_a_00001_.latent / verify_b_00001_.latent\n"
        "• 跑完把这两个文件逐位比对（tools/compare_latents.py），\n"
        "  得到严格的 PASS / FAIL，绕开 v3→v1 的 LATENT 传值问题。\n"
        "• H3LatentCompare 仍保留，它的 raw A / raw B 只是诊断信息，\n"
        "  当前预期为 INCONCLUSIVE，不要用它当结论。\n"
        "• 对照实验：把 Apply 的 mode 设为 off 再跑一次，\n"
        "  两个 latent 必须不同（比对结果应为 FAIL）。"
    ],
    "color": "#432",
    "bgcolor": "#653",
}
nodes.append(note)

for g in d.get("groups", []):
    if g["bounding"][3] < 1620:
        g["bounding"][3] = 1620

d["last_node_id"] = nid + 2
d["last_link_id"] = lid + 1

os.makedirs(os.path.dirname(DST), exist_ok=True)
with open(DST, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print("wrote", DST, "nodes:", len(nodes), "links:", len(d["links"]))
