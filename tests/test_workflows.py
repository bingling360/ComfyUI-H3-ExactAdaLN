"""Structural validation of the shipped API prompts.

The two JSON files in workflows/ are **API-format prompts** (what you POST to
ComfyUI's /prompt endpoint), not UI workflow exports. ComfyUI fails late and
confusingly when such a prompt is inconsistent (a link pointing at a missing
node, a required input absent, no reachable output node), so check it offline.

  v2_record_apply_ab.json -- A/B verification: full teacher records a table,
                             pruned base applies it in the same prompt.
  v2_apply_auto.json      -- cross-prompt lookup: pruned base only, table=(auto).

Run: python tests/test_workflows.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
WF = os.path.join(PLUGIN, "workflows")

REMOVED_NODES = {
    "H3AdaLNRecord", "H3ExactAdaLNBake", "H3ExactAdaLNSave",
    "H3ExactAdaLNLoad", "H3AdaLNMerge", "H3ExactAdaLNApply",
    "H3LatentSave",
}
OUTPUT_NODES = {"H3LatentCompare", "PreviewAny", "SaveImage", "PreviewImage",
                "SaveLatent", "SaveAnimatedWEBP", "VHS_VideoCombine"}


def load(name):
    with open(os.path.join(WF, name), "r", encoding="utf-8") as f:
        return json.load(f)


def validate(prompt):
    """Return a list of human-readable problems."""
    problems = []
    if not isinstance(prompt, dict):
        return ["top level is not a dict (UI workflows are not supported)"]
    ids = set(prompt)
    for nid, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            problems.append("node {} has no class_type".format(nid))
            continue
        cls = node["class_type"]
        if cls in REMOVED_NODES:
            problems.append("node {} uses removed v1 node {}".format(nid, cls))
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 \
                    and isinstance(value[0], str) and isinstance(value[1], int):
                src = value[0]
                if src not in ids:
                    problems.append("node {}.{} -> missing node {}".format(nid, key, src))
    if not any(n.get("class_type") in OUTPUT_NODES for n in prompt.values()
               if isinstance(n, dict)):
        problems.append("no output node -- ComfyUI would execute nothing")
    return problems


def main():
    failures = []
    files = sorted(f for f in os.listdir(WF) if f.endswith(".json"))
    if not files:
        print("FAIL no workflows found")
        return 1

    def check(name, ok, detail=""):
        print("{} {}".format("PASS" if ok else "FAIL", name) +
              ("" if ok else " -> " + str(detail)[:300]))
        if not ok:
            failures.append(name)

    for name in files:
        p = load(name)
        problems = validate(p)
        check("{}: {} nodes, valid structure".format(name, len(p)),
              not problems, problems)

    for name in files:
        bad = [nid for nid, node in load(name).items()
               if isinstance(node, dict)
               and node.get("class_type") == "H3AdaLN"
               and (node.get("inputs") or {}).get("for_model") == "(auto)"]
        check("{}: no for_model=(auto) in any H3AdaLN".format(name),
              not bad, bad)

    by_name = {n: load(n) for n in files}

    ab = by_name.get("v2_record_apply_ab.json")
    if ab is not None:
        h3 = [n for n in ab.values() if n.get("class_type") == "H3AdaLN"]
        modes = sorted(n["inputs"].get("mode") for n in h3)
        check("A/B: one record + one apply H3AdaLN", modes == ["apply", "record"], modes)
        check("A/B: apply consumes the recorded table via table_in",
              any(isinstance(n["inputs"].get("table_in"), list)
                  for n in h3 if n["inputs"].get("mode") == "apply"), "")
        check("A/B: has H3LatentCompare",
              any(n.get("class_type") == "H3LatentCompare" for n in ab.values()), "")
    else:
        check("A/B workflow present", False, "missing")

    auto = by_name.get("v2_apply_auto.json")
    if auto is not None:
        h3 = [n for n in auto.values() if n.get("class_type") == "H3AdaLN"]
        check("auto: exactly one H3AdaLN in apply mode",
              len(h3) == 1 and h3[0]["inputs"].get("mode") == "apply", h3)
        if h3:
            inp = h3[0]["inputs"]
            check("auto: uses (auto) lookup, no table_in / no after",
                  inp.get("table") == "(auto)" and "table_in" not in inp
                  and "after" not in inp, inp)
        check("auto: has an output node",
              any(n.get("class_type") in OUTPUT_NODES for n in auto.values()), "")
    else:
        check("auto workflow present", False, "missing")

    print()
    if failures:
        print("{} CHECK(S) FAILED".format(len(failures)))
        return 1
    print("ALL WORKFLOWS VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
