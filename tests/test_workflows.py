"""Structural validation of the shipped workflows.

ComfyUI fails late and confusingly when a workflow JSON is inconsistent (a
link that points at a missing slot, an input whose `link` id is stale, an
output that does not list its own link). These checks catch that offline,
before the file ever reaches a server.

It also verifies the parts of the graph that carry the correctness argument:

  * 05 pins chain B (and the saver) behind chain A via `after` edges, which is
    what guarantees the recorded table is filled before it is read;
  * every widget count matches the node's INPUT_TYPES, so a signature change
    cannot silently shift which value lands in which field.

Run: python tests/test_workflows.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
WF = os.path.join(PLUGIN, "workflows")

sys.path.insert(0, PLUGIN)

WILDCARD = "*"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_graph(path, d):
    problems = []
    nodes = {n["id"]: n for n in d["nodes"]}
    if len(nodes) != len(d["nodes"]):
        problems.append("duplicate node ids")
    link_ids = [l[0] for l in d["links"]]
    if len(set(link_ids)) != len(link_ids):
        problems.append("duplicate link ids")
    for lid, src, sslot, dst, dslot, typ in d["links"]:
        if src not in nodes or dst not in nodes:
            problems.append("link {} dangling endpoint".format(lid))
            continue
        outs = nodes[src].get("outputs") or []
        ins = nodes[dst].get("inputs") or []
        if sslot >= len(outs) or dslot >= len(ins):
            problems.append("link {} bad slot {}->{}".format(lid, sslot, dslot))
            continue
        out_t, in_t = outs[sslot]["type"], ins[dslot]["type"]
        if not (typ == out_t == in_t or WILDCARD in (out_t, in_t)):
            problems.append("link {} type {} {}->{}".format(lid, typ, out_t, in_t))
        if lid not in (outs[sslot].get("links") or []):
            problems.append("link {} missing from source outputs".format(lid))
        if ins[dslot].get("link") != lid:
            problems.append("link {} not set on target input".format(lid))
    for n in d["nodes"]:
        # Newer frontends may read the named-widget / asset metadata instead of
        # the positional array. Both must describe the same run.
        widgets = n.get("widgets_values") or []
        named = n.get("widgets_values_named")
        if isinstance(named, dict):
            for i, value in enumerate(list(named.values())[:len(widgets)]):
                if value != widgets[i]:
                    problems.append(
                        "node {} ({}) widgets_values_named[{}]={!r} != "
                        "widgets_values[{}]={!r}".format(
                            n["id"], n.get("type"), i, value, i, widgets[i]))
                    break
        for m in ((n.get("properties") or {}).get("models") or []):
            if isinstance(m, dict) and m.get("name") not in widgets:
                problems.append(
                    "node {} ({}) properties.models name {!r} is not a widget "
                    "value".format(n["id"], n.get("type"), m.get("name")))
        for i, inp in enumerate(n.get("inputs") or []):
            lid = inp.get("link")
            if lid is not None and not any(
                    l[0] == lid and l[3] == n["id"] and l[4] == i for l in d["links"]):
                problems.append("node {} input {} -> stale link {}".format(
                    n["id"], inp.get("name"), lid))
        for i, out in enumerate(n.get("outputs") or []):
            for lid in (out.get("links") or []):
                if not any(l[0] == lid and l[1] == n["id"] and l[2] == i
                           for l in d["links"]):
                    problems.append("node {} output {} -> stale link {}".format(
                        n["id"], out.get("name")))
    return problems


def widget_count(d, node):
    """Expected number of widget values for one node, from INPUT_TYPES."""
    import nodes as h3
    cls = h3.NODE_CLASS_MAPPINGS.get(node["type"])
    if cls is None:
        return None
    spec = cls.INPUT_TYPES()
    count = 0
    for group in ("required", "optional"):
        for name, decl in (spec.get(group) or {}).items():
            if isinstance(decl, tuple) and decl and isinstance(decl[0], list):
                count += 1
            elif isinstance(decl, tuple) and decl and isinstance(decl[0], str) \
                    and decl[0] in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                count += 1
    return count


def main():
    files = sorted(f for f in os.listdir(WF) if f.endswith(".json"))
    failures = []
    for name in files:
        path = os.path.join(WF, name)
        d = load(path)
        problems = check_graph(path, d)
        status = "PASS" if not problems else "FAIL"
        print("{} {} ({} nodes, {} links)".format(
            status, name, len(d["nodes"]), len(d["links"])))
        for p in problems:
            print("     " + p)
        if problems:
            failures.append(name)

    # widget-count agreement for our own nodes
    import nodes as h3
    for name in files:
        d = load(os.path.join(WF, name))
        for n in d["nodes"]:
            expected = widget_count(d, n)
            if expected is None:
                continue
            got = len(n.get("widgets_values") or [])
            if got != expected:
                msg = "{}: node {} ({}): {} widgets, INPUT_TYPES wants {}".format(
                    name, n["id"], n["type"], got, expected)
                print("FAIL " + msg)
                failures.append(msg)

    # ordering edges that carry the exactness argument
    d = load(os.path.join(WF, "05_record_strict.json"))
    nodes = {n["id"]: n for n in d["nodes"]}
    def has_after(node_type):
        hits = [n for n in d["nodes"] if n["type"] == node_type]
        return bool(hits) and all(
            any(i.get("name") == "after" and i.get("link") is not None
                for i in (n.get("inputs") or [])) for n in hits)
    for node_type in ("H3ExactAdaLNApply", "H3ExactAdaLNSave"):
        ok = has_after(node_type)
        print("{} 05: {} waits on chain A".format("PASS" if ok else "FAIL", node_type))
        if not ok:
            failures.append("05 missing after edge on " + node_type)

    print()
    if failures:
        print("{} FAILED".format(len(failures)))
        return 1
    print("ALL WORKFLOWS VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
