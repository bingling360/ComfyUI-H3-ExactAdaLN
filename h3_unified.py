"""Unified H3 AdaLN node -- one node, several modes (EXPERIMENTAL).

Design: `docs/unified-node.md`. Registered from `__init__.py` behind a
try/except so a failure here cannot take down the eight stable nodes. The parts
that need a live ComfyUI (ON_CLEANUP timing, COMBO refresh, multi-pass) are
still unverified -- see the checklist in the design note.

Everything here composes the existing engines instead of reimplementing them:
`H3AdaLNRecord` / `H3ExactAdaLNLoad` / `H3AdaLNMerge` / `H3ExactAdaLNApply`.

Tables live in `models/h3_exact_adaln/` and keep the `.safetensors` extension so
the existing Load / Merge nodes and every external tool can still read them; the
folder is what marks them as tables.
"""

import hashlib
import json
import logging
import os
import re
import time

import torch

try:  # package import
    from . import nodes as _nodes
    from . import table_io
except ImportError:  # direct-file loading
    import nodes as _nodes
    import table_io

FOLDER_KEY = "h3_exact_adaln"
TABLE_TYPE = _nodes.TABLE_TYPE
TABLE_EXT = ".safetensors"
LAST_POINTER = "_last_table.json"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------- folder
def register_folder():
    """Called once from `__init__.py`: make models/h3_exact_adaln a model folder."""
    import folder_paths
    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:  # older builds
        try:
            models_dir = os.path.dirname(
                folder_paths.get_folder_paths("diffusion_models")[0])
        except Exception:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
    base = os.path.join(models_dir, FOLDER_KEY)
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        logging.warning("H3AdaLN: cannot create %s (%s)", base, e)
    folder_paths.add_model_folder_path(FOLDER_KEY, base)
    return base


def table_folder():
    """Where tables are written; the same place table_io searches for Load."""
    try:
        return table_io.table_dir()
    except Exception:
        import folder_paths
        return os.path.join(folder_paths.models_dir, FOLDER_KEY)


def table_choices():
    """Dropdown contents: the table folder first, plus anything Load can see."""
    names = []
    try:
        folder = table_folder()
        if os.path.isdir(folder):
            names = sorted(f for f in os.listdir(folder)
                           if f.lower().endswith(TABLE_EXT) and not f.startswith("_"))
    except Exception:
        names = []
    if not names:
        try:
            names = [n for n in table_io.list_table_files()
                     if n.lower().endswith(TABLE_EXT)]
        except Exception:
            names = []
    return names or [""]


# ---------------------------------------------------------------- naming
def _tag(value, limit=32):
    text = _SAFE.sub("-", str(value or "").strip()).strip("-._")
    return text[:limit].rstrip("-._") or "unknown"


def build_name(model_tag, lora_tag, scheduler, steps, shift_v, shift_a, fp6,
               pass_tag=""):
    """`model__lora__[pass__]scheduler+steps__sh{sv}-{sa}__{fp6}.safetensors`

    One glance tells you which checkpoint, which LoRA, which scheduler/steps and
    which shift pair produced the table.

    No t-row count in the name on purpose: re-recording the same configuration
    overwrites, instead of piling up `__t7` / `__t15` siblings. The row count
    goes into the metadata and into the node report.
    """
    parts = [_tag(model_tag), _tag(lora_tag)]
    if str(pass_tag or "").strip():
        parts.append(_tag(pass_tag, 16))
    parts += [
        "{}{}".format(_tag(scheduler, 16) if str(scheduler or "").strip() else "nosched",
                      int(steps)),
        "sh{}-{}".format("%g" % float(shift_v), "%g" % float(shift_a)),
        fp6 or "nofp",
    ]
    return "__".join(parts) + TABLE_EXT


# ---------------------------------------------------------------- identity
def model_fingerprint(dm):
    """Cheap identity for "which AdaLN weights are these".

    Shapes alone are not enough -- fl2va full and ref2va full share them -- so
    the first bytes of the first three block projections are hashed in. Cached
    on the module; costs a few milliseconds.
    """
    cached = getattr(dm, "_h3_model_fp", None)
    if cached:
        return cached
    h = hashlib.sha1()
    h.update(("curve=%s;" % bool(getattr(dm, "use_adaln_curves", False))).encode())
    blocks = list(getattr(dm, "blocks", []) or [])
    h.update(("blocks=%d;" % len(blocks)).encode())
    for block in blocks[:3]:
        linear = getattr(getattr(block, "adaln_proj", None), "linear", None)
        weight = getattr(linear, "weight", None)
        if weight is None or not torch.is_tensor(weight):
            continue
        try:
            head = weight.detach().reshape(-1)[:1024].to("cpu", torch.float32)
            h.update(head.numpy().tobytes())
        except Exception:
            h.update(b"?")
    final = getattr(getattr(dm, "final_layer", None), "adaln_proj", None)
    h.update(("final_out=%s" % getattr(getattr(final, "linear", None),
                                       "out_features", 0)).encode())
    fp = h.hexdigest()[:12]
    try:
        dm._h3_model_fp = fp
    except Exception:
        pass
    return fp


# ---------------------------------------------------------------- last pointer
def _write_last(entry):
    path = os.path.join(table_folder(), LAST_POINTER)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        logging.warning("H3AdaLN: cannot write %s (%s)", path, e)


def _read_last():
    path = os.path.join(table_folder(), LAST_POINTER)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------- helpers
def _load(name):
    res = _nodes.H3ExactAdaLNLoad().load(name)
    return res["result"][0]


def _resolvable(name):
    """True when `name` points at a table Load can actually open."""
    try:
        return bool(name) and bool(table_io.find_table_file(name))
    except Exception:
        return False


def _pick_table(table, table_text, source, label):
    """Resolve the table for apply/hybrid.

    `auto` exists for the record -> apply hand-off: right after recording, the
    dropdown usually still shows the previous list, and retyping the generated
    name is exactly the friction the unified node is meant to remove.
    """
    text = (table_text or "").strip()
    picked = (table or "").strip()
    last = (_read_last() or {}).get("name", "")

    if text:
        if not _resolvable(text):
            raise RuntimeError(
                "H3AdaLN: table_text={!r} does not resolve to a table file.\n{}"
                .format(text, table_io.describe_available()))
        return text, "typed"

    if source == "last":
        if not _resolvable(last):
            raise RuntimeError(
                "H3AdaLN: table_source=last, but no table has been recorded yet "
                "(no usable {} in {}).".format(LAST_POINTER, table_folder()))
        return last, "last recorded"

    if source == "auto":
        if _resolvable(picked):
            return picked, "dropdown"
        if _resolvable(last):
            return last, "last recorded (dropdown entry not on disk yet)"
        raise RuntimeError(
            "H3AdaLN: no usable table for {}. Picked {!r}, last recorded {!r}.\n{}"
            .format(label, picked, last, table_io.describe_available()))

    if not _resolvable(picked):
        raise RuntimeError(
            "H3AdaLN: {} = {!r} does not resolve to a table file.\n{}"
            .format(label, picked, table_io.describe_available()))
    return picked, "dropdown"


def _save_table(table, name):
    folder = table_folder()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    table_io.save_table(path, table.t_values, table.raw_by_key, table.meta,
                        out_dtype=table.out_dtype)
    return path


def _is_pruned_base(dm):
    """A curve-shaped base has no time_embedder and carries adaln_t_table."""
    if dm is None:
        return False
    if bool(getattr(dm, "use_adaln_curves", False)):
        return True
    return (getattr(dm, "adaln_t_table", None) is not None
            and getattr(dm, "time_embedder", None) is None)


def _check_model(table, dm, strict):
    """Only compare fingerprints when the target still computes its own AdaLN.

    The primary use of the pack is `record on the FULL teacher -> apply to a
    PRUNED/curve base`. Those two checkpoints deliberately have different
    adaln_proj shapes ([96768,2688] vs [96768,8]), so an equality check here
    would reject the very workflow the node exists for. On a pruned base the
    stored rows *replace* the curve entirely, so the target's own AdaLN weights
    are irrelevant.
    """
    want = table.meta.get("model_fingerprint", "")
    if not want:
        return "model fp     : (table has none; recorded before fingerprints existed)"
    if _is_pruned_base(dm):
        return ("model fp     : teacher={} (target is a pruned/curve base; its "
                "own AdaLN is replaced by the table)".format(want))
    got = model_fingerprint(dm)
    if want == got:
        return "model fp     : {} (match)".format(got)
    msg = ("table was recorded on a different AdaLN branch (table={} model={}) -- "
           "the rows belong to another checkpoint".format(want, got))
    if strict:
        raise RuntimeError("H3AdaLN: " + msg + ". Re-record on this model.")
    logging.warning("H3AdaLN: %s", msg)
    return "model fp     : {} / {} MISMATCH (continuing, strictness=fallback)".format(
        want, got)


# ---------------------------------------------------------------- node
class H3AdaLN:
    """Record, apply or merge AdaLN tables -- one node, pick a mode."""

    MODES = ["apply", "record", "hybrid", "merge", "off"]
    SOURCES = ["auto", "dropdown", "last"]

    @classmethod
    def INPUT_TYPES(cls):
        choices = table_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (cls.MODES, {"default": "apply"}),
            },
            "optional": {
                "table": (choices, {"default": choices[0],
                        "tooltip": "Table for apply/hybrid/merge."}),
                "table_b": (choices, {"default": choices[0],
                        "tooltip": "Tail donor for hybrid/merge."}),
                "table_source": (cls.SOURCES, {"default": "auto",
                        "tooltip": "auto: use the dropdown entry, else fall back "
                                   "to the table recorded most recently -- so "
                                   "record -> apply needs no retyping even when "
                                   "the dropdown has not refreshed."}),
                "table_text": ("STRING", {"default": "",
                        "tooltip": "Overrides everything: filename, unique "
                                   "substring, or absolute path."}),
                "table_b_text": ("STRING", {"default": ""}),
                "split": ("INT", {"default": 25, "min": 0, "max": 50}),
                "final_from": (["a", "b"], {"default": "a"}),
                "strictness": (["strict", "fallback"], {"default": "strict"}),
                "record_name": ("STRING", {"default": "",
                        "tooltip": "Leave empty to auto-name from the tags. "
                                   "The same configuration overwrites."}),
                "model_tag": ("STRING", {"default": "",
                        "tooltip": "Goes into the table name, e.g. b25-49full-int8."}),
                "lora_tag": ("STRING", {"default": "",
                        "tooltip": "Goes into the table name, e.g. dmad8x1.0."}),
                "pass_tag": ("STRING", {"default": "",
                        "tooltip": "Only needed when one workflow records two "
                                   "models (second pass, high/low): give each "
                                   "node its own tag, e.g. p1 / p2 / high / low."}),
                "scheduler_tag": ("STRING", {"default": "",
                        "tooltip": "Free-form; the scheduler name is not visible "
                                   "to a model-side node. Steps and shifts are "
                                   "measured, not typed."}),
                "table_in": (TABLE_TYPE, {"tooltip":
                        "Optional. Connect a table produced earlier in the SAME "
                        "prompt (A/B verification); otherwise the picker wins."}),
                "after": ("*", {"tooltip":
                        "Ordering-only input. Connect the other chain's LATENT "
                        "when doing record -> apply inside one prompt with a "
                        "sampler that has no `after` socket of its own (e.g. "
                        "SelfLift / director chains): it forces that chain to "
                        "finish sampling before this node consumes the table."}),
            },
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("model", "table", "table_name", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    # -- main ---------------------------------------------------------
    def run(self, model, mode, table="", table_b="", table_source="auto",
            table_text="", table_b_text="", split=25, final_from="a",
            strictness="strict", record_name="", model_tag="", lora_tag="",
            pass_tag="", scheduler_tag="", table_in=None, after=None):
        # `after` is intentionally unused: ComfyUI only needs the edge to order
        # the graph, so a record chain can finish before an apply chain reads
        # the (in-memory) table in samplers that expose no `after` socket.
        mode = (mode or "apply").lower()
        if mode == "off":
            _nodes._clear_registered(model, "_h3_exact_runtimes")
            return (model, table_in, "", "H3AdaLN: mode=off, model untouched")
        if mode == "record":
            return self._record(model, strictness, record_name, model_tag,
                                lora_tag, pass_tag, scheduler_tag)
        if mode == "apply":
            return self._apply(model, table, table_text, table_source,
                               strictness, table_in)
        if mode in ("hybrid", "merge"):
            return self._hybrid(model, mode, table, table_b, table_text,
                                table_b_text, split, final_from, strictness,
                                record_name, model_tag, lora_tag, pass_tag,
                                scheduler_tag)
        raise RuntimeError("H3AdaLN: unknown mode %r" % mode)

    # -- record -------------------------------------------------------
    def _record(self, model, strictness, record_name, model_tag, lora_tag,
                pass_tag, scheduler_tag):
        dm = _nodes._dm_of(model)
        if dm is None:
            raise RuntimeError("H3AdaLN: expected a MiniMax-H3 MODEL")
        for attr in ("_h3_pending_table", "_h3_pending_tags", "_h3_saved"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    pass

        model, tbl, rep = _nodes.H3AdaLNRecord().run(
            model,
            mode="strict" if strictness == "strict" else "lenient",
            teacher_tag=model_tag, lora_tag=lora_tag)

        model._h3_pending_table = tbl
        model._h3_pending_tags = dict(
            model_tag=model_tag, lora_tag=lora_tag, pass_tag=pass_tag,
            scheduler_tag=scheduler_tag, record_name=record_name)
        model._h3_rec_dm = dm
        model._h3_saved = False
        try:
            import comfy.patcher_extension as pe
            model.add_callback(pe.CallbacksMP.ON_CLEANUP,
                               lambda *a, **k: _finish_record(model))
        except Exception as e:
            logging.warning("H3AdaLN: cannot hook save-on-cleanup (%s: %s); the "
                            "table will exist in memory only", type(e).__name__, e)

        preview = build_name(model_tag or "model", lora_tag or "none",
                             scheduler_tag, 1, 12, 2, "<lora fp>", pass_tag)
        return (model, tbl, "", "\n".join([
            rep,
            "",
            "=== H3AdaLN (record) ===",
            "saved at end of prompt -> models/{}/".format(FOLDER_KEY),
            "name pattern : {}".format(preview),
            "tags         : model={!r} lora={!r} pass={!r} scheduler={!r}".format(
                model_tag, lora_tag, pass_tag, scheduler_tag),
            "",
            "Switch this node to mode=apply afterwards: table_source=auto picks",
            "the table that was just recorded even if the dropdown has not",
            "refreshed. No retyping, no extra nodes.",
        ]))

    # -- apply --------------------------------------------------------
    def _apply(self, model, table, table_text, table_source, strictness, table_in):
        dm = _nodes._dm_of(model)
        if table_in is not None:
            tbl, src = table_in, "connected table_in"
        else:
            name, why = _pick_table(table, table_text, table_source, "table")
            tbl = _load(name)
            src = "{} ({})".format(name, why)
        note = _check_model(tbl, dm, strictness == "strict")
        model, rep = _nodes.H3ExactAdaLNApply().run(
            model, tbl, "strict" if strictness == "strict" else "fallback")
        return (model, tbl, tbl.source or "", "\n".join([
            "=== H3AdaLN (apply) ===",
            "table        : {}".format(src),
            note,
            rep,
        ]))

    # -- hybrid / merge -----------------------------------------------
    def _hybrid(self, model, mode, table, table_b, table_text, table_b_text,
                split, final_from, strictness, record_name, model_tag, lora_tag,
                pass_tag, scheduler_tag):
        name_a, why_a = _pick_table(table, table_text, "auto", "table A")
        name_b, why_b = _pick_table(table_b, table_b_text, "auto", "table B")
        a, b = _load(name_a), _load(name_b)
        merged, rep = _nodes.H3AdaLNMerge().run(a, b, split=split,
                                                final_from=final_from)

        rows = len(merged.t_values)
        name = (record_name or "").strip()
        if not name:
            name = build_name(model_tag or "hybrid",
                              lora_tag or "split%d" % int(split), scheduler_tag,
                              max(rows - 1, 1), merged.meta.get("shift_v", 0),
                              merged.meta.get("shift_a", 0),
                              (merged.meta.get("lora_fingerprint", "") or "")[:6],
                              pass_tag)
        if not name.lower().endswith(TABLE_EXT):
            name += TABLE_EXT
        merged.meta["model_tag"] = model_tag
        merged.meta["pass_tag"] = pass_tag
        merged.meta["t_rows"] = str(rows)
        path = _save_table(merged, name)
        _write_last({"name": name, "path": path, "created": time.time(),
                     "mode": mode, "split": int(split)})

        out = ["=== H3AdaLN ({}) ===".format(mode),
               "table A      : {} ({})".format(name_a, why_a),
               "table B      : {} ({})".format(name_b, why_b),
               rep,
               "saved        : {}".format(path)]
        if mode == "hybrid":
            note = _check_model(merged, _nodes._dm_of(model), strictness == "strict")
            model, arep = _nodes.H3ExactAdaLNApply().run(
                model, merged, "strict" if strictness == "strict" else "fallback")
            out += [note, arep]
        return (model, merged, name, "\n".join(x for x in out if x))


# ---------------------------------------------------------------- save hook
def _finish_record(model):
    """ON_CLEANUP: write the recorded table under a name derived from the run."""
    tbl = getattr(model, "_h3_pending_table", None)
    if tbl is None or getattr(model, "_h3_saved", False):
        return
    try:
        tbl.finalize(required=True)
    except Exception as e:
        logging.error("H3AdaLN: nothing saved -- %s", e)
        return

    conflicts = int(tbl.meta.get("repeat_conflicts", "0") or 0)
    if conflicts:
        logging.error(
            "H3AdaLN: refusing to save. %d timestep(s) produced different "
            "modulation in the same run (max delta %s), so the AdaLN branch "
            "changed mid-run -- record again with one model and one LoRA stack.",
            conflicts, tbl.meta.get("max_repeat_delta", "?"))
        return

    tags = getattr(model, "_h3_pending_tags", {}) or {}
    dm = getattr(model, "_h3_rec_dm", None)
    rows = len(tbl.t_values)
    name = (tags.get("record_name") or "").strip()
    if not name:
        name = build_name(tags.get("model_tag") or "model",
                          tags.get("lora_tag") or "none",
                          tags.get("scheduler_tag"), max(rows - 1, 1),
                          tbl.meta.get("shift_v", 0), tbl.meta.get("shift_a", 0),
                          (tbl.meta.get("lora_fingerprint", "") or "")[:6],
                          tags.get("pass_tag") or "")
    if not name.lower().endswith(TABLE_EXT):
        name += TABLE_EXT

    if dm is not None:
        tbl.meta["model_fingerprint"] = model_fingerprint(dm)
    tbl.meta["model_tag"] = tags.get("model_tag", "")
    tbl.meta["lora_tag"] = tags.get("lora_tag", "")
    tbl.meta["pass_tag"] = tags.get("pass_tag", "")
    tbl.meta["scheduler_tag"] = tags.get("scheduler_tag", "")
    tbl.meta["t_rows"] = str(rows)
    tbl.meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        path = _save_table(tbl, name)
    except Exception as e:
        logging.error("H3AdaLN: save failed -- %s: %s", type(e).__name__, e)
        return
    model._h3_saved = True
    _write_last({"name": name, "path": path, "created": time.time(),
                 "t_rows": rows, "tags": tags})
    logging.info("H3AdaLN: saved %s (%d t rows, %.0f MiB)", path, rows,
                 os.path.getsize(path) / (1 << 20))
