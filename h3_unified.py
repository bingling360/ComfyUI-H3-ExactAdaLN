"""Unified H3 AdaLN node -- one node, several modes (EXPERIMENTAL).

Not wired into `__init__.py` yet: the pieces that need a live ComfyUI are still
unverified (ON_CLEANUP timing, whether a COMBO over models/h3_exact_adaln
refreshes without a restart, multi-pass behaviour). Design:
`docs/unified-node.md`.

Everything here composes the existing engines instead of reimplementing them:
`H3AdaLNRecord` / `H3ExactAdaLNLoad` / `H3AdaLNMerge` / `H3ExactAdaLNApply`.
"""

import hashlib
import logging
import os
import re

import torch

try:  # package import
    from . import nodes as _nodes
    from . import table_io
except ImportError:  # direct-file loading
    import nodes as _nodes
    import table_io

TABLE_EXT = ".h3adaln"
FOLDER_KEY = "h3_exact_adaln"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------- helpers
def register_folder():
    """Make `models/h3_exact_adaln/` visible to ComfyUI's folder machinery.

    Called from `__init__.py` once. Creates the directory so a fresh install has
    somewhere to save into.
    """
    import folder_paths
    base = os.path.join(folder_paths.models_dir, FOLDER_KEY)
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        logging.warning("H3AdaLN: cannot create %s (%s)", base, e)
    folder_paths.add_model_folder_path(FOLDER_KEY, base)
    return base


def table_folder():
    import folder_paths
    paths = folder_paths.get_folder_paths(FOLDER_KEY)
    return paths[0] if paths else os.path.join(folder_paths.models_dir, FOLDER_KEY)


def table_choices():
    """Names shown in the dropdown; falls back to a single empty entry."""
    try:
        import folder_paths
        names = folder_paths.get_filename_list(FOLDER_KEY)
    except Exception:
        return [""]
    return [n for n in names if n.endswith(TABLE_EXT)] or [""]


def _tag(value, limit=32):
    text = _SAFE.sub("-", str(value or "").strip()).strip("-._")
    return text[:limit].rstrip("-._") or "unknown"


def build_name(model_tag, lora_tag, scheduler, steps, shift_v, shift_a, fp6):
    """`{model}__{lora}__{scheduler}{steps}__sh{sv}-{sa}__{fp6}.h3adaln`

    Deliberately no t-row count: re-recording the same configuration should
    overwrite the previous table, not pile up `__t7` / `__t15` siblings.
    """
    return "{}__{}__{}{}__sh{}-{}__{}{}".format(
        _tag(model_tag), _tag(lora_tag),
        _tag(scheduler, 16) if str(scheduler or "").strip() else "nosched",
        int(steps), ("%g" % float(shift_v)), ("%g" % float(shift_a)),
        (fp6 or "nofp"), TABLE_EXT)


def model_fingerprint(dm):
    """Cheap identity for "which AdaLN weights are these".

    Shapes alone are not enough (fl2va full and ref2va full share them), so the
    first bytes of the first three block projections go in. Cached on the module.
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
    out = getattr(getattr(final, "linear", None), "out_features", 0)
    h.update(("final_out=%s" % out).encode())
    fp = h.hexdigest()[:12]
    try:
        dm._h3_model_fp = fp
    except Exception:
        pass
    return fp


def _resolve_name(table, table_text, label):
    name = (table_text or "").strip() or (table or "").strip()
    if not name:
        raise RuntimeError(
            "H3AdaLN: mode needs a table, but none was picked (%s). Put a "
            "%s file in models/%s/ or type its name into table_text."
            % (label, TABLE_EXT, FOLDER_KEY))
    return name


def _load_table(name):
    table, info = _nodes.H3ExactAdaLNLoad().load(name)
    return table, info


def _save_named(table, name):
    """Write to models/h3_exact_adaln/<name>, overwriting the same config."""
    folder = table_folder()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    table_io.save_table(path, table.t_values, table.raw_by_key, table.meta,
                        out_dtype=table.out_dtype)
    return path


def _check_model(table, dm, strict):
    """Refuse a table recorded on a different AdaLN branch."""
    want = table.meta.get("model_fingerprint", "")
    if not want:
        return ""            # table predates fingerprints
    got = model_fingerprint(dm)
    if want == got:
        return "model fp     : {} (match)".format(got)
    msg = ("table was recorded on a different AdaLN branch (table={} model={}); "
           "re-record on this model".format(want, got))
    if strict:
        raise RuntimeError("H3AdaLN: " + msg)
    logging.warning("H3AdaLN: %s", msg)
    return "model fp     : {} / {} MISMATCH".format(want, got)


# ---------------------------------------------------------------- node
class H3AdaLN:
    """Record, apply, or merge AdaLN tables -- one node, pick a mode."""

    MODES = ["apply", "record", "hybrid", "merge", "off"]

    @classmethod
    def INPUT_TYPES(cls):
        choices = table_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (cls.MODES, {"default": "apply"}),
            },
            "optional": {
                "table": (choices, {"default": choices[0]}),
                "table_b": (choices, {"default": choices[0]}),
                "table_text": ("STRING", {"default": "",
                        "tooltip": "Overrides the dropdown: filename, unique "
                                   "substring or absolute path. Use this if the "
                                   "dropdown does not refresh after saving."}),
                "table_b_text": ("STRING", {"default": ""}),
                "split": ("INT", {"default": 25, "min": 0, "max": 50}),
                "final_from": (["a", "b"], {"default": "a"}),
                "strictness": (["strict", "fallback"], {"default": "strict"}),
                "record_name": ("STRING", {"default": "",
                        "tooltip": "Leave empty to auto-name from the tags. "
                                   "Same configuration overwrites."}),
                "model_tag": ("STRING", {"default": "",
                        "tooltip": "Goes into the table name, e.g. b25-49full-int8."}),
                "lora_tag": ("STRING", {"default": "",
                        "tooltip": "Goes into the table name, e.g. dmad8x1.0."}),
                "scheduler_tag": ("STRING", {"default": "",
                        "tooltip": "Goes into the table name; steps and shifts "
                                   "are measured, the scheduler name is not "
                                   "visible to a model-side node."}),
                "table_in": (table_io.TABLE_TYPE, {"tooltip":
                        "Optional. Connect a table produced earlier in the SAME "
                        "prompt (A/B verification); otherwise the dropdown wins."}),
            },
        }

    RETURN_TYPES = ("MODEL", table_io.TABLE_TYPE, "STRING")
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    # -- main ---------------------------------------------------------
    def run(self, model, mode, table="", table_b="", table_text="", table_b_text="",
            split=25, final_from="a", strictness="strict", record_name="",
            model_tag="", lora_tag="", scheduler_tag="", table_in=None):
        mode = (mode or "apply").lower()
        if mode == "off":
            _nodes._clear_registered(model, "_h3_exact_runtimes")
            return (model, table_in, "H3AdaLN: mode=off, model untouched")
        if mode == "record":
            return self._record(model, strictness, record_name, model_tag, lora_tag,
                                scheduler_tag)
        if mode == "apply":
            return self._apply(model, table, table_text, strictness, table_in)
        if mode in ("hybrid", "merge"):
            return self._hybrid(model, mode, table, table_b, table_text, table_b_text,
                                split, final_from, strictness, record_name,
                                model_tag, lora_tag, scheduler_tag)
        raise RuntimeError("H3AdaLN: unknown mode %r" % mode)

    # -- modes --------------------------------------------------------
    def _record(self, model, strictness, record_name, model_tag, lora_tag, scheduler_tag):
        dm = _nodes._dm_of(model)
        if dm is None:
            raise RuntimeError("H3AdaLN: expected a MiniMax-H3 MODEL")

        # Clear any previous pending save so a re-run starts clean.
        for attr in ("_h3_pending_table", "_h3_pending_name", "_h3_saved"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    pass

        model, tbl, rep = _nodes.H3AdaLNRecord().run(
            model, mode="strict" if strictness == "strict" else "lenient",
            teacher_tag=model_tag, lora_tag=lora_tag)

        model._h3_pending_table = tbl
        model._h3_pending_tags = (model_tag, lora_tag, scheduler_tag, record_name)
        model._h3_rec_dm = dm
        model._h3_saved = False
        try:
            import comfy.patcher_extension as pe
            model.add_callback(pe.CallbacksMP.ON_CLEANUP,
                               lambda *a, **k: _finish_record(model))
        except Exception as e:
            logging.warning("H3AdaLN: cannot hook save-on-cleanup (%s: %s); the "
                            "table will only exist in memory", type(e).__name__, e)

        return (model, tbl, "\n".join([
            rep,
            "",
            "=== H3AdaLN (record) ===",
            "the table is written to models/%s/ when the prompt finishes" % FOLDER_KEY,
            "name pattern     : {model}__{lora}__{scheduler}{steps}__sh{sv}-{sa}__{fp6}"
            + TABLE_EXT,
            "tags so far      : model=%r lora=%r scheduler=%r"
            % (model_tag, lora_tag, scheduler_tag),
        ]))

    def _apply(self, model, table, table_text, strictness, table_in):
        dm = _nodes._dm_of(model)
        if table_in is not None:
            tbl, src = table_in, "connected input"
        else:
            tbl, src = _load_table(_resolve_name(table, table_text, "apply"))
        note = _check_model(tbl, dm, strictness == "strict")
        model, rep = _nodes.H3ExactAdaLNApply().run(
            model, tbl, "strict" if strictness == "strict" else "fallback")
        return (model, tbl, "\n".join([
            "=== H3AdaLN (apply) ===",
            "table        : {} ({})".format(src, tbl.source or "(memory)"),
            note,
            rep,
        ]))

    def _hybrid(self, model, mode, table, table_b, table_text, table_b_text,
                split, final_from, strictness, record_name, model_tag, lora_tag,
                scheduler_tag):
        a, _ia = _load_table(_resolve_name(table, table_text, "table A"))
        b, _ib = _load_table(_resolve_name(table_b, table_b_text, "table B"))
        merged, rep = _nodes.H3AdaLNMerge().run(a, b, split=split, final_from=final_from)

        dm = _nodes._dm_of(model)
        name = (record_name or "").strip()
        if not name:
            rows = len(merged.t_values)
            name = build_name(model_tag or "hybrid", lora_tag or "split%d" % split,
                              scheduler_tag, max(rows - 1, 1),
                              merged.meta.get("shift_v", 0), merged.meta.get("shift_a", 0),
                              merged.meta.get("lora_fingerprint", "")[:6] or "merged")
        path = _save_named(merged, name if name.endswith(TABLE_EXT) else name + TABLE_EXT)

        out = ["=== H3AdaLN (hybrid) ===", rep, "saved        : " + path]
        if mode == "hybrid":
            note = _check_model(merged, dm, strictness == "strict")
            model, arep = _nodes.H3ExactAdaLNApply().run(
                model, merged, "strict" if strictness == "strict" else "fallback")
            out += [note, arep]
        return (model, merged, "\n".join(x for x in out if x))


# ---------------------------------------------------------------- save hook
def _finish_record(model):
    """ON_CLEANUP: write the recorded table with a name derived from the run."""
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
            "H3AdaLN: refusing to save, %d timestep(s) produced different "
            "modulation in the same run (max delta %s). This means the AdaLN "
            "branch changed mid-run -- record again with one model + one LoRA.",
            conflicts, tbl.meta.get("max_repeat_delta", "?"))
        return
    model_tag, lora_tag, scheduler_tag, record_name = getattr(
        model, "_h3_pending_tags", ("", "", "", ""))
    dm = getattr(model, "_h3_rec_dm", None)
    if record_name.strip():
        name = record_name.strip()
    else:
        rows = len(tbl.t_values)
        name = build_name(model_tag or "model", lora_tag or "none", scheduler_tag,
                          max(rows - 1, 1), tbl.meta.get("shift_v", 0),
                          tbl.meta.get("shift_a", 0),
                          (tbl.meta.get("lora_fingerprint", "") or "nofp")[:6])
    if not name.endswith(TABLE_EXT):
        name += TABLE_EXT
    if dm is not None:
        tbl.meta["model_fingerprint"] = model_fingerprint(dm)
    tbl.meta["model_tag"] = model_tag
    tbl.meta["lora_tag"] = lora_tag
    tbl.meta["t_rows"] = str(len(tbl.t_values))
    try:
        path = _save_named(tbl, name)
        model._h3_saved = True
        logging.info("H3AdaLN: saved %s (%d t rows)", path, len(tbl.t_values))
    except Exception as e:
        logging.error("H3AdaLN: save failed -- %s: %s", type(e).__name__, e)
