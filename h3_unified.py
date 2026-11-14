"""H3 AdaLN -- one node records and applies, one more assembles hybrids.

v2 of the unified node. The v1 surface was 16 widgets, and nearly all of them
existed to work around something the node can decide by itself:

  model_tag / lora_tag /    -> the table's identity is (AdaLN weights, AdaLN LoRA
  scheduler_tag / pass_tag     fingerprint, compute dtype), read off the model.
                               The file name follows from it, so nothing is typed.
  table_text / table_b_text / -> nothing to pick at apply time. `for_model` is
  table_source (3 modes)        the one explicit choice: which pruned base a
                                recording is for. Its AdaLN fingerprint goes
                                into the table header, so apply finds the table
                                by hashing the model in front of it. No "most
                                recent" rule, no auto-detection, no sidecar
                                state -- the association travels inside the
                                .safetensors.
  record_name                -> a run *merges* its rows into the table with the
                                same identity, so re-recording never needs a name
                                and two nodes on one model cannot split the t set
                                into two half-covering tables.
  strictness                 -> always strict. "fallback" only ever produced a
                                table that silently answered wrongly, which is the
                                one failure this pack exists to prevent.
  split / final_from         -> live on the hybrid node, where they are the only
                                thing to set.

What is left:

  H3 AdaLN          model, mode, report_level, for_model, table,
                    [table_in], [after]
  H3 AdaLN Hybrid   model, table_a, table_b, split, [final_from],
                    [for_model], [report_level]        bX-49 donor merge

`report_level` (silent / brief / debug) decides how much the run says. The
`report` output is a live object rather than a string: the numbers that matter
(injections per pass, which t rows a later pass used, whether a refine pass
silently reverted to native AdaLN) only exist after sampling. Feed it to
`H3 AdaLN Report` with its `after` input wired to the end of the graph.

The invariant that makes this enough:

  modulation = adaln_proj(time_embedder(t))

i.e. a pure function of (AdaLN weights + AdaLN LoRA) and t -- not of the pass,
the latent, the resolution or the seed. So one table per model covers every
sampling segment, and rows from different schedules simply accumulate.
"""

import hashlib
import logging
import os
import threading
import time

import torch

try:  # package import
    from . import apply_engine, nodes as _nodes, record_engine, sigma_plan, table_io
except ImportError:  # direct-file loading / tests
    import apply_engine
    import nodes as _nodes
    import record_engine
    import sigma_plan
    import table_io

FOLDER_KEY = "h3_exact_adaln"
TABLE_TYPE = _nodes.TABLE_TYPE
REPORT_TYPE = _nodes.REPORT_TYPE
TABLE_EXT = ".safetensors"
AUTO = "(auto)"
NONE = "(none)"
# Two rows for the same t may differ by this much and still count as the same
# observation. Anything above it means the model / LoRA / dtype changed.
# fp32 tables can be compared almost bitwise. AdaLN rows, however, are
# captured from a bf16/fp16 forward and only then stored as fp32: two runs of
# the SAME model can differ by a few low-precision ULPs (~1.5e-5 measured)
# because of kernel / compiler / mixed-precision reduction order. A real
# model or AdaLN-LoRA change moves these values by >=1e-2, so a dtype-aware
# tolerance is safe and makes re-recording possible.
CONFLICT_TOL = 1e-6
LOW_PRECISION_TOL = 2e-4
# How many ON_CLEANUP saves one record run may make. A director chain has two
# or three passes; anything beyond that is a recorder that outlived its prompt.
MAX_RECORD_PASSES = 8

_AFTER_DOC = ("Ordering only, nothing is read from it. Connect the sampler's "
              "LATENT here when a table is produced and consumed inside one "
              "prompt (ComfyUI otherwise runs this node too early).")

_FOLDER = None


# ---------------------------------------------------------------- folder
def register_folder():
    """Called once from `__init__.py`: make models/h3_exact_adaln a model folder."""
    global _FOLDER
    import folder_paths
    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:  # older builds
        try:
            models_dir = os.path.dirname(
                folder_paths.get_folder_paths("diffusion_models")[0])
        except Exception:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "models")
    base = os.path.join(models_dir, FOLDER_KEY)
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        logging.warning("H3AdaLN: cannot create %s (%s)", base, e)
    try:
        # Must not take the node down if an older build lacks this helper.
        folder_paths.add_model_folder_path(FOLDER_KEY, base)
    except Exception as e:
        logging.warning("H3AdaLN: could not register %s as a model folder (%s: %s)",
                        base, type(e).__name__, e)
    _FOLDER = base
    return base


def table_folder():
    """Where new tables are written."""
    override = os.environ.get("H3_ADALN_TABLE_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    if _FOLDER:
        return _FOLDER
    try:
        return table_io.table_dir()
    except Exception:
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", FOLDER_KEY)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def table_dirs():
    """The one folder tables live in.

    v1 also scanned `output/h3_adaln`; that let stale saves from an unrelated
    run compete with the current table, so it was removed. Tables are portable
    by copying the .safetensors itself, not by adding search paths.
    """
    return [table_folder()]


def list_tables():
    """Dropdown contents. Cheap: two flat listdir calls, no recursion."""
    names = []
    for folder in table_dirs():
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in entries:
            if name.lower().endswith(TABLE_EXT) and name not in names:
                names.append(name)
    return names


def _choices(default=AUTO):
    return [default] + [n for n in list_tables() if n != default]


# ---------------------------------------------------------------- lookup
def _table_meta(path):
    """Header only -- a few milliseconds, no tensor is loaded."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def _norm_adaln_fp(value):
    """Older tables stored sha256("")[:16] instead of "" for "no AdaLN LoRA"."""
    fp = str(value or "")
    return "" if fp == "e3b0c44298fc1c14" else fp


def _scan_tables():
    """[(path, header metadata)] for every readable table, no tensor loaded."""
    out = []
    for folder in table_dirs():
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in entries:
            if not name.lower().endswith(TABLE_EXT):
                continue
            path = os.path.join(folder, name)
            meta = _table_meta(path)
            if meta:  # not ours / unreadable
                out.append((path, meta))
    return out


def _model_choices(default=NONE):
    """Checkpoints this table can be declared for (record mode only).

    There is no `(auto)` entry: the base is an explicit choice, and ComfyUI
    keeps that value in the workflow so it only has to be made once.
    """
    names = []
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("diffusion_models"))
    except Exception:
        pass
    return [default] + [n for n in names if n != default]


def resolve_model_file(name):
    """A checkpoint name from the dropdown (or a path) -> absolute path."""
    name = str(name or "").strip()
    if not name or name == NONE:
        return None
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for key in ("diffusion_models", "unet"):
        try:
            import folder_paths
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            return path
    try:
        import folder_paths
        return folder_paths.get_full_path_or_raise("diffusion_models", name)
    except Exception:
        return None


def find_table(name, what="table"):
    """A concrete filename / path, for the hybrid node and explicit picks."""
    path = _find_named(name)
    if not path:
        raise RuntimeError(
            "H3AdaLN: {} {!r} is not in {}.\navailable: {}".format(
                what, name, " or ".join(table_dirs()),
                ", ".join(list_tables()) or "(none)"))
    return path


def resolve_for_model(model, dm, choice):
    """Which table this model gets -- by fingerprint, never by guessing.

    Order:
      1. `table` names a file -> that one (explicit override, nothing is
         remembered behind your back);
      2. exactly one table whose header declares `target_base == this model`;
      3. otherwise -> error.

    There is deliberately no "only table without a target" fallback. A v1
    table (or one recorded while the folder held several bases) may belong to
    a different teacher; using it would be a silent, unfixable wrong answer.
    """
    picked = "" if str(choice or "").strip() in ("", AUTO) else str(choice).strip()
    if picked:
        return find_table(picked), "picked"

    base_fp = base_fingerprint(dm)
    want = _norm_adaln_fp(_adaln_lora_fingerprint(model))
    scanned = _scan_tables()
    mine, untagged = [], []
    for path, meta in scanned:
        if _norm_adaln_fp(meta.get("adaln_lora_fingerprint", "")) != want:
            continue
        target = str(meta.get("target_base") or "")
        if not target:
            untagged.append(path)
        elif target == base_fp:
            mine.append(path)

    if len(mine) == 1:
        return mine[0], "declared for this model"
    if len(mine) > 1:
        raise RuntimeError(
            "H3AdaLN: {} tables are all declared for this model:\n  {}\n"
            "That means two teachers were recorded for the same base. Keep one, "
            "or pick the one you want in `table`.".format(
                len(mine), "\n  ".join(os.path.basename(p) for p in mine)))

    if not scanned:
        why = "the table folder is empty ({})".format(table_folder())
    elif untagged:
        why = ("{} table(s) exist but none declares a target base:\n  {}\n"
               "They were recorded without `for_model` (or are v1 tables). "
               "Re-record with for_model=<this checkpoint>; the plugin will "
               "never guess which teacher they belong to.".format(
                   len(untagged),
                   "\n  ".join(os.path.basename(p) for p in untagged)))
    else:
        # The old message blamed the LoRA fingerprint for every mismatch. In
        # practice the usual cause is target_base: the table was recorded for
        # a different (pruned) checkpoint. Show both fields so it is obvious.
        seen = ["  {} : target_base={} lora_fp={}".format(
                    os.path.basename(p),
                    str(m.get("target_base") or "(none)"),
                    _norm_adaln_fp(m.get("adaln_lora_fingerprint", "")) or "none")
                for p, m in scanned]
        why = ("{} table(s) scanned, none declares target_base={} with "
               "lora_fp={}:\n{}\n"
               "Usually the loaded checkpoint is not the base this table was "
               "recorded for (or the AdaLN/timestep LoRA stack changed)."
               .format(len(scanned), base_fp, want or "none", "\n".join(seen)))
    raise RuntimeError(
        "H3AdaLN: no table for this model (base fp {}) -- {}.\n"
        "Record one with mode=record + for_model=<this checkpoint>, or set "
        "mode=off to run without a table.".format(base_fp, why))


def _find_named(name):
    name = str(name or "").strip()
    if not name:
        return None
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None
    for folder in table_dirs():
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    try:  # subfolder / unique substring, for tables saved elsewhere
        return table_io.find_table_file(name)
    except Exception:
        return None


# ---------------------------------------------------------------- identity
def _f32_bytes(tensor, round_to_bf16=False):
    """Canonical float32 bytes, so a value read from a file and the same value
    read from the loaded module hash identically.

    `round_to_bf16` is used by the *base* fingerprint: checkpoint weights are
    stored as fp16 but ComfyUI may run them in bf16, and fp16 -> bf16 drops
    mantissa bits. Rounding both sides to bf16 precision first makes the two
    byte streams equal without hiding real differences between checkpoints.
    """
    try:
        flat = tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if round_to_bf16:
            flat = flat.to(torch.bfloat16).to(torch.float32)
        return flat.contiguous().numpy().tobytes()
    except Exception:
        return b"?"


def _weight_head(weight, n=256, round_to_bf16=False):
    """A few numbers that are stable across loads and cheap to read.

    Only the first row is touched: `adaln_proj.linear.weight` is
    [96768, 2688] on a full teacher, and on a quantised checkpoint even a
    `.reshape(-1)` can materialise gigabytes. A row is enough to tell two
    teachers apart in practice.

    A 1-D tensor (a bias) is hashed as its first `n` values, which is exactly
    what `_file_head` reads from the checkpoint.
    """
    if weight is None or not torch.is_tensor(weight):
        return b"-"
    try:
        row = weight[0] if weight.dim() >= 2 else weight
        return _f32_bytes(row[:n], round_to_bf16)
    except Exception:
        try:
            return repr((tuple(weight.shape), str(weight.dtype))).encode()
        except Exception:
            return b"?"


def _file_head(handle, key, n=256, round_to_bf16=False):
    """The same bytes `_weight_head` produces, read straight out of a file."""
    try:
        sl = handle.get_slice(key)
        vec = sl[0] if len(sl.get_shape()) >= 2 else sl[0:n]
        return _f32_bytes(vec[:n], round_to_bf16)
    except Exception:
        return b"-"


def model_fingerprint(dm):
    """Which AdaLN weights these are (fl2va full vs ref2va full vs b25-49...).

    Shapes are shared by every H3 variant, so a sample of the actual weights is
    required. Block 0, the middle block, the last block and final_layer are
    sampled: block 0 separates the two official teachers, the middle/last
    blocks separate a spliced bX-49 file from its base teacher.
    """
    cached = getattr(dm, "_h3_adaln_fp", None)
    if cached:
        return cached
    blocks = list(getattr(dm, "blocks", []) or [])
    h = hashlib.sha1()
    h.update("curve={};blocks={};".format(
        bool(getattr(dm, "use_adaln_curves", False)), len(blocks)).encode())
    picks = []
    if blocks:
        picks = [blocks[0], blocks[len(blocks) // 2], blocks[-1]]
    for block in picks:
        linear = getattr(getattr(block, "adaln_proj", None), "linear", None)
        h.update(_weight_head(getattr(linear, "weight", None)))
    final = getattr(getattr(dm, "final_layer", None), "adaln_proj", None)
    h.update(_weight_head(getattr(getattr(final, "linear", None), "weight", None)))
    fp = h.hexdigest()[:12]
    try:
        dm._h3_adaln_fp = fp
    except Exception:
        pass
    return fp


def base_fingerprint(dm):
    """Which base this table *replaces*: its own AdaLN branch.

    A pruned file carries no trace of the teacher it came from (b25-49 ships
    fl2va's `adaln_t_table`), so this cannot identify the table by itself. What
    it can do is identify the *model*, which is enough: record stores this value
    in the table's header (`target_base`), and apply recomputes it and looks the
    table up. Both sides therefore agree without any sidecar state.
    """
    cached = getattr(dm, "_h3_base_fp", None)
    if cached:
        return cached
    blocks = list(getattr(dm, "blocks", []) or [])
    h = hashlib.sha1()
    h.update("curve={};blocks={};".format(
        bool(getattr(dm, "use_adaln_curves", False)), len(blocks)).encode())
    tbl = getattr(dm, "adaln_t_table", None)
    if torch.is_tensor(tbl) and tbl.ndim == 2:
        h.update(_f32_bytes(tbl[:4], round_to_bf16=True))
    picks = [blocks[0], blocks[len(blocks) // 2], blocks[-1]] if blocks else []
    picks.append(getattr(dm, "final_layer", None))
    for owner in picks:
        linear = getattr(getattr(owner, "adaln_proj", None), "linear", None)
        h.update(_weight_head(getattr(linear, "weight", None), round_to_bf16=True))
        h.update(_weight_head(getattr(linear, "bias", None), round_to_bf16=True))
    fp = h.hexdigest()[:12]
    try:
        dm._h3_base_fp = fp
    except Exception:
        pass
    return fp


def base_fingerprint_from_file(path):
    """`base_fingerprint` computed from a checkpoint, without loading it.

    Reads the curve table plus four weight rows and four bias heads -- a few KB
    — so pointing at a 20 GB file costs one header read. The byte sequence fed
    into the hash is identical to the one `base_fingerprint` feeds, which is the
    whole point: it lets the record node stamp a table with the fingerprint of a
    model it is not running.
    """
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        blocks = 0
        while "blocks.{}.adaln_proj.linear.weight".format(blocks) in keys:
            blocks += 1
        picks = [0, blocks // 2, blocks - 1] if blocks else []
        prefixes = [("blocks.{}".format(i) if isinstance(i, int) else "final_layer")
                    for i in picks + ["final_layer"]]
        # A quantised checkpoint stores adaln_proj as int8 plus scale factors.
        # Hashing those bytes as float32 can never equal the fingerprint of the
        # loaded (dequantised) module, so the table would be stamped with a base
        # that no apply run can ever match -- and nothing would say why. Refuse.
        quantised = []
        for key in (["adaln_t_table"] if "adaln_t_table" in keys else []) + \
                [p + ".adaln_proj.linear.weight" for p in prefixes]:
            try:
                head = f.get_slice(key)[0]
            except Exception:
                continue
            if not torch.is_tensor(head) or head.is_floating_point():
                continue
            quantised.append("{} ({})".format(key, head.dtype))
        if quantised:
            raise RuntimeError(
                "H3AdaLN: {} stores the AdaLN branch quantised, not as floats "
                "({}). A file-side fingerprint would never match the loaded "
                "model, so this table could not be found by table=(auto). "
                "Use a checkpoint that keeps adaln_proj / adaln_t_table in "
                "bf16/fp16/fp32, or record with for_model=(none) and pick the "
                "table by name.".format(os.path.basename(path),
                                        ", ".join(quantised[:4])))
        h = hashlib.sha1()
        h.update("curve={};blocks={};".format(
            "adaln_t_table" in keys, blocks).encode())
        if "adaln_t_table" in keys:
            h.update(_f32_bytes(f.get_slice("adaln_t_table")[0:4],
                                round_to_bf16=True))
        for prefix in prefixes:
            h.update(_file_head(f, prefix + ".adaln_proj.linear.weight",
                                round_to_bf16=True))
            h.update(_file_head(f, prefix + ".adaln_proj.linear.bias",
                                round_to_bf16=True))
        return h.hexdigest()[:12]


def _adaln_lora_fingerprint(model):
    """Digest of the LoRA patches that can change AdaLN, or "" when there are none.

    `_patch_set_digest` still hashes the empty set, so the key count is what
    says "no AdaLN LoRA" -- otherwise every table would carry a meaningless
    suffix derived from sha256(b"").
    """
    try:
        digest, count = _nodes._adaln_lora_fingerprint(model)
    except Exception:
        return ""
    return digest if count else ""


def _is_pruned(dm):
    if dm is None:
        return False
    if bool(getattr(dm, "use_adaln_curves", False)):
        return True
    return (getattr(dm, "adaln_t_table", None) is not None
            and getattr(dm, "time_embedder", None) is None)


def _short_dtype(name):
    key = str(name or "").split(".")[-1]
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
            "float64": "fp64"}.get(key, key or "?")


def identity_of(model_fp, adaln_fp, out_dtype):
    """What a table *is*: same model + same AdaLN LoRA + same compute dtype.

    Not part of the identity, on purpose: the schedule, the shifts, the
    resolution, the seed, and any LoRA that does not touch time_embedder /
    adaln_proj. Those only decide which t rows exist, and rows accumulate.
    """
    return "{}:{}:{}".format(model_fp, adaln_fp or "base", _short_dtype(out_dtype))


SIG_VERSION = "sig2"


def runtime_signature(model, dm):
    """Environment that changes the numeric path of the teacher's AdaLN branch.

    Deliberately excludes anything derived from `dm` (the teacher and the
    pruned base have different structures), so the same value can be computed
    at record time and at apply time. It is a *staleness hint*, not a proof:
    only a live teacher re-check can prove the rows still match.

    It must also exclude everything that varies between two ordinary sessions.
    `vram_state` used to be in here, and it is a process-global decided from
    whatever VRAM happened to be free at startup -- so recording on Monday and
    applying on Tuesday reported STALE for a perfectly good table, and the
    verdict became noise. The version prefix lets an old signature be reported
    as "legacy" instead of "mismatch".
    """
    parts = [SIG_VERSION]
    for fn in (lambda: _nodes._comfy_version(),
               lambda: torch.__version__,
               lambda: str(_nodes._model_compute_dtype(model, dm)),
               lambda: str(_adaln_lora_fingerprint(model)),
               lambda: str(getattr(dm, "manual_cast_dtype", ""))):
        try:
            parts.append(str(fn()))
        except Exception:
            parts.append("?")
    # The version prefix is part of the RETURNED value, not only of the hashed
    # input: the audit compares `baked.startswith(SIG_VERSION)` to tell "this
    # table predates the current signature rules" from "this table is stale".
    return "{}-{}".format(SIG_VERSION, hashlib.sha1(
        "|".join(parts).encode()).hexdigest()[:10])


def auto_name(dm, model_fp, adaln_fp, out_dtype):
    parts = ["h3", "pruned" if _is_pruned(dm) else "full", model_fp[:6],
             _short_dtype(out_dtype)]
    if adaln_fp:
        parts.append("L" + adaln_fp[:4])
    return "_".join(parts) + TABLE_EXT


# ---------------------------------------------------------------- storage
def _find_by_identity(identity):
    """Existing table with this identity, wherever the user moved/renamed it."""
    try:
        from safetensors import safe_open
    except Exception:
        return None
    for folder in table_dirs():
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in entries:
            if not name.lower().endswith(TABLE_EXT):
                continue
            path = os.path.join(folder, name)
            try:
                with safe_open(path, framework="pt", device="cpu") as f:
                    if (f.metadata() or {}).get("identity") == identity:
                        return path
            except Exception:
                continue
    return None


def _unique_path(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(name)
    i = 2
    while True:
        candidate = os.path.join(folder, "{}_{}{}".format(stem, i, ext))
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _atomic_write(path, t_values, raw_by_key, meta, out_dtype):
    tmp = path + ".tmp"
    try:
        table_io.save_table(tmp, t_values, raw_by_key, meta, out_dtype=out_dtype)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _union(existing, t_new, raw_new, out_dtype):
    """Rows the old table does not have yet, plus a conflict check on shared t.

    Re-recording the same schedule is a no-op; recording a *different* schedule
    grows the table instead of replacing it, which is what keeps a later apply
    from failing with "t-set mismatch".
    """
    t_old, raw_old, meta_old = table_io.load_table(existing)
    if set(raw_old) != set(raw_new):
        raise RuntimeError(
            "H3AdaLN: {} has a different projection set ({} vs {} keys); refusing "
            "to mix rows.".format(os.path.basename(existing),
                                  len(raw_old), len(raw_new)))
    old_dtype = str(meta_old.get("out_dtype", ""))
    if old_dtype and out_dtype and old_dtype != out_dtype:
        raise RuntimeError(
            "H3AdaLN: {} was recorded in {} but this run produced {}; the rows are "
            "not interchangeable. Delete that file to re-record in the new "
            "dtype.".format(os.path.basename(existing), _short_dtype(old_dtype),
                            _short_dtype(out_dtype)))

    index_old = {sigma_plan.f32_image(t): i for i, t in enumerate(t_old)}
    added = []
    for i, t in enumerate(t_new):
        key_t = sigma_plan.f32_image(t)
        j = index_old.get(key_t)
        if j is None:
            added.append(i)
            continue
        for key, rows in raw_new.items():
            delta = float((raw_old[key][j].float() - rows[i].float()).abs().max())
            tol = (CONFLICT_TOL if "float32" in str(out_dtype)
                   else max(CONFLICT_TOL, LOW_PRECISION_TOL))
            if delta > tol:
                raise RuntimeError(
                    "H3AdaLN: this run produced different AdaLN values for t={:.7f} "
                    "than {} (max delta {:.3e} on {}). The model, the AdaLN LoRA or "
                    "the compute dtype must have changed. Nothing was written; "
                    "delete {} if you really want a fresh table.".format(
                        t, os.path.basename(existing), delta, key, existing))

    if not added:
        return list(t_old), raw_old, 0, meta_old

    added_pos = {sigma_plan.f32_image(t_new[i]): i for i in added}
    t_all = sorted(list(t_old) + [t_new[i] for i in added])
    raw_all = {}
    for key, rows in raw_new.items():
        stacked = []
        for t in t_all:
            key_t = sigma_plan.f32_image(t)
            j = index_old.get(key_t)
            stacked.append(raw_old[key][j] if j is not None
                           else rows[added_pos[key_t]])
        raw_all[key] = torch.stack(stacked).contiguous()
    # `raw_all` holds copies; drop the mmap-backed tensors before the caller
    # rewrites the file they came from (Windows refuses to replace an open file).
    del raw_old
    return t_all, raw_all, len(added), meta_old


def save_table_smart(t_values, raw_by_key, meta, out_dtype, name_hint):
    """Create the table, or merge this run's rows into the one already on disk.

    Returns `(path, action, rows_added, rows_total)`.
    """
    identity = meta["identity"]
    existing = _find_by_identity(identity)
    if existing:
        t_all, raw_all, added, meta_old = _union(existing, t_values, raw_by_key,
                                                out_dtype)
        if not added:
            # Same schedule -> no new t rows, but a re-record MUST still
            # refresh the provenance header. Otherwise a legacy table stays
            # legacy forever (no runtime_signature / shift_source) and the
            # VERDICT can never certify it, no matter how many times you
            # re-record. One atomic rewrite, tensor rows unchanged.
            refresh = ("runtime_signature", "shift_source", "shift_v",
                       "shift_a", "record_mode", "repeat_conflicts",
                       "max_repeat_delta",
                       # Re-recording the same teacher for a DIFFERENT base used
                       # to leave the old declaration in place (the identity does
                       # not include target_base), so the table silently kept
                       # pointing at the previous checkpoint.
                       "target_base", "target_name", "target_bases")
            if any(str(meta_old.get(k, "")) != str(meta.get(k, ""))
                   for k in refresh if k in meta):
                merged = dict(meta_old)
                for k in refresh:
                    if k in meta:
                        merged[k] = meta[k]
                merged["runs"] = str(int(meta_old.get("runs", "1") or 1) + 1)
                _atomic_write(existing, t_all, raw_all, merged, out_dtype)
                return existing, "header-refreshed", 0, len(t_all)
            return existing, "unchanged", 0, len(t_all)
        merged = dict(meta)
        merged["created"] = meta_old.get("created", meta.get("created", ""))
        merged["runs"] = str(int(meta_old.get("runs", "1") or 1) + 1)
        merged["shift_v"] = meta_old.get("shift_v", meta.get("shift_v", ""))
        merged["shift_a"] = meta_old.get("shift_a", meta.get("shift_a", ""))
        # A merge rewrites the whole header with this run's fresh meta. Keep the
        # existing base declaration unless this run explicitly declares a new
        # one, otherwise every re-record silently drops the association and
        # `table=(auto)` stops finding the table.
        for _key in ("target_base", "target_name", "target_bases"):
            _old_value = meta_old.get(_key, "")
            if _old_value and not meta.get(_key):
                merged[_key] = _old_value
        merged["t_rows"] = str(len(t_all))
        _atomic_write(existing, t_all, raw_all, merged, out_dtype)
        return existing, "merged", added, len(t_all)

    path = _unique_path(table_folder(), name_hint)
    _atomic_write(path, t_values, raw_by_key, meta, out_dtype)
    return path, "saved", len(t_values), len(t_values)


# ---------------------------------------------------------------- runtime
def _clear_runtime(model, seal=True):
    """Detach (and by default release) every runtime this model still carries."""
    rt = getattr(model, "_h3_exact_runtime", None)
    if rt is not None:
        try:
            rt.seal() if seal else rt.remove()
        except Exception:
            pass
        model._h3_exact_runtime = None
    dm = _nodes._dm_of(model)
    for obj in list(getattr(dm, "_h3_exact_runtimes", []) or []):
        if obj is rt:
            continue
        try:
            obj.seal() if seal else obj.remove()
        except Exception:
            pass
    for obj in list(getattr(dm, "_h3_exact_runtimes", []) or []) + ([rt] if rt else []):
        try:
            _LIVE_RUNTIMES.remove(obj)
        except ValueError:
            pass
    _nodes._clear_registered(model, "_h3_exact_runtimes")


def _set_callback_once(model, key, fn):
    """Register exactly ONE ON_CLEANUP callback under `key`, replacing the old.

    ComfyUI's `add_callback` only appends and nothing ever removes, so a fresh
    closure per apply meant four applies in one session left four callbacks --
    each holding an ExactAdaLNRuntime with its ~335 MB host copy of the table,
    and each printing a stale verdict on every later run. Writing into the
    callback dict by key is what ComfyUI itself does; `add_callback` stays as
    the fallback for builds that do not expose `callbacks`.
    """
    try:
        import comfy.patcher_extension as pe
        bucket = model.callbacks.setdefault(pe.CallbacksMP.ON_CLEANUP, {})
        bucket[key] = [fn]
        return True
    except Exception:
        try:
            model.add_callback(pe.CallbacksMP.ON_CLEANUP, fn)
            return False
        except Exception:
            return False


def _pass_signature(runtime):
    """What a single pass looked like -- used to diff pass 1 against pass 2.

    The row identity has to come from the trace: `sigma` / `t` are only
    refreshed by the DIFFUSION_MODEL wrapper, and a custom sampler that calls
    the model directly never runs it. Comparing `t_emb.shape[0]` instead (the
    old `rows` field, always 3) made such a pass look identical to the one
    before it while its table rows had in fact changed. `sigma` / `t` follow
    the matched row (1 - t) when there is one, so a frozen wrapper value cannot
    leak into the diff.
    """
    st = runtime.state
    shift = st.get("last_shift")
    last = {}
    _start, _end = getattr(runtime, "_trace_slice", None) or (0, len(runtime.trace))
    if _end > _start:
        last = runtime.trace[_end - 1]
    indices = last.get("indices")
    t_hit = last.get("t_hit")
    if t_hit:
        # A matched row carries the real t of that forward; sigma is 1 - t by
        # construction. The wrapper's own values are only trustworthy when it
        # ran, and a custom sampler never calls it.
        t = tuple(t_hit)
        sigma = round(1.0 - float(t_hit[0]), 7)
    else:
        t = tuple(round(x, 7) for x in (st.get("last_t") or ()))
        sigma = st.get("last_sigma")
    return {
        "sigma": sigma,
        "t": t,
        "indices": tuple(indices) if indices is not None else None,
        "tokens": st.get("last_tokens"),
        "segs": st.get("last_segs"),
        "shift": shift,
        "width": last.get("width") or st.get("last_emb_width"),
    }


def _describe_pass(sig):
    if not sig:
        return "none"
    shift = sig.get("shift")
    return ("sigma={} indices={} t={} tokens={} segs={} shift={}/{}".format(
        "{:.7f}".format(sig["sigma"]) if isinstance(sig["sigma"], float) else "?",
        sig.get("indices"),
        ",".join("{:.7f}".format(t) for t in sig["t"][:6]) or "?",
        sig["tokens"], sig["segs"],
        "{:g}".format(shift[0]) if shift else "?",
        "{:g}".format(shift[1]) if shift else "?"))


def _audit(runtime, table, delta, model, dm, lora_mismatch=False, pass_no=1):
    """Audit one sampling pass. Returns (lines, summary dict)."""
    st = runtime.state
    injections, matched, failures, native, wrapper = delta
    missing = sorted(st["missing"])
    meta = table.meta or {}
    sampled = injections > 0 or wrapper > 0 or native > 0
    measured = bool(st.get("injection_delta_measured"))
    # `missing` comes from the sigma-reconstruction fallback, which rebuilds t
    # from `timestep / 1000`. That division rounds differently from the way the
    # DiT derived t (`1 - sigma` on the schedule's own float32), so a run whose
    # t_emb match was EXACT can still produce a handful of "missing" values that
    # differ in the 7th decimal. Treating those as an injection failure made a
    # perfect 408/408 pass report FAIL and NOT ESTABLISHED.
    real_gap = (failures > 0) or (native > 0)

    injection_problems = []
    superseded = bool(getattr(runtime, "superseded", False))
    if not sampled and superseded:
        # Another Apply node on the same diffusion model took this runtime's
        # hooks away on purpose. Reporting FAIL here was a false alarm: the
        # node that took over did inject, and its own report says so.
        injection_state = "N/A (superseded by another Apply node)"
    elif not sampled:
        # Not "no sampling": this runtime cannot tell. What it does know is that
        # none of its hooks (or its own DIFFUSION_MODEL wrapper) ever saw a
        # forward. On a real run that means the chain sampled a different model
        # object -- a rebuilt ModelPatcher, or a checkpoint instantiated inside
        # the custom sampling node -- and AdaLN was NOT injected.
        injection_state = "FAIL (no forward reached this runtime)"
        injection_problems.append(
            "this runtime saw zero forwards. If this chain really sampled, it "
            "did not sample the model this Apply node is attached to (a "
            "rebuilt patcher or a checkpoint reloaded inside a custom sampler "
            "has no hooks), so AdaLN was NOT injected for this pass")
    else:
        if injections == 0:
            injection_problems.append("no table row was injected this pass")
        if native:
            injection_problems.append(
                "{} step(s) used the base's native AdaLN".format(native))
        if failures:
            injection_problems.append(
                "{} hook call(s) could not be matched to a table row"
                .format(failures))
        if missing and real_gap:
            injection_problems.append(
                "{} timestep(s) absent from the table".format(len(missing)))
        if lora_mismatch:
            injection_problems.append(
                "table was recorded under a different AdaLN LoRA")
        injection_state = "PASS" if not injection_problems else "FAIL"
    injection_ok = sampled and not injection_problems

    provenance_problems = []
    record_mode = str(meta.get("record_mode", ""))
    merged = record_mode.startswith("merged from two recorded tables")
    if record_mode != "observed during sampling" and not merged:
        provenance_problems.append(
            "table provenance is {!r}, not an observed teacher run".format(record_mode))
    if str(meta.get("repeat_conflicts", "0")) != "0":
        provenance_problems.append("recording had repeat conflicts")
    try:
        if float(meta.get("max_repeat_delta", 0) or 0) > 0.0:
            provenance_problems.append("recording was non-deterministic")
    except (TypeError, ValueError):
        pass

    baked_sig = str(meta.get("runtime_signature", "") or "")
    current_sig = runtime_signature(model, dm)
    if not baked_sig:
        runtime_state = "UNKNOWN (table predates runtime signatures)"
        provenance_problems.append("no runtime signature -- equivalence not provable")
    elif not baked_sig.startswith(SIG_VERSION):
        runtime_state = "LEGACY ({}, current {})".format(baked_sig, current_sig)
        provenance_problems.append(
            "table carries a pre-{} signature; re-record to certify it".format(SIG_VERSION))
    elif baked_sig != current_sig:
        runtime_state = "STALE (table={} now={})".format(baked_sig, current_sig)
        provenance_problems.append("table was recorded under a different runtime")
    else:
        runtime_state = "MATCH ({})".format(baked_sig)

    shift_note = "header {}/{} (source: {})".format(
        meta.get("shift_v", "?"), meta.get("shift_a", "?"),
        meta.get("shift_source", "unknown"))
    if meta.get("shift_source") not in (None, "transformer_options"):
        shift_note += " -- NOT authoritative, the header shift is a fallback"

    provenance_ok = not provenance_problems
    if not sampled and superseded:
        equivalence = ("NOT EVALUATED -- this runtime was superseded by another "
                       "Apply node on the same model; that node did the injection")
    elif not injection_ok:
        equivalence = ("NOT ESTABLISHED -- the table did not replace AdaLN in "
                       "this pass ({})".format(
                           "; ".join(injection_problems[:2]) or "see above"))
    elif provenance_ok:
        equivalence = ("ESTABLISHED BY CONSTRUCTION: every step injected a row "
                       "observed from the teacher, under a matching runtime")
    else:
        equivalence = ("NOT PROVEN -- {}; re-record on this exact runtime to "
                       "certify it".format("; ".join(provenance_problems)))

    def _fmt(items):
        return ", ".join(items) if items else "none"

    lines = [
        "INJECTION   : {} | injected={} matched={} failures={} native={} "
        "wrapper_steps={}".format(injection_state, injections, matched,
                                  failures, native, wrapper),
        "  problems  : {}".format(_fmt(injection_problems)),
        "  max delta : {}".format(
            "{:.6e} (cumulative, table row vs base AdaLN)".format(
                st["max_injection_delta"]) if measured
            else "UNAVAILABLE (could not read the base output)"),
        "  match     : exact={} approx={} max_dist={:.3e} first={} @ {}".format(
            st["match_hits"] - st["match_approx"], st["match_approx"],
            st["max_match_distance"], st["first_key"],
            "?" if st["first_dist"] is None else "{:.3e}".format(st["first_dist"])),
        "  layout    : {} segs / {} tokens, {} change(s); shift changes={}".format(
            st["last_segs"], st["last_tokens"], st["layout_changes"],
            st["shift_changes"]),
        "  per-row t : {}".format(
            "YES -- denoise_mask; only the t_emb match is reliable"
            if st["per_row_timestep"] else "no"),
        "  recon     : {}".format(
            "clean" if not missing else
            ("{} t value(s) the sigma rebuild could not place, but every t_emb "
             "row matched exactly -- timestep/1000 rounding, not a gap"
             .format(len(missing)) if not real_gap else
             "{} t value(s) missing from the table".format(len(missing)))),
        "PROVENANCE  : {} | mode={} conflicts={}".format(
            "OK" if provenance_ok else "DEGRADED", record_mode or "?",
            meta.get("repeat_conflicts", "?")),
        "  problems  : {}".format(_fmt(provenance_problems)),
        "RUNTIME     : {}".format(runtime_state),
        "SHIFT META  : {}".format(shift_note),
        "EQUIVALENCE : {}".format(equivalence),
    ]
    if injections and measured and st["max_injection_delta"] <= 0.0:
        lines.append(
            "NOTE        : the injected rows are bit-identical to this model's own "
            "AdaLN -- expected when the table was recorded from THIS model, "
            "suspicious when it was recorded from a teacher")
    if lora_mismatch:
        lines.insert(0, "LORA        : MISMATCH -- table fp={} model fp={} "
                        "(explicit `table` pick, injection allowed)".format(
                            meta.get("adaln_lora_fingerprint") or "none",
                            _adaln_lora_fingerprint(model) or "none"))
    verdict = ("SUPERSEDED" if (not sampled and superseded)
               else "PASS" if (injection_ok and provenance_ok) else "FAIL")
    summary = {
        "ok": bool(injection_ok and provenance_ok),
        "verdict": verdict,
        "injected": injections, "native": native, "missing": len(missing),
        "equivalence": equivalence,
    }
    return lines, summary


def _report_pass(runtime, model, dm, table, rep, lora_mismatch=False,
                 deferred=False):
    """Emit the audit block for the counters since the previous report.

    Called from ON_CLEANUP (once per sampling pass) and, with `deferred=True`,
    from the watchdog for a pass whose cleanup never reached us -- a chain that
    samples with a patcher we are not attached to leaves its numbers unclaimed.
    """
    delta = runtime.begin_pass()
    pass_no = runtime._report_count
    prev_sig = runtime._pass_sig
    sig = _pass_signature(runtime)
    runtime._pass_sig = sig
    try:
        section = "pass {}{}".format(pass_no, " (deferred)" if deferred else "")
        lines, summary = _audit(runtime, table, delta, model, dm,
                                lora_mismatch, pass_no)
        if delta[4] == 0 and delta[0] > 0:
            # This pass injected, but never went through the DIFFUSION_MODEL
            # wrapper. Everything in `state` that only the wrapper writes is
            # therefore left over from the previous pass -- say so instead of
            # presenting a frozen sigma as this pass's schedule.
            lines.append(
                "  wrapper   : 0 calls this pass -- sigma / t / token above are "
                "the last values the wrapper saw, NOT this pass's; the row "
                "identity comes from the t_emb match")
        if prev_sig is not None:
            changed = [k for k in ("sigma", "t", "indices", "tokens", "segs",
                                   "shift", "width")
                       if prev_sig.get(k) != sig.get(k)]
            lines.append("PASS DIFF   : {} vs pass {} -> changed: {}".format(
                "identical" if not changed else "DIFFERENT",
                pass_no - 1, ", ".join(changed) or "nothing"))
            lines.append("  pass {}: {}".format(pass_no - 1, _describe_pass(prev_sig)))
            lines.append("  pass {}: {}".format(pass_no, _describe_pass(sig)))
        else:
            lines.append("PASS DIFF   : first pass in this prompt ({})".format(
                _describe_pass(sig)))
        rep.add(section, "this pass     : injections={} matched={} "
                         "failures={} native={} wrapper_steps={}".format(*delta))
        rep.extend(section, lines)
        trace_section = "{} · trace".format(section)
        counters_section = "{} · counters".format(section)
        if rep.debug:
            rep.extend(trace_section, runtime.trace_lines())
            rep.extend(counters_section, runtime.summary().splitlines())
        if pass_no == 1:
            rep.add("summary", "{} : {}".format(
                summary.get("verdict") or
                ("PASS" if summary["ok"] else "FAIL"),
                summary["equivalence"]))
        rep.log_section(section)
        rep.log_section(trace_section)
        rep.log_section(counters_section)
        # The reconstruction's "missing" set is per-pass: keeping the union
        # would attribute pass 1's rounding noise to a genuine pass-2 gap.
        runtime.state["missing"] = set()
    except Exception as e:  # never let reporting break the prompt
        logging.warning("H3AdaLN: could not build the apply report (%s: %s)",
                        type(e).__name__, e)
    finally:
        # ON_CLEANUP fires after the FIRST sampling pass, but a director chain
        # then runs a SECOND pass (refine / SelfLift / custom sigmas) with a
        # different t set. Removing the hooks here silently drops that pass from
        # injection and stats. Keep them alive until the next prompt re-arms via
        # _clear_runtime. The malloc-graph repair only makes sense on the
        # sampler thread, never from the watchdog.
        if not deferred:
            apply_engine.recover_stale_malloc_graph()


def _inject(model, dm, table, rep=None, source="", allow_lora_mismatch=False):
    """Replace all 51 AdaLN projections with the table's rows."""
    if rep is None:
        rep = _nodes.Report("H3 AdaLN · apply", "silent")
    table.finalize(required=True)

    baked = _norm_adaln_fp(table.meta.get("adaln_lora_fingerprint", ""))
    current = _norm_adaln_fp(_adaln_lora_fingerprint(model))
    lora_mismatch = baked != current
    if lora_mismatch and not allow_lora_mismatch:
        raise RuntimeError(
            "H3AdaLN: this table was recorded with a different AdaLN/timestep "
            "LoRA (table={} model={}). Only time_embedder / adaln_proj LoRAs "
            "matter -- re-record on the full teacher with that LoRA loaded, or "
            "set `table` to a specific file to override this check."
            .format(baked or "none", current or "none"))
    if lora_mismatch:
        # Explicit override. Never silent: the report and the verdict both say
        # so, and the rows are known to belong to a different modulation.
        logging.warning(
            "H3AdaLN: injecting a table recorded with AdaLN LoRA %s into a "
            "model with %s -- allowed because `table` was picked explicitly, "
            "but the output will not match either side.",
            baked or "none", current or "none")

    warnings = []
    model_dtype = _nodes._model_compute_dtype(model, dm)
    if model_dtype is not None:
        want = _short_dtype(table.out_dtype)
        got = _short_dtype(model_dtype)
        if want != got:
            warnings.append("table is {} but this model computes in {} -- the "
                            "result will NOT be bitwise".format(want, got))

    # Two Apply nodes on one shared diffusion model: only the last armed one
    # ever injects, and the other one silently becomes a no-op that still
    # reports "this pass saw no sampling". Say it instead of hiding it.
    other = [r for r in (getattr(dm, "_h3_exact_runtimes", None) or [])
             if getattr(r, "hook_handles", None)
             and getattr(r, "_wrapper_model", model) is not model]
    if other:
        msg = ("another Apply node was already armed on this shared diffusion "
               "model; its hooks are detached now, so only this node injects. "
               "Two Apply nodes on one model cannot both run.")
        rep.add("exec", "TOOK OVER   : " + msg)
        logging.warning("H3AdaLN: %s", msg)
        for r in other:
            # Its ON_CLEANUP still fires (its patcher is cleaned as usual), and
            # without this flag it would report "zero forwards -> FAIL" for a
            # model that the node in front of it is injecting correctly.
            try:
                r.superseded = True
            except Exception:
                pass
    _clear_runtime(model)
    runtime = apply_engine.ExactAdaLNRuntime(
        table.t_values, table.raw_by_key, table.meta, dm, mode="strict")
    runtime.verbose = rep.debug
    hooked = runtime.register(model)
    _nodes._track_registered(model, "_h3_exact_runtimes", runtime)
    model._h3_exact_runtime = runtime

    def _done(*_a, **_k):
        _report_pass(runtime, model, dm, table, rep, lora_mismatch)

    runtime.pending_report = (
        lambda: _report_pass(runtime, model, dm, table, rep, lora_mismatch,
                             deferred=True))
    _LIVE_RUNTIMES.append(runtime)
    _ensure_autosave()
    _set_callback_once(model, "h3_adaln_apply", _done)

    rep.add("exec", "source      : {}".format(source or "(table_in)"))
    rep.add("exec", "table       : {}".format(
        os.path.basename(table.source or "") or "(in memory)"))
    rep.add("exec", "identity    : {}".format(table.meta.get("identity", "?")))
    rep.add("exec", "t rows      : {} {}".format(
        len(table.t_values),
        "[" + ", ".join("{:.7f}".format(t) for t in table.t_values[:8]) +
        (", ..." if len(table.t_values) > 8 else "") + "]"))
    rep.add("exec", "projections : {}".format(table.projections()))
    rep.add("exec", "dtype       : table {} / model {}".format(
        _short_dtype(table.out_dtype),
        _short_dtype(model_dtype) if model_dtype is not None else "unknown"))
    rep.add("exec", "lora fp     : table {} / model {} {}".format(
        baked or "none", current or "none",
        "<- MISMATCH (explicit pick)" if lora_mismatch else ""))
    rep.add("exec", "target base : {} ({})".format(
        table.meta.get("target_base") or "none",
        table.meta.get("target_name") or "-"))
    rep.add("exec", "signature   : table {} / now {}".format(
        table.meta.get("runtime_signature") or "none", runtime_signature(model, dm)))
    rep.add("exec", "shift       : {}/{} [{}]".format(
        table.meta.get("shift_v", "?"), table.meta.get("shift_a", "?"),
        table.meta.get("shift_source", "unknown")))
    rep.add("exec", "hooked      : {} projections armed".format(hooked))
    for w in warnings:
        rep.add("exec", "WARNING     : {}".format(w))
        logging.warning("H3AdaLN: %s", w)

    lines = [
        "  table  : {} · {} t rows · {} projections · {} {}".format(
            os.path.basename(table.source or "") or "(in memory)",
            len(table.t_values), table.projections(),
            _short_dtype(table.out_dtype),
            "· LoRA MISMATCH" if lora_mismatch else ""),
        "  armed  : {} projections; per-pass stats are in the report".format(hooked),
    ] + ["  " + n for n in warnings]
    return lines



# ---------------------------------------------------------------- record
# The target base is always an explicit `for_model` choice. There is
# deliberately no directory scan and no remembered binding: when the model
# folder holds several similar pruned checkpoints, the node cannot know which
# one is meant, and a hidden guess can silently produce a table that answers
# wrongly. The choice is saved with the workflow, so it is made only once.


def target_from_choice(choice):
    """(base fingerprint, file name) for a `for_model` value, or None."""
    choice = str(choice or NONE).strip()
    if choice in ("", NONE, AUTO):
        legacy = ("(auto) is no longer supported; " if choice == AUTO else "")
        logging.warning(
            "H3AdaLN: no for_model was set (%spick the pruned base in the "
            "for_model dropdown). This table carries no target_base, so "
            "table=(auto) will NOT find it -- select it in `table` instead, "
            "or record again with for_model set.", legacy)
        return None

    path = resolve_model_file(choice)
    if not path:
        logging.warning("H3AdaLN: for_model=%r does not resolve to a "
                        "checkpoint; the table will not carry a base "
                        "fingerprint", choice)
        return None

    try:
        return (base_fingerprint_from_file(path), os.path.basename(path))
    except Exception as e:
        logging.warning("H3AdaLN: cannot fingerprint %s (%s: %s); the table "
                        "will not carry a base fingerprint", path,
                        type(e).__name__, e)
        return None


def _target_of(model):
    """(base fingerprint, file name) this table is for, or None.

    `for_model` is an explicit choice -- there is no auto mode. Without one the
    table carries no `target_base`, and `table=(auto)` at apply time will refuse
    to use it rather than guess which teacher it belongs to. The value is saved
    with the workflow, so it only has to be chosen once.
    """
    cached = getattr(model, "_h3_rec_target", None)
    if cached is not None:
        return cached or None
    target = target_from_choice(getattr(model, "_h3_rec_for_model", NONE))
    model._h3_rec_target = target or ()
    return target


# One live recorder per record-node instance (UNIQUE_ID), not one per model.
# Two nodes in one prompt must capture at the same time: they share the
# diffusion module, and clearing "every recorder on the dm" made the last node
# evict the others. The id is stable across prompts for the same node, which is
# exactly what stale-hook cleanup needs.
_ACTIVE_RECORDERS = {}


def _stop_recording(model):
    """Drop this node's read-only hooks and its registry entry."""
    recorder = getattr(model, "_h3_recorder", None)
    uid = str(getattr(model, "_h3_rec_uid", "") or "")
    if recorder is not None:
        try:
            recorder.remove()
        except Exception:
            pass
        try:
            _LIVE_RECORDERS.remove(recorder)
        except ValueError:
            pass
        model._h3_recorder = None
    if uid:
        _ACTIVE_RECORDERS.pop(uid, None)
    try:
        model._h3_rec_uid = ""
    except Exception:
        pass
    # Drop the registrations we keep on the *shared* diffusion module. The
    # hooks are gone after remove(), but leaving the objects in these buckets
    # makes a later prompt look like it still has live recorders.
    dm = getattr(model, "_h3_rec_dm", None)
    if dm is not None:
        try:
            dm._h3_recorders = []
        except Exception:
            pass
        try:
            active = getattr(dm, "_h3_active_recorders", None)
            if isinstance(active, dict) and uid:
                active.pop(uid, None)
        except Exception:
            pass


def _finish_record(model):
    """ON_CLEANUP: merge this pass's rows into the table, keep the hooks.

    A director chain fires ON_CLEANUP after the first sampling pass and then
    runs a refine / SelfLift pass with its own custom sigmas. Removing the
    recorder here silently drops those t rows, and sealing the *save* after the
    first pass does the same: the rows are captured but never written. So every
    pass saves what is new, and the recorder only detaches when a pass produced
    nothing new (i.e. the sampling that just ended was not part of this
    recording) or the pass budget is exhausted.
    """
    rec = getattr(model, "_h3_recorder", None)
    if rec is not None and getattr(rec, "abandoned", False):
        _stop_recording(model)
        return
    _save_recorded(model, seal_when_idle=True)
    # Keep the hook handles for later passes, but clear the tracking bucket so
    # the shared module does not accumulate stale objects.
    dm = getattr(model, "_h3_rec_dm", None)
    if dm is not None:
        try:
            dm._h3_recorders = []
        except Exception:
            pass


def _save_recorded(model, seal_when_idle=False, quiet=False):
    """Merge this run's rows into the table. Re-entrancy guarded.

    Called from ON_CLEANUP (main thread) and from the deferred-save watchdog
    (background thread), so it must never run twice at once.
    """
    if getattr(model, "_h3_saving", False):
        return
    model._h3_saving = True
    try:
        _save_recorded_inner(model, seal_when_idle, quiet)
    finally:
        model._h3_saving = False


def _save_recorded_inner(model, seal_when_idle=False, quiet=False):
    table = getattr(model, "_h3_pending_table", None)
    if table is None:
        return
    rec = table.recorder
    rep = getattr(model, "_h3_report", None)
    if rec is None or getattr(rec, "sealed", False):
        return
    if getattr(model, "_h3_recorder", None) is not rec:
        return

    rows_now = len(getattr(rec, "rows", {}) or {})
    saved_rows = int(getattr(model, "_h3_saved_rows", 0) or 0)
    if rows_now <= saved_rows:
        if seal_when_idle and saved_rows:
            # This sampling run added nothing: it is either a repeat pass of the
            # same prompt (nothing to write) or a later prompt reusing the same
            # model. Either way the recording is finished -- detach, so the
            # hooks stop filling a dict nobody will ever save.
            _seal_recorder(model, rep,
                           "a sampling pass produced no new rows")
        return

    try:
        table.finalize(required=True)
    except Exception as e:
        if quiet:
            # A watchdog flush can land between two projections of one forward;
            # the row sets are inconsistent for a moment and build() refuses.
            # Nothing is wrong, and the next tick will try again.
            logging.info("H3AdaLN: deferred save skipped -- %s", e)
            return
        logging.error("H3AdaLN: nothing saved -- %s", e)
        if rep is not None:
            rep.add("record", "SAVE FAILED : {}".format(e))
        return

    conflicts = int(table.meta.get("repeat_conflicts", "0") or 0)
    if conflicts:
        logging.error(
            "H3AdaLN: refusing to save. %d timestep(s) produced two different "
            "modulations in one run (max delta %s), so the AdaLN branch changed "
            "mid-run -- record again with one model and one LoRA stack.",
            conflicts, table.meta.get("max_repeat_delta", "?"))
        if rep is not None:
            rep.add("record", "REFUSED     : {} repeat conflict(s), max delta {} "
                              "-- the AdaLN branch changed mid-run; nothing written"
                              .format(conflicts, table.meta.get("max_repeat_delta", "?")))
            rep.extend("record · conflicts", [
                "t={:.7f}  {}  delta={:.3e}".format(t, k, d)
                for t, k, d in rec.conflict_detail])
            rep.log_section("record")
            rep.log_section("record · conflicts")
        _seal_recorder(model, rep, "repeat conflicts made this run unusable")
        return

    dm = getattr(model, "_h3_rec_dm", None)
    model_fp = getattr(model, "_h3_rec_fp", None) or model_fingerprint(dm)
    adaln_fp = getattr(model, "_h3_rec_adaln_fp", "")
    out_dtype = table.out_dtype
    table.meta.update({
        "identity": identity_of(model_fp, adaln_fp, out_dtype),
        "model_fingerprint": model_fp,
        "adaln_lora_fingerprint": adaln_fp or "",
        "teacher": "full fp {} ({} projections)".format(
            model_fp, table.projections()),
        "t_rows": str(len(table.t_values)),
        "created": table.meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": "1",
        "comfyui_version": _nodes._comfy_version(),
    })
    target = _target_of(model)
    if target:
        table.meta["target_base"] = target[0]
        table.meta["target_name"] = target[1]

    try:
        path, action, added, total = save_table_smart(
            table.t_values, table.raw_by_key, table.meta, out_dtype,
            auto_name(dm, model_fp, adaln_fp, out_dtype))
    except Exception as e:
        logging.error("H3AdaLN: save failed -- %s: %s", type(e).__name__, e)
        if rep is not None:
            rep.add("record", "SAVE FAILED : {}: {}".format(type(e).__name__, e))
        _seal_recorder(model, rep, "the save failed")
        return

    # Count AFTER finalize, not before. `finalize()` folds the pre-computed
    # reference rows into `rec.rows`, so the pre-build count is short by the
    # number of pins -- the watchdog then saw "2 rows never saved", re-read the
    # whole 462 MiB file just to answer "unchanged", and printed a phantom
    # second pass.
    model._h3_saved_rows = len(getattr(rec, "rows", {}) or {})
    model._h3_save_passes = int(getattr(model, "_h3_save_passes", 0) or 0) + 1
    rec.saved_once = True
    rec.passes = model._h3_save_passes
    table.source = path
    logging.info("H3AdaLN: %s %s (%d new rows, %d total, %.0f MiB)", action, path,
                 added, total, os.path.getsize(path) / (1 << 20))
    if total > 200:
        logging.warning(
            "H3AdaLN: %s now holds %d t rows (~%.1f GiB). Every extra schedule "
            "adds rows; delete the file and re-record if it gets out of hand.",
            os.path.basename(path), total, os.path.getsize(path) / (1 << 30))

    if rep is not None:
        section = "pass {}".format(model._h3_save_passes)
        rep.add(section, "record      : {} {}".format(action, os.path.basename(path)))
        rep.add(section, "this pass   : +{} rows ({} total, {:.0f} MiB)".format(
            added, total, os.path.getsize(path) / (1 << 20)))
        rep.add(section, "table       : {} t rows / {} projections / {}".format(
            len(table.t_values), table.projections(), _short_dtype(out_dtype)))
        rep.add(section, "t values    : {}".format(
            ", ".join("{:.7f}".format(t) for t in table.t_values[:12]) +
            (" ... ({} more)".format(len(table.t_values) - 12)
             if len(table.t_values) > 12 else "")))
        rep.add(section, "pins        : {}".format(rec.pins_report or "-"))
        rep.add(section, "forwards    : {} seen, {} compared, {} skipped".format(
            rec.forwards, rec.compared, rec.skipped))
        rep.add(section, "conflicts   : {} (max delta {:.3e})".format(
            rec.conflicts, rec.max_repeat_delta))
        if rec.warnings:
            rep.extend(section, ["WARNING     : " + w for w in rec.warnings[-6:]])
        if rep.debug and rec.forward_trace:
            rep.extend("{} · forwards".format(section), [
                "forward {:<3} rows={} t={}".format(i, n, ", ".join(
                    "{:.7f}".format(v) for v in ts))
                for i, ts, n in rec.forward_trace])
        rep.log_section(section)
        rep.log_section("{} · forwards".format(section))

    pin = sigma_plan.f32_image(sigma_plan.VISUAL_COND_TIMESTEP)
    saved_t = table.t_values
    try:
        saved_t = table_io.load_table(path)[0]
    except Exception:
        pass
    if not any(abs(t - pin) < 1e-6 for t in saved_t):
        msg = ("{} has no row for t={:.3f} after this run (checked the merged "
               "file, not just this run's rows). Keyframe/reference workflows "
               "will error before sampling.").format(os.path.basename(path), pin)
        if rec.pin_targets:
            logging.error("H3AdaLN: %s", msg)
            if rep is not None:
                rep.add("record", "WARNING     : " + msg)
        else:
            # fill_reference_row=False is a documented choice; reporting it as an
            # ERROR made every text-only recording look like a failure.
            logging.info("H3AdaLN: %s (fill_reference_row=False, so this is "
                         "expected)", msg)
            if rep is not None:
                rep.add("record", "note        : " + msg +
                        " -- expected, fill_reference_row=False")

    if seal_when_idle and added == 0:
        _seal_recorder(model, rep, "this pass added no rows")
    elif model._h3_save_passes >= MAX_RECORD_PASSES:
        _seal_recorder(model, rep, "pass budget reached ({} saves)"
                       .format(model._h3_save_passes))


def _seal_recorder(model, rep, why):
    """Detach the recorder for good and say why."""
    rec = getattr(model, "_h3_recorder", None)
    if rec is None:
        return
    passes = int(getattr(model, "_h3_save_passes", 0) or 0)
    if passes >= MAX_RECORD_PASSES:
        why = "pass budget reached ({} saves)".format(passes)
    try:
        rec.seal()
    except Exception:
        pass
    model._h3_recorder = None
    try:
        _LIVE_RECORDERS.remove(rec)
    except ValueError:
        pass
    uid = str(getattr(model, "_h3_rec_uid", "") or "")
    if uid:
        _ACTIVE_RECORDERS.pop(uid, None)
    dm = getattr(model, "_h3_rec_dm", None)
    if dm is not None:
        try:
            active = getattr(dm, "_h3_active_recorders", None)
            if isinstance(active, dict) and uid:
                active.pop(uid, None)
        except Exception:
            pass
    if rep is not None:
        rep.add("record", "sealed      : {} -- hooks detached; a new record run "
                          "re-arms automatically".format(why))
    logging.info("H3AdaLN: recording closed (%s)", why)


# ---------------------------------------------------------------- deferred save
# ON_CLEANUP is the only save trigger, and it comes from
# `CFGGuider.outer_sample`'s finally block. A custom node that samples on its
# own (a director "second pass" / SelfLift step) never goes through that path,
# so its timesteps are captured into the recorder and then sit there forever:
# observed on a real 4080 run, where the refine pass added 8 t rows that never
# reached the table.
#
# The watchdog below closes that hole without hooking anything global: as soon
# as the recorder has been quiet for AUTOSAVE_IDLE seconds with rows that are
# not on disk yet, they are merged in. It only touches CPU tensors and file I/O,
# never CUDA, and `_save_recorded` is re-entrancy guarded, so a flush can never
# race an in-flight save.
AUTOSAVE_IDLE = 60.0
AUTOSAVE_TICK = 5.0
_AUTOSAVE = {"thread": None, "lock": threading.Lock()}
# Every recorder that is still capturing, independent of the UNIQUE_ID registry
# (a headless / scripted `run()` has no unique_id, and the watchdog must still
# find it).
_LIVE_RECORDERS = []
# Same idea for apply runtimes: needed to emit a report for a pass whose
# ON_CLEANUP never fired.
_LIVE_RUNTIMES = []


def _ensure_autosave():
    if _AUTOSAVE["thread"] is not None:
        return
    with _AUTOSAVE["lock"]:
        if _AUTOSAVE["thread"] is not None:
            return
        thread = threading.Thread(target=_autosave_loop,
                                  name="h3-adaln-autosave", daemon=True)
        _AUTOSAVE["thread"] = thread
        thread.start()


def _autosave_loop():
    while True:
        time.sleep(AUTOSAVE_TICK)
        try:
            _autosave_tick()
        except Exception:  # a watchdog must never take the process down
            continue


def _autosave_tick():
    now = time.time()
    for rec in list(_LIVE_RECORDERS):
        if rec is None or getattr(rec, "sealed", False) or getattr(rec, "abandoned", False):
            try:
                _LIVE_RECORDERS.remove(rec)
            except ValueError:
                pass
            continue
        model = getattr(rec, "model", None)
        if model is None or getattr(model, "_h3_recorder", None) is not rec:
            continue
        rows = len(getattr(rec, "rows", {}) or {})
        if rows <= int(getattr(model, "_h3_saved_rows", 0) or 0):
            continue
        last = float(getattr(rec, "last_row_at", 0.0) or 0.0)
        if now - last < AUTOSAVE_IDLE:
            continue
        if getattr(model, "_h3_saving", False):
            continue
        logging.info("H3AdaLN: %d row(s) were captured but never saved (the "
                     "sampling path did not fire ON_CLEANUP) -- merging them now",
                     rows - int(getattr(model, "_h3_saved_rows", 0) or 0))
        _save_recorded(model, seal_when_idle=False, quiet=True)

    # Apply side: a pass whose ON_CLEANUP never reached us still has counters
    # worth reporting. Only after the same idle window, so a run in progress is
    # never split in half.
    for rt in list(_LIVE_RUNTIMES):
        report = getattr(rt, "pending_report", None)
        if report is None or getattr(rt, "superseded", False):
            continue
        st = rt.state
        snap = (st["injected"], st["match_hits"], st["match_failures"],
                st["native_steps"], st["wrapper_steps"])
        if snap == getattr(rt, "_last_report", (0, 0, 0, 0, 0)):
            continue
        if now - float(st.get("last_hook_at", 0.0) or 0.0) < AUTOSAVE_IDLE:
            continue
        logging.info("H3AdaLN: a sampling pass ended without ON_CLEANUP -- "
                     "reporting its counters now")
        try:
            report()
        except Exception:
            continue


# ---------------------------------------------------------------- nodes
class H3AdaLN:
    """Record a table while the full teacher samples, or apply one to a pruned base."""

    # ComfyUI caches node outputs by input signature. Record has side effects
    # (hooks + ON_CLEANUP), so it must execute on every prompt even when its
    # inputs look unchanged; otherwise a re-run silently produces no table.
    # NOT_IDEMPOTENT only appends the node id to the cache signature
    # (comfy_execution/caching.py:118); IS_CHANGED is what invalidates the
    # cache across prompts. Record/apply have side effects (hooks, ON_CLEANUP,
    # injection), so neither may be served from a cached output.
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    MODES = ["apply", "record", "off"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (cls.MODES, {
                    "default": "apply",
                    "tooltip": "apply: inject a recorded table into this model.\n"
                               "record: capture one while this (full) model samples; "
                               "it is written when the run ends.\n"
                               "off: pass through, for an A/B control run."}),
            },
            "optional": {
                # Optional on purpose: ComfyUI's /prompt validation rejects a
                # graph that omits a *required* input, so prompts written before
                # this widget existed (scripts, saved API workflows) would fail
                # with "Required input is missing: report_level". As an optional
                # widget it still shows up in the UI with its default.
                "report_level": (_nodes.REPORT_LEVELS, {
                    "default": "brief",
                    "tooltip": "How much this node says.\n"
                               "silent: nothing but errors -- fastest, no "
                               "per-step bookkeeping.\n"
                               "brief: one block per sampling pass (what was "
                               "injected, what was not).\n"
                               "debug: per-forward traces, key distances, "
                               "timings, full counter dump, plus a .txt under "
                               "output/h3_adaln/reports/.\n"
                               "The same text is available on the `report` "
                               "output: connect it to `H3 AdaLN Report` and put "
                               "that node AFTER sampling."}),
                "for_model": (_model_choices(), {
                    "default": NONE,
                    "tooltip": "record only: choose the pruned base this table "
                               "is for. Its AdaLN fingerprint is written into "
                               "the table header, so apply's table=(auto) finds "
                               "it without any pick. The choice is saved with "
                               "the workflow, so set it once.\n"
                               "(none) = no target base is stored; apply then "
                               "needs an explicit `table` pick."}),
                "fill_reference_row": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "record only: also capture the fixed rows a "
                               "keyframe/reference workflow needs (t=0.999, and "
                               "1.0 for reference audio), so one recording covers "
                               "both text-only and reference runs.\n"
                               "Computed through the model's own modules in one "
                               "extra batch; the sampling result is untouched."}),
                "table": (_choices(), {
                    "default": AUTO,
                    "tooltip": "(auto) = the only table whose header declares "
                               "this model (for_model at record time). Nothing "
                               "is guessed: an undeclared table is refused.\n"
                               "Pick a file only to override, for this run -- "
                               "an explicit pick also allows a table whose AdaLN "
                               "LoRA fingerprint does not match (it is reported "
                               "as a MISMATCH, never silently)."}),
                "table_in": (TABLE_TYPE, {
                    "tooltip": "Advanced: take the table from a node earlier in "
                               "this same prompt (A/B verification)."}),
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, _nodes.REPORT_TYPE)
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    def run(self, model, mode, report_level="brief", for_model=NONE,
            fill_reference_row=True, table=AUTO, table_in=None, after=None,
            unique_id=None):
        # `after` is intentionally unused: the edge alone orders the graph.
        mode = str(mode or "apply").lower()
        level = report_level if report_level in _nodes.REPORT_LEVELS else "brief"
        dm = _nodes._dm_of(model)
        if mode == "off":
            _clear_runtime(model)
            _stop_recording(model)
            rep = _nodes.Report("H3 AdaLN · off", level)
            rep.add("exec", "model untouched (A/B control); any previous "
                            "runtime and recorder on this model were detached")
            rep.log()
            return (model, table_in, rep)
        if dm is None:
            raise RuntimeError("H3AdaLN: expected a ComfyUI ModelPatcher")
        if mode == "record":
            return self._record(model, dm, for_model, fill_reference_row,
                                unique_id, level)
        if mode == "apply":
            return self._apply(model, dm, table, table_in, level)
        raise RuntimeError("H3AdaLN: unknown mode {!r}".format(mode))

    # -- record -------------------------------------------------------
    def _record(self, model, dm, for_model=NONE, fill_reference_row=True,
                unique_id=None, level="brief"):
        for attr in ("_h3_pending_table", "_h3_rec_dm", "_h3_rec_fp",
                     "_h3_rec_adaln_fp", "_h3_saved", "_h3_saved_rows",
                     "_h3_save_passes", "_h3_rec_target", "_h3_rec_bound",
                     "_h3_rec_for_model", "_h3_rec_uid"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    pass
        model._h3_rec_for_model = str(for_model or NONE)
        rep = _nodes.Report("H3 AdaLN · record", level)
        model._h3_report = rep

        uid = str(unique_id or "")
        stale = _ACTIVE_RECORDERS.pop(uid, None) if uid else None
        if stale is not None:
            logging.warning(
                "H3AdaLN: node %s still had a recorder from an earlier run; "
                "dropping it before starting a new one", uid)
            try:
                stale.seal()
            except Exception:
                pass
        # Only the recorder with THIS node id is cleared above. Any other
        # recorder left on the shared diffusion module (another record node in
        # this prompt, or a zombie from a failed prompt) is handled below by
        # taking ownership instead of raising, so the prompt always completes.

        # Two record nodes on one diffusion module used to be allowed. Their
        # hooks are attached to the SAME nn.Modules, so one recorder's pin
        # batch is seen by the other as a real forward and the strict row/t
        # check kills the 0.999 / 1.0 rows. A failed prompt can also leave a
        # recorder behind (ON_CLEANUP never ran), which is why turning the
        # second node to `off` did not make this error go away.
        #
        # Policy: never raise. There must be exactly one live recorder, so
        # this node takes ownership by detaching the others (including
        # zombies). The superseded recorders skip their own save quietly.
        active = getattr(dm, "_h3_active_recorders", None)
        if active is None:
            active = {}
            try:
                dm._h3_active_recorders = active
            except Exception:
                active = {}
        live = {k: r for k, r in active.items() if getattr(r, "handles", None)}
        others = {k: r for k, r in live.items() if k != uid}
        if others:
            for k, r in others.items():
                try:
                    r.abandon()
                except Exception:
                    try:
                        r.remove()
                    except Exception:
                        pass
                active.pop(k, None)
                _ACTIVE_RECORDERS.pop(k, None)
            logging.warning(
                "H3AdaLN: node %s took over the single recorder slot from %s "
                "on this shared diffusion model (the earlier hooks were "
                "detached, including any left by a failed prompt). Only this "
                "node records now, so the 0.999/1.0 pins stay intact. Set the "
                "other chain to off/apply to avoid the handover.",
                uid, ", ".join(sorted(others)))
            rep.add("exec", "TOOK OVER   : detached recorder(s) of node(s) {} -- "
                            "only one recorder may live on a shared diffusion "
                            "model, otherwise the 0.999/1.0 pins get corrupted"
                            .format(", ".join(sorted(others))))
        # Hooks now survive ON_CLEANUP for later passes, so the tracking bucket
        # on the shared module is reset here (the new prompt's start).
        try:
            dm._h3_recorders = []
        except Exception:
            pass
        pins = ((float(sigma_plan.VISUAL_COND_TIMESTEP),
                 float(sigma_plan.AUDIO_COND_TIMESTEP))
                if fill_reference_row else ())
        recorder = record_engine.AdaLNRecorder(model, dm, mode="strict", tag="",
                                               pin_targets=pins)
        hooked = recorder.register()
        _nodes._track_registered(model, "_h3_recorders", recorder)
        model._h3_recorder = recorder
        model._h3_rec_uid = uid
        _LIVE_RECORDERS.append(recorder)
        if uid:
            _ACTIVE_RECORDERS[uid] = recorder
            if active is not None:
                active[uid] = recorder

        model_fp = model_fingerprint(dm)
        adaln_fp = _adaln_lora_fingerprint(model)
        shift_v, shift_a, shift_src = _nodes._auto_shifts(model, with_source=True)
        sig = runtime_signature(model, dm)
        meta = {
            "teacher": "full fp {} ({} projections)".format(model_fp, hooked),
            "sigmas": "[]",
            "shift_v": str(float(shift_v)),
            "shift_a": str(float(shift_a)),
            "shift_source": shift_src,
            "runtime_signature": sig,
            "visual_aug": str(sigma_plan.VISUAL_COND_TIMESTEP),
            "audio_aug": str(sigma_plan.AUDIO_COND_TIMESTEP),
            "compute_dtype": "recorded",
            "out_dtype": "unknown",
            "record_mode": "observed during sampling",
            "format": table_io.FORMAT,
            # Stamped up front, not only at save time: a table handed straight to
            # an Apply node in the same prompt (table_in) is injected before any
            # save happened, and without these it looked like a table recorded
            # under a different LoRA.
            "identity": identity_of(
                model_fp, adaln_fp,
                _nodes._model_compute_dtype(model, dm) or torch.bfloat16),
            "model_fingerprint": model_fp,
            "adaln_lora_fingerprint": adaln_fp or "",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": "1",
        }
        table = table_io.AdaLNTable([], {}, meta, recorder=recorder)

        model._h3_pending_table = table
        model._h3_rec_dm = dm
        model._h3_rec_fp = model_fp
        model._h3_rec_adaln_fp = adaln_fp
        model._h3_saved = False
        model._h3_saved_rows = 0
        model._h3_save_passes = 0
        _ensure_autosave()
        if not _set_callback_once(model, "h3_adaln_record",
                                  lambda *a, **k: _finish_record(model)):
            logging.warning("H3AdaLN: cannot hook save-on-cleanup; the table "
                            "will exist in memory only")

        hint = auto_name(dm, model_fp, adaln_fp, "<dtype>")
        target = _target_of(model)
        if target:
            meta["target_base"] = target[0]
            meta["target_name"] = target[1]

        rep.add("exec", "teacher     : full fp {} · {} projections hooked".format(
            model_fp, hooked))
        rep.add("exec", "adaln lora  : {}".format(adaln_fp or "none"))
        rep.add("exec", "shift       : {}/{} [{}]".format(
            "%g" % shift_v, "%g" % shift_a, shift_src))
        rep.add("exec", "signature   : {}".format(sig))
        rep.add("exec", "extra rows  : {}".format(
            "t={:.3f}{} pre-computed".format(
                pins[0], " / t={:.3f}".format(pins[1]) if len(pins) > 1 else "")
            if pins else "off -- a text-only table will lack the t=0.999 row"))
        rep.add("exec", "for base    : {}".format(
            "{} (fp {})".format(target[1], target[0]) if target else
            "none -- table=(auto) will NOT find this table"))
        rep.add("exec", "saves to    : {}/{}".format(table_folder(), hint))
        rep.add("exec", "flows       : rows are captured live; every ON_CLEANUP "
                        "merges what is new into that file, so a refine / "
                        "SelfLift pass adds its own timesteps instead of losing "
                        "them")

        if shift_src != "transformer_options":
            msg = ("shift %g/%g came from %s, NOT from the running forward's "
                   "transformer_options. A MiniMaxH3SigmaShift node placed "
                   "downstream of this node is invisible here, so this value is "
                   "a fallback; the captured t rows are unaffected (they come "
                   "from time_embedder's input)." % (
                       shift_v, shift_a, shift_src))
            logging.warning("H3AdaLN: %s", msg)
            rep.add("exec", "WARNING     : " + msg)
        rep.log()
        return (model, table, rep)

    # -- apply --------------------------------------------------------
    def _apply(self, model, dm, choice, table_in, level="brief"):
        rep = _nodes.Report("H3 AdaLN · apply", level)
        # Recording and injecting on the SAME diffusion module is a
        # contradiction: the recorder would capture the injected rows as if the
        # teacher had produced them. Close the recording before arming.
        if getattr(model, "_h3_recorder", None) is not None:
            _seal_recorder(model, getattr(model, "_h3_report", None),
                           "an apply node took over this model")
        model._h3_report = rep
        if table_in is not None:
            table, src = table_in, "table_in (live, from this prompt)"
            allow = False
        else:
            path, src = resolve_for_model(model, dm, choice)
            rep.add("exec", "resolved    : {}".format(path))
            try:
                table = table_io.read_table_file(path)
            except Exception as e:
                raise RuntimeError(
                    "H3AdaLN: cannot read table {}: {}: {}".format(
                        path, type(e).__name__, e))
            allow = (src == "picked")
        _inject(model, dm, table, rep, source=src, allow_lora_mismatch=allow)
        rep.log()
        return (model, table, rep)


class H3AdaLNHybrid:
    """Assemble a bX-49 table from two recordings (blocks 0..split-1 | split..49)."""

    # This node writes a .safetensors and arms injection hooks, so it must not be
    # served from a cached output: a cached run would neither re-merge nor
    # re-inject, and the sampling would silently use the base's native AdaLN.
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "table_a": (_choices(""), {
                    "default": "",
                    "tooltip": "Base teacher's table (blocks 0..split-1)."}),
                "table_b": (_choices(""), {
                    "default": "",
                    "tooltip": "Tail donor's table (blocks split..49)."}),
                "split": ("INT", {"default": 25, "min": 0, "max": 50, "step": 1,
                                  "tooltip": "First block taken from table_b."}),
            },
            "optional": {
                "final_from": (["a", "b"], {
                    "default": "a",
                    "tooltip": "Which side supplies final_layer. A bX-49 file "
                               "keeps the donor's final_layer in some recipes; "
                               "this used to be hardcoded to 'a'."}),
                "for_model": (_model_choices(), {
                    "default": NONE,
                    "tooltip": "Optional: declare the bX-49 checkpoint this "
                               "hybrid table is for, so apply's table=(auto) "
                               "finds it. Without it the hybrid table inherits "
                               "no base and must be picked by name."}),
                "report_level": (_nodes.REPORT_LEVELS, {
                    "default": "brief",
                    "tooltip": "See H3 AdaLN. `report` can be rendered after "
                               "sampling by the H3 AdaLN Report node."}),
            },
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, _nodes.REPORT_TYPE)
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    def run(self, model, table_a="", table_b="", split=25, final_from="a",
            for_model=NONE, report_level="brief"):
        level = report_level if report_level in _nodes.REPORT_LEVELS else "brief"
        rep = _nodes.Report("H3 AdaLN · hybrid", level)
        model._h3_report = rep

        split = int(split)
        final_from = str(final_from).lower()
        path_a = find_table(table_a, "table_a")
        path_b = find_table(table_b, "table_b")
        if os.path.abspath(path_a) == os.path.abspath(path_b):
            raise RuntimeError("H3AdaLNHybrid: table_a and table_b are the same file")
        a = table_io.read_table_file(path_a)
        b = table_io.read_table_file(path_b)
        merged = table_io.merge_tables(a, b, split=split, final_from=final_from)

        dm = _nodes._dm_of(model)
        out_dtype = merged.out_dtype
        identity = "hybrid:{}:{}:{}:{}".format(
            split, a.meta.get("identity", os.path.basename(path_a)),
            b.meta.get("identity", os.path.basename(path_b)), _short_dtype(out_dtype))
        merged.meta.update({
            "identity": identity,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": "1",
        })
        target = target_from_choice(for_model) if str(for_model or "").strip() not in (
            "", NONE) else None
        if target:
            merged.meta["target_base"] = target[0]
            merged.meta["target_name"] = target[1]
        name = "h3_hybrid_x{}_{}_{}_{}.safetensors".format(
            split, a.meta.get("model_fingerprint", "a")[:6],
            b.meta.get("model_fingerprint", "b")[:6], _short_dtype(out_dtype))
        path, action, added, total = save_table_smart(
            merged.t_values, merged.raw_by_key, merged.meta, out_dtype, name)
        merged.source = path

        rep.add("exec", "table_a     : {} ({} t rows)".format(
            os.path.basename(path_a), len(a.t_values)))
        rep.add("exec", "table_b     : {} ({} t rows)".format(
            os.path.basename(path_b), len(b.t_values)))
        rep.add("exec", "split       : blocks 0-{} <- A | blocks {}-49 <- B | "
                        "final_layer <- {}".format(
                            split - 1 if split > 0 else "none", split,
                            final_from))
        rep.add("exec", "lora fp     : A={} B={}".format(
            _norm_adaln_fp(a.meta.get("adaln_lora_fingerprint")) or "none",
            _norm_adaln_fp(b.meta.get("adaln_lora_fingerprint")) or "none"))
        rep.add("exec", "wrote       : {} {} (+{} rows, {} total)".format(
            action, path, added, total))
        rep.add("exec", "for base    : {}".format(
            "{} (fp {})".format(target[1], target[0]) if target
            else "none -- table=(auto) will not find this hybrid; pick it by name"))
        if dm is not None:
            _inject(model, dm, merged, rep, source="hybrid merge")
        rep.log()
        return (model, merged, rep)


NODE_CLASS_MAPPINGS = {
    "H3AdaLN": H3AdaLN,
    "H3AdaLNHybrid": H3AdaLNHybrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AdaLN": "H3 AdaLN (record / apply)",
    "H3AdaLNHybrid": "H3 AdaLN Hybrid (bX-49)",
}

