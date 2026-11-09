"""H3 AdaLN -- one node records and applies, one more assembles hybrids.

v2 of the unified node. The v1 surface was 16 widgets, and nearly all of them
existed to work around something the node can decide by itself:

  model_tag / lora_tag /    -> the table's identity is (AdaLN weights, AdaLN LoRA
  scheduler_tag / pass_tag     fingerprint, compute dtype), read off the model.
                               The file name follows from it, so nothing is typed.
  table_text / table_b_text / -> nothing to pick. At record time the node finds
  table_source (3 modes)        the pruned base by itself (the only curve-shaped
                                checkpoint in the model folder) and writes its
                                AdaLN fingerprint into the table header. At apply
                                time it hashes the model in front of it and takes
                                the table that declares that base. No "most
                                recent" rule, no sidecar state -- the association
                                travels inside the .safetensors.
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

  H3 AdaLN          model, mode, for_model, table, [table_in], [after]
  H3 AdaLN Hybrid   model, table_a, table_b, split            bX-49 donor merge

The invariant that makes this enough:

  modulation = adaln_proj(time_embedder(t))

i.e. a pure function of (AdaLN weights + AdaLN LoRA) and t -- not of the pass,
the latent, the resolution or the seed. So one table per model covers every
sampling segment, and rows from different schedules simply accumulate.
"""

import hashlib
import json
import logging
import os
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
TABLE_EXT = ".safetensors"
AUTO = "(auto)"
NONE = "(none)"
# Two rows for the same t may differ by this much and still count as the same
# observation. Anything above it means the model / LoRA / dtype changed.
CONFLICT_TOL = 1e-6

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
    """Every folder a table may live in -- new ones first, then legacy saves."""
    dirs = [table_folder()]
    try:
        import folder_paths
        legacy = os.path.join(folder_paths.get_output_directory(), "h3_adaln")
        if os.path.isdir(legacy):
            dirs.append(legacy)
    except Exception:
        pass
    return dirs


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


def _model_choices(default=AUTO):
    """Checkpoints this table can be declared for (record mode only)."""
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
    """Which table this model gets -- by fingerprint, not by choosing.

    Order:
      1. `table` names a file -> that one (an explicit override, nothing is
         remembered behind your back);
      2. a table whose header says `target_base == this model` -> exactly one;
      3. the only table recorded with this AdaLN LoRA that declares no target
         (tables recorded before `for_model` existed, or on purpose);
      4. nothing -> error.
    """
    picked = "" if str(choice or "").strip() in ("", AUTO) else str(choice).strip()
    if picked:
        return find_table(picked), "picked"

    base_fp = base_fingerprint(dm)
    want = _norm_adaln_fp(_adaln_lora_fingerprint(model))
    mine, undeclared = [], []
    for path, meta in _scan_tables():
        if _norm_adaln_fp(meta.get("adaln_lora_fingerprint", "")) != want:
            continue
        target = str(meta.get("target_base") or "")
        if not target:
            undeclared.append(path)
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
    if len(undeclared) == 1:
        return undeclared[0], "only table without a target"
    if not undeclared:
        names = list_tables()
        why = ("{} table(s) in the folder, but none is declared for this model "
               "({}) with its AdaLN LoRA:\n  {}".format(
                   len(names), base_fp, "\n  ".join(names)) if names
               else "no tables in {}".format(table_folder()))
        raise RuntimeError(
            "H3AdaLN: no table for this model -- {}.\n"
            "Record one with mode=record + for_model=<this checkpoint>, or set "
            "mode=off to run without a table.".format(why))
    raise RuntimeError(
        "H3AdaLN: {} tables fit this model and none of them declares a target:\n"
        "  {}\n"
        "Pick one in `table`, or re-record with for_model=<this checkpoint> so "
        "the table carries its base fingerprint.".format(
            len(undeclared), "\n  ".join(os.path.basename(p) for p in undeclared)))


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
        h = hashlib.sha1()
        h.update("curve={};blocks={};".format(
            "adaln_t_table" in keys, blocks).encode())
        if "adaln_t_table" in keys:
            h.update(_f32_bytes(f.get_slice("adaln_t_table")[0:4],
                                round_to_bf16=True))
        picks = [0, blocks // 2, blocks - 1] if blocks else []
        for i in picks + ["final_layer"]:
            prefix = "blocks.{}".format(i) if isinstance(i, int) else "final_layer"
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
            if delta > CONFLICT_TOL:
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
            return existing, "unchanged", 0, len(t_all)
        merged = dict(meta)
        merged["created"] = meta_old.get("created", meta.get("created", ""))
        merged["runs"] = str(int(meta_old.get("runs", "1") or 1) + 1)
        merged["shift_v"] = meta_old.get("shift_v", meta.get("shift_v", ""))
        merged["shift_a"] = meta_old.get("shift_a", meta.get("shift_a", ""))
        merged["t_rows"] = str(len(t_all))
        _atomic_write(existing, t_all, raw_all, merged, out_dtype)
        return existing, "merged", added, len(t_all)

    path = _unique_path(table_folder(), name_hint)
    _atomic_write(path, t_values, raw_by_key, meta, out_dtype)
    return path, "saved", len(t_values), len(t_values)


# ---------------------------------------------------------------- runtime
def _clear_runtime(model):
    _nodes._clear_registered(model, "_h3_exact_runtimes")
    if getattr(model, "_h3_exact_runtime", None) is not None:
        try:
            model._h3_exact_runtime.remove()
        except Exception:
            pass
        model._h3_exact_runtime = None


def _inject(model, dm, table):
    """Replace all 51 AdaLN projections with the table's rows. Returns report lines."""
    table.finalize(required=True)

    baked = table.meta.get("adaln_lora_fingerprint", "")
    current = _adaln_lora_fingerprint(model)
    if baked and current and baked != current:
        raise RuntimeError(
            "H3AdaLN: this table was recorded with a different AdaLN/timestep LoRA "
            "(table={} model={}). Only time_embedder / adaln_proj LoRAs matter -- "
            "re-record on the full teacher with that LoRA loaded.".format(baked, current))

    notes = []
    model_dtype = _nodes._model_compute_dtype(model, dm)
    if model_dtype is not None:
        want = _short_dtype(table.out_dtype)
        got = _short_dtype(model_dtype)
        if want != got:
            notes.append("WARNING: table is {} but this model computes in {} -- the "
                         "result will NOT be bitwise".format(want, got))

    _clear_runtime(model)
    runtime = apply_engine.ExactAdaLNRuntime(
        table.t_values, table.raw_by_key, table.meta, dm, mode="strict")
    hooked = runtime.register(model)
    _nodes._track_registered(model, "_h3_exact_runtimes", runtime)
    model._h3_exact_runtime = runtime

    def _done(*_a, **_k):
        try:
            logging.info("H3AdaLN: apply stats\n%s", runtime.summary())
        finally:
            runtime.remove()

    try:
        import comfy.patcher_extension as pe
        model.add_callback(pe.CallbacksMP.ON_CLEANUP, _done)
    except Exception:
        pass

    lines = [
        "  table  : {} · {} t rows · {} projections · {}".format(
            os.path.basename(table.source or "") or "(in memory)",
            len(table.t_values), table.projections(), _short_dtype(table.out_dtype)),
        "  armed  : {} projections; per-step stats are written to the log when "
        "this run ends".format(hooked),
    ] + ["  " + n for n in notes]
    logging.info("H3AdaLN: apply\n%s", "\n".join(lines))
    return lines


# ---------------------------------------------------------------- record
def _curve_checkpoints():
    """Checkpoints in the model folders that look like a pruned/curve base.

    Header only: a base ships `adaln_t_table` and no `time_embedder`, a full
    teacher is the opposite. That is what lets the record node find "the base"
    without being told -- when there is exactly one, there is nothing to ask.
    """
    paths = []
    try:
        import folder_paths
        names = list(folder_paths.get_filename_list("diffusion_models"))
    except Exception:
        return paths
    try:
        from safetensors import safe_open
    except Exception:
        return paths
    for name in names:
        path = resolve_model_file(name)
        if not path or not path.lower().endswith(TABLE_EXT):
            continue
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
        except Exception:
            continue
        if "adaln_t_table" in keys and not any(
                k.startswith("time_embedder.") for k in keys):
            paths.append(path)
    return paths


def _target_of(model):
    """(base fingerprint, file name) this table is for, or None.

    `for_model=(auto)` (the default) means "the curve base in the model folder":
    with exactly one, that is unambiguous and nothing is asked. With several,
    there is no way to tell which one you mean -- the table then carries no
    target and apply falls back to "the only table for this LoRA".
    """
    cached = getattr(model, "_h3_rec_target", None)
    if cached is not None:
        return cached or None

    choice = str(getattr(model, "_h3_rec_for_model", AUTO) or AUTO).strip()
    if choice in ("", NONE, AUTO):
        found = _curve_checkpoints()
        if len(found) != 1:
            if not found:
                logging.info("H3AdaLN: no pruned/curve checkpoint in the model "
                             "folder, so the table carries no target base")
            else:
                logging.warning(
                    "H3AdaLN: %d pruned checkpoints in the model folder, so this "
                    "table carries no target base -- set for_model to name the "
                    "right one:\n  %s", len(found),
                    "\n  ".join(os.path.basename(p) for p in found))
            model._h3_rec_target = ()
            return None
        path = found[0]
    else:
        path = resolve_model_file(choice)
        if not path:
            logging.warning("H3AdaLN: for_model=%r does not resolve to a "
                            "checkpoint; the table will not carry a base "
                            "fingerprint", choice)
            model._h3_rec_target = ()
            return None

    try:
        fp = base_fingerprint_from_file(path)
    except Exception as e:
        logging.warning("H3AdaLN: cannot fingerprint %s (%s: %s); the table will "
                        "not carry a base fingerprint", path, type(e).__name__, e)
        model._h3_rec_target = ()
        return None
    target = (fp, os.path.basename(path))
    model._h3_rec_target = target
    return target


def _stop_recording(model):
    """Drop this prompt's read-only hooks.

    Hook lifetime is per prompt, not per MODEL: ModelPatcher.clone() shares the
    underlying nn.Module, so a recorder left behind would keep copying AdaLN
    rows for every later run and hold the memory for them.
    """
    recorder = getattr(model, "_h3_recorder", None)
    if recorder is not None:
        try:
            recorder.remove()
        except Exception:
            pass
        model._h3_recorder = None
    _nodes._clear_registered(model, "_h3_recorders")


def _finish_record(model):
    """ON_CLEANUP: write the rows this run captured."""
    try:
        _save_recorded(model)
    finally:
        _stop_recording(model)


def _save_recorded(model):
    table = getattr(model, "_h3_pending_table", None)
    if table is None or getattr(model, "_h3_saved", False):
        return
    try:
        table.finalize(required=True)
    except Exception as e:
        logging.error("H3AdaLN: nothing saved -- %s", e)
        return

    conflicts = int(table.meta.get("repeat_conflicts", "0") or 0)
    if conflicts:
        logging.error(
            "H3AdaLN: refusing to save. %d timestep(s) produced two different "
            "modulations in one run (max delta %s), so the AdaLN branch changed "
            "mid-run -- record again with one model and one LoRA stack.",
            conflicts, table.meta.get("max_repeat_delta", "?"))
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
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
        return

    model._h3_saved = True
    logging.info("H3AdaLN: %s %s (%d new rows, %d total, %.0f MiB)", action, path,
                 added, total, os.path.getsize(path) / (1 << 20))
    if total > 200:
        logging.warning(
            "H3AdaLN: %s now holds %d t rows (~%.1f GiB). Every extra schedule "
            "adds rows; delete the file and re-record if it gets out of hand.",
            os.path.basename(path), total, os.path.getsize(path) / (1 << 30))
    pin = sigma_plan.f32_image(sigma_plan.VISUAL_COND_TIMESTEP)
    if not any(abs(t - pin) < 1e-6 for t in table.t_values):
        logging.warning(
            "H3AdaLN: %s has no row for t=%.3f -- a later keyframe/reference "
            "workflow will need one more recording pass (it errors before "
            "sampling, never silently). Enable fill_reference_row to add it "
            "automatically.", os.path.basename(path), pin)


# ---------------------------------------------------------------- nodes
class H3AdaLN:
    """Record a table while the full teacher samples, or apply one to a pruned base."""

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
                "for_model": (_model_choices(), {
                    "default": AUTO,
                    "tooltip": "record only, and normally left alone. (auto) = "
                               "the curve base in the model folder; its AdaLN "
                               "fingerprint is written into the table so apply "
                               "finds it without any pick.\n"
                               "Only name one explicitly when the folder holds "
                               "several pruned checkpoints."}),
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
                    "tooltip": "(auto) = the table declared for this model "
                               "(for_model at record time); failing that, the only "
                               "table recorded with this AdaLN LoRA.\n"
                               "Pick a file only to override, for this run."}),
                "table_in": (TABLE_TYPE, {
                    "tooltip": "Advanced: take the table from a node earlier in "
                               "this same prompt (A/B verification)."}),
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, "STRING")
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    def run(self, model, mode, for_model=AUTO, fill_reference_row=True,
            table=AUTO, table_in=None, after=None):
        # `after` is intentionally unused: the edge alone orders the graph.
        mode = str(mode or "apply").lower()
        dm = _nodes._dm_of(model)
        if mode == "off":
            _clear_runtime(model)
            return (model, table_in, "H3 AdaLN · off\n  model untouched (A/B control)")
        if dm is None:
            raise RuntimeError("H3AdaLN: expected a ComfyUI ModelPatcher")
        if mode == "record":
            return self._record(model, dm, for_model, fill_reference_row)
        if mode == "apply":
            return self._apply(model, dm, table, table_in)
        raise RuntimeError("H3AdaLN: unknown mode {!r}".format(mode))

    # -- record -------------------------------------------------------
    def _record(self, model, dm, for_model=AUTO, fill_reference_row=True):
        for attr in ("_h3_pending_table", "_h3_rec_dm", "_h3_rec_fp",
                     "_h3_rec_adaln_fp", "_h3_saved", "_h3_rec_target",
                     "_h3_rec_for_model"):
            if hasattr(model, attr):
                try:
                    delattr(model, attr)
                except Exception:
                    pass
        model._h3_rec_for_model = str(for_model or AUTO)

        previous = getattr(model, "_h3_recorder", None)
        if previous is not None and getattr(previous, "rows", None):
            logging.warning(
                "H3AdaLN: another record node on this MODEL is already capturing; "
                "only one recorder can be live, so this one takes over and the "
                "other saves nothing. Use one record node per model.")
        _nodes._clear_registered(model, "_h3_recorders")

        pins = ((float(sigma_plan.VISUAL_COND_TIMESTEP),
                 float(sigma_plan.AUDIO_COND_TIMESTEP))
                if fill_reference_row else ())
        recorder = record_engine.AdaLNRecorder(model, dm, mode="strict", tag="",
                                               pin_targets=pins)
        hooked = recorder.register()
        _nodes._track_registered(model, "_h3_recorders", recorder)
        model._h3_recorder = recorder

        model_fp = model_fingerprint(dm)
        adaln_fp = _adaln_lora_fingerprint(model)
        shift_v, shift_a = _nodes._auto_shifts(model)
        meta = {
            "teacher": "full fp {}".format(model_fp),
            "sigmas": "[]",
            "shift_v": str(float(shift_v)),
            "shift_a": str(float(shift_a)),
            "visual_aug": str(sigma_plan.VISUAL_COND_TIMESTEP),
            "audio_aug": str(sigma_plan.AUDIO_COND_TIMESTEP),
            "compute_dtype": "recorded",
            "out_dtype": "unknown",
            "record_mode": "observed during sampling",
            "format": table_io.FORMAT,
        }
        table = table_io.AdaLNTable([], {}, meta, recorder=recorder)

        model._h3_pending_table = table
        model._h3_rec_dm = dm
        model._h3_rec_fp = model_fp
        model._h3_rec_adaln_fp = adaln_fp
        model._h3_saved = False
        try:
            import comfy.patcher_extension as pe
            model.add_callback(pe.CallbacksMP.ON_CLEANUP,
                               lambda *a, **k: _finish_record(model))
        except Exception as e:
            logging.warning("H3AdaLN: cannot hook save-on-cleanup (%s: %s); the "
                            "table will exist in memory only",
                            type(e).__name__, e)

        hint = auto_name(dm, model_fp, adaln_fp, "<dtype>")
        target = _target_of(model)
        report = "\n".join([
            "H3 AdaLN · record",
            "  teacher : fp {} · {} projections hooked · shift {}/{}".format(
                model_fp, hooked, "%g" % shift_v, "%g" % shift_a),
            "  adaln lora : {}".format(adaln_fp or "(none)"),
            "  extra rows : {}".format(
                "t=%.3f%s pre-computed" % (pins[0], " / t=%.3f" % pins[1] if len(pins) > 1 else "")
                if pins else "off (text-only tables will lack t=0.999)"),
            "  for base   : {}".format(
                "{} (fp {}){}".format(
                    target[1], target[0],
                    " [auto]" if str(for_model or AUTO).strip() in ("", NONE, AUTO)
                    else "") if target else
                "(none) -- apply will fall back to 'the only table for this LoRA'"),
            "  saves to   : {}/{}".format(table_folder(), hint),
            "",
            "  Rows are captured live and written when this run ends. Re-recording",
            "  the same schedule changes nothing; a different schedule adds rows to",
            "  the same file instead of replacing it. Apply finds this table by",
            "  its target fingerprint -- no pick, no 'most recent' guess.",
        ])
        logging.info("%s", report)
        return (model, table, report)

    # -- apply --------------------------------------------------------
    def _apply(self, model, dm, choice, table_in):
        if table_in is not None:
            table, src = table_in, "table_in"
        else:
            path, src = resolve_for_model(model, dm, choice)
            table = table_io.read_table_file(path)
        lines = ["H3 AdaLN · apply · {}".format(src)] + _inject(model, dm, table)
        return (model, table, "\n".join(lines))


class H3AdaLNHybrid:
    """Assemble a bX-49 table from two recordings (blocks 0..split-1 | split..49)."""

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
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, "STRING")
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = _nodes.CATEGORY

    def run(self, model, table_a="", table_b="", split=25):
        split = int(split)
        path_a = find_table(table_a, "table_a")
        path_b = find_table(table_b, "table_b")
        if os.path.abspath(path_a) == os.path.abspath(path_b):
            raise RuntimeError("H3AdaLNHybrid: table_a and table_b are the same file")
        a = table_io.read_table_file(path_a)
        b = table_io.read_table_file(path_b)
        merged = table_io.merge_tables(a, b, split=split, final_from="a")

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
        name = "h3_hybrid_x{}_{}_{}_{}.safetensors".format(
            split, a.meta.get("model_fingerprint", "a")[:6],
            b.meta.get("model_fingerprint", "b")[:6], _short_dtype(out_dtype))
        path, action, added, total = save_table_smart(
            merged.t_values, merged.raw_by_key, merged.meta, out_dtype, name)
        merged.source = path

        lines = ["H3 AdaLN · hybrid"]
        if split > 0:
            lines.append("  blocks 0-{} : {}".format(split - 1,
                                                     os.path.basename(path_a)))
        if split < 50:
            lines.append("  blocks {}-49 : {}".format(split,
                                                      os.path.basename(path_b)))
        lines.append("  {} : {} (+{} rows, {} total)".format(action, path, added, total))
        if dm is not None:
            lines += _inject(model, dm, merged)
        return (model, merged, "\n".join(lines))


NODE_CLASS_MAPPINGS = {
    "H3AdaLN": H3AdaLN,
    "H3AdaLNHybrid": H3AdaLNHybrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AdaLN": "H3 AdaLN (record / apply)",
    "H3AdaLNHybrid": "H3 AdaLN Hybrid (bX-49)",
}
