"""ComfyUI nodes for exact (pre-baked / recorded) MiniMax-H3 AdaLN modulation.

Five-stage, image-like workflow
--------------------------------
  Record [full teacher MODEL + LoRA, during a real sampling run] -> H3_ADALN_TABLE
  Bake   [full teacher MODEL + LoRA + sigmas, offline]            -> H3_ADALN_TABLE
  Save   [H3_ADALN_TABLE]  -> .safetensors in output/ (shareable, like Save Image)
  Load   [upload / pick file] -> H3_ADALN_TABLE (like Load Image)
  Apply  [pruned MODEL + H3_ADALN_TABLE] -> patched MODEL

**Record is the exact path.** Bake computes the same numbers offline on
unloaded CPU weights, which cannot reproduce the kernels ComfyUI picks at
sampling time; measured drift is one bf16 ulp per projection, amplified to
~0.145 mean_abs on the latent after 8 steps. Record observes the teacher's own
output during a real run, so replaying it is bitwise by construction.

The table file is self-contained: all provenance (teacher, LoRA fingerprint,
sigma grid, shifts, dtype, ComfyUI version) lives in its safetensors header,
so it can be copied to another machine and loaded without any sidecar.

The bake path never parses a safetensors file by hand: it asks ComfyUI for the
effective (dequantised + LoRA-patched) weights, so bf16 / fp16 / fp8_scaled /
int8_convrot / nvfp4 checkpoints and any ComfyUI-applicable LoRA are handled by
ComfyUI itself.
"""

import hashlib
import json
import logging
import os

import torch

try:
    from . import apply_engine, bake_engine, record_engine, sigma_plan, table_io, latent_io
except ImportError:
    import apply_engine
    import bake_engine
    import record_engine
    import sigma_plan
    import table_io
    import latent_io

CATEGORY = "MiniMaxH3/ExactAdaLN"
TABLE_TYPE = "H3_ADALN_TABLE"
_MODES = ["strict", "fallback", "off"]

_AFTER_DOC = ("Purely an execution-order edge: nothing is read from it. ComfyUI "
              "runs nodes as soon as their inputs exist, so wire the sampler's "
              "LATENT here to make this node wait until sampling has finished.")


def _sigmas_from_model(model, steps, scheduler):
    import comfy.samplers
    ms = model.get_model_object("model_sampling")
    return comfy.samplers.calculate_sigmas(ms, scheduler, steps)


def _auto_shifts(model):
    """Video/audio shift as model.py:672-673 resolves them.

    The DiT reads `transformer_options` first and only then the module
    attribute, so reading `model_sampling` alone can disagree with what the
    forward pass actually used.
    """
    try:
        to = model.model_options.get("transformer_options", {}) or {}
        sv = to.get("minimax_h3_sigma_shift_video")
        sa = to.get("minimax_h3_sigma_shift_audio")
        if sv is not None and sa is not None:
            return float(sv), float(sa)
    except Exception:
        pass
    try:
        dm = model.model.diffusion_model
        sv = getattr(dm, "sigma_shift_video", None)
        sa = getattr(dm, "sigma_shift_audio", None)
        if sv is not None and sa is not None:
            return float(sv), float(sa)
    except Exception:
        pass
    try:
        ms = model.get_model_object("model_sampling")
        return float(getattr(ms, "shift", 12.0)), float(getattr(ms, "audio_shift", 3.0))
    except Exception:
        return sigma_plan.DEFAULT_VIDEO_SHIFT, sigma_plan.DEFAULT_AUDIO_SHIFT


def _parse_float_list(text):
    if not text:
        return []
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _patch_bytes(h, value, depth=0):
    """Fold one LoRA patch entry into a digest without cloning whole weights.

    ComfyUI stores patches as nested tuples whose first element is normally
    the strength and whose later elements hold the up/down tensors. Sampling
    only a handful of values is enough to notice a different LoRA file or a
    changed strength, while staying cheap on a 34 GB model.
    """
    if depth > 3:
        return
    if torch.is_tensor(value):
        h.update("{}|{}|{}".format(tuple(value.shape), value.dtype,
                                   value.numel()).encode())
        try:
            sample = value.detach().flatten()[:8].float().cpu().tolist()
            h.update(repr([round(float(x), 6) for x in sample]).encode())
        except Exception:
            pass
    elif isinstance(value, dict):
        for k in sorted(value, key=str):
            _patch_bytes(h, k, depth + 1)
            _patch_bytes(h, value[k], depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value[:12]:
            _patch_bytes(h, item, depth + 1)
    elif isinstance(value, (int, float, str, bool, type(None))):
        h.update(repr(value).encode())
    else:
        h.update(repr(type(value).__name__).encode())


def _patch_set_digest(model, predicate):
    patches = getattr(model, "patches", {}) or {}
    keys = sorted(k for k in patches if predicate(k))
    h = hashlib.sha256()
    for key in keys:
        h.update(key.encode())
        try:
            entries = patches.get(key, [])
            for entry in entries[:12]:
                _patch_bytes(h, entry)
        except Exception:
            pass
    return h.hexdigest()[:16], len(keys)


def _lora_fingerprint(model):
    """Digest of the LoRA stack shared by the teacher and the pruned base.

    AdaLN / time_embedder patch keys are kept out of the primary fingerprint:
    they exist on the full teacher but are dropped by ComfyUI on a pruned
    base, so requiring them to match would reject every valid table. They are
    tracked separately by `_adaln_lora_fingerprint`.

    Unlike the old key-name-only digest, this one includes patch strengths and
    a sample of the patch weights: swapping the LoRA file or changing
    `strength_model` must invalidate a recorded table.
    """
    return _patch_set_digest(
        model, lambda k: "time_embedder" not in k and "adaln_proj" not in k)


def _adaln_lora_fingerprint(model):
    return _patch_set_digest(
        model, lambda k: "time_embedder" in k or "adaln_proj" in k)


def _lora_key_fingerprint(model):
    """Legacy key-name-only digest, kept for tables from older builds.

    Those tables store this value in `lora_fingerprint`, so validating them
    against the weight-aware digest would reject a perfectly valid table.
    New tables stored by this build carry it as `lora_fingerprint_keys` and
    are validated strictly.
    """
    patches = getattr(model, "patches", {}) or {}
    keys = sorted(k for k in patches
                  if "time_embedder" not in k and "adaln_proj" not in k)
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return digest, len(keys)


def _dm_of(model):
    return getattr(getattr(model, "model", None), "diffusion_model", None)


def _model_compute_dtype(model, dm):
    """Best-effort dtype the diffusion model actually runs its maths in.

    Only two trustworthy sources: ComfyUI's own `manual_cast_dtype`, and the
    model's declared dtype. Guessing from the first floating-point parameter is
    deliberately avoided -- norms are often fp32 and would report a false
    mismatch.
    """
    d = getattr(dm, "manual_cast_dtype", None)
    if isinstance(d, torch.dtype):
        return d
    try:
        d = model.model.get_dtype()
        if isinstance(d, torch.dtype):
            return d
    except Exception:
        pass
    return None


def _resolve_checkpoint(name):
    """Absolute path for a checkpoint name/substring/path in diffusion_models."""
    name = (name or "").strip()
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    try:
        import folder_paths
        files = folder_paths.get_filename_list("diffusion_models")
    except Exception:
        return None
    hits = [f for f in files if f == name]
    if not hits:
        hits = [f for f in files if name.lower() in f.lower()]
    if len(hits) == 1:
        try:
            return folder_paths.get_full_path("diffusion_models", hits[0])
        except Exception:
            return None
    if len(hits) > 1:
        raise RuntimeError(
            "H3AdaLNRecord: time_embedder_from={!r} is ambiguous, it matches {} "
            "checkpoints: {}. Use a longer substring or the full name.".format(
                name, len(hits), ", ".join(sorted(hits)[:6])))
    return None


def _clear_time_embedder_patch(model):
    """Undo a previous `time_embedder_from` swap on this ModelPatcher."""
    prev = getattr(model, "_h3_te_object_patch", None)
    if prev is not None:
        model.add_object_patch(prev[0], prev[1])
        model._h3_te_object_patch = None
        return True
    return False


def _patch_time_embedder(model, name):
    """Substitute another checkpoint's `time_embedder.*` into this MODEL.

    A bX-49 hybrid takes the base teacher's body (and its `time_embedder`) and
    only the tail blocks' `adaln_proj` from the donor. AdaLN modulation is
    `adaln_proj(time_embedder(t))`, so recording the donor teacher as-is gives
    the donor's embedding, which is *not* what those blocks see inside a hybrid.
    Swapping the embedding in first makes the recording identical to recording
    the assembled hybrid directly -- without needing the 34 GB file.

    Returns `(n_tensors, module_or_None, mode)` where mode is "module" for the
    exact path and "weights" for the bf16 fallback (not bitwise).
    """
    path = _resolve_checkpoint(name)
    if not path:
        if os.path.isabs(name):
            raise RuntimeError(
                "H3AdaLNRecord: time_embedder_from={!r} does not exist (absolute "
                "paths are used as-is, with no search).".format(name))
        raise RuntimeError(
            "H3AdaLNRecord: time_embedder_from={!r} matches nothing in "
            "ComfyUI/models/diffusion_models. Pass a filename, a unique "
            "substring, or an absolute path to an existing file.".format(name))
    try:
        from safetensors import safe_open
    except Exception as e:  # pragma: no cover
        raise RuntimeError("H3AdaLNRecord: safetensors unavailable: %s" % e)

    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {k.split("time_embedder.", 1)[1]: f.get_tensor(k)
                   for k in f.keys() if "time_embedder." in k}
    if not tensors:
        raise RuntimeError(
            "H3AdaLNRecord: {} contains no `time_embedder.*` tensors -- it is a "
            "pruned/curve checkpoint, which cannot donate an embedding."
            .format(os.path.basename(path)))

    # Preferred: swap the whole module. Weight patches go through
    # `patch_weight_to_device`, which casts to lora_compute_dtype (bf16) and
    # would round the donor's fp32 embedding -- that alone moves the modulation
    # by ~1e-1 and is exactly what a bitwise match cannot tolerate.
    swapped = None
    try:
        import copy as _copy
        live = model.get_model_object("diffusion_model.time_embedder")
        fresh = _copy.deepcopy(live)
        own = fresh.state_dict()
        bad = [k for k in tensors if k not in own or tuple(own[k].shape) != tuple(tensors[k].shape)]
        if bad:
            # A partial swap would silently mix two embeddings -- refuse.
            raise RuntimeError(
                "H3AdaLNRecord: {} does not match this teacher's time_embedder "
                "({} of {} tensors differ in name/shape: {})".format(
                    os.path.basename(path), len(bad), len(tensors), ", ".join(bad[:4])))
        for k, t in tensors.items():
            own[k] = t.to(own[k].dtype)
        fresh.load_state_dict(own, strict=False)
        model.add_object_patch("diffusion_model.time_embedder", fresh)
        if model.get_model_object("diffusion_model.time_embedder") is not fresh:
            raise RuntimeError("object patch did not take effect")
        model._h3_te_object_patch = ("diffusion_model.time_embedder", live)
        swapped = fresh
    except Exception as e:
        logging.warning("H3AdaLNRecord: exact time_embedder swap failed (%s: %s); "
                        "falling back to weight patches, which round to bf16 and "
                        "are NOT bitwise", type(e).__name__, e)

    if swapped is not None:
        return len(tensors), swapped, "module"

    patches = {}
    for k, t in tensors.items():
        tail = "time_embedder." + k
        # ComfyUI's BaseModel state dict is namespaced under `diffusion_model.`
        # (the same prefix LoRA files use). Register both spellings and let
        # add_patches keep whichever one the live model actually has.
        patches[tail] = ("set", (t,))
        patches["diffusion_model." + tail] = ("set", (t,))
    before = set(getattr(model, "patches", {}) or {})
    model.add_patches(patches, 1.0)
    applied = [k for k in (getattr(model, "patches", {}) or {})
               if k not in before and "time_embedder." in k]
    if not applied:
        raise RuntimeError(
            "H3AdaLNRecord: time_embedder_from={!r} was read ({} tensors) but the "
            "model accepted none of the keys -- its time_embedder is not exposed "
            "to weight patches in this ComfyUI build.".format(name, len(tensors)))
    return len(applied), None, "weights"


def _clear_registered(model, attr):
    """Remove hooks left on this shared diffusion module by earlier prompts.

    ModelPatcher.clone() does not copy arbitrary attributes but it DOES share
    the underlying nn.Module. Track registrations on the module itself, or a
    later prompt (including an `off` control) can inherit a stale injector.
    """
    dm = _dm_of(model)
    if dm is None:
        return
    for obj in list(getattr(dm, attr, []) or []):
        try:
            obj.remove()
        except Exception:
            pass
    try:
        setattr(dm, attr, [])
    except Exception:
        pass


def _track_registered(model, attr, obj):
    dm = _dm_of(model)
    if dm is None:
        return
    try:
        bucket = getattr(dm, attr, None)
        if bucket is None:
            bucket = []
            setattr(dm, attr, bucket)
        bucket.append(obj)
    except Exception:
        pass


def _comfy_version():
    try:
        import comfyui_version
        return str(comfyui_version.__version__)
    except Exception:
        return "unknown"


class H3AdaLNRecord:
    """Capture the teacher's native AdaLN output while it is actually sampling.

    This is the bitwise path. The node is a pass-through for MODEL: it attaches
    read-only forward hooks and hands the same model on to the guider. The
    table object it returns is filled *during* sampling, so whatever consumes
    it must run afterwards -- connect the sampler's LATENT to that node's
    `after` input (or run Record and Save/Apply in two separate prompts).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["strict", "lenient"], {"default": "strict",
                        "tooltip": "strict: refuse to sample when rows cannot be "
                                   "labelled. lenient: log and keep going."}),
            },
            "optional": {
                "teacher_tag": ("STRING", {"default": "",
                                           "tooltip": "Recorded into the table "
                                                      "metadata, e.g. the checkpoint name."}),
                "lora_tag": ("STRING", {"default": "",
                                        "tooltip": "Free-form note about the LoRA stack."}),
                "time_embedder_from": ("STRING", {"default": "",
                        "tooltip": "Optional checkpoint whose `time_embedder.*` "
                                   "weights are substituted into this teacher "
                                   "before recording. Use it to record the tail "
                                   "half of a bX-49 hybrid from the donor teacher "
                                   "alone: AdaLN modulation is adaln_proj(t_emb), "
                                   "and a hybrid's tail blocks see the *base* "
                                   "teacher's embedding, not the donor's."}),
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
        }

    RETURN_TYPES = ("MODEL", TABLE_TYPE, "STRING")
    RETURN_NAMES = ("model", "table", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, mode="strict", teacher_tag="", lora_tag="",
            time_embedder_from="", after=None):
        dm = _dm_of(model)
        if dm is None:
            raise RuntimeError("H3AdaLNRecord: expected a ComfyUI MODEL (ModelPatcher) "
                               "for MiniMax-H3")
        prev = getattr(model, "_h3_recorder", None)
        if prev is not None and not getattr(prev, "t_values", None):
            logging.warning("H3AdaLNRecord: a previous Record node on this MODEL "
                            "had not finished; only one recorder can be live per "
                            "model, so it is being replaced (its table stays empty)")
        _clear_registered(model, "_h3_recorders")

        _clear_time_embedder_patch(model)
        te_patch, te_module, te_mode = 0, None, ""
        if (time_embedder_from or "").strip():
            # A LoRA that patches time_embedder cannot survive the module swap:
            # its delta is applied to the original module, which the swap then
            # replaces. Refuse instead of recording a table that silently lost it.
            te_keys = [k for k in (getattr(model, "patches", {}) or {})
                       if "time_embedder" in k]
            if te_keys:
                raise RuntimeError(
                    "H3AdaLNRecord: this MODEL carries {} LoRA patch(es) on "
                    "time_embedder ({}). time_embedder_from would silently drop "
                    "them -- record the donor without that LoRA, or clear "
                    "time_embedder_from.".format(
                        len(te_keys), ", ".join(sorted(te_keys)[:3])))
            te_patch, te_module, te_mode = _patch_time_embedder(model, time_embedder_from)

        recorder = record_engine.AdaLNRecorder(model, dm, mode=mode, tag=teacher_tag,
                                               te_override=te_module)
        n = recorder.register()
        _track_registered(model, "_h3_recorders", recorder)
        model._h3_recorder = recorder
        try:
            import comfy.patcher_extension
            model.add_callback(comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
                               lambda *a, **k: recorder.remove())
        except Exception:
            pass

        fp, count = _lora_fingerprint(model)
        keys_fp, _ = _lora_key_fingerprint(model)
        adaln_fp, _ = _adaln_lora_fingerprint(model)
        shift_v, shift_a = _auto_shifts(model)
        meta = {
            "teacher": teacher_tag or "(unspecified full checkpoint)",
            "lora_tag": lora_tag,
            "lora_fingerprint": fp,
            "lora_fingerprint_keys": keys_fp,
            "lora_patch_count": str(count),
            "adaln_lora_fingerprint": adaln_fp,
            "sigmas": "[]",
            "shift_v": str(float(shift_v)),
            "shift_a": str(float(shift_a)),
            "visual_aug": str(sigma_plan.VISUAL_COND_TIMESTEP),
            "audio_aug": str(sigma_plan.AUDIO_COND_TIMESTEP),
            "compute_dtype": "recorded",
            "out_dtype": "unknown",
            "comfyui_version": _comfy_version(),
            "format": table_io.FORMAT,
        }
        if te_patch:
            meta["time_embedder_from"] = time_embedder_from.strip()
            meta["time_embedder_tensors"] = str(te_patch)
            meta["time_embedder_patch"] = te_mode
        table = table_io.AdaLNTable([], {}, meta, recorder=recorder)

        report = "\n".join([
            "=== H3 Exact AdaLN record (armed) ===",
            "hooked modules : {} projections + time_embedder".format(n),
            "teacher        : {}".format(meta["teacher"]),
            "lora fp        : {} ({} non-AdaLN keys)".format(fp, count),
            "shift_v / a    : {} / {}".format(shift_v, shift_a),
        ] + ([
            "time_embedder  : PATCHED from {!r} ({} tensors, {})".format(
                meta["time_embedder_from"], te_patch,
                "module swap - exact" if te_mode == "module"
                else "weight patch - bf16, NOT bitwise"),
        ] if te_patch else []) + [
            "",
            "The table is EMPTY until the sampler runs. Anything that reads it",
            "(Save / Apply / summary) must execute after sampling -- connect the",
            "sampler's LATENT to that node's `after` input, or split the run into",
            "two prompts (record+save, then load+apply).",
            "",
            "Note: AdaLN depends only on t, so a table recorded in an earlier run",
            "is bitwise identical as long as the schedule, shifts and layout match.",
        ])
        logging.info("%s", report)
        return (model, table, report)


class H3ExactAdaLNBake:
    """Offline approximation from a FULL H3 model (+ optional LoRA). No disk IO.

    Prefer H3AdaLNRecord for bitwise results; see the module docstring.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        except Exception:
            schedulers = ["simple", "normal", "sgm_uniform", "beta", "ddim_uniform"]
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "scheduler": (schedulers, {"default": "simple"}),
                "compute_dtype": (["bfloat16", "float16", "float32"],
                                  {"default": "bfloat16"}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
                # Explicit shifts are safer: an object patch can silently
                # fall back to the model default (audio shift 3 vs DMAD's 2).
                "auto_shift": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "extra_t": ("STRING", {"default": "",
                                       "tooltip": "Extra timesteps, comma separated, "
                                                  "e.g. for denoise_mask rows"}),
                "shift_v": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0,
                                      "step": 0.1}),
                "shift_a": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0,
                                      "step": 0.1}),
                "visual_aug": ("FLOAT", {"default": 0.999, "min": 0.0, "max": 1.0,
                                         "step": 0.001}),
                "audio_aug": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                        "step": 0.001}),
                "teacher_tag": ("STRING", {"default": ""}),
                "lora_tag": ("STRING", {"default": ""}),
                "splice": ("STRING", {"default": "",
                                      "tooltip": "Hybrid teacher: FILE:FIRST-LAST, e.g. "
                                                 "/models/ref2va.safetensors:25-49. Blocks "
                                                 "in that range take their adaln_proj "
                                                 "weight/bias from the donor file. The "
                                                 "donor must be a plain float checkpoint "
                                                 "(bf16/fp16/fp32), not quantised."}),
            },
        }

    RETURN_TYPES = (TABLE_TYPE, "STRING")
    RETURN_NAMES = ("table", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, steps, scheduler, compute_dtype, device, auto_shift,
            sigmas=None, extra_t="", shift_v=12.0, shift_a=3.0,
            visual_aug=0.999, audio_aug=1.0, teacher_tag="", lora_tag="",
            splice=""):
        if sigmas is not None:
            sigma_values = sigma_plan.normalise_sigmas(sigmas)
        else:
            sigma_values = sigma_plan.normalise_sigmas(
                _sigmas_from_model(model, int(steps), scheduler))
        if not sigma_values:
            raise RuntimeError("H3ExactAdaLNBake: empty sigma schedule")

        if auto_shift:
            shift_v, shift_a = _auto_shifts(model)

        t_values = sigma_plan.bake_t_values(
            sigma_values, shift_v, shift_a, visual_aug, audio_aug,
            extra_t=_parse_float_list(extra_t))
        if not t_values:
            raise RuntimeError("H3ExactAdaLNBake: no timesteps derived from the schedule")

        dev = None
        if device == "cpu":
            dev = torch.device("cpu")
        elif device == "cuda":
            dev = torch.device("cuda")

        def progress(i, total):
            logging.info("H3ExactAdaLNBake: projection %d/%d", i, total)

        raw_by_key, info = bake_engine.bake(
            model, t_values, compute_dtype=compute_dtype, device=dev,
            progress=progress, splice=splice)

        dm = model.model.diffusion_model
        fingerprint, patch_count = _lora_fingerprint(model)
        keys_fingerprint, _ = _lora_key_fingerprint(model)
        adaln_fingerprint, _ = _adaln_lora_fingerprint(model)
        meta = {
            "teacher": teacher_tag or "(unspecified full checkpoint)",
            "lora_tag": lora_tag,
            "lora_fingerprint": fingerprint,
            "lora_fingerprint_keys": keys_fingerprint,
            "lora_patch_count": str(patch_count),
            "adaln_lora_fingerprint": adaln_fingerprint,
            "sigmas": json.dumps([float(s) for s in sigma_values]),
            "shift_v": str(float(shift_v)),
            "shift_a": str(float(shift_a)),
            "visual_aug": str(float(visual_aug)),
            "audio_aug": str(float(audio_aug)),
            "compute_dtype": compute_dtype,
            "num_blocks": str(info["projections"] - 1),
            "t_embed_dim": str(info["t_embed_dim"]),
            "applied_lora_patches": str(info["applied_patches"]),
            "comfyui_version": _comfy_version(),
            "out_dtype": bake_engine.teacher_out_dtype(dm),
            "record_mode": "offline bake (approximate, use Record for bitwise)",
            "format": table_io.FORMAT,
        }
        meta.update({k: v for k, v in info.items() if k.startswith("splice")})
        table = table_io.AdaLNTable(t_values, raw_by_key, meta)

        lines = [
            "=== H3 Exact AdaLN bake (offline, approximate) ===",
            table.summary(),
            "auto shift     : {} (v={}, a={})".format(auto_shift, shift_v, shift_a),
            "applied patches: {}/{} AdaLN+timestep keys".format(
                info["applied_patches"], info["target_patch_keys"]),
            "teacher weights: te={} ({}) | adaln={} ({}) in_features={}".format(
                info.get("teacher_te_type", "?"), info.get("teacher_te_dtype", "?"),
                info.get("teacher_adaln_type", "?"), info.get("teacher_adaln_dtype", "?"),
                info.get("teacher_adaln_in_features", "?")),
            "device         : {}".format(info["device"]),
        ]
        if info.get("splice"):
            lines.append("splice         : {} (blocks {}..{}, fingerprint {})".format(
                info["splice"], info.get("splice_first"), info.get("splice_last"),
                info.get("splice_fingerprint")))
        if meta["out_dtype"].split(".")[-1] != compute_dtype:
            lines.append("note           : compute_dtype={} but the teacher's AdaLN path "
                         "runs in {}; the table stores that dtype so replay matches "
                         "the model.".format(compute_dtype, meta["out_dtype"]))
        lines += [
            "",
            "This is the OFFLINE path: it cannot see the kernels the sampler uses, so",
            "it is not bitwise against a native run. Use H3AdaLNRecord when exactness",
            "matters; keep Bake for exploring schedules without sampling.",
            "",
            "Next: connect `table` to H3 Exact AdaLN Save.",
        ]
        report = "\n".join(lines)
        logging.info("%s", report)
        return {"ui": {"text": [report]}, "result": (table, report)}


class H3ExactAdaLNSave:
    """Save a table like Save Image: auto-numbered file under output/."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "table": (TABLE_TYPE,),
                "filename_prefix": ("STRING", {
                    "default": "h3_adaln/H3Exact",
                    "tooltip": "Like Save Image. Subfolders allowed, e.g. "
                               "h3_adaln/dmad8. Files are auto-numbered."}),
            },
            "optional": {
                "location": (["output", "models/h3_exact_adaln"],
                             {"default": "output"}),
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, table, filename_prefix="h3_adaln/H3Exact", location="output",
             after=None):
        import shutil
        import folder_paths
        from urllib.parse import quote

        table.finalize(required=True)

        prefix = str(filename_prefix).strip() or "h3_adaln/H3Exact"
        if location == "output":
            folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix, folder_paths.get_output_directory())
            name = "{}_{:05d}.safetensors".format(filename, counter)
            path = os.path.join(folder, name)
            rel = os.path.join(subfolder, name).replace("\\", "/")
            ui_type = "output"
            download_url = "/view?filename={}&subfolder={}&type=output".format(
                quote(name), quote(subfolder))
        else:
            folder = table_io.table_dir()
            base = os.path.basename(os.path.normpath(prefix)) or "H3Exact"
            counter = 1
            while True:
                name = "{}_{:05d}.safetensors".format(base, counter)
                path = os.path.join(folder, name)
                if not os.path.exists(path):
                    break
                counter += 1
            rel = name
            ui_type = "input"
            download_url = None

        table_io.write_table_file(path, table)
        table.source = os.path.abspath(path)

        # Mirror into input/ so the Load node, ComfyUI's file picker and the
        # platform's asset browser can all reach it without digging through
        # output/h3_adaln/. Saving twice the file size is acceptable (tables
        # are a few hundred MB) and avoids upload-widget format filtering.
        mirrored = None
        try:
            inp = folder_paths.get_input_directory()
            os.makedirs(inp, exist_ok=True)
            dst = os.path.join(inp, name)
            if os.path.abspath(dst) != os.path.abspath(path):
                shutil.copy2(path, dst)
                mirrored = dst
        except Exception as e:
            logging.warning("H3ExactAdaLNSave: could not mirror into input/: %s", e)

        # ComfyUI's /view endpoint serves any file under output/ with
        # application/octet-stream, so this link downloads directly.
        ui = {
            "text": ["=== H3 Exact AdaLN table saved ===",
                     "path     : {}".format(path),
                     "relative : {}".format(rel),
                     "location : {}".format(location),
                     "download : {}".format(download_url or "(use the path above)"),
                     "input mirror : {}".format(mirrored or "(unavailable)"),
                     "tip      : paste <ComfyUI host>{} into your browser to "
                                "download without opening the folder".format(download_url or ""),
                     "",
                     table.summary()],
            "h3_adaln_tables": [
                {"filename": name, "subfolder": os.path.dirname(rel),
                 "type": ui_type, "download_url": download_url,
                 "path": path}
            ],
        }
        # Also expose it through ComfyUI's generic output panel. Non-image
        # files show a broken thumbnail but the panel still offers /view
        # download; harmless if this frontend ignores unknown extensions.
        if download_url:
            ui["images"] = [{"filename": name,
                             "subfolder": os.path.dirname(rel),
                             "type": "output"}]
        report = "\n".join(ui["text"])
        logging.info("%s", report)
        return {"ui": ui, "result": (path,)}


class H3ExactAdaLNLoad:
    """Load a shared table by name / path (free text, no dropdown cache).

    A COMBO + `image_upload` widget cannot work here: ComfyUI only builds that
    list once at startup (so a table saved later never appears), and the upload
    button is image-only (so .safetensors cannot be selected). The table is
    already on the server anyway -- pointing at it by text is the reliable way.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "table_name": ("STRING", {
                    "default": "",
                    "tooltip": "Filename, partial name (e.g. dmad8), relative "
                               "path under input/ or output/, or an absolute "
                               "path to a baked .safetensors table."}),
            },
            "optional": {
                "table_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path override; takes priority over "
                               "table_name when set."}),
            },
        }

    RETURN_TYPES = (TABLE_TYPE, "STRING")
    RETURN_NAMES = ("table", "info")
    OUTPUT_NODE = True
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, table_name="", table_path=""):
        query = str(table_path).strip() or str(table_name).strip()
        if not query:
            raise RuntimeError(
                "H3ExactAdaLNLoad: enter a table filename (or part of it) in "
                "`table_name`, or an absolute path in `table_path`.\n"
                + table_io.describe_available())
        if os.path.isabs(query):
            path = query if os.path.isfile(query) else None
        else:
            path = table_io.find_table_file(query)
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                "H3ExactAdaLNLoad: cannot resolve {!r} to a table file.\n{}".format(
                    query, table_io.describe_available()))

        table = table_io.read_table_file(path)
        report = "\n".join([
            "=== H3 Exact AdaLN table loaded ===",
            table.summary(),
            "",
            "Next: connect `table` to H3 Exact AdaLN Apply on the pruned base.",
        ])
        logging.info("%s", report)
        return {"ui": {"text": [report]}, "result": (table, report)}


class H3AdaLNMerge:
    """Assemble the exact table for a bX-49 hybrid from two recorded tables.

    A `bX-49` checkpoint is curve-shaped everywhere (51 x `[96768, 8]`, no
    `time_embedder`), with blocks X..49 taken from a second teacher. Record
    cannot run on it, and its 8-wide curve cannot carry the teachers' full
    modulation -- so record each full teacher separately, then merge:
    blocks 0..X-1 from A, blocks X..49 from B. Every block stays a row its
    teacher actually produced, so replay is exact per block.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "table_a": (TABLE_TYPE, {
                    "tooltip": "Base teacher record (e.g. FL2VA full)."}),
                "table_b": (TABLE_TYPE, {
                    "tooltip": "Tail donor record (e.g. Ref2VA full)."}),
                "split": ("INT", {
                    "default": 25, "min": 0, "max": 50, "step": 1,
                    "tooltip": "First block taken from table_b. b25-49 -> 25."}),
                "final_from": (["a", "b"], {
                    "default": "a",
                    "tooltip": "Which table supplies final_layer. bX-49 keeps "
                               "the base teacher's final layer -> 'a'."}),
            },
            "optional": {
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
        }

    RETURN_TYPES = (TABLE_TYPE, "STRING")
    RETURN_NAMES = ("table", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, table_a, table_b, split=25, final_from="a", after=None):
        merged = table_io.merge_tables(
            table_a, table_b, split=split, final_from=final_from)
        source_a = merged.meta.get("hybrid_source_a", "?")
        source_b = merged.meta.get("hybrid_source_b", "?")
        seg_a = "blocks 0-{}".format(split - 1) if split > 0 else "blocks (none)"
        seg_b = "blocks {}-49".format(split) if split < 50 else "blocks (none)"
        report = "\n".join([
            "=== H3 AdaLN merge (bX-49 hybrid) ===",
            "{} : A = {}".format(seg_a, source_a),
            "{} : B = {}".format(seg_b, source_b),
            "final_layer    : {}".format("A" if final_from == "a" else "B"),
            "t rows         : {}".format(len(merged.t_values)),
            "projections    : {}".format(merged.projections()),
            "out dtype      : {}".format(merged.out_dtype),
            "table size     : {:.1f} MiB".format(merged.size_bytes() / (1 << 20)),
            "lora fp        : A={} / B={}".format(
                merged.meta.get("hybrid_lora_fp_a", "?"),
                merged.meta.get("hybrid_lora_fp_b", "?")),
            "time_embedder  : A={} / B={}{}".format(
                merged.meta.get("hybrid_te_from_a") or "(own)",
                merged.meta.get("hybrid_te_from_b") or "(own)",
                "" if (split <= 0 or merged.meta.get("hybrid_te_from_b"))
                else "  <-- tail uses the donor's own embedding: this is the OLD "
                     "recipe and differs from a real bX-49 hybrid"),
        ])
        logging.info("%s", report)
        return (merged, report)


class H3ExactAdaLNApply:
    """Replace all 51 AdaLN projections with rows from a loaded table."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "table": (TABLE_TYPE,),
                "mode": (_MODES, {"default": "strict"}),
            },
            "optional": {
                "expected_teacher": ("STRING", {"default": ""}),
                "after": ("*", {"tooltip": _AFTER_DOC}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, table, mode, expected_teacher="", after=None):
        # Also runs for `off`: a previous strict run may have left hooks on the
        # shared nn.Module. Without this the negative control would keep
        # injecting the old table and could report a false PASS.
        _clear_registered(model, "_h3_exact_runtimes")
        if mode == "off":
            return (model, "H3ExactAdaLNApply: mode=off, model untouched (A/B control)")

        table.finalize(required=True)
        dm = model.model.diffusion_model
        arch = "pruned/curve" if getattr(dm, "use_adaln_curves", False) else "full"
        current_fp, current_count = _lora_fingerprint(model)
        current_keys_fp, _ = _lora_key_fingerprint(model)
        baked_fp = table.meta.get("lora_fingerprint", "")
        baked_keys_fp = table.meta.get("lora_fingerprint_keys", "")
        fp_ok = (current_fp == baked_fp)
        if not fp_ok and not baked_keys_fp:
            # Table written by an older build: only the key-name digest is
            # stored, so that is the strongest check it can support.
            fp_ok = bool(current_keys_fp) and current_keys_fp == baked_fp
        teacher_ok = (not expected_teacher) or (
            expected_teacher in table.meta.get("teacher", ""))
        adaln_baked = table.meta.get("adaln_lora_fingerprint", "")
        adaln_current, _ = _adaln_lora_fingerprint(model)
        # A pruned base legitimately has no AdaLN patches (ComfyUI drops them),
        # so only compare when both sides actually carry an AdaLN LoRA.
        adaln_ok = not (adaln_baked and adaln_current) or adaln_baked == adaln_current

        # A table stores `adaln_proj(time_embedder(t))`. Neither of those two
        # modules is touched by a LoRA that only patches attention / MLP, so a
        # different non-AdaLN LoRA cannot invalidate a recorded table -- verified
        # bitwise (record with LoRA A, apply with LoRA B, latent diff 0). Only
        # the AdaLN / timestep patches, and an explicitly requested teacher, can.
        model_dtype = _model_compute_dtype(model, dm)
        table_dtype = str(table.out_dtype or "").split(".")[-1]
        dtype_ok = None
        if model_dtype is not None:
            dtype_ok = str(model_dtype).split(".")[-1] == table_dtype
            if not dtype_ok:
                logging.warning(
                    "H3ExactAdaLN: table out_dtype=%s but this model computes in %s; "
                    "FinalLayer.mod picks its multiply-add precision from scale.dtype, "
                    "so replay will not be bitwise",
                    table_dtype, str(model_dtype).split(".")[-1])
        teacher_note = ""
        if teacher_ok and expected_teacher and expected_teacher != table.meta.get("teacher", ""):
            teacher_note = " (substring match)"

        problems = []
        if not adaln_ok:
            problems.append("AdaLN/timestep LoRA fingerprint mismatch: "
                            "table={} model={} -- re-record with this teacher LoRA".format(
                                adaln_baked, adaln_current))
        if not teacher_ok:
            problems.append("teacher mismatch: expected {!r}, table has {!r}".format(
                expected_teacher, table.meta.get("teacher", "")))
        if problems and mode == "strict":
            raise RuntimeError("H3ExactAdaLNApply: " + " | ".join(problems) +
                               " -- re-record the table with this exact model+LoRA stack.")

        old = getattr(model, "_h3_exact_runtime", None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            model._h3_exact_runtime = None

        runtime = apply_engine.ExactAdaLNRuntime(
            table.t_values, table.raw_by_key, table.meta, dm, mode=mode)
        n = runtime.register(model)
        _track_registered(model, "_h3_exact_runtimes", runtime)
        model._h3_exact_runtime = runtime
        try:
            import comfy.patcher_extension
            model.add_callback(comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
                               lambda *a, **k: runtime.remove())
        except Exception:
            pass

        report = "\n".join([
            "=== H3 Exact AdaLN apply ===",
            "table          : {}".format(table.source or "(in-memory)"),
            "provenance     : {}".format(table.meta.get("record_mode", "offline bake")),
            "target arch    : {}".format(arch),
            "hooked modules : {}".format(n),
            "lora fp        : {} ({}){}".format(
                current_fp, "match" if fp_ok else "differs - harmless, AdaLN is unaffected",
                "" if baked_keys_fp else " [legacy table: keys-only check]"),
            "adaln lora fp  : {} / {} ({})".format(
                adaln_baked or "(none)", adaln_current or "(none)",
                "match" if adaln_ok else "MISMATCH"),
            "teacher        : {}{}".format(table.meta.get("teacher", "?"), teacher_note),
            "out dtype      : {}{}".format(
                table.out_dtype,
                "" if dtype_ok is None
                else ("  (model computes in {}, match)".format(str(model_dtype).split(".")[-1])
                      if dtype_ok else
                      "  MISMATCH: model computes in {} -- replay is NOT bitwise"
                      .format(str(model_dtype).split(".")[-1]))),
            "t rows         : {}".format(len(table.t_values)),
            "sigmas in table: {}".format(str(table.meta.get("sigmas", "?"))[:180]),
            "warnings       : {}".format("; ".join(problems) or "none"),
            "",
            runtime.summary(),
        ])
        logging.info("%s", report)
        return (model, report)


class H3LatentCompare:
    """Verification node: are two LATENT outputs bit-identical?"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "label_a": ("STRING", {"default": "full model (native)"}),
                "label_b": ("STRING", {"default": "pruned + exact table"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, latent_a, latent_b, label_a, label_b):
        # ComfyUI 0.44 packs H3's video/audio latents into NestedTensor,
        # which is not a torch.Tensor. Expand it into plain tensors first.
        def normalize(obj):
            if not isinstance(obj, dict):
                return obj
            out = {}
            for k, v in obj.items():
                if torch.is_tensor(v):
                    out[k] = v
                elif getattr(v, "is_nested", False) and hasattr(v, "unbind"):
                    for i, part in enumerate(v.unbind()):
                        out["{}__part{}".format(k, i)] = part
                else:
                    out[k] = v
            return out

        latent_a = normalize(latent_a)
        latent_b = normalize(latent_b)

        def collect(obj, prefix="", out=None, depth=0):
            """Find tensors inside dict / list / Latent-like wrapper objects."""
            if out is None:
                out = {}
            if depth > 4:
                return out
            if torch.is_tensor(obj):
                out[prefix or "tensor"] = obj
                return out
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = "{}/{}".format(prefix, k) if prefix else str(k)
                    collect(v, p, out, depth + 1)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    collect(v, "{}[{}]".format(prefix or "v", i), out, depth + 1)
            elif hasattr(obj, "__dict__"):
                for name in ("samples", "latent", "data", "x"):
                    v = getattr(obj, name, None)
                    if v is not None:
                        collect(v, "{}/{}".format(prefix, name) if prefix else name,
                                out, depth + 1)
            return out

        def strip_side(d, side):
            return {k.split("/", 1)[1] if k.startswith(side + "/") else k: v
                    for k, v in d.items()}

        raw_a = collect(latent_a, "a")
        raw_b = collect(latent_b, "b")
        ta = strip_side(raw_a, "a")
        tb = strip_side(raw_b, "b")
        # A self-comparison is always bitwise equal. Refuse to call that PASS:
        # the two chains in a verification graph must produce distinct tensors.
        same_source = latent_a is latent_b or (
            bool(raw_a) and
            {id(v) for v in raw_a.values()} == {id(v) for v in raw_b.values()})
        def describe(obj):
            if isinstance(obj, dict):
                items = [(k, type(v).__name__,
                          tuple(v.shape) if torch.is_tensor(v) else "")
                         for k, v in obj.items()]
                return "dict " + str(items)[:500]
            return "{} repr={}".format(type(obj).__name__, repr(obj)[:400])

        lines = ["=== latent comparison ===",
                 "A: {}".format(label_a), "B: {}".format(label_b),
                 "inputs : A={} keys={} | B={} keys={}".format(
                     type(latent_a).__name__, sorted(ta),
                     type(latent_b).__name__, sorted(tb)),
                 "raw A  : {}".format(describe(latent_a)),
                 "raw B  : {}".format(describe(latent_b))]
        all_equal = True
        compared = 0
        for key in sorted(set(ta) | set(tb)):
            a, b = ta.get(key), tb.get(key)
            if a is None or b is None:
                lines.append("{}: MISSING on {} side".format(
                    key, "A" if a is None else "B"))
                all_equal = False
                continue
            if a.shape != b.shape:
                lines.append("{}: SHAPE MISMATCH {} vs {}".format(
                    key, tuple(a.shape), tuple(b.shape)))
                all_equal = False
                continue
            af, bf = a.detach().float().cpu(), b.detach().float().cpu()
            eq = torch.equal(a.detach().cpu(), b.detach().cpu())
            d = (af - bf).abs()
            max_abs = float(d.max())
            mean_abs = float(d.mean())
            mse = float((af - bf).square().mean())
            peak = float(max(af.max() - af.min(), 1e-8))
            psnr = float("inf") if mse == 0 else 10.0 * torch.log10(
                torch.tensor(peak * peak / mse)).item()
            lines.append(
                "{}: equal={} max_abs={:.3e} mean_abs={:.3e} mse={:.3e} "
                "psnr={} shape={}".format(
                    key, eq, max_abs, mean_abs, mse,
                    "inf" if mse == 0 else "{:.2f} dB".format(psnr),
                    tuple(a.shape)))
            all_equal = all_equal and eq
            compared += 1
        lines.append("compared tensor pairs: {}".format(compared))
        if same_source:
            lines.append("WARNING: both LATENT inputs resolve to the same tensor "
                         "objects -- check the wiring")
        if same_source:
            verdict = ("INCONCLUSIVE - both LATENT inputs resolve to the same "
                       "tensor objects; a self-comparison always looks bitwise")
        elif compared == 0:
            verdict = ("INCONCLUSIVE - no latent tensors were compared; "
                       "check the two LATENT inputs")
        elif all_equal:
            verdict = "PASS - bitwise identical"
        else:
            verdict = "FAIL - outputs differ (see max_abs / psnr above)"
        lines.append("")
        lines.append("VERDICT: {}".format(verdict))
        report = "\n".join(lines)
        logging.info("%s", report)

        # Also persist the report so it can be read without the UI/log panel.
        try:
            import folder_paths
            from datetime import datetime
            folder = os.path.join(folder_paths.get_output_directory(), "h3_adaln")
            os.makedirs(folder, exist_ok=True)
            tag = "PASS" if (all_equal and compared) else "OTHER"
            path = os.path.join(folder, "compare_{}_{}.txt".format(
                tag, datetime.now().strftime("%Y%m%d_%H%M%S")))
            lines.append("report file: {}".format(path))
            report = "\n".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report + "\n")
        except Exception as e:
            logging.warning("H3LatentCompare: could not write report file: %s", e)
            lines.append("report file: (unavailable: {})".format(e))
            report = "\n".join(lines)
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {
    "H3AdaLNRecord": H3AdaLNRecord,
    "H3ExactAdaLNBake": H3ExactAdaLNBake,
    "H3ExactAdaLNSave": H3ExactAdaLNSave,
    "H3ExactAdaLNLoad": H3ExactAdaLNLoad,
    "H3AdaLNMerge": H3AdaLNMerge,
    "H3ExactAdaLNApply": H3ExactAdaLNApply,
    "H3LatentCompare": H3LatentCompare,
    "H3LatentSave": latent_io.H3LatentSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AdaLNRecord": "H3 AdaLN Record (exact, during sampling)",
    "H3ExactAdaLNBake": "H3 Exact AdaLN Bake (offline, approximate)",
    "H3ExactAdaLNSave": "H3 Exact AdaLN Save Table",
    "H3ExactAdaLNLoad": "H3 Exact AdaLN Load Table",
    "H3AdaLNMerge": "H3 AdaLN Merge (bX-49 hybrid)",
    "H3ExactAdaLNApply": "H3 Exact AdaLN Apply (pruned base)",
    "H3LatentCompare": "H3 Latent Compare (verification)",
    "H3LatentSave": "H3 Latent Save (NestedTensor)",
}
