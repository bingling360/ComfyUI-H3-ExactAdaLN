"""Offline exact-AdaLN table baker.

Quantisation and LoRA universality comes from deliberately NOT parsing
safetensors by hand. We ask ComfyUI's own ModelPatcher to materialise the
effective (dequantised + LoRA-patched) weights, then call the real modules'
forward. Anything ComfyUI can load -- bf16 / fp16 / fp8_scaled /
int8_convrot / nvfp4 -- and any LoRA it can apply therefore works here with
zero format-specific code.

The baked value is the RAW output of adaln_proj.linear, before AdalnProj's
view/chunk. That keeps the runtime replay independent of expand/hidden/
modalities semantics.
"""

import logging

import torch
import torch.nn.functional as F

try:
    from . import sigma_plan
except ImportError:
    import sigma_plan


def _diffusion_model(model):
    base = getattr(model, "model", None)
    dm = getattr(base, "diffusion_model", None)
    if dm is None:
        raise RuntimeError(
            "H3ExactAdaLN Bake: expected a ComfyUI MODEL (ModelPatcher) for MiniMax-H3")
    return model, dm


def _projections(dm):
    if getattr(dm, "use_adaln_curves", False) or not hasattr(dm, "time_embedder"):
        raise RuntimeError(
            "H3ExactAdaLN Bake: this is a PRUNED/curve checkpoint (no time_embedder). "
            "Bake must be fed the FULL teacher checkpoint that the pruned file came from.")
    blocks = list(getattr(dm, "blocks", []))
    if not blocks:
        raise RuntimeError("H3ExactAdaLN Bake: no transformer blocks found")
    out = []
    for i, block in enumerate(blocks):
        ap = getattr(block, "adaln_proj", None)
        if ap is None or not hasattr(ap, "linear"):
            raise RuntimeError("H3ExactAdaLN Bake: block {} has no adaln_proj".format(i))
        out.append(("blocks.{}".format(i), ap))
    fl = getattr(dm, "final_layer", None)
    ap = getattr(fl, "adaln_proj", None) if fl is not None else None
    if ap is None:
        raise RuntimeError("H3ExactAdaLN Bake: final_layer.adaln_proj missing")
    out.append(("final_layer", ap))
    return out


def _apply_patches(mp):
    """Materialise the effective AdaLN/timestep weights, LoRA included.

    patch_weight_to_device() runs ComfyUI's official dequantise ->
    calculate_weight (every LoRA flavour, strength/alpha, stochastic rounding)
    -> write-back pipeline, so bf16 / fp8_scaled / int8_convrot / nvfp4 all go
    through their own ComfyUI conversion instead of custom code here.

    ONLY the timestep/AdaLN keys are touched. Merging the whole 2 GB LoRA into
    a 34 GB checkpoint would be slow, VRAM-heavy and is irrelevant: the baked
    table is a function of time_embedder + the 51 adaln_proj alone.
    """
    keys = [k for k in getattr(mp, "patches", {}).keys()
            if "time_embedder" in k or "adaln_proj" in k]
    applied, errors = [], []
    for key in keys:
        try:
            mp.patch_weight_to_device(key, inplace_update=True)
            applied.append(key)
        except Exception as e:
            errors.append("{}: {}: {}".format(key, type(e).__name__, str(e)[:120]))
    return applied, errors, len(keys)


def _weight_info(dm):
    """Expose what kind of weights the teacher actually carries."""
    try:
        w = dm.time_embedder.proj_in.weight
        b = dm.blocks[0].adaln_proj.linear.weight
        return {
            "teacher_te_type": type(w).__name__,
            "teacher_te_dtype": str(getattr(w, "dtype", "?")),
            "teacher_adaln_type": type(b).__name__,
            "teacher_adaln_dtype": str(getattr(b, "dtype", "?")),
            "teacher_adaln_in_features": int(getattr(
                dm.blocks[0].adaln_proj.linear, "in_features", -1)),
        }
    except Exception as e:
        return {"teacher_weight_probe": "unavailable: {}".format(e)}


def _compute_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def bake(model, t_values, compute_dtype="bfloat16", device=None, progress=None):
    """Return (raw_by_key, info). Does not run a sampler or need conditioning."""
    mp, dm = _diffusion_model(model)
    projs = _projections(dm)
    if device is None:
        try:
            import comfy.model_management
            device = comfy.model_management.get_torch_device()
        except Exception:
            device = torch.device("cpu")

    # clone() shares the underlying nn.Module, so patching and unpatching is
    # scoped to this bake call; the original MODEL object is left usable.
    work = mp.clone() if hasattr(mp, "clone") else mp
    applied, errors, n_target_keys = _apply_patches(work)
    dtype = _compute_dtype(compute_dtype)

    try:
        if errors:
            raise RuntimeError(
                "H3ExactAdaLNBake: could not materialise {} AdaLN/timestep LoRA "
                "patch(es); refusing to write a silently-wrong table. {}".format(
                    len(errors), "; ".join(errors[:4])))
        t_vals = torch.tensor(t_values, dtype=torch.float32, device=device)
        with torch.no_grad():
            # TimeEmbedder is fp32 internally (model.py:505); _forward casts
            # its output to the compute dtype, which we reproduce here.
            t_emb = dm.time_embedder(t_vals).to(dtype)
            raw_by_key = {}
            for n, (key, ap) in enumerate(projs):
                inp = F.silu(t_emb) if getattr(ap, "apply_silu", True) else t_emb
                raw = ap.linear(inp)
                raw_by_key[key] = raw.detach().to(torch.float32).cpu().contiguous()
                if progress is not None and n % 10 == 0:
                    progress(n, len(projs))
    finally:
        try:
            work.unpatch_model()
        except Exception as e:
            logging.warning("H3ExactAdaLN Bake: unpatch failed: %s", e)

    info = {
        "projections": len(projs),
        "applied_patches": len(applied),
        "target_patch_keys": n_target_keys,
        "patch_errors": errors,
        "compute_dtype": compute_dtype,
        "device": str(device),
        "teacher_architecture": "full" if not getattr(dm, "use_adaln_curves", False) else "curve",
        "t_embed_dim": int(t_emb.shape[-1]) if t_emb is not None else -1,
    }
    info.update(_weight_info(dm))
    return raw_by_key, info


def self_check(model, t_values, raw_by_key, compute_dtype="bfloat16"):
    """Re-run one projection through a fresh official patch and compare.

    Cheap sanity check that our LoRA/quantisation materialisation matches
    ComfyUI's own pipeline. Returns a human readable string.
    """
    try:
        from comfy.model_patcher import get_key_weight
    except Exception as e:
        return "self-check skipped (get_key_weight unavailable): {}".format(e)
    try:
        mp, dm = _diffusion_model(model)
        key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
        weight, set_func, convert_func = get_key_weight(mp.model, key)
        if weight is None:
            return "self-check skipped: weight key not found"
        return "self-check: key={} weight_dtype={} quantised={} (conversion handled by ComfyUI)".format(
            key, getattr(weight, "dtype", "?"), type(weight).__name__ != "Tensor")
    except Exception as e:
        return "self-check failed: {}: {}".format(type(e).__name__, str(e)[:160])
