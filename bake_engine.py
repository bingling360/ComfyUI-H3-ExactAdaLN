"""Offline exact-AdaLN table baker.

**Record mode is the exact path** (see `record_engine.py`): this module cannot
reproduce the kernels ComfyUI picks at sampling time, so its output is an
approximation. It is kept because it is cheap, needs no sampling run, and is
useful for exploring a schedule. Measured on the real 34 GB teacher the
offline numbers differ from the sampling-time ones by up to one bf16 ulp
(0.015625 on `blocks.0`), which eight diffusion steps amplify to ~0.145
mean_abs on the latent. Use Record when "bitwise" is the requirement.

Quantisation and LoRA universality comes from deliberately NOT parsing
safetensors by hand: we ask ComfyUI for the effective (dequantised +
LoRA-patched) weights through its own conversion hooks, then do the matmul
ourselves. Anything ComfyUI can load -- bf16 / fp16 / fp8_scaled /
int8_convrot / nvfp4 -- and any LoRA it can apply therefore works here with
zero format-specific code.

Nothing is written back into the model: no `patch_weight_to_device`, no
`unpatch_model`, no `clone()`. The previous implementation mutated the shared
`nn.Module` (clone() shares it) and left the caller's ModelPatcher holding
restored weights from a half-finished unpatch.

The baked value is the RAW output of adaln_proj.linear, before AdalnProj's
view/chunk. That keeps the runtime replay independent of expand/hidden/
modalities semantics.
"""

import hashlib
import logging
import math
import os

import torch
import torch.nn.functional as F


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


def _compute_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _effective_weight(mp, key, dtype=None):
    """Dequantise + apply every LoRA patch for one key. Read-only.

    Mirrors `ModelPatcher.patch_weight_to_device`'s pipeline (cast -> convert ->
    calculate_weight) but stops before `set_func`, because `set_func` for a
    quantised op requantises the tensor instead of handing back floats.
    """
    try:
        from comfy.model_patcher import get_key_weight
    except Exception as e:  # pragma: no cover - very old ComfyUI
        raise RuntimeError("H3ExactAdaLN Bake: comfy.model_patcher.get_key_weight "
                           "unavailable ({})".format(e))
    weight, _set_func, convert_func = get_key_weight(mp.model, key)
    if weight is None:
        return None
    if convert_func is not None:
        try:
            weight = convert_func(weight, inplace=False)
        except TypeError:
            weight = convert_func(weight)
    patches = getattr(mp, "patches", {}) or {}
    if key in patches and patches[key]:
        import comfy.lora
        weight = comfy.lora.calculate_weight(patches[key], weight, key)
    if dtype is not None and getattr(weight, "is_floating_point", lambda: False)():
        weight = weight.to(dtype)
    return weight


def _time_embed(dm, mp, t_vals, dtype, device):
    """Run TimeEmbedder's math on effective weights (LoRA included)."""
    te = dm.time_embedder
    try:
        w_in = _effective_weight(mp, "diffusion_model.time_embedder.proj_in.weight")
        b_in = _effective_weight(mp, "diffusion_model.time_embedder.proj_in.bias")
        w_out = _effective_weight(mp, "diffusion_model.time_embedder.proj_out.weight")
        b_out = _effective_weight(mp, "diffusion_model.time_embedder.proj_out.bias")
    except Exception as e:
        logging.warning("H3ExactAdaLN Bake: falling back to the module's own "
                        "time_embedder (%s: %s)", type(e).__name__, e)
        w_in = w_out = None
        b_in = b_out = None
    if w_in is None or w_out is None:
        # No patches on the time embedder at all: the module itself is exact.
        with torch.no_grad():
            return te(t_vals.to(device)).to(dtype)

    w_in = w_in.to(device=device, dtype=torch.float32)
    b_in = b_in.to(device=device, dtype=torch.float32) if b_in is not None else None
    w_out = w_out.to(device=device, dtype=torch.float32)
    b_out = b_out.to(device=device, dtype=torch.float32) if b_out is not None else None
    half = int(te.freq_dim) // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = t_vals.to(device=device, dtype=torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    with torch.no_grad():
        h = F.silu(F.linear(emb, w_in, b_in))
        out = F.linear(h, w_out, b_out)
    del w_in, b_in, w_out, b_out
    return out.to(dtype)


# -- splice (hybrid teacher) ------------------------------------------------
def parse_splice(spec):
    """'FILE:FIRST-LAST' -> (path, first, last). Windows paths keep their colon."""
    if not spec:
        return None
    text = str(spec).strip()
    if not text:
        return None
    if ":" not in text:
        raise RuntimeError(
            "H3ExactAdaLN Bake: `splice` must look like "
            "'/path/to/ref2va.safetensors:25-49' (file:first-last). Got {!r}".format(text))
    path, span = text.rsplit(":", 1)
    path = path.strip()
    span = span.strip()
    if "-" in span:
        a, b = span.split("-", 1)
    else:
        a = b = span
    try:
        first, last = int(a), int(b)
    except ValueError:
        raise RuntimeError("H3ExactAdaLN Bake: bad splice span {!r}; expected FIRST-LAST"
                           .format(span))
    if first > last:
        first, last = last, first
    if not os.path.isfile(path):
        raise RuntimeError("H3ExactAdaLN Bake: splice donor not found: {}".format(path))
    return path, first, last


def _load_splice(spec, projs):
    """Load donor adaln weights for blocks[first..last]. Plain-float only."""
    parsed = parse_splice(spec)
    if parsed is None:
        return None
    path, first, last = parsed
    from safetensors import safe_open
    wanted = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key, ap in projs:
            if not key.startswith("blocks."):
                continue
            idx = int(key.split(".")[1])
            if not (first <= idx <= last):
                continue
            for suffix in ("weight", "bias"):
                target = "{}.adaln_proj.linear.{}".format(key, suffix)
                hits = [k for k in keys if k == target or k.endswith("." + target)]
                if not hits and suffix == "bias":
                    continue  # bias is optional in the donor
                if not hits:
                    raise RuntimeError(
                        "H3ExactAdaLN Bake: splice donor {} has no {!r}".format(path, target))
                tensor = f.get_tensor(hits[0])
                if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                    raise RuntimeError(
                        "H3ExactAdaLN Bake: splice donor tensor {!r} is {}; splice needs "
                        "a plain float donor (bf16/fp16/fp32), not a quantised one."
                        .format(hits[0], tensor.dtype))
                wanted[(key, suffix)] = tensor
    if not wanted:
        raise RuntimeError("H3ExactAdaLN Bake: splice span {}-{} selected no blocks"
                           .format(first, last))
    digest = hashlib.sha256()
    for k in sorted(wanted):
        digest.update(str(k).encode("utf-8"))
        digest.update(wanted[k].to(torch.float32).numpy().tobytes())
    info = {
        "splice": "{}:{}-{}".format(os.path.basename(path), first, last),
        "splice_first": str(first),
        "splice_last": str(last),
        "splice_donor": os.path.abspath(path),
        "splice_fingerprint": digest.hexdigest()[:16],
        "splice_blocks": str(last - first + 1),
    }
    return wanted, info


# -- bake -------------------------------------------------------------------
def bake(model, t_values, compute_dtype="bfloat16", device=None, progress=None,
         splice=None):
    """Return (raw_by_key, info). Does not run a sampler or need conditioning."""
    mp, dm = _diffusion_model(model)
    projs = _projections(dm)
    if device is None:
        try:
            import comfy.model_management
            device = comfy.model_management.get_torch_device()
        except Exception:
            device = torch.device("cpu")

    dtype = _compute_dtype(compute_dtype)
    splice_weights, splice_info = (None, {})
    if splice:
        splice_weights, splice_info = _load_splice(splice, projs)

    t_vals = torch.tensor(t_values, dtype=torch.float32)
    t_emb = _time_embed(dm, mp, t_vals, dtype, device)

    raw_by_key = {}
    errors = []
    applied = 0
    for n, (key, ap) in enumerate(projs):
        try:
            wkey = "diffusion_model.{}.adaln_proj.linear.weight".format(key)
            bkey = "diffusion_model.{}.adaln_proj.linear.bias".format(key)
            if splice_weights is not None and (key, "weight") in splice_weights:
                weight = splice_weights[(key, "weight")].to(dtype)
                bias = splice_weights.get((key, "bias"))
                bias = bias.to(dtype) if bias is not None else None
            else:
                weight = _effective_weight(mp, wkey, dtype=dtype)
                bias = _effective_weight(mp, bkey, dtype=dtype)
            if weight is None:
                raise RuntimeError("weight key not found: {}".format(wkey))
            if wkey in (getattr(mp, "patches", {}) or {}) or \
                    bkey in (getattr(mp, "patches", {}) or {}):
                applied += 1
            inp = F.silu(t_emb) if getattr(ap, "apply_silu", True) else t_emb
            weight = weight.to(device=device, dtype=dtype)
            bias = bias.to(device=device, dtype=dtype) if bias is not None else None
            with torch.no_grad():
                raw = F.linear(inp.to(device=device, dtype=dtype), weight, bias)
            raw_by_key[key] = raw.detach().to(torch.float32).cpu().contiguous()
            del weight, bias, raw
        except Exception as e:
            errors.append("{}: {}: {}".format(key, type(e).__name__, str(e)[:160]))
        if progress is not None and n % 5 == 0:
            progress(n, len(projs))

    if errors:
        raise RuntimeError(
            "H3ExactAdaLN Bake: failed on {} projection(s); refusing to write a "
            "table with holes. {}".format(len(errors), "; ".join(errors[:4])))

    info = {
        "projections": len(projs),
        "applied_patches": applied,
        "target_patch_keys": len([
            k for k in getattr(mp, "patches", {}) or {}
            if "time_embedder" in k or "adaln_proj" in k]),
        "patch_errors": errors,
        "compute_dtype": compute_dtype,
        "device": str(device),
        "t_embed_dim": int(t_emb.shape[-1]),
    }
    info.update(_weight_info(dm))
    info.update(splice_info)
    return raw_by_key, info


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


def teacher_out_dtype(dm):
    """The dtype the teacher's own AdaLN path produces.

    model.py:495-496 picks `adaln_dtype = fp32` for curve checkpoints and the
    model dtype otherwise. Reading `linear.weight.dtype` is not usable here --
    a quantised op reports its storage dtype (uint8/int8), not the dtype the
    matmul runs in -- so mirror the constructor instead.

    Replaying a table in a different dtype changes the precision of
    `FinalLayer.mod`'s multiply-add (it switches on `scale.dtype`), which is
    visible in the latent.
    """
    if getattr(dm, "use_adaln_curves", False):
        return "torch.float32"
    dt = getattr(dm, "dtype", None)
    return str(dt) if dt is not None else "torch.bfloat16"


def self_check(model, t_values, raw_by_key, compute_dtype="bfloat16"):
    """Report what the teacher's AdaLN weights actually look like."""
    try:
        mp, dm = _diffusion_model(model)
        key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
        weight = _effective_weight(mp, key)
        if weight is None:
            return "self-check skipped: weight key not found"
        patched = key in (getattr(mp, "patches", {}) or {})
        return ("self-check: key={} dtype={} quantised={} lora_patched={}".format(
            key, getattr(weight, "dtype", "?"), type(weight).__name__ != "Tensor",
            patched))
    except Exception as e:
        return "self-check failed: {}: {}".format(type(e).__name__, str(e)[:160])
