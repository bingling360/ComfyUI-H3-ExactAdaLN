"""Shared helpers and the verification node for ComfyUI-H3-ExactAdaLN.

The plugin exposes three nodes (see h3_unified.py and __init__.py):

  H3 AdaLN          record / apply / off  -- the only node needed day to day
  H3 AdaLN Hybrid   bX-49 donor merge (only when no full hybrid teacher exists)
  H3 Latent Compare A/B verification of two LATENT outputs

Everything that used to live here (Record / Bake / Save / Load / Merge / Apply /
LatentSave single-purpose nodes, and the offline bake engine) was removed:
record is the only exact path, the picker was replaced by fingerprint lookup,
and the v1 workflows that referenced those nodes are gone.
"""

import hashlib
import logging
import os

import torch

try:
    from . import sigma_plan
except ImportError:  # pragma: no cover - direct-file loading fallback
    import sigma_plan

CATEGORY = "MiniMaxH3/ExactAdaLN"
TABLE_TYPE = "H3_ADALN_TABLE"


def _auto_shifts(model, with_source=False):
    """Video/audio shift as model.py:672-673 resolves them.

    The DiT reads `transformer_options` first and only then the module
    attribute, so reading `model_sampling` alone can disagree with what the
    forward pass actually used.

    `with_source=True` also returns where the value came from. A director
    workflow often places MiniMaxH3SigmaShift *downstream* of this node, so at
    record time transformer_options carries no shift and the old code silently
    reported the model default (12/3) as if it were the workflow's shift.
    """
    def _ret(sv, sa, src):
        return (float(sv), float(sa), src) if with_source else (float(sv), float(sa))

    try:
        to = model.model_options.get("transformer_options", {}) or {}
        sv = to.get("minimax_h3_sigma_shift_video")
        sa = to.get("minimax_h3_sigma_shift_audio")
        if sv is not None and sa is not None:
            return _ret(sv, sa, "transformer_options")
    except Exception:
        pass
    try:
        dm = model.model.diffusion_model
        sv = getattr(dm, "sigma_shift_video", None)
        sa = getattr(dm, "sigma_shift_audio", None)
        if sv is not None and sa is not None:
            return _ret(sv, sa, "model_default_attribute")
    except Exception:
        pass
    try:
        ms = model.get_model_object("model_sampling")
        return _ret(getattr(ms, "shift", 12.0), getattr(ms, "audio_shift", 3.0),
                    "model_sampling")
    except Exception:
        return _ret(sigma_plan.DEFAULT_VIDEO_SHIFT,
                    sigma_plan.DEFAULT_AUDIO_SHIFT, "hardcoded_default")


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


def _adaln_lora_fingerprint(model):
    return _patch_set_digest(
        model, lambda k: "time_embedder" in k or "adaln_proj" in k)


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
    "H3LatentCompare": H3LatentCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentCompare": "H3 Latent Compare (verification)",
}
