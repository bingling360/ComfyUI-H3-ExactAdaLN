"""Runtime injection of an exact AdaLN table into a MiniMax-H3 MODEL.

Two mechanisms only, both already used by official ComfyUI H3 code:
  * set_model_patch_replace(..., "dit", "double_block", 0) -- runs before any
    block and receives `layout` and `transformer_options`, letting us rebuild
    model.py:709 `unique_t` and map it to table row indices.
  * register_forward_hook on all 51 adaln_proj modules -- returns replacement
    modulation tables, skipping adaln_t_table interpolation and the folded
    [96768, 8] projection entirely.

We deliberately do NOT use set_model_attn1_patch / transformer_options
["patches"]: H3's model.py never reads them.
"""

import logging

import torch

try:
    from . import sigma_plan
except ImportError:
    import sigma_plan


def _parse_dtype(name):
    return {"torch.bfloat16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16, "float16": torch.float16,
            "torch.float32": torch.float32, "float32": torch.float32}.get(
                str(name), torch.bfloat16)


class ExactAdaLNRuntime:
    def __init__(self, t_values, raw_by_key, meta, dm, mode="strict"):
        self.t_values = list(t_values)
        self.raw_by_key = dict(raw_by_key)
        self.meta = dict(meta)
        self.dm = dm
        self.mode = mode
        self.lookup = sigma_plan.build_index(self.t_values)
        self.out_dtype = _parse_dtype(meta.get("out_dtype", "bfloat16"))
        self.visual_aug = float(meta.get("visual_aug", sigma_plan.VISUAL_COND_TIMESTEP))
        self.audio_aug = float(meta.get("audio_aug", sigma_plan.AUDIO_COND_TIMESTEP))
        self.previous_patch = None
        self.hook_handles = []
        self._device_cache = {}
        self.state = {"t_indices": None, "hits": 0, "fallbacks": 0,
                      "missing": set(), "last_sigma": None}

    def _previous(self, model):
        try:
            to = model.model_options.get("transformer_options", {})
            return to.get("patches_replace", {}).get("dit", {}).get(("double_block", 0))
        except Exception:
            return None

    def dit_patch(self, args, original_block):
        to = args.get("transformer_options", {}) or {}
        try:
            sigma = sigma_plan.sigma_from_options(to, None)
            self.state["last_sigma"] = sigma
            shift_v, shift_a = sigma_plan.shifts_from_options(to, self.dm)
            layout = args.get("layout")
            unique_t = sigma_plan.runtime_unique_t(
                sigma, layout, shift_v, shift_a, self.visual_aug, self.audio_aug)
            indices = [self.lookup(t) for t in unique_t]
            missing = [t for t, i in zip(unique_t, indices) if i is None]
            if missing:
                self.state["missing"].update(round(float(t), 7) for t in missing)
                if self.mode == "strict":
                    raise RuntimeError(
                        "H3ExactAdaLN: table has no rows for t={} at sigma={:.8f}. "
                        "Re-bake with these values in extra_t (or use mode=fallback). "
                        "Most likely cause: a denoise_mask / per-row timestep or a "
                        "different sigma schedule.".format(
                            ", ".join("{:.7f}".format(t) for t in missing), sigma))
                self.state["t_indices"] = None
                self.state["fallbacks"] += 1
            else:
                self.state["t_indices"] = indices
                self.state["hits"] += 1
        except RuntimeError:
            raise
        except Exception as e:
            logging.warning("H3ExactAdaLN: could not resolve timesteps (%s: %s); "
                            "falling back to the native AdaLN path for this step",
                            type(e).__name__, e)
            self.state["t_indices"] = None
            self.state["fallbacks"] += 1

        if self.previous_patch is not None:
            return self.previous_patch(args, {"original_block": original_block})
        return original_block(args)

    def _raw(self, key, device):
        cached = self._device_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        tensor = self.raw_by_key[key].to(device=device, dtype=self.out_dtype).contiguous()
        self._device_cache[key] = tensor
        return tensor

    def make_hook(self, key):
        def hook(module, inputs, output):
            indices = self.state["t_indices"]
            if indices is None:
                return None
            t_emb = inputs[0]
            raw = self._raw(key, t_emb.device)
            index = torch.tensor(indices, dtype=torch.long, device=raw.device)
            gathered = raw.index_select(0, index)
            m = gathered.shape[0]
            x = gathered.reshape(m * module.modalities,
                                 module.expand * module.hidden)
            return x.chunk(module.expand, dim=-1)
        return hook

    def register(self, model):
        """Attach to `model` (a ModelPatcher). Returns number of hooked modules."""
        dm = self.dm
        targets = []
        for i, block in enumerate(dm.blocks):
            ap = getattr(block, "adaln_proj", None)
            if ap is not None:
                targets.append(("blocks.{}".format(i), ap))
        fl_ap = getattr(getattr(dm, "final_layer", None), "adaln_proj", None)
        if fl_ap is None:
            raise RuntimeError("H3ExactAdaLN Apply: final_layer.adaln_proj missing "
                               "(50 blocks + final = 51 projections required)")
        targets.append(("final_layer", fl_ap))

        missing_keys = [k for k, _ in targets if k not in self.raw_by_key]
        if missing_keys:
            raise RuntimeError("H3ExactAdaLN Apply: table is missing projections: {}".format(
                ", ".join(missing_keys[:8])))

        self.previous_patch = self._previous(model)
        model.set_model_patch_replace(self.dit_patch, "dit", "double_block", 0)
        for key, module in targets:
            self.hook_handles.append(module.register_forward_hook(self.make_hook(key)))
        return len(targets)

    def remove(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles = []
        self._device_cache.clear()

    def summary(self):
        lines = [
            "mode             : {}".format(self.mode),
            "t rows in table  : {}".format(len(self.t_values)),
            "projections      : {}".format(len(self.raw_by_key)),
            "out dtype        : {}".format(self.out_dtype),
            "steps injected   : {}".format(self.state["hits"]),
            "steps fallen back: {}".format(self.state["fallbacks"]),
            "last sigma       : {}".format(self.state["last_sigma"]),
            "missing t values : {}".format(
                ", ".join("{:.7f}".format(t) for t in sorted(self.state["missing"])[:12])
                or "none"),
            "chained patch    : {}".format(self.previous_patch is not None),
        ]
        return "\n".join(lines)
