"""Runtime injection of an exact AdaLN table into a MiniMax-H3 MODEL.

The injection must not depend on how the sampler is invoked. Custom sampler
nodes (SequenceForge, HyperFlow adapters, etc.) often call
`comfy.sample.sample()` directly with their own model_options, so a
`set_model_patch_replace(..., "dit", "double_block", 0)` callback may never
run. Instead we identify timesteps from the `t_emb` tensor itself:

  * pruned / curve checkpoint: t_emb is the 8-wide lerped curve coordinate,
    rebuilt from `adaln_t_table`
  * full checkpoint: t_emb is `time_embedder(t).to(dtype)`, 2688-wide,
    rebuilt by running the model's own `time_embedder` on the baked t values

Both give a [T, k] key matrix. Each `adaln_proj` forward hook nearest-matches
the incoming t_emb rows against those keys and returns the exact baked rows.
No layout, no transformer_options["sigmas"], no patches_replace required.

A dit patch is kept only as a last-resort fallback. ComfyUI calls it as
patch(args, {"original_block": callable}) -- a dict, not a function -- so both
that form and a bare callable are accepted.
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
        self.match_keys, self.match_mode = self._build_match_keys(dm)
        self.use_match = self.match_keys is not None
        self.state = {
            "t_indices": None, "hits": 0, "fallbacks": 0,
            "missing": set(), "last_sigma": None,
            "match_hits": 0, "match_failures": 0,
            "max_match_distance": 0.0, "last_diag": "",
            "dit_hits": 0, "native_steps": 0,
        }

    def _build_match_keys(self, dm):
        """Return ([T, k] keys, human readable mode) or (None, 'dit patch')."""
        t = torch.tensor(self.t_values, dtype=torch.float32)
        try:
            if getattr(dm, "use_adaln_curves", False):
                table = getattr(dm, "adaln_t_table", None)
                if table is not None and table.ndim == 2 and table.shape[0] >= 2:
                    tbl = table.detach().to(device="cpu", dtype=torch.float32).contiguous()
                    pos = t.clamp(0.0, 1.0) * (tbl.shape[0] - 1)
                    i0 = pos.floor().long().clamp(max=tbl.shape[0] - 2)
                    frac = (pos - i0).unsqueeze(1)
                    keys = torch.lerp(tbl[i0], tbl[i0 + 1], frac)
                    return keys.contiguous(), "curve-coordinate (8-wide)"

            te = getattr(dm, "time_embedder", None)
            if te is not None:
                dev = next(te.parameters()).device
                with torch.no_grad():
                    keys = te(t.to(dev)).to(self.out_dtype)
                keys = keys.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if keys.ndim == 2 and keys.shape[0] == len(self.t_values):
                    return keys, "time-embedder ({} -wide)".format(keys.shape[-1])
                logging.warning("H3ExactAdaLN: unexpected time_embedder shape %s",
                                tuple(keys.shape))
        except Exception as e:
            logging.warning("H3ExactAdaLN: could not build match keys (%s: %s); "
                            "falling back to dit patch", type(e).__name__, e)
        return None, "dit patch fallback"

    def _match_indices(self, t_emb):
        """Return (indices, max_distance, tolerance) for t_emb rows."""
        keys = self.match_keys.to(device=t_emb.device, dtype=torch.float32)
        x = t_emb.detach().to(torch.float32)
        if x.ndim != 2 or x.shape[-1] != keys.shape[-1]:
            return None, float("inf"), 0.0
        dist = torch.cdist(x, keys)
        idx = dist.argmin(dim=1)
        mind = dist.gather(1, idx[:, None]).squeeze(1)
        scale = float(keys.abs().max().clamp(min=1.0))
        tol = max(1e-3, 2e-3 * scale)
        return idx.detach().cpu().tolist(), float(mind.max()), tol

    def _previous(self, model):
        try:
            to = model.model_options.get("transformer_options", {})
            return to.get("patches_replace", {}).get("dit", {}).get(("double_block", 0))
        except Exception:
            return None

    def dit_patch(self, args, block_info=None):
        """Fallback resolver; only used when numeric matching is unavailable."""
        if isinstance(block_info, dict):
            info = block_info
            fn = block_info.get("original_block")
        else:
            info = {"original_block": block_info}
            fn = block_info

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
                        "Re-bake with these values in extra_t (or use mode=fallback).".format(
                            ", ".join("{:.7f}".format(t) for t in missing), sigma))
                self.state["t_indices"] = None
                self.state["fallbacks"] += 1
            else:
                self.state["t_indices"] = indices
                self.state["hits"] += 1
        except RuntimeError:
            raise
        except Exception as e:
            logging.warning("H3ExactAdaLN: could not resolve timesteps (%s: %s)",
                            type(e).__name__, e)
            self.state["t_indices"] = None
            self.state["fallbacks"] += 1

        if not callable(fn):
            raise RuntimeError(
                "H3ExactAdaLN: ComfyUI did not supply a callable original_block "
                "(got {}). Cannot run the dit fallback.".format(type(fn).__name__))
        if self.previous_patch is not None:
            return self.previous_patch(args, info)
        return fn(args)

    def _raw(self, key, device):
        cached = self._device_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        tensor = self.raw_by_key[key].to(device=device, dtype=self.out_dtype).contiguous()
        self._device_cache[key] = tensor
        return tensor

    def make_hook(self, key):
        def hook(module, inputs, output):
            t_emb = inputs[0]
            m = t_emb.shape[0]
            indices = None
            maxd, tol = 0.0, 0.0

            if self.use_match and t_emb.shape[-1] == self.match_keys.shape[-1]:
                indices, maxd, tol = self._match_indices(t_emb)
                self.state["max_match_distance"] = max(
                    self.state["max_match_distance"], maxd)
                if maxd > tol:
                    self.state["match_failures"] += 1
                    self.state["last_diag"] = (
                        "t_emb={} keys={} max_dist={:.3e} tol={:.3e}".format(
                            tuple(t_emb.shape), tuple(self.match_keys.shape), maxd, tol))
                    indices = None
                else:
                    self.state["match_hits"] += 1

            if indices is None:
                indices = self.state["t_indices"]
                if indices is not None:
                    self.state["dit_hits"] += 1

            if indices is None or len(indices) != m:
                self.state["native_steps"] += 1
                if self.mode == "strict":
                    raise RuntimeError(
                        "H3ExactAdaLN: could not map t_emb rows to baked table rows "
                        "(mode=strict). match_mode={}. {} "
                        "This usually means the table was baked for a DIFFERENT "
                        "sigma schedule / shift than this workflow uses. Re-bake "
                        "with this workflow's sigmas, or set mode=fallback to let "
                        "unknown timesteps use the native AdaLN path.".format(
                            self.match_mode,
                            self.state["last_diag"] or "t_emb={}".format(tuple(t_emb.shape))))
                return None

            # One explicit log line proving hooks fired during sampling;
            # Apply's own report is printed before the sampler runs.
            if self.state["match_hits"] + self.state["dit_hits"] == 1:
                logging.info(
                    "H3ExactAdaLN: first injection confirmed during sampling "
                    "(%s, %d rows, indices=%s, max_dist=%.3e)",
                    key, m, indices, self.state["max_match_distance"])

            raw = self._raw(key, t_emb.device)
            index = torch.as_tensor(indices, dtype=torch.long, device=raw.device)
            gathered = raw.index_select(0, index)
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

        if not self.use_match:
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
        return "\n".join([
            "mode             : {}".format(self.mode),
            "injection mode   : {}".format(
                self.match_mode if self.use_match else "dit patch fallback"),
            "t rows in table  : {}".format(len(self.t_values)),
            "projections      : {}".format(len(self.raw_by_key)),
            "out dtype        : {}".format(self.out_dtype),
            "match hits       : {}".format(self.state["match_hits"]),
            "match failures   : {}".format(self.state["match_failures"]),
            "max match dist   : {:.3e}".format(self.state["max_match_distance"]),
            "last diagnostic  : {}".format(self.state["last_diag"] or "none"),
            "dit hits         : {}".format(self.state["dit_hits"]),
            "native steps     : {}".format(self.state["native_steps"]),
            "dit steps        : {}/fallback {}".format(
                self.state["hits"], self.state["fallbacks"]),
            "last sigma       : {}".format(self.state["last_sigma"]),
            "missing t values : {}".format(
                ", ".join("{:.7f}".format(t) for t in sorted(self.state["missing"])[:12])
                or "none"),
            "chained patch    : {}".format(self.previous_patch is not None),
        ])
