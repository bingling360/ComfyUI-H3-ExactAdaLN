"""Record the teacher's native AdaLN output during a real sampling run.

Why recording instead of baking offline
---------------------------------------
`bake_engine.bake()` runs `time_embedder` + `adaln_proj.linear` on weights that
are still sitting on CPU in their checkpoint form. At sampling time the very
same modules run on GPU with weights ComfyUI dequantised through its own CUDA
kernels, so the two paths disagree. Measured on the real 34 GB teacher:

    blocks.0.adaln_proj        equal=False  max_abs=0.015625  mean_abs=5.2e-4

0.015625 is exactly one bf16 ulp, and eight diffusion steps turn that into a
latent offset of 0.145 mean_abs. No amount of offline care can reproduce a
kernel that is chosen at run time, so the only way to a bitwise table is to
*observe* the values instead of recomputing them.

How it stays exact
------------------
Two read-only forward hooks:

  * `dm.time_embedder` -- captures `t_vals`, the float32 tensor of distinct
    timesteps that `model.py:743` builds from `unique_t`. Row i of every AdaLN
    projection therefore belongs to `unique_t[i]`; nothing needs to be
    reconstructed from sigmas, layouts or shifts.
  * every `adaln_proj.linear` -- captures the raw `[M, expand*hidden*modalities]`
    output, which is exactly what a table stores (the module's own
    view/chunk is replayed later, not baked in).

Replaying those rows through the same view/chunk is bitwise by construction,
including on a pruned base: the gathered rows are cast back to the dtype they
were recorded in, so `FinalLayer.mod`'s `scale.dtype`-driven arithmetic sees
the same dtype the teacher used.
"""

import logging

import torch

try:
    from . import sigma_plan
except ImportError:
    import sigma_plan


def _adln_projections(dm):
    """[(key, adaln_proj module)] for blocks.N and final_layer."""
    out = []
    for i, block in enumerate(getattr(dm, "blocks", []) or []):
        ap = getattr(block, "adaln_proj", None)
        if ap is not None and hasattr(ap, "linear"):
            out.append(("blocks.{}".format(i), ap))
    ap = getattr(getattr(dm, "final_layer", None), "adaln_proj", None)
    if ap is not None and hasattr(ap, "linear"):
        out.append(("final_layer", ap))
    return out


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


class AdaLNRecorder:
    """Live, read-only capture of AdaLN rows keyed by the float32 timestep."""

    def __init__(self, model, dm, mode="strict", tag=""):
        self.model = model
        self.dm = dm
        self.mode = mode
        self.tag = tag
        self.rows = {}          # t -> {projection key: [F] fp32 cpu}
        self.t_order = []       # first-seen order, for the report
        self.proj_keys = []
        self.handles = []
        self.out_dtype = None
        self.forwards = 0
        self.finished_t = []
        self.conflicts = 0
        self.max_repeat_delta = 0.0
        self.sigma_seen = []
        self.errors = []
        self._cur_t = None
        self._cur_forward = -1
        self._canary_key = None

    # -- registration ----------------------------------------------------
    def register(self):
        te = getattr(self.dm, "time_embedder", None)
        if te is None or not hasattr(te, "register_forward_hook"):
            raise RuntimeError(
                "H3AdaLNRecord: this checkpoint has no `time_embedder` -- it is a "
                "pruned/curve base. Record must run on the FULL teacher model "
                "(same checkpoint you would feed to Bake).")
        self.handles.append(te.register_forward_hook(self._time_embedder_hook))

        for key, ap in _adln_projections(self.dm):
            expected = int(getattr(ap, "expand", 0)) * int(getattr(ap, "hidden", 0)) \
                * int(getattr(ap, "modalities", 0))
            self.proj_keys.append(key)
            self.handles.append(
                ap.linear.register_forward_hook(self._linear_hook(key, ap, expected)))
        if not self.proj_keys:
            raise RuntimeError("H3AdaLNRecord: no adaln_proj modules found")
        self._canary_key = "blocks.0" if "blocks.0" in self.proj_keys else self.proj_keys[0]
        return len(self.proj_keys)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    # -- hooks -----------------------------------------------------------
    def _time_embedder_hook(self, module, inputs, output):
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        t = inputs[0].detach()
        if t.ndim != 1:
            t = t.reshape(-1)
        self._cur_t = t.reshape(-1).to(torch.float32).cpu().tolist()
        self._cur_forward += 1
        self.forwards += 1

    def _fail(self, message):
        self.errors.append(message)
        if self.mode == "strict":
            raise RuntimeError(message)
        logging.warning("H3AdaLNRecord: %s", message)

    def _linear_hook(self, key, ap, expected):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                self._fail("H3AdaLNRecord: {}.linear returned {} instead of a tensor"
                           .format(key, type(output).__name__))
                return None
            if output.ndim != 2:
                self._fail("H3AdaLNRecord: {}.linear output has shape {}, expected 2-D"
                           .format(key, tuple(output.shape)))
                return None
            m = int(output.shape[0])
            if expected and int(output.shape[1]) != expected:
                self._fail(
                    "H3AdaLNRecord: {}.linear output width {} != expand*hidden*modalities {}"
                    .format(key, int(output.shape[1]), expected))
                return None
            t_values = self._cur_t
            if t_values is None:
                self._fail("H3AdaLNRecord: {} fired before time_embedder; cannot label rows"
                           .format(key))
                return None
            if len(t_values) != m:
                self._fail(
                    "H3AdaLNRecord: {} got {} rows but time_embedder saw {} timesteps. "
                    "denoise_mask per-row timesteps are not supported by record mode."
                    .format(key, m, len(t_values)))
                return None

            if self.out_dtype is None:
                self.out_dtype = output.dtype
            elif output.dtype != self.out_dtype:
                self._fail("H3AdaLNRecord: dtype changed mid-run ({} -> {})"
                           .format(self.out_dtype, output.dtype))

            labels = [sigma_plan.f32_image(t) for t in t_values]
            new_rows = [i for i, t in enumerate(labels) if key not in self.rows.get(t, {})]
            check_rows = [i for i, t in enumerate(labels)
                          if key == self._canary_key and t in self.rows and key in self.rows[t]]
            if not new_rows and not check_rows:
                return None

            # One bulk device->host copy per projection instead of one per row:
            # the first forward moves ~336 MB (51 x [T, 96768] fp32) and every
            # later forward only the canary projection.
            rows_cpu = output.detach().to(torch.float32).to("cpu")
            if new_rows and self._canary_key == key:
                logging.info("H3AdaLNRecord: first rows captured at t=%s",
                             ", ".join("{:.7f}".format(t_values[i]) for i in new_rows[:4]))
            for i in new_rows:
                t = labels[i]
                slot = self.rows.setdefault(t, {})
                slot[key] = rows_cpu[i]
                if len(slot) == 1:
                    self.t_order.append(t)
            for i in check_rows:
                # The same t must produce the same row on every step / every
                # CFG pass. A non-zero delta means the run is not deterministic
                # and the table would be a coin flip, so surface it instead of
                # silently keeping sample 1.
                delta = float((self.rows[labels[i]][key] - rows_cpu[i]).abs().max())
                if delta > 0.0:
                    self.conflicts += 1
                    self.max_repeat_delta = max(self.max_repeat_delta, delta)
            return None
        return hook

    # -- results ---------------------------------------------------------
    def build(self):
        if not self.rows:
            raise RuntimeError(
                "H3AdaLNRecord: nothing was recorded -- the sampler never ran with "
                "this model. Wire the recorder between the model and the guider, and "
                "make sure the node that consumes `table` executes AFTER sampling "
                "(connect the sampler's LATENT to its `after` input).")
        t_values = sorted(self.rows)
        raw = {}
        for key in self.proj_keys:
            missing = [t for t in t_values if key not in self.rows[t]]
            if missing:
                raise RuntimeError(
                    "H3AdaLNRecord: {} is missing rows for {} timestep(s) (interrupted "
                    "run?). Nothing was saved.".format(key, len(missing)))
            raw[key] = torch.stack([self.rows[t][key] for t in t_values]).contiguous()
        te = getattr(self.dm, "time_embedder", None)
        t_embed_dim = int(getattr(getattr(te, "proj_out", None), "out_features", 0) or 0)
        extra = {
            "record_mode": "observed during sampling",
            "out_dtype": "torch." + _dtype_name(self.out_dtype or torch.bfloat16),
            "recorded_forwards": str(self.forwards),
            "repeat_conflicts": str(self.conflicts),
            "max_repeat_delta": "{:.3e}".format(self.max_repeat_delta),
            "t_embed_dim": str(t_embed_dim),
            "num_blocks": str(len(self.proj_keys) - 1),
        }
        return t_values, raw, extra

    def summary(self):
        return "\n".join([
            "recorder         : {} projections hooked".format(len(self.proj_keys)),
            "forwards seen    : {}".format(self.forwards),
            "distinct t rows  : {}".format(len(self.rows)),
            "recorded dtype   : {}".format(self.out_dtype or "(not yet)"),
            "repeat conflicts : {} (max_delta={:.3e})".format(
                self.conflicts, self.max_repeat_delta),
            "t values         : {}".format(
                ", ".join("{:.7f}".format(t) for t in sorted(self.rows)[:10]) +
                (" ..." if len(self.rows) > 10 else "") or "none yet"),
        ])
