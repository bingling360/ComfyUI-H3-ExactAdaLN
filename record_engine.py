"""Record the teacher's native AdaLN output during a real sampling run.

Why recording instead of baking offline
---------------------------------------
Offline recomputation runs `time_embedder` + `adaln_proj.linear` on weights
that are still sitting on CPU in their checkpoint form. At sampling time the very
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
import threading
import time

import torch

try:
    from . import apply_engine, sigma_plan
except ImportError:
    import apply_engine
    import sigma_plan


# Pin calls are process-wide, not per-recorder.
#
# A pin is an EXTRA forward through the model's own time_embedder /
# adaln_proj.linear modules, with one more batch row than the real forward.
# Every recorder hooked on this diffusion module sees that call. If the guard
# were an instance attribute, recorder B would treat recorder A's extra batch
# as a genuine forward and fail its strict row/timestep check, which is exactly
# what silently killed the 0.999 / 1.0 rows on two-chain director workflows.
_PIN_STATE = threading.local()


def _pin_active():
    return getattr(_PIN_STATE, "depth", 0) > 0


class _PinGuard:
    def __enter__(self):
        _PIN_STATE.depth = getattr(_PIN_STATE, "depth", 0) + 1

    def __exit__(self, *_exc):
        _PIN_STATE.depth = max(0, getattr(_PIN_STATE, "depth", 1) - 1)
        return False


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

    def __init__(self, model, dm, mode="strict", tag="", te_override=None,
                 pin_targets=(), full_check_forwards=2):
        self.model = model
        self.dm = dm
        # Timesteps the run itself may never produce but a later workflow can ask
        # for: keyframe/reference rows are pinned at 0.999, reference-audio rows
        # at 1.0. They are computed through the model's own modules in one extra
        # call whose batch is [this step's t] + [pin] -- i.e. exactly the batch a
        # reference run would have used, which is what keeps the row bitwise.
        self.pin_targets = [float(v) for v in (pin_targets or ())]
        self.pinned = {}        # pin t -> {projection key: [F] fp32 cpu}
        self._pin_emb = {}      # pin t -> tensor on the model's device
        self._pin_pending = {}  # pin t -> keys still to capture in this forward
        self._pin_step = None   # index of the forward the pins belong to
        self._in_pin = False    # kept for compatibility; guards use _pin_active()
        self.pin_errors = {}    # pin t -> why the pre-computation failed
        self.pin_skipped = set()  # pins this run produced by itself
        self.abandoned = False  # another record node took over our module
        self.sealed = False     # hooks detached; nothing more will be captured
        self.saved_once = False  # a save already happened for this prompt
        self.passes = 0         # ON_CLEANUP invocations that produced a save
        # Optional override when a caller swapped `time_embedder` for another
        # checkpoint's; the hook must land on the object the forward will
        # actually run, not on the original module.
        self.te_override = te_override
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
        self.conflict_detail = []   # (t, key, delta) -- which block disagreed
        self.max_repeat_delta = 0.0
        self.sigma_seen = []
        self.errors = []
        self.warnings = []
        self._cur_t = None
        self._cur_forward = -1
        self._canary_key = None
        # Comparing the same t on EVERY projection costs a full device->host
        # copy of all 51 rows, so it is done for the first N forwards only:
        # that is where a mid-run AdaLN change shows up. Afterwards only the
        # canary block is cross-checked, which is free.
        self.full_check_forwards = max(1, int(full_check_forwards))
        self.compared = 0
        self.skipped = 0
        self.forward_trace = []     # [(forward idx, [t...], n_rows)]
        self.trace_cap = 64
        self._labels_cache = None
        self.pins_report = ""
        # Wall clock of the last captured row. The deferred-save watchdog uses
        # it to tell "a pass is still running" from "sampling stopped and the
        # rows are sitting in memory, unsaved".
        self.last_row_at = 0.0

    # -- registration ----------------------------------------------------
    def register(self):
        te = self.te_override if self.te_override is not None else getattr(self.dm, "time_embedder", None)
        if te is None or not hasattr(te, "register_forward_hook"):
            raise RuntimeError(
                "H3AdaLN: this checkpoint has no `time_embedder` -- it is a "
                "pruned/curve base. Record must run on the FULL teacher model "
                "(same checkpoint you would feed to Bake).")
        # Same reason as the injection hooks: these run inside the DiT forward,
        # which ComfyUI 0.44+ wraps in a comfy-aimdo malloc graph. The bulk
        # `output.to(torch.float32)` below is a GPU allocation made inside that
        # graph -- it is recorded as a "rogue" and breaks the compiled prefetch
        # allocation plan ("Comfy model compiler graph breaks: N, rogues: N").
        outside = apply_engine._outside_malloc_graph
        self.handles.append(te.register_forward_hook(
            outside(self._time_embedder_hook)))

        for key, ap in _adln_projections(self.dm):
            expected = int(getattr(ap, "expand", 0)) * int(getattr(ap, "hidden", 0)) \
                * int(getattr(ap, "modalities", 0))
            self.proj_keys.append(key)
            self.handles.append(
                ap.linear.register_forward_hook(
                    outside(self._linear_hook(key, ap, expected))))
        if not self.proj_keys:
            raise RuntimeError("H3AdaLN: no adaln_proj modules found")
        self._canary_key = "blocks.0" if "blocks.0" in self.proj_keys else self.proj_keys[0]
        return len(self.proj_keys)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def abandon(self):
        """Detach cleanly because another recorder owns this module now.

        A superseded recorder must not try to save a half-empty table during
        ON_CLEANUP; `_finish_record` checks this flag and quietly skips it.
        """
        self.abandoned = True
        self.remove()

    def seal(self):
        """Detach for good: this recorder will not capture another row.

        ON_CLEANUP fires once per sampling pass, so the hooks have to survive
        the first one (a director chain runs a refine pass afterwards). But a
        recorder that outlives its prompt keeps writing rows into a dict that
        is never saved, and -- because `mode="strict"` used to raise -- could
        abort a completely unrelated prompt. Sealing is the explicit end.
        """
        self.sealed = True
        self.remove()

    # -- hooks -----------------------------------------------------------
    def _time_embedder_hook(self, module, inputs, output):
        if _pin_active():  # any recorder's extra call, not the model's
            return
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        t = inputs[0].detach()
        if t.ndim != 1:
            t = t.reshape(-1)
        self._cur_t = t.reshape(-1).to(torch.float32).cpu().tolist()
        self._cur_forward += 1
        self.forwards += 1
        if self.pin_targets and self._pin_step is None:
            self._pin_step = self._cur_forward
            self._make_pins(module, t.reshape(-1), self._cur_t)

    def _make_pins(self, te, t, t_list):
        """Compute the pinned embeddings in an [M+1] batch, once per run."""
        for pin in self.pin_targets:
            if any(abs(v - pin) < 1e-9 for v in t_list):
                self.pin_skipped.add(pin)
                continue  # this run produces that row by itself
            try:
                with _PinGuard(), torch.no_grad():
                    extra = torch.tensor([pin], dtype=t.dtype, device=t.device)
                    emb = te(torch.cat([t, extra], 0))[len(t_list):]
                self._pin_emb[pin] = emb.detach()
                self._pin_pending[pin] = set(self.proj_keys)
            except Exception as e:
                self.pin_errors[pin] = "{}: {}".format(type(e).__name__, e)
                logging.warning(
                    "H3AdaLN: cannot pre-compute the t=%.3f row (%s: %s); "
                    "this run will NOT be saved", pin, type(e).__name__, e)

    def _pin_row(self, key, ap, module, x):
        """One extra projection call per pin; the extra batch row is discarded."""
        if not self._pin_pending or self._cur_forward != self._pin_step:
            return
        for pin, pending in list(self._pin_pending.items()):
            if key not in pending:
                continue
            pending.discard(key)
            emb = self._pin_emb.get(pin)
            if emb is None:
                continue
            try:
                with _PinGuard(), torch.no_grad():
                    # model.py casts t_emb to the compute dtype *before* AdalnProj
                    # applies silu; doing it the other way round rounds twice.
                    emb = emb.to(x.dtype)
                    if getattr(ap, "apply_silu", True):
                        emb = torch.nn.functional.silu(emb)
                    aug = torch.cat([x, emb], 0)
                    out = module(aug)
                    row = out[len(x):].detach().to(torch.float32).to("cpu")
                self.pinned.setdefault(pin, {})[key] = row[0]
            except Exception as e:
                self.pin_errors[pin] = "{}: {}".format(type(e).__name__, e)
                self._pin_pending.pop(pin, None)
                logging.warning(
                    "H3AdaLN: cannot pre-compute %s at t=%.3f (%s: %s); "
                    "this run will NOT be saved", key, pin, type(e).__name__, e)

    def _fail(self, message, fatal=None):
        """Record a problem, and raise only while this recorder is authoritative.

        After the first save the hooks may still be attached for a later pass
        of the same prompt -- but they also survive into unrelated prompts that
        happen to sample the same diffusion module. Raising there crashed a
        prompt that had nothing to do with recording, so past the first save a
        strict recorder reports instead of raising.
        """
        self.errors.append(message)
        if fatal is None:
            fatal = (self.mode == "strict") and not self.saved_once
        if fatal:
            raise RuntimeError(message)
        self.warnings.append(message)
        logging.warning("H3AdaLN: %s", message)

    def _linear_hook(self, key, ap, expected):
        def hook(module, inputs, output):
            if _pin_active():  # any recorder's extra call
                return None
            if not torch.is_tensor(output):
                self._fail("H3AdaLN: {}.linear returned {} instead of a tensor"
                           .format(key, type(output).__name__))
                return None
            if output.ndim != 2:
                self._fail("H3AdaLN: {}.linear output has shape {}, expected 2-D"
                           .format(key, tuple(output.shape)))
                return None
            m = int(output.shape[0])
            if expected and int(output.shape[1]) != expected:
                self._fail(
                    "H3AdaLN: {}.linear output width {} != expand*hidden*modalities {}"
                    .format(key, int(output.shape[1]), expected))
                return None
            t_values = self._cur_t
            if t_values is None:
                self._fail("H3AdaLN: {} fired before time_embedder; cannot label rows"
                           .format(key))
                return None
            if len(t_values) != m:
                self._fail(
                    "H3AdaLN: {} got {} rows but time_embedder saw {} timesteps. "
                    "denoise_mask per-row timesteps are not supported by record mode."
                    .format(key, m, len(t_values)))
                return None

            if inputs and torch.is_tensor(inputs[0]):
                self._pin_row(key, ap, module, inputs[0])

            if self.out_dtype is None:
                self.out_dtype = output.dtype
            elif output.dtype != self.out_dtype:
                self._fail("H3AdaLN: dtype changed mid-run ({} -> {})"
                           .format(self.out_dtype, output.dtype))

            labels = self._labels(t_values)
            new_rows = [i for i, t in enumerate(labels) if key not in self.rows.get(t, {})]
            repeat_rows = [i for i, t in enumerate(labels)
                           if t in self.rows and key in self.rows[t]]
            # Past the warm-up forwards a projection that has nothing new to
            # learn only needs the canary cross-check, and even that is skipped
            # for the other 50 blocks -- copying 336 MB per forward to re-check
            # rows that never change is what made recording slow.
            cold = self.forwards > self.full_check_forwards
            if not new_rows and (key != self._canary_key or cold):
                self.skipped += 1
                return None
            if not new_rows and not repeat_rows:
                self.skipped += 1
                return None

            if key == self._canary_key and len(self.forward_trace) < self.trace_cap:
                self.forward_trace.append(
                    (self._cur_forward, tuple(round(t, 7) for t in t_values[:8]), m))

            # One bulk device->host copy per projection instead of one per row.
            rows_cpu = output.detach().to(torch.float32).to("cpu")
            if new_rows and self._canary_key == key:
                logging.info("H3AdaLN: first rows captured at t=%s",
                             ", ".join("{:.7f}".format(t_values[i]) for i in new_rows[:4]))
            for i in new_rows:
                t = labels[i]
                slot = self.rows.setdefault(t, {})
                slot[key] = rows_cpu[i]
                if len(slot) == 1:
                    self.t_order.append(t)
            if new_rows:
                self.last_row_at = time.time()
            # The same t must produce the same row on every step / every CFG
            # pass, for EVERY block -- a LoRA or a weight change that only
            # touches blocks 20..49 left blocks.0 identical, so a canary-only
            # check reported "0 conflicts" and silently kept sample 1.
            if cold and key != self._canary_key:
                repeat_rows = []
            for i in repeat_rows:
                delta = float((self.rows[labels[i]][key] - rows_cpu[i]).abs().max())
                self.compared += 1
                if delta > 0.0:
                    self.conflicts += 1
                    self.max_repeat_delta = max(self.max_repeat_delta, delta)
                    if len(self.conflict_detail) < 8:
                        self.conflict_detail.append((labels[i], key, delta))
            return None
        return hook

    def _labels(self, t_values):
        """float32 image of the current forward's timesteps, cached per forward."""
        cached = self._labels_cache
        if cached is not None and cached[0] == self._cur_forward:
            return cached[1]
        labels = [sigma_plan.f32_image(t) for t in t_values]
        self._labels_cache = (self._cur_forward, labels)
        return labels

    # -- results ---------------------------------------------------------
    def build(self):
        if not self.rows:
            raise RuntimeError(
                "H3AdaLN: nothing was recorded -- the sampler never ran with "
                "this model. Wire the recorder between the model and the guider, and "
                "make sure the node that consumes `table` executes AFTER sampling "
                "(connect the sampler's LATENT to its `after` input).")
        # A pin failure used to be swallowed: the run saved a table without the
        # reference rows and then printed a *misleading* "enable
        # fill_reference_row" warning. Fail loudly instead -- a table that lacks
        # 0.999 must never be presented as a successful recording.
        if self.pin_errors:
            raise RuntimeError(
                "H3AdaLN: reference-row pre-computation failed for t={} ({}). "
                "Nothing was saved. Most common cause: two record nodes sharing "
                "one diffusion model; keep exactly one recorder per model.".format(
                    ", ".join("{:.3f}".format(p) for p in sorted(self.pin_errors)),
                    "; ".join(sorted(self.pin_errors.values()))[:300]))
        missing_pins = [p for p in self.pin_targets
                        if p not in self.pin_skipped
                        and sigma_plan.f32_image(p) not in self.rows
                        and p not in self.pinned]
        if missing_pins:
            raise RuntimeError(
                "H3AdaLN: reference rows t={} were never captured. Nothing was "
                "saved; re-record with fill_reference_row enabled and a single "
                "record node.".format(
                    ", ".join("{:.3f}".format(p) for p in missing_pins)))
        # Computed BEFORE the pre-computed rows are folded into self.rows --
        # otherwise every successful pin reports "captured", which is exactly the
        # ambiguity the line exists to remove.
        self.pins_report = self._pins_report()
        for pin, entries in sorted(self.pinned.items()):
            key_t = sigma_plan.f32_image(pin)
            if key_t in self.rows:
                continue  # the run itself produced this row
            if len(entries) != len(self.proj_keys):
                logging.warning(
                    "H3AdaLN: dropping the pre-computed t=%.3f row -- only "
                    "%d of %d projections were captured", pin, len(entries),
                    len(self.proj_keys))
                continue
            self.rows[key_t] = dict(entries)
            self.t_order.append(key_t)

        t_values = sorted(self.rows)
        raw = {}
        for key in self.proj_keys:
            missing = [t for t in t_values if key not in self.rows[t]]
            if missing:
                raise RuntimeError(
                    "H3AdaLN: {} is missing rows for {} timestep(s) (interrupted "
                    "run?). Nothing was saved.".format(key, len(missing)))
            raw[key] = torch.stack([self.rows[t][key] for t in t_values]).contiguous()
        te = self.te_override if self.te_override is not None else getattr(self.dm, "time_embedder", None)
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

    def _pins_report(self):
        parts = []
        for pin in self.pin_targets:
            if pin in self.pin_skipped:
                state = "produced by this run"
            elif sigma_plan.f32_image(pin) in self.rows:
                state = "captured"
            elif pin in self.pinned:
                state = "pre-computed"
            else:
                state = "MISSING"
            parts.append("t={:.3f} {}".format(pin, state))
        return ", ".join(parts) if parts else "off (fill_reference_row=False)"

    def summary(self):
        rows = sorted(self.rows)
        lines = [
            "recorder         : {} projections hooked (canary {})".format(
                len(self.proj_keys), self._canary_key),
            "forwards seen    : {}".format(self.forwards),
            "distinct t rows  : {}".format(len(rows)),
            "recorded dtype   : {}".format(self.out_dtype or "(not yet)"),
            "repeat conflicts : {} (max_delta={:.3e}, {} re-compared, {} skipped)".format(
                self.conflicts, self.max_repeat_delta, self.compared, self.skipped),
        ]
        if self.conflict_detail:
            lines.append("conflict detail  : {}".format(
                "; ".join("t={:.7f} {} d={:.3e}".format(t, k, d)
                          for t, k, d in self.conflict_detail[:4])))
        lines.append("t values         : {}".format(
            ", ".join("{:.7f}".format(t) for t in rows[:12]) +
            (" ... ({} more)".format(len(rows) - 12) if len(rows) > 12 else "")
            or "none yet"))
        if self.pins_report:
            lines.append("pinned rows      : {}".format(self.pins_report))
        return "\n".join(lines)
