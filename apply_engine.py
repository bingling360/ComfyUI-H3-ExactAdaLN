"""Runtime injection of an exact AdaLN table into a MiniMax-H3 MODEL.

Two independent resolvers, in this order:

1. **t_emb matching (primary).** Every table row is a function of one distinct
   timestep, and the model builds exactly one `t_emb` row per distinct
   timestep. So the incoming `t_emb` rows can be matched against per-row keys
   rebuilt from the *target* model:

     * pruned / curve checkpoint -> lerp of `adaln_t_table` (fp32, 8-wide)
     * full checkpoint           -> `time_embedder(t).to(dtype)` (2688-wide)

   Keys are rebuilt lazily on the tensor's own device and cast to its own
   dtype, because that is what the model does; computing them on CPU or in
   another dtype makes the distance non-zero for reasons that have nothing to
   do with the table. Because Record stores the float32 image of t, this
   match is expected to be **exactly zero**, and anything else is reported
   instead of silently snapped to the nearest row.

2. **sigma reconstruction (fallback).** A `WrappersMP.DIFFUSION_MODEL` wrapper
   wraps `_forward` -- unlike `patches_replace`, which custom samplers that
   call `comfy.sample.sample()` never populate. It sees the real `timestep`
   and the packed `layout`, so `unique_t` can be rebuilt and mapped by value.
   It also produces the "which timesteps are missing from the table" list used
   by the strict-mode error message.

No layout, no patches_replace and no transformer_options are required for the
primary path.
"""

import contextlib
import functools
import logging
import time

import torch

try:
    from . import sigma_plan
except ImportError:
    import sigma_plan


def _outside_malloc_graph(fn):
    """Run a forward hook OUTSIDE comfy-aimdo's malloc graph recording.

    ComfyUI 0.44 wraps MiniMaxH3.forward in malloc_graph_begin/end and
    compiles every CUDA allocation made inside it for DynamicVRAM prefetch.
    A plain PyTorch hook that allocates (cdist, .to(device), index_select,
    or a first-use 446 MB table upload) shows up as a "rogue" allocation,
    breaks the graph, and after an OOM retry makes malloc_graph_push fail
    with the opaque "aimdo memory compile error".
    pause_malloc_graph() is ComfyUI's own escape hatch for exactly this.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            from comfy import model_prefetch
            ctx = model_prefetch.pause_malloc_graph()
        except Exception:
            ctx = contextlib.nullcontext()
        with ctx:
            return fn(*args, **kwargs)
    return wrapper


def recover_stale_malloc_graph():
    """Adapt to ComfyUI 0.44: retire a graph left active by a failed forward.

    MiniMaxH3.forward calls malloc_graph_begin() without a try/finally, so an
    exception inside the recorded region (OOM, interrupt) leaves an active,
    un-popped graph in comfy.model_prefetch.MALLOC_GRAPHS. The next sampling
    attempt pushes onto that stale handle and aimdo reports "memory compile
    error". ON_CLEANUP runs on both success and failure, so this is the right
    place for the node to clean up after itself instead of patching ComfyUI.
    """
    try:
        import threading
        from comfy import model_prefetch as mp
        graphs = getattr(mp, "MALLOC_GRAPHS", None)
        if not graphs:
            return
        g = graphs.get(threading.get_ident())
        if g is not None and getattr(g, "_comfy_active", False):
            mp.cleanup_malloc_graph()
            logging.info("H3AdaLN: retired a stale comfy-aimdo malloc graph "
                         "left by an interrupted forward")
    except Exception:
        pass


def _parse_dtype(name):
    return {"torch.bfloat16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16, "float16": torch.float16,
            "torch.float32": torch.float32, "float32": torch.float32}.get(
                str(name), torch.bfloat16)


# ---------------------------------------------------------------------------
# Why no cache of the whole table on the GPU
#
# The ~446 MB table is never uploaded whole. `_gather` copies only the 1-2 rows
# a step needs (a few MB) straight from the CPU copy, so nothing large is
# allocated inside comfy-aimdo's malloc graph -- a first-use upload there shows
# up as a "rogue" allocation and breaks the compiled prefetch graph.
#
# It also means no CUDA tensor is owned by this module, so there is nothing to
# free from ON_CLEANUP. That matters: cleanup can run while DynamicVRAM is
# tearing down the allocator, and freeing a CUDA tensor at that moment makes the
# C++ destructor throw, which ends in std::terminate -> SIGABRT -> the whole
# ComfyUI process dies. Keeping the rows on the host sidesteps it entirely.


def _sigma_text(sigma, t_hit):
    """Sigma for one trace line.

    `sigma` comes from the DIFFUSION_MODEL wrapper, which a custom sampler that
    calls the model directly never runs -- it then holds the *previous* pass's
    last step (0.6315789 in the 2026-11-14 run), so four refine steps all
    printed the same value. The matched table row is immune to that: its t is
    the value the model actually used, and sigma = 1 - t by construction. When
    the two disagree, the row wins and says so.
    """
    derived = 1.0 - float(t_hit[0]) if t_hit else None
    if derived is not None and (not isinstance(sigma, float)
                                or abs(sigma - derived) > 1e-5):
        return "{:.7f}(row)".format(derived)
    return "{:.7f}".format(sigma) if isinstance(sigma, float) else "?"


class ExactAdaLNRuntime:
    def __init__(self, t_values, raw_by_key, meta, dm, mode="strict"):
        self.t_values = list(t_values)
        self.raw_by_key = dict(raw_by_key)
        # Frozen count: detach_payload() drops raw_by_key after the first
        # cleanup, but later passes must still report the real projection count.
        self._projections = len(raw_by_key)
        self.meta = dict(meta)
        self.dm = dm
        self.mode = mode
        self.lookup = sigma_plan.build_index(self.t_values)
        self.out_dtype = _parse_dtype(meta.get("out_dtype", "bfloat16"))
        self.visual_aug = float(meta.get("visual_aug", sigma_plan.VISUAL_COND_TIMESTEP))
        self.audio_aug = float(meta.get("audio_aug", sigma_plan.AUDIO_COND_TIMESTEP))
        self.previous_patch = None
        self.hook_handles = []
        self._key_specs, self.match_mode = self._discover_key_sets(dm)
        self._key_cache = {}
        self.use_match = bool(self._key_specs)
        self.wrapper_key = None
        self.state = {
            "t_indices": None, "sigma_hits": 0, "fallbacks": 0,
            "missing": set(), "last_sigma": None, "last_t": None,
            "match_hits": 0, "match_failures": 0, "match_approx": 0,
            "max_match_distance": 0.0, "last_diag": "",
            "injected": 0, "native_steps": 0, "wrapper_steps": 0,
            "sigma_misses": 0, "max_injection_delta": 0.0,
            "injection_delta_measured": False,
            # Diagnostics that only the report reads. Kept as plain numbers so
            # the hot path pays nothing for them beyond an int increment.
            "last_segs": 0, "last_tokens": 0, "per_row_timestep": False,
            "layout_changes": 0, "shift_changes": 0, "hook_seconds": 0.0,
            "first_dist": None, "first_key": None, "gather_bytes": 0,
            "last_shift": None, "last_hook_at": 0.0,
        }
        self.trace = []            # one entry per forward, blocks.0 only
        self.trace_cap = 96
        self._trace_key = ("blocks.0" if "blocks.0" in raw_by_key
                           else (next(iter(raw_by_key)) if raw_by_key else ""))
        self._last_report = (0, 0, 0, 0, 0)
        self._report_count = 0
        self._pass_sig = None
        # Which slice of `trace` the pass that is currently being reported owns.
        # The trace is appended to for the whole lifetime of the runtime, so a
        # pass block that printed all of it claimed forwards it never ran.
        self._trace_mark = 0
        self._trace_slice = None
        # Set when another Apply node takes over the shared diffusion model.
        # This runtime's hooks are gone by design, so "zero forwards" here is
        # not a failure -- the other node did the injection.
        self.superseded = False
        # Set by the node: emits the pass report for counters that no
        # ON_CLEANUP has claimed. A chain that samples with a patcher we are not
        # attached to never fires our callback for that pass, and its numbers
        # were otherwise lost.
        self.pending_report = None
        # Extra per-forward console line in debug mode, so a pass that never
        # gets a report block is still visible while it runs.
        self.verbose = False

    # -- key construction ------------------------------------------------
    def _discover_key_sets(self, dm):
        """{t_emb width: kind} for every way this model can build t_emb."""
        specs = {}
        modes = []
        tbl = getattr(dm, "adaln_t_table", None)
        if torch.is_tensor(tbl) and tbl.ndim == 2 and tbl.shape[0] >= 2:
            specs[int(tbl.shape[1])] = "curve"
            modes.append("curve-coordinate ({}-wide)".format(int(tbl.shape[1])))
        te = getattr(dm, "time_embedder", None)
        if te is not None:
            width = int(getattr(getattr(te, "proj_out", None), "out_features", 0) or 0)
            if width:
                specs[width] = "time_embedder"
                modes.append("time-embedder ({}-wide)".format(width))
        return specs, " + ".join(modes) if modes else "unavailable"

    def _keys_for(self, t_emb):
        """Per-row keys on t_emb's device/dtype, or None if width is unknown."""
        width = int(t_emb.shape[-1])
        kind = self._key_specs.get(width)
        if kind is None:
            return None, None
        cache_key = (width, t_emb.device.type, str(t_emb.dtype))
        cached = self._key_cache.get(cache_key)
        if cached is not None:
            return cached, kind
        dev = t_emb.device
        dtype = t_emb.dtype
        t = torch.tensor(self.t_values, dtype=torch.float32, device=dev)
        with torch.no_grad():
            if kind == "curve":
                tbl = getattr(self.dm, "adaln_t_table")
                try:
                    import comfy.model_management
                    tbl = comfy.model_management.cast_to(tbl, device=dev)
                except Exception:
                    tbl = tbl.to(device=dev)
                tbl = tbl.to(torch.float32)
                pos = t.clamp(0.0, 1.0) * (tbl.shape[0] - 1)
                i0 = pos.floor().long().clamp(max=tbl.shape[0] - 2)
                keys = torch.lerp(tbl[i0], tbl[i0 + 1], (pos - i0).unsqueeze(1))
            else:
                keys = self.dm.time_embedder(t)
            keys = keys.to(dtype).contiguous()
        self._key_cache[cache_key] = keys
        return keys, kind

    def _match_indices(self, t_emb):
        """(indices|None, max_distance, soft_tolerance, note)."""
        keys, kind = self._keys_for(t_emb)
        if keys is None:
            return None, float("inf"), 0.0, "no key set for width {}".format(t_emb.shape[-1])
        x = t_emb.detach()
        if x.ndim != 2 or x.shape[-1] != keys.shape[-1]:
            return None, float("inf"), 0.0, "shape {} vs keys {}".format(
                tuple(x.shape), tuple(keys.shape))
        # Both sides must be in ONE dtype: torch.cdist refuses mixed input, and
        # casting only t_emb to float32 while the keys stayed bf16 raised
        # "expected scalar type Float but found BFloat16" on the very first
        # step. bf16 -> fp32 is lossless, so the distance stays exact.
        dist = torch.cdist(x.to(torch.float32), keys.to(torch.float32))
        idx = dist.argmin(dim=1)
        mind = dist.gather(1, idx[:, None]).squeeze(1)
        # float32 rounding only; anything above one ulp of the key magnitude is
        # a real difference and must not be swallowed by a tolerance.
        scale = float(keys.abs().max().clamp(min=1.0))
        return (idx.detach().cpu().tolist(), float(mind.max()),
                1e-6 * scale, "{} keys".format(kind))

    # -- fallback wrapper -------------------------------------------------
    def _build_t_indices(self, timestep, transformer_options, minimax_payload):
        """Rebuild model.py:709 unique_t and map it to table rows."""
        payload = minimax_payload or {}
        layout = payload.get("layout")
        if layout is None:
            layout = (transformer_options or {}).get("minimax_h3_layout")
        # Diagnostics for the second pass. A refine / SelfLift step changes the
        # token count (higher resolution) while keeping the same t grid, so a
        # mismatch here separates "the schedule moved" from "only the picture
        # got bigger" -- two failures that used to look identical.
        segs = getattr(layout, "segments", None) or []
        tokens = 0
        for item in segs:
            try:
                tokens += int(item[1]) - int(item[0])
            except (TypeError, IndexError, ValueError):
                pass
        if segs and self.state["last_segs"] and (
                len(segs), tokens) != (self.state["last_segs"], self.state["last_tokens"]):
            self.state["layout_changes"] += 1
        self.state["last_segs"] = len(segs)
        self.state["last_tokens"] = tokens
        if torch.is_tensor(timestep) and timestep.numel() > 1:
            flat = timestep.detach().flatten()
            if float(flat.max()) != float(flat.min()):
                # Per-row timesteps (denoise_mask) are not reconstructable from
                # one sigma; only the t_emb match handles them. Say so instead
                # of silently mapping every row with row 0's timestep.
                self.state["per_row_timestep"] = True
        sigma = sigma_plan.sigma_from_timestep(timestep)
        shift_v, shift_a = sigma_plan.shifts_from_options(transformer_options, self.dm)
        prev_shift = self.state.get("last_shift")
        if prev_shift is not None and (
                abs(prev_shift[0] - shift_v) > 1e-9 or abs(prev_shift[1] - shift_a) > 1e-9):
            self.state["shift_changes"] += 1
        self.state["last_shift"] = (shift_v, shift_a)
        unique_t = sigma_plan.runtime_unique_t(
            sigma, layout, shift_v, shift_a, self.visual_aug, self.audio_aug)
        indices = [self.lookup(t) for t in unique_t]
        missing = [t for t, i in zip(unique_t, indices) if i is None]
        self.state["last_sigma"] = sigma
        self.state["last_t"] = unique_t
        if missing:
            self.state["missing"].update(missing)
            self.state["t_indices"] = None
            return None
        self.state["t_indices"] = indices
        return indices

    def diffusion_wrapper(self, executor, *args, **kwargs):
        """Tolerant wrapper: never changes the forward, only observes it.

        Signature is `(executor, *args, **kwargs)` on purpose. ComfyUI has
        changed the exact kwarg set of `_forward` between releases, and a
        wrapper that raises here would break sampling for no benefit.
        """
        try:
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            transformer_options = (args[3] if len(args) > 3
                                   else kwargs.get("transformer_options", {})) or {}
            payload = kwargs.get("minimax_payload")
            if payload is None and len(args) > 4:
                payload = args[4]
            self.state["wrapper_steps"] += 1
            self._build_t_indices(timestep, transformer_options, payload)
        except Exception as e:
            logging.warning("H3AdaLN: sigma reconstruction failed (%s: %s)",
                            type(e).__name__, e)
            self.state["t_indices"] = None
        return executor(*args, **kwargs)

    # -- legacy dit patch -------------------------------------------------
    def _previous(self, model):
        try:
            to = model.model_options.get("transformer_options", {})
            return to.get("patches_replace", {}).get("dit", {}).get(("double_block", 0))
        except Exception:
            return None

    def dit_patch(self, args, block_info=None):
        """Legacy resolver for engines that only populate patches_replace."""
        if isinstance(block_info, dict):
            info = block_info
            fn = block_info.get("original_block")
        else:
            info = {"original_block": block_info}
            fn = block_info

        to = args.get("transformer_options", {}) or {}
        try:
            indices = self._build_t_indices(
                to.get("sigmas"), to, {"layout": args.get("layout")})
            if indices is None:
                if self.mode == "strict":
                    raise RuntimeError(
                        "H3AdaLN: table has no row for t={} (sigma={}). Re-record "
                        "with this workflow's sigmas, or use mode=fallback.".format(
                            ", ".join("{:.7f}".format(t)
                                      for t in sorted(self.state["missing"])[:8]),
                            self.state["last_sigma"]))
                self.state["sigma_misses"] += 1
            else:
                self.state["sigma_hits"] += 1
        except RuntimeError:
            raise
        except Exception as e:
            logging.warning("H3AdaLN: could not resolve timesteps (%s: %s)",
                            type(e).__name__, e)
            self.state["t_indices"] = None
            self.state["sigma_misses"] += 1

        if not callable(fn):
            raise RuntimeError(
                "H3AdaLN: ComfyUI did not supply a callable original_block "
                "(got {}). Cannot run the dit fallback.".format(type(fn).__name__))
        if self.previous_patch is not None:
            return self.previous_patch(args, info)
        return fn(args)

    # -- injection --------------------------------------------------------
    def _gather(self, key, indices, device):
        """Upload only the 1-2 rows this step needs, from the CPU-side table.

        Keeping the whole ~446 MB table resident on the GPU (or allocating it
        during the first forward, inside comfy-aimdo's malloc graph) eats VRAM
        that DynamicVRAM cannot account for and destabilises the compiled
        allocation graph. Host RAM is cheap here; per-step H2D is a few MB.
        """
        idx = torch.as_tensor(indices, dtype=torch.long)
        rows = self.raw_by_key[key].index_select(0, idx)
        return rows.to(device=device, dtype=self.out_dtype).contiguous()

    def _strict_error(self, key, t_emb, detail):
        missing = sorted(self.state["missing"])
        rows = ["H3AdaLN: cannot map {} rows to table rows.".format(
            tuple(t_emb.shape)),
            "reason: {}".format(detail),
            "table t rows ({}): {}".format(
                len(self.t_values),
                ", ".join("{:.7f}".format(t) for t in self.t_values[:12]) +
                (" ..." if len(self.t_values) > 12 else ""))]
        if missing:
            rows.append("missing from table ({}): {}".format(
                len(missing), ", ".join("{:.7f}".format(t) for t in missing[:12])))
            rows.append("This is a t-set mismatch, not a precision problem: the running "
                        "schedule/shift produces timesteps the table was never recorded "
                        "for. Re-record on this exact schedule (or add them via extra_t).\n"
                        "Steps / scheduler / shifts change the timestep set; resolution, "
                        "frame count and seed do not.")
        else:
            rows.append("No timestep reconstruction was available either -- record the "
                        "table on this exact model + schedule (steps / scheduler / "
                        "shifts). Resolution and frame count do not matter.")
        return "\n".join(rows)

    def make_hook(self, key):
        def hook(module, inputs, output):
            t_emb = inputs[0]
            m = int(t_emb.shape[0])
            indices = None
            detail = ""
            how = "native"
            maxd, tol, note = float("inf"), 0.0, ""

            if self.use_match:
                indices, maxd, tol, note = self._match_indices(t_emb)
                self.state["max_match_distance"] = max(
                    self.state["max_match_distance"], maxd)
                if indices is not None and maxd <= tol:
                    self.state["match_hits"] += 1
                    how = "exact" if maxd == 0.0 else "approx"
                    if maxd > 0.0:
                        self.state["match_approx"] += 1
                else:
                    self.state["match_failures"] += 1
                    keys, _kind = self._keys_for(t_emb)
                    self.state["last_diag"] = (
                        "t_emb={} keys={} max_dist={:.3e} tol={:.3e} ({})".format(
                            tuple(t_emb.shape),
                            tuple(keys.shape) if keys is not None else "?",
                            maxd, tol, note))
                    indices = None

            if indices is None:
                fallback = self.state["t_indices"]
                if fallback is not None and len(fallback) == m:
                    indices = fallback
                    self.state["sigma_hits"] += 1
                    how = "sigma-fallback"
                else:
                    detail = self.state["last_diag"] or "no key set and no sigma mapping"

            self._trace(key, t_emb, indices, maxd, tol, how, note)

            if indices is None or len(indices) != m:
                self.state["native_steps"] += 1
                if self.mode == "strict":
                    raise RuntimeError(self._strict_error(key, t_emb, detail))
                return None

            self.state["injected"] += 1
            if self.state["first_key"] is None:
                # Report the distance of THIS hook, not the running maximum over
                # all 51 (which was already contaminated by later blocks).
                self.state["first_key"] = key
                self.state["first_dist"] = maxd
                logging.info(
                    "H3AdaLN: first injection confirmed during sampling "
                    "(%s, %d rows, indices=%s, max_dist=%.3e, via %s)",
                    key, m, indices, maxd, how)

            gathered = self._gather(key, indices, t_emb.device)
            x = gathered.reshape(m * module.modalities,
                                 module.expand * module.hidden)
            out = x.chunk(module.expand, dim=-1)
            # Proof that the table actually replaced the base's native AdaLN
            # output. AdalnProj returns a TUPLE of chunks, not a tensor, so the
            # old `torch.is_tensor(output)` test never matched and the delta was
            # permanently "UNAVAILABLE" on the real model.
            self._measure_delta(key, output, out)
            return out
        return _outside_malloc_graph(hook)

    def _measure_delta(self, key, old, new):
        """Record |table row - base AdaLN| for this step, if measurable."""
        if torch.is_tensor(old) and torch.is_tensor(new):
            pairs = [(old, new)]
        elif isinstance(old, (tuple, list)) and isinstance(new, (tuple, list)) \
                and len(old) == len(new):
            pairs = [(a, b) for a, b in zip(old, new)
                     if torch.is_tensor(a) and torch.is_tensor(b)]
        else:
            return
        delta = 0.0
        for a, b in pairs:
            if a.shape != b.shape:
                return
            delta = max(delta, float((b.detach().to(torch.float32)
                                      - a.detach().to(torch.float32)).abs().max()))
        if not pairs:
            return
        self.state["max_injection_delta"] = max(
            self.state["max_injection_delta"], delta)
        self.state["injection_delta_measured"] = True

    def _trace(self, key, t_emb, indices, maxd, tol, how, note):
        """One report line per forward -- recorded on the first projection only."""
        self.state["last_emb_width"] = int(t_emb.shape[-1])
        self.state["last_hook_at"] = time.time()
        if key != self._trace_key or len(self.trace) >= self.trace_cap:
            return
        # Only a match tells us which row (hence which t) this forward used.
        # A sigma-fallback reuses `state["t_indices"]`, which may be the
        # previous pass's, so it must not be presented as this forward's t.
        t_hit = None
        if how in ("exact", "approx") and indices is not None:
            try:
                t_hit = [round(float(self.t_values[i]), 7) for i in indices]
            except (IndexError, TypeError, ValueError):
                t_hit = None
        self.trace.append({
            "forward": self.state["wrapper_steps"],
            "module": key,
            "rows": int(t_emb.shape[0]),
            "width": int(t_emb.shape[-1]),
            "indices": list(indices) if indices is not None else None,
            "t_hit": t_hit,
            "max_dist": float(maxd),
            "tol": float(tol),
            "how": how,
            "note": note,
            "sigma": self.state.get("last_sigma"),
            "tokens": self.state.get("last_tokens"),
            "segs": self.state.get("last_segs"),
            "shift": self.state.get("last_shift"),
        })
        if self.verbose:
            logging.info(
                "H3AdaLN: fwd %d %s t_emb=%dx%d -> rows=%s max_dist=%.3e "
                "sigma=%s%s",
                len(self.trace) - 1, how,
                int(t_emb.shape[0]), int(t_emb.shape[-1]),
                indices, maxd,
                _sigma_text(self.state.get("last_sigma"), t_hit),
                "" if how != "native" else "  (NATIVE AdaLN -- not injected)")

    # -- registration ------------------------------------------------------
    def register(self, model, use_sigma_wrapper=True, use_dit_patch=False):
        """Attach to `model` (a ModelPatcher). Returns number of hooked modules."""
        dm = self.dm
        targets = []
        for i, block in enumerate(dm.blocks):
            ap = getattr(block, "adaln_proj", None)
            if ap is not None:
                targets.append(("blocks.{}".format(i), ap))
        fl_ap = getattr(getattr(dm, "final_layer", None), "adaln_proj", None)
        if fl_ap is None:
            raise RuntimeError("H3AdaLN: final_layer.adaln_proj missing "
                               "(50 blocks + final = 51 projections required)")
        targets.append(("final_layer", fl_ap))

        missing_keys = [k for k, _ in targets if k not in self.raw_by_key]
        if missing_keys:
            raise RuntimeError("H3AdaLN: table is missing projections: {}".format(
                ", ".join(missing_keys[:8])))

        if use_sigma_wrapper:
            self._register_sigma_wrapper(model)
        if use_dit_patch:
            self.previous_patch = self._previous(model)
            model.set_model_patch_replace(self.dit_patch, "dit", "double_block", 0)

        for key, module in targets:
            self.hook_handles.append(module.register_forward_hook(self.make_hook(key)))
        return len(targets)

    def _register_sigma_wrapper(self, model):
        try:
            import comfy.patcher_extension as pe
            self.wrapper_key = "h3_exact_apply"
            self._wrapper_model = model
            pe.add_wrapper_with_key(
                pe.WrappersMP.DIFFUSION_MODEL, self.wrapper_key,
                self.diffusion_wrapper, model.model_options, is_model_options=True)
        except Exception as e:
            logging.warning("H3AdaLN: could not register the sigma wrapper "
                            "(%s: %s); relying on t_emb matching only",
                            type(e).__name__, e)
            self.wrapper_key = None

    def remove(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles = []
        # Key tensors are per-runtime and cheap; dropping the reference is
        # enough (nothing here owns a CUDA allocation).
        self._key_cache = {}
        # Drop the wrapper too: re-running Apply on the same MODEL object would
        # otherwise stack one wrapper per run, each holding a dead runtime.
        model = getattr(self, "_wrapper_model", None)
        if model is not None and self.wrapper_key:
            try:
                wrappers = (model.model_options.get("transformer_options", {})
                            .get("wrappers", {}).get("diffusion_model", {}))
                bucket = wrappers.get(self.wrapper_key)
                if bucket:
                    # bound methods are created fresh on every attribute
                    # access, so identity has to go through __self__.
                    wrappers[self.wrapper_key] = [
                        f for f in bucket if getattr(f, "__self__", None) is not self]
                    if not wrappers[self.wrapper_key]:
                        wrappers.pop(self.wrapper_key, None)
            except Exception:
                pass
            self._wrapper_model = None

    def begin_pass(self):
        """Close out one sampling pass; returns this pass's counter deltas.

        Called from ON_CLEANUP, i.e. once per sampling call, so a director
        chain reports pass 1 and pass 2 separately instead of one cumulative
        blob where a pass that injected nothing hides inside the total.

        Also freezes which slice of `trace` belongs to this pass. The trace is
        appended to for the whole runtime, so printing all of it under one
        block made a 4-forward refine look like a 12-forward pass (the first
        eight forwards belonged to the pass before it).
        """
        snap = (self.state["injected"], self.state["match_hits"],
                self.state["match_failures"], self.state["native_steps"],
                self.state["wrapper_steps"])
        prev = self._last_report
        self._last_report = snap
        self._report_count += 1
        self._trace_slice = (self._trace_mark, len(self.trace))
        self._trace_mark = len(self.trace)
        return [b - a for a, b in zip(prev, snap)]

    def detach_payload(self):
        """Drop heavy references; ON_CLEANUP callbacks are never unregistered."""
        self.raw_by_key = {}
        self.dm = None
        self.meta = {}

    def seal(self):
        """Detach the hooks AND release the table rows.

        A runtime that is never re-armed otherwise keeps its ~335 MB host copy
        alive through the ON_CLEANUP closure forever -- four applies in one
        session was a 1.3 GB leak.
        """
        self.remove()
        self.detach_payload()

    def trace_lines(self, limit=None, start=None, end=None):
        """Per-forward mapping trace, for the debug report.

        Defaults to the pass `begin_pass()` just closed (`_trace_slice`), not
        the whole runtime. The index printed is still the absolute trace
        position, so a pass that starts at `fwd 8` is visibly a continuation of
        the one before it -- it just no longer re-prints that pass's forwards.
        """
        if start is None and end is None and self._trace_slice:
            start, end = self._trace_slice
        if start is None:
            start = 0
        if end is None:
            end = len(self.trace)
        entries = self.trace[start:end]
        if limit:
            entries = entries[:limit]
        out = []
        used_row = False
        for i, item in enumerate(entries, start=start):
            sigma = item.get("sigma")
            shift = item.get("shift")
            sigma_txt = _sigma_text(sigma, item.get("t_hit"))
            used_row = used_row or sigma_txt.endswith("(row)")
            # `forward` is the wrapper's step counter, which stays 0 when the
            # wrapper does not run (a chain that samples without model_options).
            # Printing it as "step" produced eight identical `step 0` lines.
            out.append(
                "fwd {:<3} wrapper#{} {:<14} sigma={} shift={}/{} tokens={} "
                "segs={} t_emb={}x{} -> indices={} {} max_dist={:.3e}".format(
                    i, item.get("forward", 0), item["how"], sigma_txt,
                    "{:g}".format(shift[0]) if shift else "?",
                    "{:g}".format(shift[1]) if shift else "?",
                    item.get("tokens"), item.get("segs"),
                    item.get("rows"), item.get("width"),
                    item.get("indices"), "" if item["how"] != "approx"
                    else "(nearest)", item.get("max_dist", 0.0)))
        if used_row:
            out.append("  (row) = 1 - t of the matched table row: the wrapper did "
                       "not run for that forward, so its own sigma was stale")
        if len(self.trace) >= self.trace_cap:
            out.append("... trace truncated at {} forwards".format(self.trace_cap))
        return out

    def summary(self):
        st = self.state
        lines = [
            "scope            : cumulative since arm -- the delta for this pass "
            "is the `this pass` line above",
            "mode             : {}".format(self.mode),
            "injection mode   : {}".format(self.match_mode),
            "t rows in table  : {}".format(len(self.t_values)),
            "projections      : {}".format(
                getattr(self, "_projections", len(self.raw_by_key))),
            "out dtype        : {}".format(self.out_dtype),
            "exact matches    : {}".format(st["match_hits"] - st["match_approx"]),
            "approx matches   : {} (max_dist={:.3e})".format(
                st["match_approx"], st["max_match_distance"]),
            "match failures   : {}".format(st["match_failures"]),
            "injections       : {}".format(st["injected"]),
            "native steps     : {}".format(st["native_steps"]),
            "sigma fallbacks  : {} hit / {} miss (wrapper steps {})".format(
                st["sigma_hits"], st["sigma_misses"], st["wrapper_steps"]),
            "first injection  : {} at max_dist={}".format(
                st["first_key"], "?" if st["first_dist"] is None
                else "{:.3e}".format(st["first_dist"])),
            "max injection D  : {}".format(
                "{:.6e} (table row vs native AdaLN)".format(st["max_injection_delta"])
                if st["injection_delta_measured"]
                else "UNAVAILABLE (another Apply hook already replaced output)"),
            "last wrapper sigma/t: {} / {}".format(
                st["last_sigma"],
                ",".join("{:.7f}".format(t) for t in (st["last_t"] or [])[:6])
                or "none"),
            "last layout      : {} segments, {} tokens".format(
                st["last_segs"], st["last_tokens"]),
            "layout changes   : {}  shift changes: {}".format(
                st["layout_changes"], st["shift_changes"]),
            "per-row t        : {}".format(
                "YES (denoise_mask: only the t_emb match is reliable)"
                if st["per_row_timestep"] else "no"),
            "last diagnostic  : {}".format(st["last_diag"] or "none"),
            "missing t values : {}".format(
                ", ".join("{:.7f}".format(t) for t in sorted(st["missing"])[:12])
                or "none"),
        ]
        return "\n".join(lines)
