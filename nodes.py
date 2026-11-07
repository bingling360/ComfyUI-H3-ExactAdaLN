"""ComfyUI nodes for exact (pre-baked) MiniMax-H3 AdaLN modulation.

Four-stage, image-like workflow
--------------------------------
  Bake   [full teacher MODEL + LoRA + sigmas] -> H3_ADALN_TABLE (in memory)
  Save   [H3_ADALN_TABLE]  -> .safetensors in output/ (shareable, like Save Image)
  Load   [upload / pick file] -> H3_ADALN_TABLE (like Load Image)
  Apply  [pruned MODEL + H3_ADALN_TABLE] -> patched MODEL

The table file is self-contained: all provenance (teacher, LoRA fingerprint,
sigma grid, shifts, dtype, ComfyUI version) lives in its safetensors header,
so it can be copied to another machine and loaded without any sidecar.

The bake path never parses a safetensors file by hand: it goes through
ComfyUI's ModelPatcher, so bf16 / fp16 / fp8_scaled / int8_convrot / nvfp4
checkpoints and any ComfyUI-applicable LoRA are handled by ComfyUI itself.
"""

import hashlib
import json
import logging
import os

import torch

try:
    from . import apply_engine, bake_engine, sigma_plan, table_io, latent_io
except ImportError:
    import apply_engine, bake_engine, sigma_plan, table_io, latent_io

CATEGORY = "MiniMaxH3/ExactAdaLN"
TABLE_TYPE = "H3_ADALN_TABLE"
_MODES = ["strict", "fallback", "off"]


def _sigmas_from_model(model, steps, scheduler):
    import comfy.samplers
    ms = model.get_model_object("model_sampling")
    return comfy.samplers.calculate_sigmas(ms, scheduler, steps)


def _auto_shifts(model):
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


def _lora_fingerprint(model):
    """Stable id of the non-AdaLN LoRA target set.

    AdaLN / time_embedder keys exist on the full teacher but are absent on a
    pruned base, so they are excluded: the remaining key sets must match
    between the bake chain and the apply chain.
    """
    keys = sorted(
        k for k in getattr(model, "patches", {}).keys()
        if "time_embedder" not in k and "adaln_proj" not in k
    )
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]
    return digest, len(keys)


def _comfy_version():
    try:
        import comfyui_version
        return str(comfyui_version.__version__)
    except Exception:
        return "unknown"


class H3ExactAdaLNBake:
    """Compute an exact table from a FULL H3 model (+ optional LoRA). No disk IO."""

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
            },
        }

    RETURN_TYPES = (TABLE_TYPE, "STRING")
    RETURN_NAMES = ("table", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, steps, scheduler, compute_dtype, device, auto_shift,
            sigmas=None, extra_t="", shift_v=12.0, shift_a=3.0,
            visual_aug=0.999, audio_aug=1.0, teacher_tag="", lora_tag=""):
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
            progress=progress)

        fingerprint, patch_count = _lora_fingerprint(model)
        meta = {
            "teacher": teacher_tag or "(unspecified full checkpoint)",
            "lora_tag": lora_tag,
            "lora_fingerprint": fingerprint,
            "lora_patch_count": str(patch_count),
            "sigmas": json.dumps([round(float(s), 9) for s in sigma_values]),
            "shift_v": str(float(shift_v)),
            "shift_a": str(float(shift_a)),
            "visual_aug": str(float(visual_aug)),
            "audio_aug": str(float(audio_aug)),
            "compute_dtype": compute_dtype,
            "num_blocks": str(info["projections"] - 1),
            "t_embed_dim": str(info["t_embed_dim"]),
            "applied_lora_patches": str(info["applied_patches"]),
            "comfyui_version": _comfy_version(),
            "out_dtype": "torch." + compute_dtype,
        }
        table = table_io.AdaLNTable(t_values, raw_by_key, meta)

        report = "\n".join([
            "=== H3 Exact AdaLN bake (in memory) ===",
            table.summary(),
            "auto shift     : {} (v={}, a={})".format(auto_shift, shift_v, shift_a),
            "applied patches: {}/{} AdaLN+timestep keys".format(
                info["applied_patches"], info["target_patch_keys"]),
            "patch errors   : {}".format("; ".join(info["patch_errors"][:3]) or "none"),
            "teacher weights: te={} ({}) | adaln={} ({}) in_features={}".format(
                info.get("teacher_te_type", "?"), info.get("teacher_te_dtype", "?"),
                info.get("teacher_adaln_type", "?"), info.get("teacher_adaln_dtype", "?"),
                info.get("teacher_adaln_in_features", "?")),
            "device         : {}".format(info["device"]),
            "",
            "Next: connect `table` to H3 Exact AdaLN Save.",
        ])
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, table, filename_prefix="h3_adaln/H3Exact", location="output"):
        import shutil
        import folder_paths
        from urllib.parse import quote

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
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, table, mode, expected_teacher=""):
        if mode == "off":
            return (model, "H3ExactAdaLNApply: mode=off, model untouched (A/B control)")

        dm = model.model.diffusion_model
        arch = "pruned/curve" if getattr(dm, "use_adaln_curves", False) else "full"
        current_fp, current_count = _lora_fingerprint(model)
        baked_fp = table.meta.get("lora_fingerprint", "")
        fp_ok = (current_fp == baked_fp)
        teacher_ok = (not expected_teacher) or (
            expected_teacher in table.meta.get("teacher", ""))

        problems = []
        if not fp_ok:
            problems.append("LoRA fingerprint mismatch: table={} model={} ({} vs {} non-AdaLN keys)".format(
                baked_fp, current_fp, table.meta.get("lora_patch_count", "?"), current_count))
        if not teacher_ok:
            problems.append("teacher mismatch: expected {!r}, table has {!r}".format(
                expected_teacher, table.meta.get("teacher", "")))
        if problems and mode == "strict":
            raise RuntimeError("H3ExactAdaLNApply: " + " | ".join(problems) +
                               " -- re-bake the table with this exact model+LoRA stack.")

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
            "target arch    : {}".format(arch),
            "hooked modules : {}".format(n),
            "lora fp        : {} ({})".format(current_fp, "match" if fp_ok else "MISMATCH"),
            "teacher        : {}".format(table.meta.get("teacher", "?")),
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

        ta = strip_side(collect(latent_a, "a"), "a")
        tb = strip_side(collect(latent_b, "b"), "b")
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
        if compared == 0:
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
    "H3ExactAdaLNBake": H3ExactAdaLNBake,
    "H3ExactAdaLNSave": H3ExactAdaLNSave,
    "H3ExactAdaLNLoad": H3ExactAdaLNLoad,
    "H3ExactAdaLNApply": H3ExactAdaLNApply,
    "H3LatentCompare": H3LatentCompare,
    "H3LatentSave": latent_io.H3LatentSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ExactAdaLNBake": "H3 Exact AdaLN Bake (full teacher)",
    "H3ExactAdaLNSave": "H3 Exact AdaLN Save Table",
    "H3ExactAdaLNLoad": "H3 Exact AdaLN Load Table",
    "H3ExactAdaLNApply": "H3 Exact AdaLN Apply (pruned base)",
    "H3LatentCompare": "H3 Latent Compare (verification)",
    "H3LatentSave": "H3 Latent Save (NestedTensor)",
}
