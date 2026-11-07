# ComfyUI-H3-ExactAdaLN

Pre-bake the **full** MiniMax-H3 model's AdaLN modulation (including every LoRA
delta) into an exact timestep-indexed table, then replay that table on a
**pruned / curve-form** checkpoint instead of the compressed `[1025, 8]` curve.

在完整模型上预计算精确的 AdaLN 调制表，生成时直接注入剪枝版 MiniMax-H3，
完全跳过 `adaln_t_table` 插值与折叠后的 `adaln_proj [96768, 8]`。

## Why

H3 has ~13B parameters (39%) in AdaLN branches whose output depends only on the
timestep. Comfy-Org's pruned checkpoints fold that whole branch into an 8-wide
curve table. The rank-8 truncation is mathematically exact (SVD residual 0),
but the shipped path still introduces:

- linear interpolation on a 1025-point grid,
- fp32/fp16 storage of the curve and folded projection,
- quantisation of `adaln_proj.linear` in some pruned builds,
- and, most importantly, **silent loss of all 51 AdaLN/timestep LoRA keys**:
  `time_embedder.*` no longer exists, and a `[96768, 2688]` LoRA delta cannot be
  reshaped onto `[96768, 8]`, so ComfyUI logs an error and drops it.

This node pack bakes the **final modulation tensors** for every timestep used
by a fixed schedule, from the full teacher with the LoRA stack applied. It does
not constrain the result to an 8-dimensional subspace, so the baked values are
exact for any LoRA that ComfyUI can apply.

## Install

```bash
cd ComfyUI/custom_nodes
git clone --depth 1 -b main https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git
```

If `github.com:443` times out (common on Chinese cloud platforms), use one of
these fallbacks, both verified end-to-end:

```bash
# 1) reverse proxy - full git clone, keeps .git so `git pull` works
git clone --depth 1 -b main https://gh-proxy.com/https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git

# 2) GitHub's own codeload CDN - no third-party proxy involved
curl -4 -L --retry 3 -o /tmp/h3exact.zip \
  https://codeload.github.com/bingling360/ComfyUI-H3-ExactAdaLN/zip/refs/heads/main \
  && unzip -q /tmp/h3exact.zip \
  && mv ComfyUI-H3-ExactAdaLN-main ComfyUI-H3-ExactAdaLN \
  && rm -f /tmp/h3exact.zip
```

The ComfyUI root differs between images; find it with
`find / -maxdepth 5 -type d -name custom_nodes 2>/dev/null`.

Restart ComfyUI. Requirements: ComfyUI 0.42.5+ (the `MiniMaxH3Model` /
`patches_replace` implementation), PyTorch, `safetensors`. No extra pip packages.
Verified against ComfyUI 0.44.0 / PyTorch 2.9.1 / RTX 4080 SUPER.

## Nodes

| Node | In | Out | Behaviour |
|---|---|---|---|
| `H3 Exact AdaLN Bake` | full MODEL (+ LoRA), SIGMAS or steps/scheduler | `H3_ADALN_TABLE` | Computes the table in memory. No disk IO. |
| `H3 Exact AdaLN Save Table` | `H3_ADALN_TABLE` | file path | Like **Save Image**: auto-numbered `.safetensors` under `output/h3_adaln/`, or into `models/h3_exact_adaln/`. Also mirrors into `input/`. |
| `H3 Exact AdaLN Load Table` | `table_name` (free text) / `table_path` | `H3_ADALN_TABLE` | Resolves a filename, substring, relative path or absolute path. Scans `input/`, `output/`, `models/h3_exact_adaln/`. |
| `H3 Exact AdaLN Apply` | pruned MODEL + `H3_ADALN_TABLE` | MODEL | Replaces all 51 AdaLN projections at runtime. `strict` / `fallback` / `off`. |
| `H3 Latent Compare` | two LATENT | report | `equal`, `max_abs`, `PSNR`; PASS = bitwise identical. |

A saved table is **self-contained**: its safetensors header carries the
teacher, LoRA fingerprint, sigma grid, shifts, compute dtype and ComfyUI
version. Copy the file anywhere to share it; the receiving side just Loads it.
`strict` mode refuses a table baked with a different LoRA stack.

## Sharing and downloading tables

ComfyUI's asset/output panel only understands media types, so a raw
`.safetensors` produced by a custom node is normally not downloadable from the
UI. Save Table works around this in three ways:

1. **Direct download link** — the Save report prints a relative URL like
   `/view?filename=h3_exact_dmad8_v1_00001.safetensors&subfolder=h3_adaln&type=output`.
   Paste `<your ComfyUI host>` + that URL into a browser; `/view` serves any
   file under `output/` as `application/octet-stream`, so it downloads directly.
2. **Output panel entry** — Save also reports the file through ComfyUI's
   generic `ui.images` channel, so a download card appears without opening the
   folder (the thumbnail may be blank for a non-image file).
3. **input/ mirror** — the same file is copied into `ComfyUI/input/`, which
   most platform file browsers and the Load node can reach directly.

Load Table deliberately uses **free text**, not a dropdown + `image_upload`
widget. A COMBO list is built once at ComfyUI startup (so a table saved later
never appears), and the image upload control only accepts image formats (so a
`.safetensors` cannot be selected). The table is already on the server; point
at it by name or path instead. If the name cannot be resolved, the error message
lists every table file it can see.

## Workflows

- `workflows/01_bake_exact_adaln.json` — Bake → Save. Queue once on its own.
- `workflows/02_verify_full_vs_pruned.json` — Load → Apply on chain B, chain A
  runs the full model natively; both chains share LoRA, sigmas, seed, sampler,
  conditioning and shift. Compare the two latents.

Defaults match the DMAD 8-step LoRA: 8 steps, `euler`, `simple`, strength 1.0,
no CFG, video shift 12 / **audio shift 2**. Change the LoRA filename to your
own before running. The short verification clip (832x480, 39 frames) can be
changed to the full 1344x768 / 124 frames as long as both chains stay in sync
and the table is re-baked.

## Implementation notes

- Bake never parses a safetensors file by hand. It calls
  `ModelPatcher.patch_weight_to_device()` to materialise the effective
  (dequantised + LoRA-patched) weights and then runs the real modules, so
  bf16 / fp16 / fp8_scaled / int8_convrot / nvfp4 checkpoints all work without
  format-specific code. Only `time_embedder` and the 51 `adaln_proj` LoRA keys
  are materialised.
- The table stores the **raw** `adaln_proj.linear` output, before
  `view`/`chunk`. Runtime gathers rows by `t` and replays the module's own
  reshape, so it does not depend on `expand`/`hidden`/`modalities` semantics.
- **Injection is sampler-agnostic**: Apply identifies timesteps from `t_emb`
  itself, so it works whether the sampler is `SamplerCustomAdvanced` or a
  custom node calling `comfy.sample.sample()` directly (SequenceForge,
  HyperFlow adapters, LCM, ...), which never reaches `patches_replace`.
  - pruned / curve base: `t_emb` is the 8-wide lerped curve coordinate,
    rebuilt from `adaln_t_table` for every baked `t`;
  - full base: `t_emb = time_embedder(t).to(dtype)` (2688-wide), rebuilt by
    running the model's own `time_embedder` on the baked `t` values.
  Each of the 51 `adaln_proj` forward hooks nearest-matches the incoming
  `t_emb` rows against that `[T, k]` key matrix and returns the exact rows.
  No `layout`, no `transformer_options["sigmas"]`, no `patches_replace`.
- A dit patch (`set_model_patch_replace(..., "dit", "double_block", 0)`) is
  kept only as a last resort. ComfyUI passes its second argument as
  `{"original_block": callable}` (a dict, not a function); both that form and
  a bare callable are handled.
- `strict` reports diagnostic info (t_emb shape, key shape, max distance,
  tolerance) when timesteps cannot be matched; the usual cause is a table
  baked for a **different sigma schedule / shift** than the running workflow.
- Tables are stored fp32; bf16 costs ~2.5e-2 relative AdaLN error.

## Known limits

- A table is a function of (base weights, LoRA set **and strengths**, sigma
  grid, shifts). Changing any of them requires a re-bake. Strength cannot be
  interpolated at runtime when a LoRA touches `time_embedder` (two SiLUs make
  it non-linear).
- `compute_dtype` defaults to `bfloat16` and must match the actual inference
  dtype, or the two chains will not be bitwise identical.
- HyperFlow-style `(t, r)` endpoint conditioning is not reproduced by Bake;
  per-row `denoise_mask` timesteps work at runtime but must exist in the baked
  table (add them via `extra_t`, or use `mode=fallback`).
- Python hooks cannot reach DiTs executed by a C++ engine (e.g. QuantFunc) or
  fused kernels that bypass `module.forward`.
- Status: baking was verified end-to-end on ComfyUI 0.44.0 (51 projections,
  17 t rows, 314.5 MiB table). The curve-matching injection and the dit-patch
  signature fix passed local regression against that real table with synthetic
  modules; **a fresh end-to-end sampling run after this fix is still pending**.

## Related work

`ComfyUI-HyperFlow-H3` (Create Curve Fit), `H3 AdaLN LoRA Fix`,
`ComfyUI-VDN-H3-Plus` and `indhic-ai/curve_prune.py` all project or fit LoRA
AdaLN deltas **back into the 8-wide curve basis**, and their authors note the
result is an approximation. This pack instead stores the final modulation
values directly, at the cost of a larger, schedule-bound table.
