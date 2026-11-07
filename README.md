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
git clone https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git
```

Restart ComfyUI. Requirements: ComfyUI 0.42.5+ (the `MiniMaxH3Model` /
`patches_replace` implementation), PyTorch, `safetensors`. No extra pip packages.

## Nodes

| Node | In | Out | Behaviour |
|---|---|---|---|
| `H3 Exact AdaLN Bake` | full MODEL (+ LoRA), SIGMAS or steps/scheduler | `H3_ADALN_TABLE` | Computes the table in memory. No disk IO. |
| `H3 Exact AdaLN Save Table` | `H3_ADALN_TABLE` | file path | Like **Save Image**: auto-numbered `.safetensors` under `output/h3_adaln/`, or into `models/h3_exact_adaln/`. |
| `H3 Exact AdaLN Load Table` | upload button / dropdown / absolute path | `H3_ADALN_TABLE` | Like **Load Image**: scans `input/`, `output/`, `models/h3_exact_adaln/`. |
| `H3 Exact AdaLN Apply` | pruned MODEL + `H3_ADALN_TABLE` | MODEL | Replaces all 51 AdaLN projections at runtime. `strict` / `fallback` / `off`. |
| `H3 Latent Compare` | two LATENT | report | `equal`, `max_abs`, `PSNR`; PASS = bitwise identical. |

A saved table is **self-contained**: its safetensors header carries the
teacher, LoRA fingerprint, sigma grid, shifts, compute dtype and ComfyUI
version. Copy the file anywhere to share it; the receiving side just Loads it.
`strict` mode refuses a table baked with a different LoRA stack.

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
- Injection uses `set_model_patch_replace(..., "dit", "double_block", 0)` to
  rebuild `unique_t` and 51 `adaln_proj` forward hooks. It chains with any
  pre-existing dit patch. Generic `set_model_attn1_patch`-style options are not
  used because H3's `model.py` does not read them.
- Tables are stored fp32; bf16 costs ~2.5e-2 relative AdaLN error.

## Known limits

- A table is a function of (base weights, LoRA set **and strengths**, sigma
  grid, shifts). Changing any of them requires a re-bake. Strength cannot be
  interpolated at runtime when a LoRA touches `time_embedder` (two SiLUs make
  it non-linear).
- `compute_dtype` defaults to `bfloat16` and must match the actual inference
  dtype, or the two chains will not be bitwise identical.
- Per-row `denoise_mask` timesteps and HyperFlow-style `(t, r)` conditioning
  are not enumerated by default. `strict` reports the missing `t` values; add
  them to Bake's `extra_t` and re-bake, or use `mode=fallback` to let those
  steps use the native path.
- Python hooks cannot reach DiTs executed by a C++ engine (e.g. QuantFunc) or
  fused kernels that bypass `module.forward`.
- This is an experimental v0.1. Local validation covered schemas, table
  round-trips, hook replacement, sigma/t coverage and workflow wiring with
  synthetic tensors; **end-to-end GPU verification is still pending**.

## Related work

`ComfyUI-HyperFlow-H3` (Create Curve Fit), `H3 AdaLN LoRA Fix`,
`ComfyUI-VDN-H3-Plus` and `indhic-ai/curve_prune.py` all project or fit LoRA
AdaLN deltas **back into the 8-wide curve basis**, and their authors note the
result is an approximation. This pack instead stores the final modulation
values directly, at the cost of a larger, schedule-bound table.
