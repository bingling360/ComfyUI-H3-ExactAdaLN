# ComfyUI-H3-ExactAdaLN

Make a **pruned** MiniMax-H3 checkpoint reproduce the **full** model's AdaLN
modulation exactly, so that

> pruned base + exact table + LoRA  ==  full teacher + the same LoRA  (bitwise)

在剪枝版 MiniMax-H3 上逐位复刻完整模型的 AdaLN 调制：生成结果与「完整模型 +
同一 LoRA」完全一致，速度维持剪枝版水平，表可 Save/Load 分享。

## Why

H3 puts ~13B parameters (39%) into AdaLN branches whose output depends only on
the timestep. Comfy-Org's pruned checkpoints fold that whole branch into a
small curve basis. The rank-8 truncation is mathematically exact (SVD residual
0), but the shipped path still introduces:

- linear interpolation between curve rows,
- fp32/fp16 storage differences along the curve,
- quantisation of `adaln_proj.linear` in some pruned builds,
- and, most importantly, **silent loss of all 51 AdaLN/timestep LoRA keys**:
  `time_embedder.*` no longer exists, and a `[96768, 2688]` LoRA delta cannot be
  reshaped onto `[96768, 8]`, so ComfyUI logs an error and drops it.

This pack stores the **final modulation tensors** for every timestep a fixed
schedule uses, taken from the full teacher with the LoRA stack applied. It does
not constrain the result to an 8-dimensional subspace, so the stored values are
exact for any LoRA ComfyUI can apply.

## The exactness contract

Two ways to produce a table, and only one of them is exact.

| | `H3 AdaLN Record` | `H3 Exact AdaLN Bake` |
|---|---|---|
| when | during a real sampling run | offline, no sampling |
| source of numbers | the teacher's own forward pass | recomputed on unloaded CPU weights |
| bitwise vs native | **yes, by construction** | no |
| needs a sampling run | yes | no |
| use it for | anything you will compare or ship | exploring a schedule cheaply |

Why the offline path cannot be exact, even with perfect code: at sampling time
ComfyUI runs the AdaLN GEMM on GPU with weights it dequantised through its own
CUDA kernels (`comfy_kitchen`), while an offline pass sees the checkpoint's
CPU-side form and eager dequantisation. Measured on the real 34 GB teacher:

```
blocks.0.adaln_proj   equal=False   max_abs=0.015625   mean_abs=5.2e-4
```

`0.015625` is exactly one bf16 ulp, and eight diffusion steps amplify it into a
latent offset of `mean_abs=0.145` (`PSNR≈31 dB`). No amount of offline care can
predict a kernel that is chosen at run time — so Record replaces prediction with
observation.

Record attaches two **read-only** forward hooks:

- `dm.time_embedder` — captures `t_vals`, the float32 tensor of distinct
  timesteps that `model.py:743` builds from `unique_t`. Row *i* of every AdaLN
  projection therefore belongs to `unique_t[i]`; nothing has to be
  reconstructed from sigmas, layouts or shifts.
- every `adaln_proj.linear` — captures the raw
  `[M, expand*hidden*modalities]` output, which is exactly what a table stores
  (the module's own view/chunk is replayed later, not baked in).

Replay is then bitwise by construction: gather rows, `reshape` exactly like
`AdalnProj.forward`, `chunk`, and cast back to **the dtype the rows were
recorded in**. That last part matters — `FinalLayer.mod` switches its
multiply-add precision on `scale.dtype`, so injecting fp32 into a model whose
teacher ran bf16 would change the arithmetic even with identical numbers.

## Install

```bash
cd ComfyUI/custom_nodes
git clone --depth 1 -b main https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git
```

If `github.com:443` times out (common on Chinese cloud platforms):

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

Restart ComfyUI. Requirements: ComfyUI 0.42.5+ (the `MiniMaxH3Model` /
`patches_replace` implementation), PyTorch, `safetensors`. No extra pip
packages. Verified against ComfyUI 0.44.0 / PyTorch 2.9.1 / RTX 4080 SUPER.

## Nodes

| Node | In | Out | Behaviour |
|---|---|---|---|
| `H3 AdaLN Record` | full MODEL (+ LoRA) | MODEL, `H3_ADALN_TABLE` | **Exact.** Read-only hooks; the table fills up *during* sampling. |
| `H3 Exact AdaLN Bake` | full MODEL (+ LoRA), SIGMAS or steps/scheduler | `H3_ADALN_TABLE` | Offline approximation. Cheap, no sampling. |
| `H3 Exact AdaLN Save Table` | `H3_ADALN_TABLE` | file path | Like **Save Image**: auto-numbered `.safetensors` under `output/h3_adaln/`, or into `models/h3_exact_adaln/`. Mirrors into `input/`. |
| `H3 Exact AdaLN Load Table` | `table_name` (free text) / `table_path` | `H3_ADALN_TABLE` | Resolves a filename, substring, relative path or absolute path. Scans `input/`, `output/`, `models/h3_exact_adaln/`. |
| `H3 AdaLN Merge` | two `H3_ADALN_TABLE` | `H3_ADALN_TABLE` | **bX-49 hybrids.** Blocks `0..split-1` from A, `split..49` from B, `final_layer` from A. Refuses mismatched t-sets / projections / dtype / LoRA. |
| `H3 Exact AdaLN Apply` | pruned MODEL + `H3_ADALN_TABLE` | MODEL | Replaces all 51 AdaLN projections at runtime. `strict` / `fallback` / `off`. |
| `H3 Latent Compare` | two LATENT | report | `equal`, `max_abs`, `PSNR`; PASS = bitwise identical. |
| `H3 Latent Save` | LATENT | file | Saves H3's `NestedTensor` latents (stock `SaveLatent` cannot). |

A saved table is **self-contained**: its safetensors header carries the
teacher, provenance (record vs bake), LoRA fingerprint, sigma grid, shifts,
out dtype and ComfyUI version. Copy the file anywhere to share it; the
receiving side just Loads it. `strict` mode refuses a table recorded with a
different LoRA stack.

## Quick start

1. **Record once.** Load the *full* teacher + your LoRA, put `H3 AdaLN Record`
   between the shift node and the guider, and run the exact workflow you will
   actually use (same steps, scheduler, shifts, resolution, frames).
2. **Save.** Wire the recorder's `table` into `H3 Exact AdaLN Save Table` and
   its `after` input to the sampler's `LATENT`, so the save happens after
   sampling rather than before.
3. **Reuse forever.** On the pruned base, load the table and put
   `H3 Exact AdaLN Apply` between the shift node and the guider.

### The `after` input (important)

ComfyUI runs a node as soon as its inputs exist. A recorded table is **empty
until chain A has sampled**, so every consumer must be ordered behind it.
`after` is an ignored wildcard input whose only job is to create that edge:

```
sampler_A.output ──► Apply.after      (chain B may then start)
sampler_A.output ──► Save.after       (the table may then be written)
```

Without it, `Apply`/`Save` run before any row exists and fail with an explicit
"nothing was recorded" error. If your frontend refuses to connect a `LATENT`
into a `*` socket, split the run in two prompts (record + save, then load +
apply) — AdaLN depends only on `t`, so a table recorded in an earlier run is
bitwise identical as long as the schedule, shifts and layout match.

## Workflows

- `workflows/01_bake_exact_adaln.json` — Bake → Save. Queue once on its own.
  The offline path; good for a quick look, not for bitwise claims.
- `workflows/02_verify_full_vs_pruned.json` — Load → Apply on chain B, chain A
  runs the full model natively; both chains share LoRA, sigmas, seed, sampler,
  conditioning and shift.
- `workflows/03_verify_save_latent.json` — same, plus two `H3LatentSave` taps
  that write both latents to disk for an independent cross-check.
- `workflows/04_verify_strict.json` — single prompt, two chains, offline table.
  Kept as the A/B comparison of the bake path.
- `workflows/05_record_strict.json` — **the exact one.** Single prompt, two
  chains, `after`-ordered: A records while it samples, B applies the recorded
  rows, Compare must print `equal=True / max_abs=0 / VERDICT: PASS`. Set
  `Apply.mode=off` and rerun for the negative control (must FAIL).
- `workflows/08_record_fl2va.json` — records the **FL2VA full** teacher, the
  base half of a bX-49 hybrid. Run this once before 06.
- `workflows/06_b25-49_hybrid.json` — **bX-49 hybrid, one run.** Records the
  Ref2VA full teacher, merges it with the FL2VA table (`split=25`), and saves
  both the donor table and the hybrid table.
- `workflows/07_merge_tables.json` — CPU-only merge of two saved tables, for
  re-combining after either side was re-recorded.

Defaults match the DMAD 8-step LoRA: 8 steps, `euler`, `simple`, strength 1.0,
no CFG, video shift 12 / **audio shift 2**. Change the LoRA filename to your
own before running. The short verification clip (832x480, 39 frames) can be
changed to the full 1344x768 / 124 frames as long as both chains stay in sync
**and the table is recorded on the same shape** (the layout fixes which
timesteps exist).

## Sharing and downloading tables

ComfyUI's asset/output panel only understands media types, so a raw
`.safetensors` produced by a custom node is normally not downloadable from the
UI. Save Table works around this in three ways:

1. **Direct download link** — the Save report prints a relative URL like
   `/view?filename=h3_exact_dmad8_v1_00001.safetensors&subfolder=h3_adaln&type=output`.
   Paste `<your ComfyUI host>` + that URL into a browser; `/view` serves any
   file under `output/` as `application/octet-stream`.
2. **Output panel entry** — Save also reports the file through ComfyUI's
   generic `ui.images` channel, so a download card appears without opening the
   folder (the thumbnail may be blank for a non-image file).
3. **input/ mirror** — the same file is copied into `ComfyUI/input/`, which
   most platform file browsers and the Load node can reach directly.

Load Table deliberately uses **free text**, not a dropdown + `image_upload`
widget: a COMBO list is built once at ComfyUI startup (so a table saved later
never appears), and the image upload control only accepts images. If the name
cannot be resolved, the error lists every table file it can see.

## Implementation notes

- **Injection is sampler-agnostic.** Apply identifies timesteps from `t_emb`
  itself, so it works whether the sampler is `SamplerCustomAdvanced` or a
  custom node calling `comfy.sample.sample()` directly (SequenceForge,
  HyperFlow adapters, LCM, ...), which never reaches `patches_replace`.
  - curve base: keys are the `adaln_t_table` lerp, rebuilt on the tensor's own
    device and in its own dtype (CPU or another dtype makes the distance
    non-zero for reasons unrelated to the table);
  - full base: keys are `time_embedder(t).to(dtype)`.
- **Matching is exact, not fuzzy.** Table keys are the *float32 image* of `t`
  (`struct.pack`), because the DiT feeds `torch.tensor(unique_t, dtype=float32)`
  into the embedder. An exact hit is therefore the normal case; the tolerance
  is `1e-6 × |keys|`, i.e. far below one bf16 ulp. Anything above that fails
  loudly instead of snapping to a neighbouring row — a silent 1-row shift was
  the failure mode that made a wrong table look like a numerical difference.
- A `WrappersMP.DIFFUSION_MODEL` wrapper wraps `_forward` (as a *fallback*)
  to rebuild `unique_t` from the real `timestep` + packed layout. Unlike
  `patches_replace`, it fires for custom samplers too. Its signature is
  `(executor, *args, **kwargs)` on purpose: ComfyUI has changed `_forward`'s
  kwarg set between releases, and a wrapper that raises would break sampling
  for no benefit.
- `transformer_options["sigmas"]` is set by `comfy/samplers.py:325` to the
  **timestep** (`sigma * 1000`), not to sigma and not to the schedule (that is
  `"sample_sigmas"`). `sigma_plan.sigma_from_options` divides by 1000, and all
  t arithmetic is done on float32 tensors to mirror `model.py:674-676`.
- Bake never parses safetensors by hand and never mutates the model: it asks
  ComfyUI's own `convert_weight` / `calculate_weight` hooks for the effective
  (dequantised + LoRA-patched) weight, does the matmul itself, and lets the
  transient weight go. The previous implementation called
  `patch_weight_to_device(inplace_update=True)` — which asserts on quantised
  ops — on a `clone()` that shares the same `nn.Module`, then `unpatch`ed it,
  leaving the caller's ModelPatcher holding restored weights from a
  half-finished patch.
- Tables are stored fp32; the recorded/produced dtype travels in `out_dtype`
  and is replayed exactly. bf16 *storage* of AdaLN values costs ~2.5e-2
  relative error, so fp32 storage is not negotiable even when the values are
  bf16-precision.
- `round(t, 7)` is gone. Rounding the baked t by up to 5e-8 shifts `t_emb`
  visibly after `exp(-log(10000)*k/128)`, i.e. it re-introduced the very
  mismatch the table is supposed to remove. Dedup now happens on the exact
  float32 value.
- On ComfyUI 0.44 H3 latents are `comfy.nested_tensor.NestedTensor` (video +
  audio tensors), not a `torch.Tensor`. Both `H3LatentCompare` and
  `H3LatentSave` expand it via `.unbind()` into `samples__part0`,
  `samples__part1`, ... ComfyUI's stock `SaveLatent` calls `.contiguous()` on
  it and crashes.

## t-set coverage

A table is valid for exactly the timesteps it contains. Strict mode therefore
reports a **t-set mismatch** (listing the missing values) rather than a generic
"different schedule" hint:

```
H3ExactAdaLN: cannot map (17, 8) rows to table rows.
reason: t_emb=(17, 8) keys=(17, 8) max_dist=4.1e-3 tol=1e-6 (curve keys)
missing from table (3): 0.9230769, 0.9411765, 0.9600000
This is a t-set mismatch, not a precision problem: the running schedule/shift
produces timesteps the table was never recorded for.
```

Re-record on the exact schedule you intend to use. `shift_v`/`shift_a` being
equal is neither necessary nor sufficient: what matters is that *every runtime
t is in the table*, and that the t→t_emb mapping is the same one that produced
it.

## Hybrid (spliced) teachers

`b25-49`-style hybrids are **not** "full blocks + pruned blocks" stitched
together: they take FL2VA as the base and swap the `adaln_proj.linear`
weight/bias of a trailing block range for Ref2VA's tensors. Because the base is
a pruned file, it has no `time_embedder`, so it cannot be recorded from
directly.

`H3 Exact AdaLN Bake` accepts a `splice` string, `FILE:FIRST-LAST`
(e.g. `/models/ref2va.safetensors:25-49`), which swaps those blocks' AdaLN
weights for the donor's before computing the table, and records the donor
fingerprint in the metadata. The donor must be a **plain float** checkpoint
(bf16/fp16/fp32); quantised donors are rejected with an explicit error, since
their storage format is not a weight.

Standing caveat: a table baked from an FL2VA teacher replaces all 51 blocks
with FL2VA modulation, which erases the reference-conditioning behaviour of the
hybrid's tail blocks. Splice is what keeps that behaviour; without it the table
is numerically fine but semantically wrong for a hybrid.

### The exact path for bX-49

A `b25-49` file is curve-shaped **everywhere** — 51 x `[96768, 8]`, one
`adaln_t_table [1025, 8]`, no `time_embedder` — and the trailing blocks' AdaLN
tensors come from a second teacher. Record cannot run on it, and its 8-wide
curve cannot carry either teacher's full modulation. Two ways out, and they are
**not** the same target.

#### A. Record the assembled hybrid (exact, one pass)

Point Record at a full bX-49 checkpoint — a full model whose body is the base
teacher's and whose blocks X..49 `adaln_proj` come from the donor, i.e. exactly
the same recipe as the pruned file. One recording run, no merge, and the table
reproduces that model **bitwise**. A ready-made `b25-49` int8 teacher:
[`binglingzhimeng/MiniMax-H3-b25-49-full`](https://huggingface.co/binglingzhimeng/MiniMax-H3-b25-49-full).

Measured end to end (both chains, same LoRA, 8 steps, `simple`, shift 12/2):

```
A: full b25-49 hybrid (native)
B: pruned b25-49 + table recorded on A
samples__part0: equal=True max_abs=0.000e+00 psnr=inf
samples__part1: equal=True max_abs=0.000e+00 psnr=inf
VERDICT: PASS - bitwise identical
```

#### B. Two teachers + `time_embedder_from` (same target, no merged file)

If you do not want a second 34 GB checkpoint, record the donor teacher with the
base teacher's embedding swapped in first:

1. record the **base** teacher (FL2VA full) → A;
2. record the **tail donor** (Ref2VA full) with
   `time_embedder_from = <base teacher file>` → B';
3. `H3 AdaLN Merge`, `split=25`, `final_from=a`.

Why the extra input is mandatory: AdaLN output is `adaln_proj(t_emb)`, and the
tail blocks of a hybrid see the **base** teacher's `t_emb`. The two official
teachers do not share a `time_embedder` (`proj_out.weight` differs by up to
`0.104`), so recording the donor as-is bakes the donor's embedding into the
table. Measured against the assembled hybrid:

| tail source | max abs delta on blocks 25-49 |
|---|---|
| donor recorded as-is (old recipe) | 0.09 – 0.19 |
| donor + `time_embedder_from` (this recipe) | **0** (25/25 projections bitwise) |

`time_embedder_from` swaps the whole module, not the weights: a weight patch
would run through `patch_weight_to_device`, be cast to `lora_compute_dtype`
(bf16) and stay there, while the real model keeps the embedding in fp32 —
that alone moves the modulation by ~1e-1.

If the merged table and the hybrid disagree, the numbers are large enough to
see: applying the old merged table to the pruned b25-49 and comparing against
the full hybrid gives `psnr=22.20 dB` (video) / `31.10 dB` (audio), while the
table from A gives `inf`.

#### What the shipped b25-49 actually is

Verified tensor by tensor on `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8`:
blocks 0–24 `adaln_proj` are byte-identical to `fl2va_pruned`, blocks 25–49 to
`ref2va_pruned`, and `adaln_t_table` to `fl2va_pruned`. So the file is
"base body + donor tail" in the literal sense — which is why recipe A is the
faithful one.

Note that its own curve output happens to sit *closer* to the old merged table
than to the assembled hybrid (mean abs 0.003 vs 0.008 on blocks 25/30/35/40/45/49,
18 of 18 sampled rows): reusing the base teacher's coordinate table for the
donor's 8-wide tail weights is itself an approximation. Neither table equals the
pruned file's raw output — the plugin exists precisely because that output is
not exact.

Both teachers must be **full** checkpoints; a pruned donor has already lost the
information the table needs.

## Which LoRAs need a fresh table

A table stores `adaln_proj(time_embedder(t))`. Only two module groups can change
that: `time_embedder` and `adaln_proj`. Every other LoRA — attention, MLP,
norms — leaves the modulation untouched, so **one table serves all of them**.

Scanned 104 unique MiniMax-H3 LoRAs on a working node:

| keys touching AdaLN | count | shape | consequence |
|---|---|---|---|
| none | 87 (84 %) | — | table reusable as-is |
| `adaln_proj` only | 7 (7 %) | `lora_A` in-width **8** (curve form) | applies to the pruned base, really changes modulation -> re-record |
| `adaln_proj` only | 10 (10 %) | in-width **2688** (full form) | dropped by ComfyUI on a pruned base -> record with it on the teacher to keep its AdaLN effect |
| `time_embedder` | 0 | — | — |

In practice the split is by *kind*, not by author: every acceleration /
distillation LoRA in the scan (turbo, 4-step, 8-step Acc, dmd) carries AdaLN
keys, because it was trained against the whole denoising trajectory. Styling,
motion, effect and slider LoRAs carried none, and the DMAD 8-step LoRA
(`dmad_minimax_h3_8step_full_critic_comfyui`) is among the clean ones.

Measured, not inferred — table recorded with LoRA *X*, applied to the pruned
base running a different LoRA *Y*:

```
A: full hybrid + turbo 8-step LoRA (native)
B: pruned + turbo 8-step LoRA + table recorded with the DMAD 8-step LoRA
samples__part0: equal=True max_abs=0.000e+00 psnr=inf
samples__part1: equal=True max_abs=0.000e+00 psnr=inf
```

Because of that, `H3 Exact AdaLN Apply` no longer fails `strict` on a non-AdaLN
LoRA mismatch — it reports it as harmless and keeps going. `strict` still
refuses on an **AdaLN/timestep** fingerprint mismatch, on an explicit
`expected_teacher` mismatch, and on a t-set mismatch. Before this change the
cross-LoRA case above could only be run with `mode=fallback`.

## Known limits

- A table is a function of (base weights, LoRA set **and strengths**, sigma
  grid, shifts, and the packed layout that decides which timesteps exist).
  Changing any of them requires a re-record. Strength cannot be interpolated at
  runtime when a LoRA touches `time_embedder` (two SiLUs make it non-linear).
- `out_dtype` is fixed when the table is produced and must match the teacher's
  inference dtype. Recording a bf16 teacher and applying into a model forced to
  fp32 gives different arithmetic than the teacher used.
- Per-row `denoise_mask` timesteps are not supported by Record (it labels rows
  by the `unique_t` the forward actually used, and per-row masks multiply that
  set unpredictably); the offline Bake path covers them via `extra_t`.
- HyperFlow-style `(t, r)` endpoint conditioning is not reproduced.
- Python hooks cannot reach DiTs executed by a C++ engine or fused kernels that
  bypass `module.forward`.
- Record needs one sampling run with the full teacher, so peak VRAM is
  whatever the teacher needs. The table itself is ~315 MiB for 17 t rows.

## Testing

```bash
# structure: every workflow JSON, link/slot consistency, widget counts
python tests/test_workflows.py

# numbers: real MiniMaxH3Model at tiny scale, full teacher vs curve student
python tests/test_exact.py

# merge: block split, t alignment, metadata round-trip, rejection paths (CPU)
python tests/test_merge.py
```

`test_exact.py` builds a full teacher and a curve student that shares every
non-AdaLN weight byte, records 4 steps, applies, and asserts that all steps are
`torch.equal` — then asserts the untouched student does **not** match (the
control), and that an unrecorded sigma is rejected with a t-set message.

Set `COMFYUI_DIR` if your ComfyUI lives somewhere other than the default path.

## Repo layout

```
record_engine.py   observe AdaLN during sampling (the exact path)
bake_engine.py     offline recomputation + splice donors
apply_engine.py    runtime injection: t_emb matching + sigma fallback
sigma_plan.py      float32-exact t arithmetic, mirrors model.py:671-751
table_io.py        safetensors container, live (recording) tables
nodes.py           the seven nodes
workflows/         01 bake, 02/03 load+apply A/B, 04 offline strict,
                   05 record strict, 06 bX-49 hybrid, 07 merge only,
                   08 record FL2VA base
tests/             test_exact.py (numbers), test_merge.py (hybrid merge),
                   test_workflows.py (graph structure)
tools/             workflow generators (incl. build_hybrid.py) + fetch helpers
HANDOFF.md         repair plan, root-cause evidence and per-phase status
```

The workflow JSONs are **generated**, never hand-edited:

```bash
python tools/build_workflows.py   # 01, 02
python tools/add_savelatent.py    # 03 (from 02)
python tools/build_strict.py      # 04
python tools/build_record.py      # 05
python tools/build_hybrid.py      # 06, 07
```

They read node slot definitions from ComfyUI's own
`blueprints/Image to Video (MiniMax H3).json`; set `COMFYUI_DIR` or
`H3_BLUEPRINT` if it is not in the default place. Regenerating on a clean
checkout reproduces the committed files byte-for-byte, which is a cheap way to
tell whether a workflow was hand-edited.

## Status

- Record → Apply verified bitwise on a real `MiniMaxH3Model` (tiny config):
  4/4 steps `equal=True`, `max_dist=0.000e+00`, 12 exact t_emb matches, control
  fails as expected.
- End-to-end on ComfyUI 0.44.0 / RTX 4080 SUPER: the offline bake produced 51
  projections / 17 t rows / 314.5 MiB and the first injection was confirmed
  (`max_dist=0.000e+00`), yet the latents still differ — that measurement is
  what identified the offline path as the root cause and motivated Record.
- bX-49 hybrid support: `H3 AdaLN Merge` assembles the table from two full
  teachers (blocks 0..24 from FL2VA, 25..49 from Ref2VA). `tests/test_merge.py`
  covers the split, t alignment, metadata round-trip and every rejection path
  on CPU; the real 20 GiB b25-49 run still needs a GPU instance.
- End-to-end on the 34 GB teacher is done: `05_record_strict.json` reports
  `equal=True / max_abs=0 / VERDICT: PASS`, `Apply.mode=off` reports `FAIL`
  with no injection log, and strict still passes after the control run. Root
  cause, evidence and the verification log live in [`HANDOFF.md`](HANDOFF.md).

## Related work

`ComfyUI-HyperFlow-H3` (Create Curve Fit), `H3 AdaLN LoRA Fix`,
`ComfyUI-VDN-H3-Plus` and `indhic-ai/curve_prune.py` all project or fit LoRA
AdaLN deltas **back into the curve basis**, and their authors note the result
is an approximation. This pack instead stores the final modulation values
directly, at the cost of a larger, schedule-bound table.
