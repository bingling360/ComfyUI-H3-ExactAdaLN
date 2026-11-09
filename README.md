# ComfyUI-H3-ExactAdaLN

Make a **pruned** MiniMax-H3 checkpoint reproduce the **full** model's AdaLN
modulation exactly, so that

> pruned base + exact table + LoRA  ==  full teacher + the same LoRA  (bitwise)

在剪枝版 MiniMax-H3 上逐位复刻完整模型的 AdaLN 调制：生成结果与「完整模型 +
同一 LoRA」完全一致，速度维持剪枝版水平，表自动落盘、可直接拷贝分享。

## Why

> **New here?** [The node guide](docs/NODE_GUIDE.md) has the two-step usage, the
> naming/identity rules, worked recipes and the pitfall checklist from the
> 2026-11-08/09 verification runs. [The design note](docs/unified-node.md)
> explains why one node is enough.


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

There is exactly one way to produce a table: `mode=record` on the full teacher
during a real sampling run. Read-only hooks capture the tensors the teacher
itself computed, so replaying them is bitwise by construction.

Offline recomputation (the v1 "Bake" node) was removed. It runs on unloaded CPU
weights and cannot reproduce the kernels ComfyUI selects at sampling time:
measured drift is one bf16 ulp per projection, amplified to ~0.145 mean_abs on
the latent after 8 steps. There is no offline fallback.


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

Two nodes, plus a diagnostic. Everything the old Save / Load / Merge plumbing
did now happens inside the node that needs it.

| Node | In | Out | Behaviour |
|---|---|---|---|
| `H3 AdaLN` | `model`, `mode` (`apply` / `record` / `off`), `for_model`, `table` | MODEL, table, report | `record`: read-only hooks on the full teacher; `for_model=(auto)` finds the pruned base and stamps its fingerprint. `apply`: replaces all 51 AdaLN projections on the model that table declares. `off`: pass-through for an A/B control. |
| `H3 AdaLN Hybrid` | `model`, `table_a`, `table_b`, `split` | MODEL, table, report | **bX-49 only.** Blocks `0..split-1` from A, `split..49` from B, saves and injects. |
| `H3 Latent Compare` | two LATENT | report | `equal`, `max_abs`, `PSNR`; PASS = bitwise identical. |

Why there is nothing to type:

- **The name comes from the model.** A table's identity is its AdaLN weights
  (fingerprinted from block 0, the middle block, the last block and
  `final_layer`) + its **AdaLN** LoRA + its compute dtype. The file lands in
  `models/h3_exact_adaln/` as `h3_full_<fp>_<dtype>[_L<lora>].safetensors`.
  Rename it freely — the identity lives in the safetensors header and the node
  keeps finding the file.
- **The table learns which base it belongs to.** At record time the node finds
  the pruned checkpoint in your model folder by itself (a base ships
  `adaln_t_table` and no `time_embedder`; a teacher is the opposite) and writes
  its AdaLN fingerprint into the table header (`target_base`). At apply time it
  hashes the model in front of it and takes the table that declares that base —
  no picker, no sidecar file, and the association travels inside the
  `.safetensors`. Two model chains in one workflow each find their own table,
  and there is no "most recent" rule to go stale.
- **Re-recording merges.** Same schedule: nothing changes. Different schedule:
  the new timesteps are *added* to the same table. Only a genuine contradiction
  — the same `t` producing different numbers, which means the model, the AdaLN
  LoRA or the dtype changed — is refused, and then nothing is written.
- **One table per model, not per pass.** `modulation = adaln_proj(time_embedder(t))`
  contains no latent, resolution, seed or pass, so every sampling segment of the
  same model hits the same rows.

A table is **self-contained**: the safetensors header carries the identity,
teacher fingerprint, provenance, t rows, shift pair, out dtype
and ComfyUI version. Copy the file anywhere to share it.

The v1 single-purpose nodes (Record / Bake / Save / Load / Merge / Apply /
LatentSave) and the offline bake engine have been **removed**. Only three nodes
remain: `H3 AdaLN`, `H3 AdaLN Hybrid`, `H3 Latent Compare`. Old v1 workflows no
longer load; the two shipped API prompts below use the current nodes.

## Quick start

1. **Record.** Load the *full* teacher + your LoRA, put `H3 AdaLN` between the
   shift node and the guider, set `mode=record`, and run the workflow you will
   actually use. The table is written when the prompt finishes — no Save node,
   no `after` edge. Nothing else to fill in: with one pruned checkpoint in the
   folder, `for_model=(auto)` picks it up and stamps it.
2. **Apply.** Swap in that pruned base, set `mode=apply`, leave `table` at
   `(auto)`. Done: the table declares this base, so nothing is picked.
3. **Verify (optional).** `mode=off` on the same graph must produce a visibly
   different latent — if it does not, the table was never injected.

**Make the recording run cheap.** The table depends only on the timesteps, not
on resolution or frame count, so record at the smallest size your workflow
accepts: H3's latent node allows `width`/`height` down to `32` (step `32`) and
`length` down to `5` (step `17`, snapped to the `17k+5` grid). `256x128` or
`320x192` with `5` frames is a safe practical floor — the full teacher then
takes seconds per step instead of minutes. Keep the steps, scheduler, shifts
and reference/keyframe setup identical to the run you will actually use. See
[the node guide](docs/NODE_GUIDE.md) §4.6 for the exact limits and why.

### The `after` input (only for A/B in one prompt)

A recorded table is empty until the sampler has run, so anything reading it in
the *same* prompt has to be ordered behind it. `after` is an ignored wildcard
input whose only job is to create that edge:

```
sampler_A.output ──► H3 AdaLN(apply).after      (chain B may then start)
```

This is only needed for the `v2_record_apply_ab.json` A/B prompt. Daily use
records in one prompt and applies in the next — AdaLN depends only on `t`, so a
table recorded earlier is bitwise identical as long as the t values match.

## Workflows

Two **API-format prompts** live in `workflows/`. They are plain JSON you POST to
ComfyUI's `/prompt` endpoint (they are not UI workflow exports), and both were
run end to end on the cloud instance before shipping.

- `workflows/v2_record_apply_ab.json` — single prompt, two chains. Chain A
  records on the full 34 GB teacher while chain B applies that same table to
  the pruned 20 GB base; `H3 Latent Compare` must report
  `VERDICT: PASS - bitwise identical`. Set chain B's `mode=off` and rerun for
  the negative control (must FAIL).
- `workflows/v2_apply_auto.json` — cross-prompt lookup. Only the pruned base,
  `table=(auto)`, no `table_in`: the node must find the table by its
  `target_base` header with no picker. `PreviewAny` shows the apply report.

Both default to the smallest practical recording size (256x128, 5 frames),
8 steps, `euler/simple`, the DMAD 8-step LoRA and shift 12/2. Change the LoRA
and checkpoint names to your own. The two chains in the A/B prompt must stay
comparable; resolution and frame count do not affect the table.

### What a table depends on

| | value stored for a given t | which t values exist |
|---|---|---|
| teacher weights, AdaLN LoRA, compute dtype | **yes** | — |
| steps, scheduler, custom sigmas | no | **yes** (t = 1 − sigma) |
| video / audio shift | no | **yes** |
| keyframes / references / reference audio | no | no — those two rows (`0.999`, `1.0`) are pre-computed at record time |
| which image, how many, first vs last frame | no | no |
| resolution, frame count, seed, sampler | no | no |

So: change the resolution, swap a reference image, add a first/last frame, feed
three references instead of one — none of it needs a new recording. The two
“pinned” rows a reference workflow asks for (`t=0.999`, and `1.0` for reference
audio) are pre-computed during recording (`fill_reference_row`, on by default):
the node appends that timestep to the step's embedding batch, runs the model's
own projections, and throws the extra row away. Same batch as a real reference
run, so the row is exact, and the sampling output is untouched.

Change the steps, scheduler or shift and the existing rows stay correct, but the
new schedule may need timesteps the table does not have — one more recording run
adds them (`merged`, see below), it does not replace the table.

## Sharing tables

Tables live in `ComfyUI/models/h3_exact_adaln/` (registered as a model folder,
so the platform's file browser can reach it). A table is a normal
`.safetensors` file that carries its own provenance: `target_base` (the model it
was recorded for), `identity` (teacher + AdaLN LoRA + dtype) and the t-row set.
Copy it to another machine, drop it in the same folder, and apply finds it with
no configuration at all. Renaming is safe — nothing is keyed on the filename,
and the node keeps merging into a renamed file.

There is no state outside the table files. The only thing the plugin ever needs
to know about a model is its AdaLN fingerprint, and it reads that itself — at
record time from the checkpoint on disk, at apply time from the loaded model.
`for_model` therefore defaults to `(auto)` and only has to be touched when the
folder holds more than one pruned checkpoint, where "which base did you mean"
is genuinely not derivable. **A table without `target_base` is never used
automatically**: it could belong to a different teacher, and guessing would be
a silent wrong answer. Re-record it with `for_model` set, or pick it explicitly
in `table` for that run. Only `models/h3_exact_adaln/` is searched; v1 saves
under `output/h3_adaln/` are ignored.

The picker is a plain dropdown of the table folder plus `(auto)`, which is
the default and the only value the node writes into a graph. Picking a file is a
one-shot override for that run — nothing is remembered, so a graph left on
`(auto)` keeps resolving by fingerprint.

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
- Record never mutates the model: it only reads what the teacher's forward
  produced, so quantised checkpoints and LoRA patches are handled by ComfyUI
  itself.
- Tables are stored fp32; the recorded/produced dtype travels in `out_dtype`
  and is replayed exactly. bf16 *storage* of AdaLN values costs ~2.5e-2
  relative error, so fp32 storage is not negotiable even when the values are
  bf16-precision.
- t values are stored as exact float32 images; dedup happens on the exact
  value, never on a rounded one (a 5e-8 shift is visible after the sinusoidal
  embedding).
- On ComfyUI 0.44 H3 latents are `comfy.nested_tensor.NestedTensor` (video +
  audio tensors), not a `torch.Tensor`. Both `H3LatentCompare` and
  `H3LatentSave` expand it via `.unbind()` into `samples__part0`,
  `samples__part1`, ... ComfyUI's stock `SaveLatent` calls `.contiguous()` on
  it and crashes.

## t-set coverage

A table is valid for exactly the timesteps it contains, and applying it to a
schedule with other timesteps fails loudly instead of snapping to a neighbour:

```
H3AdaLN: cannot map (17, 8) rows to table rows.
reason: t_emb=(17, 8) keys=(17, 8) max_dist=4.1e-3 tol=1e-6 (curve keys)
missing from table (3): 0.9230769, 0.9411765, 0.9600000
This is a t-set mismatch, not a precision problem: the running schedule/shift
produces timesteps the table was never recorded for.
```

That is no longer a one-way door: run `mode=record` once on the new schedule
and the missing rows are appended to the same table. `shift_v`/`shift_a` being
equal is neither necessary nor sufficient — what matters is that *every runtime
t is in the table*, and that the t→t_emb mapping is the same one that produced
it.

## Hybrid (spliced) teachers

`b25-49`-style hybrids are **not** "full blocks + pruned blocks" stitched
together: they take FL2VA as the base and swap the `adaln_proj.linear`
weight/bias of a trailing block range for Ref2VA's tensors. The shipped pruned
file is curve-shaped everywhere (51 x `[96768, 8]`, one
`adaln_t_table [1025, 8]`, no `time_embedder`), so it cannot be recorded from
directly — record on a **full** bX-49 teacher instead. `H3 AdaLN Hybrid` can
still splice two recorded tables, but see the caveat in section B below.

### The exact path for bX-49

A `b25-49` file is curve-shaped **everywhere** — 51 x `[96768, 8]`, one
`adaln_t_table [1025, 8]`, no `time_embedder` — and the trailing blocks' AdaLN
tensors come from a second teacher. Record cannot run on it, and its 8-wide
curve cannot carry either teacher's full modulation. Two ways out, and they are
**not** the same target.

#### A. Record the assembled hybrid (exact, one pass)

Point `H3 AdaLN` (`mode=record`) at a full bX-49 checkpoint — a full model
whose body is the base teacher's and whose blocks X..49 `adaln_proj` come from
the donor, i.e. exactly the same recipe as the pruned file. One recording run,
no merge, and the table reproduces that model **bitwise**. A ready-made `b25-49`
int8 teacher:
[`binglingzhimeng/MiniMax-H3-b25-49-full`](https://huggingface.co/binglingzhimeng/MiniMax-H3-b25-49-full).

Measured end to end (both chains, same LoRA, 8 steps, `simple`, shift 12/2):

```
A: full b25-49 hybrid (native)
B: pruned b25-49 + table recorded on A
samples__part0: equal=True max_abs=0.000e+00 psnr=inf
samples__part1: equal=True max_abs=0.000e+00 psnr=inf
VERDICT: PASS - bitwise identical
```

#### B. If you have no full hybrid teacher

`H3 AdaLN Hybrid` can still splice two recorded tables, but the two official
teachers do not share a `time_embedder` (`proj_out.weight` differs by up to
`0.104`). Recording the tail donor as-is bakes the donor's embedding into the
table: measured `0.09–0.19` off on blocks 25–49. The v1
`time_embedder_from` workaround lived on the removed `H3 AdaLN Record` node
and is gone. **Record the full bX-49 teacher directly** (published as
`binglingzhimeng/MiniMax-H3-b25-49-full`); that is the only exact route.

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

- A table is a function of (AdaLN weights, **AdaLN** LoRA set and strengths,
  compute dtype) and of the timesteps it contains. Every other LoRA is
  irrelevant, and a new schedule only costs one more recording run — its rows
  are appended, not substituted. Strength cannot be interpolated at runtime
  when a LoRA touches `time_embedder` (two SiLUs make it non-linear).
- `out_dtype` is fixed when the table is produced and must match the teacher's
  inference dtype, so it is part of the identity: a bf16 recording and an fp32
  recording of the same teacher are two tables, not one. Apply compares it
  against the target model's compute dtype and warns
  `replay is NOT bitwise` instead of leaving you to notice it in the pixels.
- Per-row `denoise_mask` timesteps are not supported (Record labels rows by
  the `unique_t` the forward actually used, and per-row masks multiply that set
  unpredictably).
- HyperFlow-style `(t, r)` endpoint conditioning is not reproduced.
- Python hooks cannot reach DiTs executed by a C++ engine or fused kernels that
  bypass `module.forward`.
- Record needs one sampling run with the full teacher, so peak VRAM is
  whatever the teacher needs. The table itself is ~315 MiB for 17 t rows, and it
  grows with every extra schedule you record.

## Testing

```bash
# the node surface: naming, identity, row merging, record -> save -> inject (CPU)
python tests/test_simple.py

# structure: every workflow JSON, link/slot consistency, widget counts
python tests/test_workflows.py

# numbers: real MiniMaxH3Model at tiny scale, full teacher vs curve student
python tests/test_exact.py

# merge: block split, t alignment, metadata round-trip, rejection paths (CPU)
python tests/test_merge.py

# inside a real ComfyUI: real ModelPatcher, real ON_CLEANUP, real table folder
COMFYUI_DIR=<checkout> python tests/test_runtime.py
```

`test_exact.py` builds a full teacher and a curve student that shares every
non-AdaLN weight byte, records 4 steps, applies, and asserts that all steps are
`torch.equal` — then asserts the untouched student does **not** match (the
control), and that an unrecorded sigma is rejected with a t-set message.

Set `COMFYUI_DIR` if your ComfyUI lives somewhere other than the default path.

## Repo layout

```
h3_unified.py      H3 AdaLN (record/apply/off) + H3 AdaLN Hybrid
nodes.py           shared helpers + H3 Latent Compare
record_engine.py   observe AdaLN during sampling (the exact path)
apply_engine.py    runtime injection: t_emb matching + sigma fallback
sigma_plan.py      float32-exact t arithmetic, mirrors model.py:671-751
table_io.py        safetensors container, live (recording) tables, row merge
workflows/         two API-format prompts (A/B verification, auto lookup)
tests/             test_simple.py (node surface + lookup), test_exact.py
                   (numbers), test_merge.py (hybrid merge), test_workflows.py
                   (prompt structure), test_runtime.py (real ComfyUI process)
HANDOFF.md         root-cause evidence, cloud verification log, v1 history
docs/unified-node.md  why one node is enough: invariants, identity, row merging
docs/NODE_GUIDE.md    node tutorial + lessons learned + pitfall checklist
```

The two files in `workflows/` are API prompts, not UI graph exports. They were
hand-built from the ComfyUI node signatures and validated on the cloud
instance; `tests/test_workflows.py` checks their structure offline.

## Status

- Record → Apply verified bitwise on a real `MiniMaxH3Model` (tiny config):
  4/4 steps `equal=True`, `max_dist=0.000e+00`, 12 exact t_emb matches, control
  fails as expected.
- Why the offline path was removed: on ComfyUI 0.44.0 / RTX 4080 SUPER it
  produced 51 projections / 17 t rows / 314.5 MiB and the first injection was
  confirmed (`max_dist=0.000e+00`), yet the latents still differed. That
  measurement identified offline recomputation as the root cause.
- bX-49 hybrid support: `H3 AdaLN Hybrid` assembles the table from two full
  teachers (blocks 0..24 from FL2VA, 25..49 from Ref2VA). `tests/test_merge.py`
  covers the split, t alignment, metadata round-trip and every rejection path
  on CPU; the real 20 GiB b25-49 run still needs a GPU instance.
- v1 end-to-end on the 34 GB teacher (historical): the old strict A/B graph reports
  `equal=True / max_abs=0 / VERDICT: PASS`, `Apply.mode=off` reports `FAIL`
  with no injection log, and strict still passes after the control run. Root
  cause, evidence and the verification log live in [`HANDOFF.md`](HANDOFF.md).
- The v2 node surface is verified on CPU (`tests/test_simple.py`) and inside a
  real ComfyUI 0.44.2 process (`tests/test_runtime.py`: real `ModelPatcher`,
  real `ON_CLEANUP`, real table folder, plus the invariant that a fingerprint
  computed from a checkpoint file is byte-identical to the one computed from the
  loaded module). `tests/test_exact.py` passes 21/21 against the real
  `MiniMaxH3Model` at tiny scale, including record → apply bitwise and the
  `mode=off` control.
- **v2 end-to-end on the cloud instance (2026-11-09, ComfyUI 0.44.0 /
  RTX 4080 SUPER 32 GB): PASS.** A two-chain A/B prompt used the new node on
  the real 34 GB `minimax_h3_ref2va_int8_convrot` teacher and the 20 GB
  `minimax_h3_ref2va_pruned_int8_convrot` base at **256×128 / 5 frames**, 8
  steps, `euler/simple`, DMAD 8-step LoRA, shift 12/2:
  `VERDICT: PASS - bitwise identical` on both the video and audio latent
  (`max_abs=0`), `exact matches=408` (51 × 8), `match failures=0`,
  `native steps=0`. The table landed as `h3_full_03692f_bf16.safetensors`
  (17 t rows, 314 MiB) with `target_base` stamped and the pinned `0.999` /
  `1.0` rows present. A **second, separate prompt** with only the pruned base
  and `table=(auto)` (no `table_in`) resolved `declared for this model` and
  produced the same 408 exact matches — i.e. cross-prompt lookup works with no
  picker. That run also confirmed the smallest recommended recording size is
  usable. See [`HANDOFF.md`](HANDOFF.md) §0.1 for the full log.
- The local box only has a *pruned* b25-49 (20 GB) with 6 GB VRAM and no full
  teacher; it is used for the CPU/structural suites and for reading real
  checkpoint headers (`adaln_t_table [1025,8]`, no `time_embedder`,
  51 × `[96768,8]` fp16 projections).

## Related work

`ComfyUI-HyperFlow-H3` (Create Curve Fit), `H3 AdaLN LoRA Fix`,
`ComfyUI-VDN-H3-Plus` and `indhic-ai/curve_prune.py` all project or fit LoRA
AdaLN deltas **back into the curve basis**, and their authors note the result
is an approximation. This pack instead stores the final modulation values
directly, at the cost of a larger, schedule-bound table.
