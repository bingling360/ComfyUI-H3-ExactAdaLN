"""Numerical regression for the exact-AdaLN pipeline.

Everything here runs against the REAL `comfy.ldm.minimax.model.MiniMaxH3Model`
with tiny dimensions, not against mocks. Three models are built:

  * teacher -- a full checkpoint (has `time_embedder`, AdaLN in bf16)
  * student -- a curve/pruned checkpoint (has `adaln_t_table`, AdaLN in fp32)
    with byte-identical non-AdaLN weights and a random curve table, so its
    native AdaLN is garbage and any equality must come from the injected table
  * the same student with Apply(mode=off), used as the negative control

Checks
------
1. record labels rows by the float32 timestep (row count == distinct t).
2. record -> apply reproduces the teacher's forward bitwise, step by step.
3. Apply(mode=off) does NOT reproduce it (the control must not pass).
4. Table round-trip through safetensors keeps every byte.
5. sigma_plan builds the same t values the DiT builds.
6. Strict mode rejects a schedule the table was not recorded for.
7. Node schemas: all nodes import, inputs/outputs are consistent, and the
   `after` ordering edges exist where documented.

Run:
    python tests/test_exact.py
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
COMFY = os.environ.get("COMFYUI_DIR", "D:/ComfyUI_windows_portable/ComfyUI")

sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402

import torch.nn.functional as F  # noqa: E402

import record_engine  # noqa: E402
import sigma_plan  # noqa: E402
import table_io  # noqa: E402
import apply_engine  # noqa: E402


HIDDEN = 32
T_EMBED = 16
CURVE_ROWS = 33
LATENT_T, LATENT_H, LATENT_W, AUDIO_T = 2, 4, 4, 4
TEXT_LEN = 5

VIDEO = None
AUDIO = None
CONTEXT = None


def _common_kwargs(dtype):
    return dict(
        hidden_size=HIDDEN, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64,
        latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=16,
        timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=T_EMBED,
        rope_inv_freq_len=2, dtype=dtype, device="cpu",
        operations=comfy.ops.disable_weight_init,
    )


def _init_params(dm, seed):
    """`disable_weight_init` leaves every weight as torch.empty -- fill it.

    Uninitialised memory happens to read back as zeros on this allocator, which
    would make the AdaLN projections a bias-only passthrough and quietly weaken
    the negative control. Deterministic random weights keep both models live.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for _name, param in dm.named_parameters():
            param.normal_(0.0, 0.08)
    return dm


def make_teacher():
    dm = MiniMaxH3Model(**_common_kwargs(torch.bfloat16)).eval()
    dm.requires_grad_(False)
    return _init_params(dm, 1234)


def make_student(teacher, seed=99):
    """Curve-form twin: same non-AdaLN weights, fp32 AdaLN, random curve."""
    torch.manual_seed(seed)
    dm = MiniMaxH3Model(**_common_kwargs(torch.bfloat16),
                        adaln_curve_grid=CURVE_ROWS).eval()
    dm.requires_grad_(False)
    target = teacher.state_dict()
    shared = {}
    for key, value in dm.state_dict().items():
        # AdaLN / time_embedder are exactly what differs between the two
        # checkpoints, so they must never be copied -- otherwise the control
        # would "pass" for the wrong reason.
        if "adaln_t_table" in key or "adaln_proj" in key or "time_embedder" in key:
            continue
        if key in target and target[key].shape == value.shape:
            shared[key] = target[key]
    dm.load_state_dict(shared, strict=False)
    # Everything that was not copied (the curve AdaLN projections) still holds
    # whatever `torch.empty` left there; fill it with a deterministic spread so
    # the model is finite and clearly different from the teacher.
    with torch.no_grad():
        for name, param in dm.named_parameters():
            if name not in shared:
                param.normal_(0.0, 0.08)
        dm.adaln_t_table.normal_(0.0, 1.0)
    missing = [k for k in dm.state_dict() if k not in shared]
    return dm, missing


def make_inputs():
    global VIDEO, AUDIO, CONTEXT
    torch.manual_seed(7)
    VIDEO = torch.randn(1, 24, LATENT_T, LATENT_H, LATENT_W)
    AUDIO = torch.randn(1, 32, 2, AUDIO_T)
    CONTEXT = torch.randn(1, TEXT_LEN, 16, dtype=torch.bfloat16)
    return VIDEO, AUDIO, CONTEXT


def run_forward(dm, sigma):
    ts = torch.tensor([float(sigma) * 1000.0])
    with torch.no_grad():
        return dm._forward([VIDEO.clone(), AUDIO.clone()], ts, CONTEXT.clone(), {})


def compare(a, b):
    out = []
    equal = True
    for i, (x, y) in enumerate(zip(a, b)):
        eq = torch.equal(x, y)
        d = (x.float() - y.float()).abs()
        out.append("part{}: equal={} max_abs={:.3e} mean_abs={:.3e}".format(
            i, eq, float(d.max()), float(d.mean())))
        equal = equal and eq
    return equal, out


def main():
    failures = []

    def check(name, ok, detail=""):
        print("{} {}".format("PASS" if ok else "FAIL", name))
        for line in detail.splitlines():
            if line:
                print("     " + line)
        if not ok:
            failures.append(name)

    make_inputs()
    teacher = make_teacher()
    ref_weight_snapshot = {
        "blocks.0": teacher.blocks[0].adaln_proj.linear.weight.detach().clone()}
    student, missing = make_student(teacher)

    sigmas = [1.0, 0.75, 0.5, 0.25]
    t_pairs = [sigma_plan.sigma_to_t_pair(s, 12.0, 3.0) for s in sigmas]

    # ---- 1. record ----------------------------------------------------
    recorder = record_engine.AdaLNRecorder(None, teacher, mode="strict")
    n_hooked = recorder.register()
    teacher_out = [run_forward(teacher, s) for s in sigmas]
    t_values, raw_by_key, extra = recorder.build()
    recorder.remove()

    # This layout is [text | audio | video]: every segment is a target stream,
    # so unique_t is exactly {1-sigma, 1-shift(sigma)} per step. The pinned
    # cond/ref timesteps (0.999 / 1.0) only appear when the layout carries
    # cond or reference segments, which needs keyframes/refs latents.
    expected_t = set()
    for t_v, t_a in t_pairs:
        expected_t.update({sigma_plan.f32_image(t_v), sigma_plan.f32_image(t_a)})
    check("record: 3 projections hooked (2 blocks + final)", n_hooked == 3,
          "hooked={}".format(n_hooked))
    check("record: t rows == distinct float32 t of the schedule",
          set(t_values) == expected_t,
          "table={}\nexpect={}".format(
              ["{:.9f}".format(t) for t in t_values],
              ["{:.9f}".format(t) for t in sorted(expected_t)]))
    check("record: projection set", sorted(raw_by_key) ==
          ["blocks.0", "blocks.1", "final_layer"], str(sorted(raw_by_key)))
    check("record: out dtype follows the teacher",
          extra["out_dtype"] == "torch.bfloat16", extra["out_dtype"])

    # ---- 2. table round-trip ------------------------------------------
    table = table_io.AdaLNTable(t_values, raw_by_key, dict(
        format=table_io.FORMAT, teacher="tiny-teacher", lora_fingerprint="none",
        out_dtype=extra["out_dtype"]))
    # Not TemporaryDirectory: load_file() keeps an mmap on the .safetensors for
    # as long as the tensors live, and Windows refuses to unlink an open file.
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "t.safetensors")
        table_io.write_table_file(path, table)
        back = table_io.read_table_file(path)
        same_t = back.t_values == table.t_values
        same_rows = all(torch.equal(back.raw_by_key[k], table.raw_by_key[k])
                        for k in table.raw_by_key)
    finally:
        back = None
        shutil.rmtree(tmp, ignore_errors=True)
    check("table: safetensors round-trip is byte-identical", same_t and same_rows,
          "t_equal={} rows_equal={} size={:.2f} MiB".format(
              same_t, same_rows, table.size_bytes() / (1 << 20)))

    # ---- 3. apply on the curve student --------------------------------
    runtime = apply_engine.ExactAdaLNRuntime(
        table.t_values, table.raw_by_key, table.meta, student, mode="strict")
    n_reg = runtime.register(student, use_sigma_wrapper=False)
    student_out = [run_forward(student, s) for s in sigmas]
    ok, detail = compare(teacher_out[-1], student_out[-1])
    check("apply: student matches teacher bitwise (last step)", ok, "\n".join(detail))
    ok_all = True
    step_detail = []
    for i, (a, b) in enumerate(zip(teacher_out, student_out)):
        eq, lines = compare(a, b)
        ok_all = ok_all and eq
        if not eq:
            step_detail.append("step {}".format(i))
            step_detail.extend(lines)
    check("apply: every step matches bitwise", ok_all, "\n".join(step_detail))
    check("apply: exact t_emb matches (not approximate)",
          runtime.state["match_hits"] > 0 and runtime.state["match_approx"] == 0
          and runtime.state["match_failures"] == 0,
          "hits={} approx={} failures={} max_dist={:.3e}".format(
              runtime.state["match_hits"], runtime.state["match_approx"],
              runtime.state["match_failures"], runtime.state["max_match_distance"]))
    check("apply: 3 modules hooked", n_reg == 3, str(n_reg))
    runtime.remove()

    # ---- 4. negative control ------------------------------------------
    control = [run_forward(student, s) for s in sigmas]
    eq, lines = compare(teacher_out[-1], control[-1])
    check("control: untouched curve student does NOT match", not eq,
          "\n".join(lines))

    # ---- 5. strict mode rejects an unknown schedule --------------------
    runtime2 = apply_engine.ExactAdaLNRuntime(
        table.t_values, table.raw_by_key, table.meta, student, mode="strict")
    runtime2.register(student, use_sigma_wrapper=False)
    rejected = False
    try:
        run_forward(student, 0.62)  # never recorded
    except RuntimeError as e:
        rejected = "cannot map" in str(e) or "t-set" in str(e)
        msg = str(e).splitlines()[:3]
    runtime2.remove()
    check("strict: unknown sigma raises with a t-set message", rejected,
          "" if rejected else "no error raised")

    # ---- 6. sigma_plan agrees with the model -------------------------
    ts = torch.tensor([float(0.5) * 1000.0])
    sigma = sigma_plan.sigma_from_timestep(ts)
    t_v, t_a = sigma_plan.runtime_t_pair(sigma, 12.0, 3.0)
    from comfy.ldm.minimax.model import time_shift_sigma as _ref
    ref_t_a = float(1.0 - _ref(torch.tensor(sigma).float().clamp(min=1e-6), 12.0, 3.0))
    check("sigma_plan: t_audio matches model.py float32 arithmetic",
          t_a == ref_t_a, "planner={:.17g} model={:.17g}".format(t_a, ref_t_a))

    # ---- 7. node schemas ---------------------------------------------
    import h3_unified as H
    import nodes as h3nodes

    spec = H.H3AdaLN.INPUT_TYPES()
    check("nodes: H3AdaLN schema (model/mode + optional inputs)",
          list(spec["required"]) == ["model", "mode"] and
          set(spec["optional"]) == {"for_model", "fill_reference_row", "table",
                                    "table_in", "after"}, str(spec))
    check("nodes: exactly three core nodes",
          set(h3nodes.NODE_CLASS_MAPPINGS) == {"H3LatentCompare"} and
          set(H.NODE_CLASS_MAPPINGS) == {"H3AdaLN", "H3AdaLNHybrid"}, "")
    check("nodes: every registered class has a display name",
          set(H.NODE_CLASS_MAPPINGS) == set(H.NODE_DISPLAY_NAME_MAPPINGS) and
          set(h3nodes.NODE_CLASS_MAPPINGS) ==
          set(h3nodes.NODE_DISPLAY_NAME_MAPPINGS), "")

    print()
    if failures:
        print("{} FAILED: {}".format(len(failures), ", ".join(failures)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
