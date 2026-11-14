"""Runtime smoke test inside a real ComfyUI process (opt-in).

`test_simple.py` mocks the model holder. This one does not: it imports the
ComfyUI that is actually installed, builds a `MiniMaxH3Model` at tiny scale,
wraps it in a real `ModelPatcher`, and drives the node through the same
`ON_CLEANUP` path a prompt uses. What it proves that the CPU tests cannot:

  * the plugin loads next to every other custom node and registers its folders;
  * `record` hooks the real modules, `ModelPatcher.cleanup()` writes the table;
  * `(latest)` resolves to it, `apply` arms the real patcher, cleanup unhooks;
  * `H3 AdaLN Hybrid` merges two real recordings and lands a third file.

Run (paths are environment-specific, so this is not part of the default suite):

    COMFYUI_DIR=<ComfyUI checkout> [COMFYUI_BASE=<--base-directory>] \\
        python tests/test_runtime.py

It uses `--cpu` and ~1 MB of weights; nothing is sampled and no checkpoint is
loaded. Tables written by the test are removed again at the end.
"""

import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)

COMFY = os.environ.get("COMFYUI_DIR", "").strip()
if not COMFY or not os.path.isdir(COMFY):
    print("SKIP: set COMFYUI_DIR to a ComfyUI checkout to run this test")
    sys.exit(0)

sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

# ComfyUI only parses argv after main.py enables it; folder_paths reads
# `--base-directory` at import time, so both have to happen before the import.
_argv = ["main.py", "--cpu"]
_base = os.environ.get("COMFYUI_BASE", "").strip()
if _base:
    _argv += ["--base-directory", _base]
sys.argv = _argv

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
import folder_paths  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.model_management as mm  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "h3exact", os.path.join(PLUGIN, "__init__.py"),
    submodule_search_locations=[PLUGIN])
plugin = importlib.util.module_from_spec(spec)
sys.modules["h3exact"] = plugin
spec.loader.exec_module(plugin)

H = plugin.NODE_CLASS_MAPPINGS["H3AdaLN"]
HY = plugin.NODE_CLASS_MAPPINGS["H3AdaLNHybrid"]

HIDDEN, T_EMBED = 32, 16
checks = []


def check(name, fn):
    try:
        fn()
        print("PASS " + name)
        checks.append(True)
    except Exception as e:  # noqa: BLE001
        print("FAIL {} -> {!r}".format(name, e))
        checks.append(False)


def tiny_model(dtype=torch.bfloat16, curve=None, seed=1):
    kw = dict(
        hidden_size=HIDDEN, num_layers=2, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=16, ffn_hidden_size=64,
        latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=16,
        timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=T_EMBED,
        rope_inv_freq_len=2, dtype=dtype, device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    if curve:
        kw["adaln_curve_grid"] = curve
    dm = MiniMaxH3Model(**kw).eval()
    dm.requires_grad_(False)
    torch.manual_seed(seed)
    with torch.no_grad():
        for _n, p in dm.named_parameters():
            p.normal_(0.0, 0.08)
    return dm


def patcher_for(dm):
    holder = types.SimpleNamespace(diffusion_model=dm, device=None)
    return ModelPatcher(holder, mm.get_torch_device(), mm.unet_offload_device())


def feed(dm, t_list):
    """One AdaLN pass: what the DiT does per denoising step.

    Calls `adaln_proj(...)`, not `adaln_proj.linear(...)`: that is the real
    call site (it applies silu and returns chunks), and it is the one the apply
    hook replaces. Recording hooks the inner linear, so both fire here.
    """
    t = torch.tensor(t_list, dtype=torch.float32)
    dtype = dm.blocks[0].adaln_proj.linear.weight.dtype
    emb = dm.time_embedder(t).to(dtype)  # model.py:751 does exactly this cast
    for block in dm.blocks:
        block.adaln_proj(emb)
    dm.final_layer.adaln_proj(emb)


FOLDER = os.path.join(folder_paths.models_dir, "h3_exact_adaln")
CKPT_DIR = tempfile.mkdtemp(prefix="h3_ckpt_")
# Keep every table this test writes out of the user's models folder. Set before
# the plugin import so `table_folder()` picks it up.
TABLE_DIR = tempfile.mkdtemp(prefix="h3_tables_")
os.environ["H3_ADALN_TABLE_DIR"] = TABLE_DIR
_STUDENT = {}


def checkpoint_of(dm, name):
    """Write a tiny model out as a .safetensors the way a real checkpoint is."""
    from safetensors.torch import save_file
    path = os.path.join(CKPT_DIR, name)
    if not os.path.exists(path):
        save_file({k: v.detach().contiguous() for k, v in dm.state_dict().items()},
                  path)
    return path


def student():
    """The curve base every apply test resolves against (built + saved once)."""
    if "dm" not in _STUDENT:
        dm = tiny_model(curve=33, seed=7)
        _STUDENT["dm"] = dm
        _STUDENT["path"] = checkpoint_of(dm, "tiny-base.safetensors")
    return _STUDENT["dm"], _STUDENT["path"]


def test_registration():
    assert "H3AdaLN" in plugin.NODE_CLASS_MAPPINGS
    assert "H3AdaLNHybrid" in plugin.NODE_CLASS_MAPPINGS
    spec_i = H.INPUT_TYPES()
    assert list(spec_i["required"]) == ["model", "mode"], list(spec_i["required"])
    assert "report_level" in spec_i["optional"], spec_i["optional"]
    assert list(spec_i["optional"]) == ["report_level", "for_model",
                                        "fill_reference_row", "table",
                                        "table_in", "after"]
    assert os.path.isdir(FOLDER), "register_folder() did not create {}".format(FOLDER)
    paths = folder_paths.get_folder_paths("h3_exact_adaln")
    assert FOLDER in paths, "not registered as a model folder: {}".format(paths)


def test_base_fingerprint_from_file_matches_model():
    """The whole design rests on this: file bytes == loaded-module bytes."""
    teacher = tiny_model()
    assert (plugin.h3_unified.base_fingerprint_from_file(
        checkpoint_of(teacher, "tiny-teacher.safetensors"))
        == plugin.h3_unified.base_fingerprint(teacher))
    dm, path = student()
    assert (plugin.h3_unified.base_fingerprint_from_file(path)
            == plugin.h3_unified.base_fingerprint(dm))


def test_record_saves_on_cleanup():
    dm = tiny_model()
    patcher = patcher_for(dm)
    _base_dm, base_path = student()
    _, table, report = H().run(patcher, "record", for_model=base_path,
                                fill_reference_row=False)
    assert "record" in report, report
    assert os.path.basename(base_path) in report, report
    assert not table.raw_by_key, "table should be empty before sampling"

    feed(dm, [1.0, 0.5])
    patcher.cleanup()  # this is what a finished prompt does

    files = sorted(f for f in os.listdir(TABLE_DIR) if f.endswith(".safetensors"))
    mine = [f for f in files if f.startswith("h3_full_")]
    assert len(mine) == 1, files
    assert mine[0].endswith("_bf16.safetensors"), mine[0]

    _t, raw, meta = plugin.table_io.load_table(os.path.join(TABLE_DIR, mine[0]))
    assert len(raw) == 3 and meta["t_rows"] == "2", (len(raw), meta.get("t_rows"))
    assert meta["identity"].endswith(":base:bf16"), meta["identity"]
    assert meta["target_base"] == plugin.h3_unified.base_fingerprint(student()[0])
    assert meta["target_name"] == os.path.basename(student()[1])
    # the recorder must not outlive the prompt
    assert getattr(dm, "_h3_recorders", []) == [], "recorder hooks were left behind"


def test_second_schedule_merges():
    dm2 = tiny_model()  # same seed -> same identity as the first teacher
    patcher = patcher_for(dm2)
    H().run(patcher, "record", for_model=student()[1], fill_reference_row=False)
    feed(dm2, [1.0, 0.75, 0.25])  # 0.75 / 0.25 are new
    patcher.cleanup()
    base, _path = student()  # the table is declared for the base, not the teacher
    path, why = plugin.h3_unified.resolve_for_model(
        patcher_for(base), base, plugin.h3_unified.AUTO)
    t_values, _raw, meta = plugin.table_io.load_table(path)
    assert len(t_values) == 4, t_values
    assert meta["runs"] == "2", meta["runs"]
    assert "declared" in why, why


def test_no_auto_base_detection():
    """No for_model = no target_base, even with exactly one curve checkpoint.

    Auto-detection was removed: which base a recording belongs to is an
    explicit choice saved with the workflow, never a scan of the model folder
    or a remembered per-teacher binding.
    """
    other = tiny_model(curve=33, seed=71)  # a second base, so the first one stays clean
    path = checkpoint_of(other, "tiny-base2.safetensors")
    original = folder_paths.get_filename_list
    folder_paths.get_filename_list = lambda k: (
        [path] if k == "diffusion_models" else original(k))
    try:
        before = set(os.listdir(TABLE_DIR))
        dm = tiny_model(seed=77)
        patcher = patcher_for(dm)
        _, _table, report = H().run(patcher, "record")  # for_model defaults to (none)
        assert "[auto]" not in report, report
        feed(dm, [1.0])
        patcher.cleanup()

        new = sorted(set(os.listdir(TABLE_DIR)) - before)
        assert len(new) == 1, new
        _t, _raw, meta = plugin.table_io.load_table(
            os.path.join(TABLE_DIR, new[0]))
        assert "target_base" not in meta, meta
        assert not os.path.isfile(
            os.path.join(TABLE_DIR, "_default_bases.json")), \
            "no hidden binding state may be written"
    finally:
        folder_paths.get_filename_list = original


def test_pinned_rows_match_a_real_reference_batch():
    """The pre-computed 0.999 row equals what a reference run would produce."""
    import torch.nn.functional as F

    before = set(os.listdir(TABLE_DIR))
    dm = tiny_model(seed=131)
    patcher = patcher_for(dm)
    H().run(patcher, "record")  # fill_reference_row defaults to on
    feed(dm, [1.0, 0.5])
    patcher.cleanup()

    new = sorted(set(os.listdir(TABLE_DIR)) - before)
    assert len(new) == 1, new
    t_values, raw, _meta = plugin.table_io.load_table(os.path.join(TABLE_DIR, new[0]))
    assert any(abs(v - 0.999) < 1e-6 for v in t_values), t_values

    dtype = dm.blocks[0].adaln_proj.linear.weight.dtype
    with torch.no_grad():
        aug = torch.tensor([1.0, 0.5, 0.999], dtype=torch.float32)
        want = dm.blocks[0].adaln_proj.linear(
            F.silu(dm.time_embedder(aug).to(dtype)))[-1]
    idx = [i for i, v in enumerate(t_values) if abs(v - 0.999) < 1e-6][0]
    assert torch.equal(raw["blocks.0"][idx], want.float()), "pinned row differs"

    # The report must say HOW each pin got there: 0.999 was pre-computed in an
    # extra batch, 1.0 was produced by the run itself. Both used to print
    # "captured", which hides a failed pin pre-computation.
    pins = patcher._h3_recorder.pins_report
    assert "t=0.999 pre-computed" in pins, pins
    assert "t=1.000 produced by this run" in pins, pins

    # The saved-row watermark must be the count AFTER finalize(): finalize folds
    # the pre-computed pins into rec.rows, so a pre-build count leaves the
    # watchdog thinking 2 rows were never written, and it re-reads the whole
    # 462 MiB table to answer "unchanged".
    assert patcher._h3_saved_rows == len(patcher._h3_recorder.rows), (
        patcher._h3_saved_rows, len(patcher._h3_recorder.rows))
    passes = patcher._h3_save_passes
    patcher._h3_recorder.last_row_at -= (plugin.h3_unified.AUTOSAVE_IDLE + 1)
    plugin.h3_unified._autosave_tick()
    assert patcher._h3_save_passes == passes, "phantom save after the pins"


def test_missing_reference_row_is_reported():
    """A text-only run should say, at save time, that t=0.999 is absent."""
    import logging

    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    handler = Capture()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        dm = tiny_model(seed=91)
        patcher = patcher_for(dm)
        H().run(patcher, "record", fill_reference_row=False)
        feed(dm, [1.0])  # a text-only step: no keyframe / reference rows
        patcher.cleanup()
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    assert any("no row for t=0.999" in m for m in seen), seen[-4:]


def test_pinned_run_does_not_warn_about_a_missing_pin():
    """f32_image(0.999) is 0.999000013 -- the check must not miss it."""
    import logging

    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    handler = Capture()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        dm = tiny_model(seed=141)
        patcher = patcher_for(dm)
        H().run(patcher, "record")  # pins on by default
        feed(dm, [1.0, 0.5])
        patcher.cleanup()
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    assert not any("no row for t=0.999" in m for m in seen), seen[-6:]


def test_apply_arms_and_cleanup_unhooks():
    base, _path = student()
    patcher = patcher_for(base)
    _, _table, report = H().run(patcher, "apply", table=plugin.h3_unified.AUTO)
    assert "armed" in report, report
    assert "declared for this model" in report, report
    runtime = base._h3_exact_runtimes[0]
    assert len(runtime.hook_handles) == 3, len(runtime.hook_handles)

    patcher.cleanup()
    # Injection hooks deliberately survive the FIRST pass's ON_CLEANUP: a
    # director chain runs a refine pass afterwards with its own sigmas. The
    # next prompt detaches them via _clear_runtime.
    assert len(runtime.hook_handles) == 3, "refine pass must still be hooked"
    assert "[pass 1]" in report, report          # per-pass audit exists
    assert "INJECTION" in report and "EQUIVALENCE" in report, report
    assert "zero forwards" in report, report      # armed but nothing sampled

    patcher.cleanup()
    assert "[pass 2]" in report, "a second pass must get its own section"
    assert "PASS DIFF" in report, report

    plugin.h3_unified._clear_runtime(patcher)
    assert runtime.hook_handles == [], "next prompt must detach the runtime"
    assert not runtime.raw_by_key, "next prompt must release the table rows"


def test_apply_hook_end_to_end():
    """Drive the real AdalnProj.forward so the hook, cdist and delta all run.

    This is the path the tests used to skip: recording hooks `.linear`, so a
    fixture that only called `.linear` never exercised the injection hook -- and
    the injection hook crashed on the first step because torch.cdist refuses a
    float32 query against bf16 keys.
    """
    dm = tiny_model(seed=777)
    patcher = patcher_for(dm)
    _, table, _rep = H().run(patcher, "record", fill_reference_row=False)
    feed(dm, [1.0, 0.5])
    patcher.cleanup()
    name = os.path.basename(table.source)
    assert name.endswith(".safetensors"), name

    _, _t, rep = H().run(patcher, "apply", table=name, report_level="debug")
    feed(dm, [1.0, 0.5])
    patcher.cleanup()

    assert "injected=3" in rep, rep          # 2 blocks + final_layer
    assert "exact=3" in rep, rep
    assert "max delta : UNAVAILABLE" not in rep, rep
    assert "NOT ESTABLISHED" not in rep, rep
    plugin.h3_unified._clear_runtime(patcher)


def test_deferred_save_covers_a_pass_without_cleanup():
    """A custom sampling node that never fires ON_CLEANUP must not lose rows.

    The real 4080 run did exactly this: the refine pass went through a
    hand-written sampler that unloads models itself, so `ModelPatcher.cleanup()`
    never ran and the 8 timesteps it captured never reached the table.
    """
    dm = tiny_model(seed=1212)
    patcher = patcher_for(dm)
    _, table, _rep = H().run(patcher, "record", fill_reference_row=False)
    feed(dm, [1.0, 0.5])
    patcher.cleanup()                              # pass 1 -> saved
    assert len(plugin.table_io.load_table(table.source)[0]) == 2

    feed(dm, [0.25, 0.1])                          # refine, NO cleanup
    rec = patcher._h3_recorder
    assert len(rec.rows) == 4, len(rec.rows)

    rec.last_row_at -= (plugin.h3_unified.AUTOSAVE_IDLE + 1)
    plugin.h3_unified._autosave_tick()

    t_values = plugin.table_io.load_table(table.source)[0]
    assert len(t_values) == 4, t_values
    assert patcher._h3_saved_rows == 4, patcher._h3_saved_rows


def test_quantised_checkpoint_is_refused():
    """A file-side fingerprint over int8 bytes can never match the loaded model."""
    import torch as _t
    from safetensors.torch import save_file

    path = os.path.join(CKPT_DIR, "quantised.safetensors")
    save_file({"blocks.0.adaln_proj.linear.weight":
               _t.zeros((8, 8), dtype=_t.int8),
               "blocks.0.adaln_proj.linear.bias": _t.zeros(8)}, path)
    try:
        plugin.h3_unified.base_fingerprint_from_file(path)
    except RuntimeError as e:
        assert "quantised" in str(e), str(e)
        return
    raise AssertionError("a quantised checkpoint silently produced a fingerprint")


def test_deferred_apply_report_without_cleanup():
    """A pass with no ON_CLEANUP still gets a report, from the watchdog.

    Real case: the 8-step first pass reports at its own cleanup, then the refine
    pass runs through a sibling patcher and no cleanup ever reaches the armed
    runtime -- its 4 steps (204 injections) were invisible in the console.
    """
    dm = tiny_model(seed=1313)
    patcher = patcher_for(dm)
    _, table, _rep = H().run(patcher, "record", fill_reference_row=False)
    feed(dm, [1.0, 0.5])
    patcher.cleanup()

    _, _t, rep = H().run(patcher, "apply", report_level="debug",
                         table=os.path.basename(table.source))
    feed(dm, [1.0, 0.5])                    # sampled, but no patcher.cleanup()
    runtime = patcher._h3_exact_runtime
    assert runtime is not None
    assert runtime.state["injected"] == 3, runtime.state["injected"]

    runtime.state["last_hook_at"] -= (plugin.h3_unified.AUTOSAVE_IDLE + 1)
    plugin.h3_unified._autosave_tick()

    assert "deferred" in rep, rep
    assert "injected=3" in rep, rep
    plugin.h3_unified._clear_runtime(patcher)


def test_two_passes_are_both_saved():
    """ON_CLEANUP fires per pass; a refine pass must add rows, not be dropped."""
    before = set(os.listdir(TABLE_DIR))
    dm = tiny_model(seed=909)
    patcher = patcher_for(dm)
    _, _table, report = H().run(patcher, "record", fill_reference_row=False)
    feed(dm, [1.0, 0.5])
    patcher.cleanup()                      # pass 1
    feed(dm, [0.25, 0.1])                  # refine pass, different t
    patcher.cleanup()                      # pass 2

    new = sorted(set(os.listdir(TABLE_DIR)) - before)
    assert len(new) == 1, new
    t_values = plugin.table_io.load_table(os.path.join(TABLE_DIR, new[0]))[0]
    assert len(t_values) == 4, t_values
    assert "pass 2" in report, report
    assert "merged" in str(report) or "saved" in str(report), report


def test_each_pass_reports_only_its_own_forwards():
    """A pass block must not inherit the previous pass's forwards or row ids.

    Real case (2026-11-14): the 8-step first pass and the 4-step refine both
    injected, but the refine's deferred block printed 12 `fwd` lines -- 8 of
    them the first pass's -- and said `PASS DIFF: identical`. Two reasons:
    the refine bypasses the DIFFUSION_MODEL wrapper, so sigma/t stayed frozen
    at the previous pass's last step, and the signature compared
    `t_emb.shape[0]` (always 3) instead of the injected row ids.
    """
    dm = tiny_model(seed=4242)
    patcher = patcher_for(dm)
    _, table, _ = H().run(patcher, "record", fill_reference_row=False)
    feed(dm, [1.0, 0.5])
    patcher.cleanup()
    feed(dm, [0.25, 0.1])                  # second schedule -> four rows
    patcher.cleanup()

    _, _t, rep = H().run(patcher, "apply", report_level="debug",
                         table=os.path.basename(table.source))
    feed(dm, [1.0, 0.5])                   # pass 1 -> rows 3, 2
    patcher.cleanup()
    feed(dm, [0.25, 0.1])                  # pass 2 -> rows 1, 0
    patcher.cleanup()

    text = str(rep)
    # One forward per pass -- not 1 + 2. The trace is sliced per pass now.
    assert text.count("fwd ") == 2, text
    assert "indices=[3, 2]" in text, text
    assert "indices=[1, 0]" in text, text
    # Row ids changed, so the diff must not claim the passes were identical.
    assert "DIFFERENT" in text, text
    assert "indices=(3, 2)" in text, text
    assert "indices=(1, 0)" in text, text
    # This sampler never calls the wrapper; say it instead of printing a
    # frozen sigma as if it were this pass's schedule.
    assert "0 calls this pass" in text, text
    # The counters block is cumulative and now says so.
    assert "cumulative since arm" in text, text
    plugin.h3_unified._clear_runtime(patcher)


def test_lora_mismatch_is_reported_not_silent():
    base, _path = student()
    path, _why = plugin.h3_unified.resolve_for_model(
        patcher_for(base), base, plugin.h3_unified.AUTO)
    table = plugin.table_io.read_table_file(path)
    table.meta["adaln_lora_fingerprint"] = "deadbeefdeadbeef"

    patcher = patcher_for(base)
    try:
        plugin.h3_unified._inject(patcher, base, table)
    except RuntimeError as e:
        assert "LoRA" in str(e), str(e)
    else:
        raise AssertionError("a table with a foreign AdaLN LoRA was injected silently")

    rep = plugin.nodes.Report("t", "brief")
    plugin.h3_unified._inject(patcher, base, table, rep, allow_lora_mismatch=True)
    assert "MISMATCH" in rep, rep
    plugin.h3_unified._clear_runtime(patcher)


def test_silent_level_stays_silent():
    base, _path = student()
    patcher = patcher_for(base)
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    handler = Capture()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        _, _, rep = H().run(patcher, "apply", table=plugin.h3_unified.AUTO,
                            report_level="silent")
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    assert "diagnostics are off" in str(rep), str(rep)
    noise = [m for m in seen if m.startswith("H3AdaLN:")]
    assert not noise, noise
    plugin.h3_unified._clear_runtime(patcher)


def test_off_clears():
    base, _path = student()
    patcher = patcher_for(base)
    H().run(patcher, "apply", table=plugin.h3_unified.AUTO)
    assert getattr(base, "_h3_exact_runtimes", [])
    H().run(patcher, "off")
    assert not getattr(base, "_h3_exact_runtimes", []), "off left a live runtime"
    assert getattr(base, "_h3_exact_runtime", None) is None


def test_hybrid_merges_two_real_tables():
    # two teachers recorded here, so the merge does not depend on what earlier
    # tests happened to leave in the folder
    before = set(os.listdir(TABLE_DIR))
    for seed in (4242, 4343):
        dm = tiny_model(seed=seed)
        patcher = patcher_for(dm)
        H().run(patcher, "record", fill_reference_row=False)
        feed(dm, [1.0, 0.5])
        patcher.cleanup()

    new = sorted(set(os.listdir(TABLE_DIR)) - before)
    assert len(new) == 2, new
    base_name, other = new

    target = tiny_model(curve=33, seed=9)
    patcher2 = patcher_for(target)
    _, merged, report = HY().run(patcher2, base_name, other, split=1)
    assert "hybrid" in report, report
    assert merged.source and os.path.isfile(merged.source), merged.source
    assert os.path.basename(merged.source).startswith("h3_hybrid_x1_"), merged.source
    assert getattr(target, "_h3_exact_runtimes", []), "hybrid did not inject"
    patcher2.cleanup()


for name, fn in [
    ("load: registers nodes, table folder, schema", test_registration),
    ("fingerprint: file bytes == loaded-module bytes",
     test_base_fingerprint_from_file_matches_model),
    ("record: real ModelPatcher.cleanup() writes the table",
     test_record_saves_on_cleanup),
    ("record: a second schedule merges into the same file",
     test_second_schedule_merges),
    ("record: no auto base detection; for_model must be explicit",
     test_no_auto_base_detection),
    ("record: the pinned 0.999 row equals a real reference batch",
     test_pinned_rows_match_a_real_reference_batch),
    ("record: a pinned run never reports a missing 0.999 row",
     test_pinned_run_does_not_warn_about_a_missing_pin),
    ("record: a text-only run reports the missing 0.999 row",
     test_missing_reference_row_is_reported),
    ("apply: resolves by declared base, hooks, cleanup unhooks",
     test_apply_arms_and_cleanup_unhooks),
    ("apply: a pass with no ON_CLEANUP is reported by the watchdog",
     test_deferred_apply_report_without_cleanup),
    ("record: a second sampling pass adds rows instead of losing them",
     test_two_passes_are_both_saved),
    ("record: a pass with no ON_CLEANUP is saved by the watchdog",
     test_deferred_save_covers_a_pass_without_cleanup),
    ("fingerprint: a quantised checkpoint is refused",
     test_quantised_checkpoint_is_refused),
    ("apply: real AdalnProj forward is replaced (cdist + delta)",
     test_apply_hook_end_to_end),
    ("apply: a foreign AdaLN LoRA is refused, not injected silently",
     test_lora_mismatch_is_reported_not_silent),
    ("report: report_level=silent emits nothing",
     test_silent_level_stays_silent),
    ("off: drops a live runtime", test_off_clears),
    ("hybrid: merges two real tables and injects",
     test_hybrid_merges_two_real_tables),
]:
    check(name, fn)

shutil.rmtree(TABLE_DIR, ignore_errors=True)
shutil.rmtree(CKPT_DIR, ignore_errors=True)

print()
if all(checks):
    print("ALL RUNTIME CHECKS PASSED ({})".format(COMFY))
    sys.exit(0)
print("{} CHECK(S) FAILED".format(checks.count(False)))
sys.exit(1)
