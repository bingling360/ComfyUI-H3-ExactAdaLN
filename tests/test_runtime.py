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
    """One AdaLN pass: what the DiT does per denoising step."""
    t = torch.tensor(t_list, dtype=torch.float32)
    dtype = dm.blocks[0].adaln_proj.linear.weight.dtype
    emb = dm.time_embedder(t).to(dtype)  # model.py:751 does exactly this cast
    for block in dm.blocks:
        block.adaln_proj.linear(emb)
    dm.final_layer.adaln_proj.linear(emb)


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
    assert list(spec_i["optional"]) == ["for_model", "fill_reference_row",
                                      "table", "table_in", "after"]
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
    assert runtime.hook_handles == [], "apply hooks survived cleanup"


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
