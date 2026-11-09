"""Checks for the v2 node surface: naming, identity, row merging. Pure CPU.

Run: python tests/test_simple.py

The behaviour these tests pin down is the reason the node lost its tag fields:

  * a table's name comes from the model + AdaLN LoRA + dtype, never from a
    widget, and it keeps working after the file is renamed;
  * re-recording the same schedule is a no-op;
  * re-recording a *different* schedule adds rows instead of replacing them
    (this is what stops a later apply from failing with a t-set mismatch);
  * the same t with different numbers is refused, loudly.
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
sys.path.insert(0, PLUGIN)

TABLE_DIR = tempfile.mkdtemp()
os.environ["H3_ADALN_TABLE_DIR"] = TABLE_DIR

import torch  # noqa: E402
import h3_unified as H  # noqa: E402

KEYS = ["blocks.0", "blocks.1", "final_layer"]
GAIN = 100.0
checks = []


def check(name, fn):
    try:
        fn()
        print("PASS " + name)
        checks.append(True)
    except Exception as e:  # noqa: BLE001
        print("FAIL {} -> {!r}".format(name, e))
        checks.append(False)


def rows(t_list, offset=0.0):
    """Row values encode the projection and t, so a mis-merged row is visible."""
    out = {}
    for ki, key in enumerate(KEYS):
        out[key] = torch.stack([
            torch.full((4,), offset + GAIN * ki + t, dtype=torch.float32)
            for t in t_list]).contiguous()
    return out


def meta(identity="m1:base:bf16", dtype="torch.bfloat16", **extra):
    base = {"identity": identity, "out_dtype": dtype, "format": "h3_exact_adaln/v2"}
    base.update(extra)
    return base


def clean():
    for name in os.listdir(TABLE_DIR):
        try:
            os.remove(os.path.join(TABLE_DIR, name))
        except OSError:
            pass


# ---------------------------------------------------------------- schema
def test_schema():
    spec = H.H3AdaLN.INPUT_TYPES()
    assert list(spec["required"]) == ["model", "mode"], spec["required"]
    assert list(spec["optional"]) == ["for_model", "fill_reference_row", "table",
                                      "table_in", "after"], spec["optional"]
    assert spec["required"]["mode"][0] == ["apply", "record", "off"]
    assert spec["optional"]["table"][0][0] == H.AUTO
    assert spec["optional"]["for_model"][0][0] == H.AUTO
    assert spec["optional"]["fill_reference_row"][1]["default"] is True
    assert H.H3AdaLN.RETURN_TYPES == ("MODEL", H.TABLE_TYPE, "STRING")

    hybrid = H.H3AdaLNHybrid.INPUT_TYPES()
    assert list(hybrid["required"]) == ["model", "table_a", "table_b", "split"]
    assert hybrid["required"]["split"][1]["default"] == 25


def test_legacy_nodes_hidden():
    import nodes
    for name in ("H3AdaLNRecord", "H3ExactAdaLNBake", "H3ExactAdaLNSave",
                 "H3ExactAdaLNLoad", "H3AdaLNMerge", "H3ExactAdaLNApply"):
        assert getattr(nodes.NODE_CLASS_MAPPINGS[name], "DEPRECATED", False), name
    assert not getattr(nodes.H3LatentCompare, "DEPRECATED", False)


# ---------------------------------------------------------------- identity
def test_auto_name():
    dm = type("DM", (), {"blocks": [], "use_adaln_curves": False,
                         "time_embedder": None, "adaln_t_table": None})()
    name = H.auto_name(dm, "001c09abcdef", "", "torch.bfloat16")
    assert name == "h3_full_001c09_bf16.safetensors", name
    named = H.auto_name(dm, "001c09abcdef", "a91f22", "torch.float32")
    assert named == "h3_full_001c09_fp32_La91f.safetensors", named
    dm.use_adaln_curves = True
    assert H.auto_name(dm, "001c09abcdef", "", "torch.bfloat16").startswith("h3_pruned_")


def test_identity_is_model_lora_dtype():
    a = H.identity_of("m1", "", "torch.bfloat16")
    assert a == "m1:base:bf16"
    assert H.identity_of("m1", "x", "torch.bfloat16") != a
    assert H.identity_of("m1", "", "torch.float32") != a


# ---------------------------------------------------------------- storage
def test_create_then_noop():
    clean()
    t = [0.0, 0.5, 1.0]
    path, action, added, total = H.save_table_smart(
        t, rows(t), meta(), "torch.bfloat16", "h3_full_aaa_bf16.safetensors")
    assert action == "saved" and added == 3 and total == 3, (action, added, total)
    assert os.path.basename(path) == "h3_full_aaa_bf16.safetensors"
    assert os.path.isfile(path)

    again, action, added, total = H.save_table_smart(
        t, rows(t), meta(), "torch.bfloat16", "h3_full_aaa_bf16.safetensors")
    assert again == path and action == "unchanged" and added == 0, (action, added)
    assert len(os.listdir(TABLE_DIR)) == 1, os.listdir(TABLE_DIR)


def test_different_schedule_adds_rows():
    clean()
    first = [0.0, 0.5, 1.0]
    H.save_table_smart(first, rows(first), meta(), "torch.bfloat16", "t.safetensors")
    # second run: two shared timesteps + two new ones
    second = [0.5, 0.75, 1.0, 1.25]
    path, action, added, total = H.save_table_smart(
        second, rows(second), meta(), "torch.bfloat16", "t.safetensors")
    assert action == "merged" and added == 2 and total == 5, (action, added, total)
    t2, raw2, meta2 = H.table_io.load_table(path)
    assert t2 == [0.0, 0.5, 0.75, 1.0, 1.25], t2
    assert meta2["runs"] == "2"
    for ki, key in enumerate(KEYS):
        for j, t in enumerate(t2):
            assert float(raw2[key][j, 0]) == GAIN * ki + t, (key, t, raw2[key][j, 0])
    assert len(os.listdir(TABLE_DIR)) == 1, os.listdir(TABLE_DIR)


def test_conflict_is_refused():
    clean()
    t = [0.0, 0.5]
    H.save_table_smart(t, rows(t), meta(), "torch.bfloat16", "c.safetensors")
    try:
        H.save_table_smart(t, rows(t, offset=7.0), meta(), "torch.bfloat16",
                           "c.safetensors")
    except RuntimeError as e:
        assert "different AdaLN values" in str(e), str(e)
        assert "Nothing was written" in str(e), str(e)
    else:
        raise AssertionError("conflicting rows were accepted")
    # the original file must be untouched
    t2, raw2, _ = H.table_io.load_table(os.path.join(TABLE_DIR, "c.safetensors"))
    assert t2 == t and float(raw2["blocks.0"][0, 0]) == 0.0


def test_rename_is_followed():
    clean()
    t = [0.0, 0.5]
    path, _, _, _ = H.save_table_smart(t, rows(t), meta(), "torch.bfloat16",
                                       "orig.safetensors")
    renamed = os.path.join(TABLE_DIR, "renamed_by_hand.safetensors")
    os.replace(path, renamed)
    t2 = [0.5, 0.9]
    path2, action, added, total = H.save_table_smart(
        t2, rows(t2), meta(), "torch.bfloat16", "orig.safetensors")
    assert path2 == renamed, (path2, renamed)
    assert action == "merged" and added == 1 and total == 3, (action, added, total)
    assert not os.path.exists(os.path.join(TABLE_DIR, "orig.safetensors"))


def test_dtype_branch_of_union():
    clean()
    t = [0.0, 0.5]
    path, _, _, _ = H.save_table_smart(t, rows(t), meta(), "torch.bfloat16",
                                       "d.safetensors")
    try:
        H._union(path, t, rows(t), "torch.float32")
    except RuntimeError as e:
        assert "not interchangeable" in str(e), str(e)
    else:
        raise AssertionError("a dtype change was accepted")


def _two_tables(declared_for=""):
    """Two tables with the same AdaLN LoRA, different teacher identities."""
    t = [0.0, 0.5]
    m1 = meta()
    m2 = meta(identity="m2:base:bf16")
    if declared_for:
        m1["target_base"] = declared_for
    a, _, _, _ = H.save_table_smart(t, rows(t), m1, "torch.bfloat16",
                                    "aaa.safetensors")
    b, _, _, _ = H.save_table_smart(t, rows(t, offset=50.0), m2,
                                    "torch.bfloat16", "bbb.safetensors")
    return a, b


def test_resolve_by_declared_target():
    clean()
    model, dm = _fake_model()
    fp = H.base_fingerprint(dm)
    a, b = _two_tables(declared_for=fp)
    got, why = H.resolve_for_model(model, dm, H.AUTO)
    assert got == a and "declared" in why, (got, why)


def test_resolve_ignores_a_table_declared_for_another_model():
    clean()
    model, dm = _fake_model()
    a, b = _two_tables(declared_for="000000000000")
    got, why = H.resolve_for_model(model, dm, H.AUTO)
    assert got != a, "a table declared for another base was used"
    assert got == b and "without a target" in why, (got, why)


def test_resolve_two_declared_for_one_base_errors():
    clean()
    model, dm = _fake_model()
    fp = H.base_fingerprint(dm)
    t = [0.0]
    m1 = meta(); m1["target_base"] = fp
    m2 = meta(identity="m2:base:bf16"); m2["target_base"] = fp
    H.save_table_smart(t, rows(t), m1, "torch.bfloat16", "a.safetensors")
    H.save_table_smart(t, rows(t, 9.0), m2, "torch.bfloat16", "b.safetensors")
    try:
        H.resolve_for_model(model, dm, H.AUTO)
    except RuntimeError as e:
        assert "declared for this model" in str(e), str(e)
        assert "a.safetensors" in str(e) and "b.safetensors" in str(e), str(e)
    else:
        raise AssertionError("two tables claiming one base were accepted")


def test_resolve_accepts_legacy_empty_digest():
    """Tables recorded before v2 stored sha256("")[:16] for "no AdaLN LoRA"."""
    clean()
    model, dm = _fake_model()
    m = meta()
    m["adaln_lora_fingerprint"] = "e3b0c44298fc1c14"
    path, _, _, _ = H.save_table_smart([0.0], rows([0.0]), m, "torch.bfloat16",
                                       "legacy.safetensors")
    got, why = H.resolve_for_model(model, dm, H.AUTO)
    assert got == path and "without a target" in why, (got, why)


def test_resolve_ambiguous_without_a_target():
    clean()
    model, dm = _fake_model()
    a, b = _two_tables()
    try:
        H.resolve_for_model(model, dm, H.AUTO)
    except RuntimeError as e:
        assert "none of them declares a target" in str(e), str(e)
        assert "for_model" in str(e), str(e)
    else:
        raise AssertionError("an ambiguous lookup returned a table")
    got, why = H.resolve_for_model(model, dm, os.path.basename(b))
    assert got == b and why == "picked", (got, why)


def test_resolve_missing_table_says_what_to_do():
    clean()
    model, dm = _fake_model()
    try:
        H.resolve_for_model(model, dm, H.AUTO)
    except RuntimeError as e:
        assert "no table for this model" in str(e), str(e)
        assert "mode=off" in str(e), str(e)
    else:
        raise AssertionError("an empty folder resolved to a table")


def test_two_models_resolve_their_own_tables():
    clean()
    m1, d1 = _fake_model()
    torch.manual_seed(11)
    with torch.no_grad():
        for p in d1.parameters():
            p.normal_(0.0, 0.02)
    d1.__dict__.pop("_h3_base_fp", None)
    m2, d2 = _fake_model()
    torch.manual_seed(22)
    with torch.no_grad():
        for p in d2.parameters():
            p.normal_(0.0, 0.02)
    d2.__dict__.pop("_h3_base_fp", None)
    f1, f2 = H.base_fingerprint(d1), H.base_fingerprint(d2)
    assert f1 != f2, "two bases share a fingerprint"

    t = [0.0]
    a = H.save_table_smart(t, rows(t), dict(meta(), target_base=f1),
                           "torch.bfloat16", "one.safetensors")[0]
    b = H.save_table_smart(t, rows(t, 9.0), dict(meta(identity="m2:base:bf16"),
                                                 target_base=f2),
                           "torch.bfloat16", "two.safetensors")[0]
    # no pick, no state: each base finds the table that declares it
    assert H.resolve_for_model(m1, d1, H.AUTO)[0] == a
    assert H.resolve_for_model(m2, d2, H.AUTO)[0] == b


def test_base_fingerprint_tracks_the_adaln_branch():
    clean()
    model, dm = _fake_model()
    k1 = H.base_fingerprint(dm)
    with torch.no_grad():
        dm.blocks[-1].adaln_proj.linear.weight[0, 0] += 0.5
    dm.__dict__.pop("_h3_base_fp", None)  # a real model does not change under us
    k2 = H.base_fingerprint(dm)
    assert k1 != k2, "the base fingerprint ignored the AdaLN weights"


def test_choices_put_auto_first():
    clean()
    H.save_table_smart([0.0], rows([0.0]), meta(), "torch.bfloat16",
                       "h3_full_ccc_bf16.safetensors")
    names = H._choices()
    assert names[0] == H.AUTO
    assert "h3_full_ccc_bf16.safetensors" in names


# ---------------------------------------------------------------- end to end
class _TE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_out = torch.nn.Linear(8, 16)

    def forward(self, t):
        # stand-in for TimeEmbedder's sinusoid: [n] -> [n, 8] -> [n, 16]
        x = torch.stack([t * (i + 1) for i in range(8)], dim=-1)
        return self.proj_out(x)


class _Proj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.expand, self.hidden, self.modalities = 1, 8, 1


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_proj = _Proj()


class _DM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(2)])
        final = torch.nn.Module()
        final.adaln_proj = _Proj()
        self.final_layer = final
        self.time_embedder = _TE()
        self.use_adaln_curves = False
        self.adaln_t_table = None


def _fake_model():
    import types
    dm = _DM()
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=dm),
        patches={}, model_options={})
    return model, dm


def test_record_then_save():
    """The real path: hook, sample, ON_CLEANUP write, then inject."""
    clean()
    model, dm = _fake_model()
    _, table, report = H.H3AdaLN().run(model, "record", fill_reference_row=False)
    assert "record" in report, report

    t = torch.tensor([1.0, 0.5], dtype=torch.float32)
    emb = dm.time_embedder(t)
    for block in dm.blocks:
        block.adaln_proj.linear(emb)
    dm.final_layer.adaln_proj.linear(emb)

    recorder = model._h3_recorder
    H._finish_record(model)
    assert recorder.handles == [], "record hooks outlived the prompt"
    assert model._h3_recorder is None
    files = [f for f in os.listdir(TABLE_DIR) if f.endswith(".safetensors")]
    assert len(files) == 1, files
    assert files[0].startswith("h3_full_") and files[0].endswith("_fp32.safetensors"), files

    t2, raw2, meta2 = H.table_io.load_table(os.path.join(TABLE_DIR, files[0]))
    assert t2 == [0.5, 1.0], t2
    assert sorted(raw2) == ["blocks.0", "blocks.1", "final_layer"], sorted(raw2)
    assert meta2["identity"].endswith(":base:fp32"), meta2["identity"]
    assert meta2["adaln_lora_fingerprint"] == "", meta2["adaln_lora_fingerprint"]
    assert "target_base" not in meta2, "no for_model was given, so no target"

    loaded = H.table_io.read_table_file(os.path.join(TABLE_DIR, files[0]))
    lines = H._inject(model, dm, loaded)
    assert len(model._h3_exact_runtime.hook_handles) == 3, lines
    H._clear_runtime(model)
    assert not model._h3_exact_runtime


def test_record_pins_the_reference_rows():
    """One recording covers reference workflows: 0.999 / 1.0 are pre-computed."""
    clean()
    model, dm = _fake_model()
    _, _table, report = H.H3AdaLN().run(model, "record")  # switch defaults to on
    assert "extra rows" in report and "0.999" in report, report

    t = torch.tensor([1.0, 0.5], dtype=torch.float32)
    dtype = dm.blocks[0].adaln_proj.linear.weight.dtype
    emb = dm.time_embedder(t).to(dtype)
    for block in dm.blocks:
        block.adaln_proj.linear(emb)
    dm.final_layer.adaln_proj.linear(emb)
    H._finish_record(model)

    name = [f for f in os.listdir(TABLE_DIR) if f.endswith(".safetensors")][0]
    t_values, raw, _meta = H.table_io.load_table(os.path.join(TABLE_DIR, name))
    pinned = [v for v in t_values if abs(v - 0.999) < 1e-6]
    assert len(pinned) == 1, t_values
    assert any(abs(v - 1.0) < 1e-9 for v in t_values), t_values

    # what the model would have produced had the run carried a reference row:
    # the same modules, one extra row in the batch
    with torch.no_grad():
        aug = torch.tensor([1.0, 0.5, 0.999], dtype=torch.float32)
        want = dm.blocks[0].adaln_proj.linear(
            torch.nn.functional.silu(dm.time_embedder(aug).to(dtype)))[-1]
    row = raw["blocks.0"][t_values.index(pinned[0])]
    assert torch.equal(row, want.float()), (row[:4], want[:4])


def test_record_without_pins_has_no_reference_row():
    clean()
    model, dm = _fake_model()
    H.H3AdaLN().run(model, "record", fill_reference_row=False)
    t = torch.tensor([1.0], dtype=torch.float32)
    emb = dm.time_embedder(t)
    for block in dm.blocks:
        block.adaln_proj.linear(emb)
    dm.final_layer.adaln_proj.linear(emb)
    H._finish_record(model)
    name = [f for f in os.listdir(TABLE_DIR) if f.endswith(".safetensors")][0]
    t_values, _raw, _meta = H.table_io.load_table(os.path.join(TABLE_DIR, name))
    assert t_values == [1.0], t_values


def test_off_clears_a_previous_injection():
    clean()
    model, dm = _fake_model()
    _, table, _ = H.H3AdaLN().run(model, "record")
    t = torch.tensor([1.0], dtype=torch.float32)
    emb = dm.time_embedder(t)
    for block in dm.blocks:
        block.adaln_proj.linear(emb)
    dm.final_layer.adaln_proj.linear(emb)
    H._finish_record(model)
    name = [f for f in os.listdir(TABLE_DIR) if f.endswith(".safetensors")][0]
    loaded = H.table_io.read_table_file(os.path.join(TABLE_DIR, name))
    H._inject(model, dm, loaded)
    assert model._h3_exact_runtime.hook_handles

    H.H3AdaLN().run(model, "off")
    assert getattr(model, "_h3_exact_runtime", None) is None
    assert loaded is not None


def test_t_set_ignores_shape_but_not_schedule():
    """Resolution / frame count must not change the timesteps a table needs."""
    S = H.sigma_plan

    class Layout:
        """A packed layout: only the segment *kinds* matter, never the counts."""
        def __init__(self, rows, kinds=("text", "video", "audio")):
            self.segments, r = [], 0
            for kind in kinds:
                self.segments.append((r, r + rows, kind))
                r += rows

    small, big = Layout(4), Layout(400_000)
    assert S.runtime_unique_t(0.7, small) == S.runtime_unique_t(0.7, big), \
        "the timestep set moved with the latent size"
    assert S.runtime_unique_t(0.7, small) != S.runtime_unique_t(0.4, small), \
        "a different sigma did not change the timestep set"
    assert (S.runtime_unique_t(0.7, small, shift_v=12)
            != S.runtime_unique_t(0.7, small, shift_v=6)), \
        "a different video shift did not change the timestep set"
    base = S.runtime_unique_t(0.7, small)
    refs = ("text", "video", "audio", "ref_img")
    one = S.runtime_unique_t(0.7, Layout(4, refs))
    assert set(one) - set(base) == {0.999}, \
        "a reference segment should add exactly the pinned 0.999 row"
    # how many references, and whether they are keyframes or images, is irrelevant
    many = S.runtime_unique_t(0.7, Layout(4, ("text", "video", "audio")
                                         + ("ref_img",) * 9))
    keyframe = S.runtime_unique_t(0.7, Layout(4, ("text", "video", "audio", "cond")))
    assert one == many == keyframe, "the number/kind of reference changed the t set"
    # reference *audio* pins a different value
    a_ref = S.runtime_unique_t(0.7, Layout(4, ("text", "video", "audio", "ref_audio")))
    assert set(a_ref) - set(base) == {1.0}, \
        "reference audio should add exactly the pinned 1.0 row"


check("schema: main node has model + mode + one picker", test_schema)
check("schema: legacy single-purpose nodes are marked deprecated",
      test_legacy_nodes_hidden)
check("name: derived from model fp + dtype (+ AdaLN LoRA)", test_auto_name)
check("identity: model / AdaLN LoRA / dtype, nothing else", test_identity_is_model_lora_dtype)
check("t-set: shape-free, but steps/scheduler/shift change it",
      test_t_set_ignores_shape_but_not_schedule)
check("save: first write creates one file, same run is a no-op", test_create_then_noop)
check("save: a new schedule grows the table instead of replacing it",
      test_different_schedule_adds_rows)
check("save: same t with different numbers is refused", test_conflict_is_refused)
check("save: a hand-renamed table is still found and grown", test_rename_is_followed)
check("save: a dtype change is refused by the union path", test_dtype_branch_of_union)
check("resolve: a table declaring this base wins", test_resolve_by_declared_target)
check("resolve: a table declared for another base is skipped",
      test_resolve_ignores_a_table_declared_for_another_model)
check("resolve: two tables claiming one base is an error",
      test_resolve_two_declared_for_one_base_errors)
check("resolve: a legacy table's empty digest still counts",
      test_resolve_accepts_legacy_empty_digest)
check("resolve: several undeclared tables need one pick",
      test_resolve_ambiguous_without_a_target)
check("resolve: an empty folder errors and names the way out",
      test_resolve_missing_table_says_what_to_do)
check("resolve: two models each find their own declared table",
      test_two_models_resolve_their_own_tables)
check("resolve: the fingerprint follows the AdaLN branch",
      test_base_fingerprint_tracks_the_adaln_branch)
def test_base_fingerprint_survives_fp16_to_bf16_cast():
    """ComfyUI may run fp16 checkpoint weights in bf16; the lookup must survive."""
    import copy
    from safetensors.torch import save_file

    class Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 16)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = Proj()

    class DM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])
            self.final_layer = Block()
            self.use_adaln_curves = True
            self.adaln_t_table = torch.randn(1025, 8)

    def cast(dm, dtype):
        for m in dm.modules():
            for _n, p in m.named_parameters(recurse=False):
                p.data = p.data.to(dtype)
        dm.adaln_t_table = dm.adaln_t_table.to(dtype)

    torch.manual_seed(5)
    a = DM()
    cast(a, torch.float16)
    b = copy.deepcopy(a)
    cast(b, torch.bfloat16)  # what ComfyUI actually exposes at compute time

    assert H.base_fingerprint(a) == H.base_fingerprint(b), \
        "a fp16/bf16 cast changed the base fingerprint"

    path = os.path.join(TABLE_DIR, "dtype_probe.safetensors")
    sd = {k: v.detach().contiguous() for k, v in a.state_dict().items()}
    sd["adaln_t_table"] = a.adaln_t_table.contiguous()
    save_file(sd, path)
    assert H.base_fingerprint_from_file(path) == H.base_fingerprint(b), \
        "file-side and model-side base fingerprints differ"


check("base fp: identical across fp16 file / bf16 model",
      test_base_fingerprint_survives_fp16_to_bf16_cast)
check("picker: (auto) first, then the files on disk", test_choices_put_auto_first)
check("record: pins 0.999/1.0 through the model's own batch",
      test_record_pins_the_reference_rows)
check("record: with the switch off, no reference row is written",
      test_record_without_pins_has_no_reference_row)
check("e2e: record -> ON_CLEANUP save -> inject", test_record_then_save)
check("e2e: mode=off drops a previously injected runtime",
      test_off_clears_a_previous_injection)

shutil.rmtree(TABLE_DIR, ignore_errors=True)

print()
if all(checks):
    print("ALL SIMPLE-NODE CHECKS PASSED")
    sys.exit(0)
print("{} CHECK(S) FAILED".format(checks.count(False)))
sys.exit(1)
