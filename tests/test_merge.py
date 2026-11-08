"""Numerical checks for H3AdaLNMerge. Pure CPU: no ComfyUI, no GPU.

Two synthetic tables with the same timesteps in *opposite* row order, so any
implementation that merges by row index instead of by t value fails loudly.

Run: python tests/test_merge.py
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
sys.path.insert(0, PLUGIN)

import torch  # noqa: E402
import table_io  # noqa: E402

TA = [0.0, 0.25, 0.5, 0.75]
TB = list(reversed(TA))          # deliberately different order
KEYS = ["blocks.0", "blocks.1", "blocks.2", "final_layer"]
WIDTH = 4

checks = []


def check(name, fn):
    try:
        fn()
        print("PASS " + name)
        checks.append(True)
    except Exception as e:  # noqa: BLE001
        print("FAIL {} -> {!r}".format(name, e))
        checks.append(False)


def value(side, ki, t):
    """side 1=A 2=B; encode block, side and the *t value* in every element."""
    return float(side * 1000 + ki * 10 + int(round(t * 100)))


def make_table(t_list, side, fp="fp-same", dtype="torch.bfloat16", keys=None,
               keys_fp=None):
    keys = keys or KEYS
    raw = {}
    for ki, key in enumerate(keys):
        rows = [torch.full((WIDTH,), value(side, ki, t), dtype=torch.float32)
                for t in t_list]
        raw[key] = torch.stack(rows)
    meta = {
        "teacher": "teacher-{}".format("A" if side == 1 else "B"),
        "lora_fingerprint": fp,
        "out_dtype": dtype,
        "format": table_io.FORMAT,
    }
    if keys_fp is not None:
        meta["lora_fingerprint_keys"] = keys_fp
    return table_io.AdaLNTable(list(t_list), raw, meta)


def expect(merged, key, j):
    ki = KEYS.index(key)
    side = 2 if (key == "blocks.2") else 1      # split=2 -> blocks.2 from B
    want = value(side, ki, TA[j])
    got = float(merged.raw_by_key[key][j, 0])
    assert got == want, "{} row {}: got {} want {} (t={})".format(
        key, j, got, want, TA[j])


A = make_table(TA, 1)
B = make_table(TB, 2)


def test_block_split_and_t_alignment():
    m = table_io.merge_tables(A, B, split=2, final_from="a")
    assert m.t_values == TA, m.t_values
    assert len(m.raw_by_key) == len(KEYS)
    for key in KEYS:
        assert m.raw_by_key[key].shape == (len(TA), WIDTH)
        for j in range(len(TA)):
            expect(m, key, j)
    assert m.meta["hybrid_split"] == "2"
    assert m.meta["hybrid_final_from"] == "a"
    assert "blocks 0-1" in m.meta["teacher"] and "blocks 2-49" in m.meta["teacher"]


def test_split_zero_and_fifty():
    m0 = table_io.merge_tables(A, B, split=0, final_from="a")
    for j in range(len(TA)):
        for key in ("blocks.0", "blocks.1", "blocks.2"):
            assert float(m0.raw_by_key[key][j, 0]) == value(2, KEYS.index(key), TA[j])
        assert float(m0.raw_by_key["final_layer"][j, 0]) == value(1, 3, TA[j])
    # split=50 means "no block comes from B"; final_from still selects final.
    m50 = table_io.merge_tables(A, B, split=50, final_from="b")
    for j in range(len(TA)):
        for key in ("blocks.0", "blocks.1", "blocks.2"):
            assert float(m50.raw_by_key[key][j, 0]) == value(1, KEYS.index(key), TA[j])
        assert float(m50.raw_by_key["final_layer"][j, 0]) == value(2, 3, TA[j])


def test_save_load_roundtrip():
    import shutil
    m = table_io.merge_tables(A, B, split=2)
    # Not TemporaryDirectory: safe_open/load_file keep an mmap on Windows, so
    # the automatic cleanup at block exit would fail with PermissionError.
    td = tempfile.mkdtemp()
    try:
        path = os.path.join(td, "hybrid.safetensors")
        table_io.write_table_file(path, m)
        t2, raw2, meta2 = table_io.load_table(path)
        assert t2 == TA
        assert meta2["hybrid_split"] == "2"
        assert meta2["out_dtype"] == "torch.bfloat16"
        for key in KEYS:
            assert torch.equal(raw2[key], m.raw_by_key[key]), key
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_rejects_missing_timestep():
    B_missing = make_table(TB[:-1], 2)
    try:
        table_io.merge_tables(A, B_missing, split=2)
    except RuntimeError as e:
        assert "missing" in str(e).lower(), str(e)
        return
    raise AssertionError("merge accepted a table with missing timesteps")


def test_rejects_lora_mismatch():
    B_fp = make_table(TB, 2, fp="fp-other")
    try:
        table_io.merge_tables(A, B_fp, split=2)
    except RuntimeError as e:
        assert "lora" in str(e).lower(), str(e)
        return
    raise AssertionError("merge accepted a different LoRA stack")


def test_rejects_dtype_and_projection_mismatch():
    B_dt = make_table(TB, 2, dtype="torch.float16")
    try:
        table_io.merge_tables(A, B_dt, split=2)
    except RuntimeError as e:
        assert "dtype" in str(e).lower(), str(e)
    else:
        raise AssertionError("merge accepted mismatched out_dtype")
    B_keys = make_table(TB, 2, keys=["blocks.0", "blocks.1", "blocks.9",
                                     "final_layer"])
    try:
        table_io.merge_tables(A, B_keys, split=2)
    except RuntimeError as e:
        assert "projection" in str(e).lower(), str(e)
        return
    raise AssertionError("merge accepted mismatched projections")


def test_legacy_fingerprint_compat():
    # A table recorded before the weight-aware digest only has the key digest
    # in `lora_fingerprint`; it must still merge with a new table.
    a_legacy = make_table(TA, 1, fp="legacy-keys")
    b_new = make_table(TB, 2, fp="weight-aware", keys_fp="legacy-keys")
    m = table_io.merge_tables(a_legacy, b_new, split=2)
    assert m.meta["hybrid_split"] == "2"
    # Two new tables with the same keys but different weights must be rejected.
    b_other = make_table(TB, 2, fp="weight-aware-other",
                         keys_fp="legacy-keys")
    try:
        table_io.merge_tables(b_new, b_other, split=2)
    except RuntimeError as e:
        assert "lora" in str(e).lower(), str(e)
        return
    raise AssertionError("merge accepted two new tables with different weights")


def test_mid_version_digest_only():
    # A build in between stored the weight-aware digest but not the key digest.
    # Identical digests must still merge (this is what the real instance hit).
    a_new = make_table(TA, 1, fp="weight-aware", keys_fp="legacy-keys")
    b_mid = make_table(TB, 2, fp="weight-aware")
    m = table_io.merge_tables(a_new, b_mid, split=2)
    assert m.meta["hybrid_split"] == "2"


def test_node_schema():
    import nodes as h3
    spec = h3.H3AdaLNMerge.INPUT_TYPES()
    assert list(spec["required"]) == ["table_a", "table_b", "split", "final_from"]
    assert spec["required"]["split"][1]["default"] == 25
    assert "after" in spec["optional"]
    assert h3.H3AdaLNMerge.RETURN_TYPES[:2] == ("H3_ADALN_TABLE", "STRING")
    assert "H3AdaLNMerge" in h3.NODE_CLASS_MAPPINGS
    info = h3.H3AdaLNMerge().run(A, B, split=2)
    assert "merge" in info[1].lower()
    assert info[0].meta["hybrid_split"] == "2"


check("merge: blocks >= split come from B, rest from A", test_block_split_and_t_alignment)
check("merge: t alignment follows t values, not row order", test_block_split_and_t_alignment)
check("merge: split=0 / split=50 edge cases", test_split_zero_and_fifty)
check("merge: safetensors round-trip keeps hybrid metadata", test_save_load_roundtrip)
check("merge: rejects missing timestep", test_rejects_missing_timestep)
check("merge: rejects a different LoRA stack", test_rejects_lora_mismatch)
check("merge: rejects dtype / projection mismatch", test_rejects_dtype_and_projection_mismatch)
check("merge: legacy key-only table still merges, new tables stay strict",
      test_legacy_fingerprint_compat)
check("merge: digest-only (mid-version) table merges on identical digest",
      test_mid_version_digest_only)
check("merge: node schema and report", test_node_schema)

print()
if all(checks):
    print("ALL MERGE CHECKS PASSED")
    sys.exit(0)
print("{} CHECK(S) FAILED".format(checks.count(False)))
sys.exit(1)
