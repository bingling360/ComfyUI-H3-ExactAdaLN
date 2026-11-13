"""Serialisation and in-memory container for exact AdaLN tables.

File layout (safetensors, fp32, t-indexed):

  t_values                 [T]                fp32, ascending
  blocks.0 .. blocks.49    [T, out_features] fp32, raw adaln_proj.linear output
  final_layer              [T, out_features] fp32

Storing the RAW linear output (before AdalnProj.view/chunk) means the runtime
does not need to know `expand`, `hidden` or `modalities`: it just gathers rows
and replays the module's own view/chunk. fp32 is mandatory -- bf16 storage of
AdaLN values costs ~2.5e-2 relative error (measured by indhic-ai).

A table file is self-contained: all provenance (teacher, LoRA fingerprint,
sigma grid, shifts, dtype, ComfyUI version) lives in the safetensors header,
so a saved file can be copied/shared like an image and loaded elsewhere.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT = "h3_exact_adaln/v2"
LEGACY_FORMATS = ("h3_exact_adaln/v1",)
BLOCK_PREFIX = "blocks."
FINAL_KEY = "final_layer"
# sha256("")[:16] -- what a pre-2026-11 recorder wrote when the model had no
# AdaLN LoRA. Normalised to "" so old and new tables compare equal.
EMPTY_DIGEST = "e3b0c44298fc1c14"


def _adaln_fp_of(meta):
    """The AdaLN-LoRA digest of a table, from either header generation.

    v2 recorders write `adaln_lora_fingerprint`. The merge check used to read
    `lora_fingerprint`, a key nothing ever wrote, so `"" == ""` made every
    merge pass -- two tables recorded under different AdaLN LoRAs merged
    silently. Read the field that exists, and treat a missing one as empty.
    """
    if "adaln_lora_fingerprint" in meta:
        fp = meta.get("adaln_lora_fingerprint") or ""
    elif "lora_fingerprint" in meta:
        fp = meta.get("lora_fingerprint") or ""   # v1 / pre-2026-11 header
    else:
        fp = ""
    fp = str(fp)
    return "" if fp == EMPTY_DIGEST else fp


def _lora_identity(meta):
    """(weight_digest, key_digest, raw_digest) for a header, any generation.

    Three header generations exist and the merge check has to cope with all of
    them, because a bX-49 hybrid is assembled from whatever recordings happen
    to be on disk:

      v2   `adaln_lora_fingerprint`            weight-aware, no key digest
      mid  `lora_fingerprint` = weight digest,
           `lora_fingerprint_keys`            key digest alongside
      v1   `lora_fingerprint` = key digest     names only

    A lone `lora_fingerprint` cannot be told apart as weight- or name-digest,
    so `raw_digest` carries it: if both sides literally agree on it, that is
    as much proof as either generation can offer.
    """
    raw = str(meta.get("lora_fingerprint") or "")
    raw = "" if raw == EMPTY_DIGEST else raw
    if "adaln_lora_fingerprint" in meta:
        return _adaln_fp_of(meta), "", raw
    keys = str(meta.get("lora_fingerprint_keys") or "")
    keys = "" if keys == EMPTY_DIGEST else keys
    if keys:
        return raw, keys, raw
    return "", raw, raw


class AdaLNTable:
    """In-memory value passed between nodes as the H3_ADALN_TABLE type.

    A table produced by Record stays *live*: the node hands the sampler a
    recorder handle and the rows only exist once sampling has run.
    `finalize()` pulls them in and can be called again after a later sampling
    pass has captured more timesteps -- a director chain fires ON_CLEANUP once
    per pass, and each pass must be able to add its rows.
    """

    def __init__(self, t_values, raw_by_key, meta, source=None, recorder=None):
        self.t_values = [float(t) for t in t_values]
        self.raw_by_key = dict(raw_by_key)
        self.meta = dict(meta)
        self.source = source
        self.recorder = recorder
        self._pulled_rows = 0

    @property
    def is_live(self):
        return self.recorder is not None

    def finalize(self, required=True):
        """Materialise (or refresh) a live table. Safe to call repeatedly."""
        rec = self.recorder
        if rec is None:
            return self
        rows = len(getattr(rec, "rows", {}) or {})
        if rows == 0:
            if required:
                rec.build()  # raises with the actionable message
            return self
        if self.raw_by_key and rows == self._pulled_rows:
            return self
        t_values, raw, extra = rec.build()
        self.t_values = t_values
        self.raw_by_key = raw
        self.meta.update(extra)
        self.meta["t_count"] = str(len(t_values))
        # `build()` may add the pinned rows, so the row count that now matters is
        # the one after the build, not the one that triggered it.
        self._pulled_rows = len(getattr(rec, "rows", {}) or {})
        return self

    def release(self):
        """Drop the recorder handle once this table will not grow any more."""
        self.recorder = None
        return self

    @property
    def out_dtype(self):
        return self.meta.get("out_dtype", "torch.bfloat16")

    def size_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.raw_by_key.values())

    def projections(self):
        return len(self.raw_by_key)

    def summary(self):
        self.finalize(required=False)
        meta = self.meta
        rows = len(self.t_values)
        return "\n".join([
            "format       : {}".format(meta.get("format", FORMAT)),
            "source       : {}".format(self.source or "(in-memory, not saved yet)"),
            "teacher      : {}".format(meta.get("teacher", "?")),
            "provenance   : {}".format(meta.get("record_mode", "?")),
            "identity     : {}".format(meta.get("identity", "?")),
            "adaln lora fp: {}".format(meta.get("adaln_lora_fingerprint") or "none"),
            "target base  : {}".format(meta.get("target_base") or "none"),
            "sigmas       : {}".format(str(meta.get("sigmas", "[]"))[:180]),
            "shift_v/a    : {} / {}".format(meta.get("shift_v", "?"), meta.get("shift_a", "?")),
            "compute dtype: {}".format(meta.get("compute_dtype", "?")),
            "out dtype    : {}".format(self.out_dtype),
            "t rows       : {}{}".format(
                rows, "  (live recorder, may grow)" if self.is_live else ""),
            "projections  : {}".format(self.projections()),
            "table size   : {:.1f} MiB".format(self.size_bytes() / (1 << 20)),
        ])


def _meta_value(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def save_table(path, t_values, raw_by_key, meta, out_dtype="torch.bfloat16"):
    """Write a table. `raw_by_key` maps 'blocks.N'/'final_layer' -> [T, F]."""
    tensors = {"t_values": torch.tensor(t_values, dtype=torch.float32).contiguous()}
    t_count = len(t_values)
    if not t_count:
        raise RuntimeError("H3AdaLN: refusing to save an empty table (0 t rows)")
    for key, value in raw_by_key.items():
        v = value.detach().to(torch.float32).cpu().contiguous()
        if v.ndim != 2 or v.shape[0] != t_count:
            raise ValueError("bad shape for {}: {} (expected [{}, F])".format(
                key, tuple(v.shape), t_count))
        tensors[key] = v
    metadata = {k: _meta_value(v) for k, v in meta.items()}
    metadata["format"] = FORMAT
    metadata["t_count"] = str(t_count)
    metadata["out_dtype"] = out_dtype
    metadata["storage_dtype"] = "float32"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return path


def load_table(path, device="cpu"):
    """Return (t_values, raw_by_key, meta)."""
    sd = load_file(path, device=device)
    # load_file() returns a plain dict of tensors; metadata lives in the file
    # header and must be read with safe_open().
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
    if meta.get("format") != FORMAT and meta.get("format") not in LEGACY_FORMATS:
        raise RuntimeError("H3AdaLN: unsupported table format {!r} in {}".format(
            meta.get("format"), path))
    t_values = sd["t_values"].to(torch.float32).tolist()
    raw = {k: v for k, v in sd.items() if k != "t_values"}
    return t_values, raw, meta


def _block_index(key):
    """'blocks.17' -> 17, 'final_layer' -> None, anything else -> None."""
    if key == FINAL_KEY:
        return None
    if key.startswith(BLOCK_PREFIX):
        try:
            return int(key[len(BLOCK_PREFIX):])
        except ValueError:
            return None
    return None


def merge_tables(table_a, table_b, split=25, final_from="a"):
    """Combine two recorded tables block-wise (the bX-49 hybrid recipe).

    A `bX-49` checkpoint is not "full blocks + pruned blocks": the whole file
    is curve-shaped, with `adaln_proj` of blocks X..49 taken from a second
    teacher. Its own 8-wide curve cannot carry the teacher's full modulation,
    so the exact table must be assembled from two records made on the two full
    teachers: blocks 0..X-1 from `table_a`, blocks X..49 from `table_b`, and
    `final_layer` from whichever side the checkpoint kept.

    Both tables must have been recorded on the same timesteps and the same
    LoRA stack; every block is still a row the corresponding teacher actually
    produced, so replaying the merged table is exact per block.
    """
    a, b = table_a, table_b
    a.finalize(required=True)
    b.finalize(required=True)

    split = int(split)
    if not 0 <= split <= 50:
        raise RuntimeError("H3AdaLN: split must be 0..50, got {}".format(split))
    final_from = str(final_from).lower()
    if final_from not in ("a", "b"):
        raise RuntimeError("H3AdaLN: final_from must be 'a' or 'b'")

    keys_a, keys_b = set(a.raw_by_key), set(b.raw_by_key)
    if keys_a != keys_b:
        raise RuntimeError(
            "H3AdaLN: projection sets differ -- only in A: {} | only in B: {}"
            .format(sorted(keys_a - keys_b)[:6], sorted(keys_b - keys_a)[:6]))

    if a.out_dtype != b.out_dtype:
        raise RuntimeError(
            "H3AdaLN: out_dtype differs (A={} B={}); re-record both tables "
            "with the same teacher dtype".format(a.out_dtype, b.out_dtype))

    wa, ka, ra = _lora_identity(a.meta)
    wb, kb, rb = _lora_identity(b.meta)
    if not (wa or ka or ra) and not (wb or kb or rb):
        fp_ok, fp_a, fp_b = True, "", ""          # neither declares an AdaLN LoRA
    elif ra and rb and ra == rb:
        fp_ok, fp_a, fp_b = True, ra, rb          # same literal digest
    elif wa and wb:
        fp_ok, fp_a, fp_b = (wa == wb), wa, wb    # both weight-aware: strict
    elif ka and kb:
        fp_ok, fp_a, fp_b = (ka == kb), ka, kb    # both name-only: compare names
    else:
        # Mixed generations: accept only if a value both sides actually carry
        # agrees. Otherwise the two sides cannot be shown to match.
        common_a = {v for v in (wa, ka, ra) if v}
        common_b = {v for v in (wb, kb, rb) if v}
        fp_ok = bool(common_a & common_b)
        fp_a, fp_b = (wa or ka or ra), (wb or kb or rb)
    if not fp_ok:
        raise RuntimeError(
            "H3AdaLN: the two tables were recorded with different AdaLN/timestep "
            "LoRAs (A: fp={} | B: fp={} | A: {} | B: {}). Blocks from the two "
            "sides would not describe the same modulation, so the merged table "
            "would be wrong everywhere except at t=0. Re-record both on their "
            "full teachers with the same LoRA stack.".format(
                fp_a or "none", fp_b or "none",
                a.meta.get("teacher", "?"), b.meta.get("teacher", "?")))

    try:
        from . import sigma_plan
    except ImportError:  # pragma: no cover - direct-file loading
        import sigma_plan

    index_b = {}
    for i, t in enumerate(b.t_values):
        index_b[sigma_plan.f32_image(t)] = i
    order, missing = [], []
    for t in a.t_values:
        j = index_b.get(sigma_plan.f32_image(t))
        if j is None:
            missing.append(t)
        else:
            order.append(j)
    if missing:
        raise RuntimeError(
            "H3AdaLN: B is missing {} of A's timesteps (first: {}). "
            "Record both tables on the same steps / scheduler / shifts / "
            "resolution.".format(len(missing), ["{:.7f}".format(t) for t in missing[:4]]))
    if len(order) != len(a.t_values) or len(set(order)) != len(order):
        raise RuntimeError("H3AdaLN: timestep alignment is not one-to-one")

    index = torch.as_tensor(order, dtype=torch.long)
    merged = {}
    for key in a.raw_by_key:
        block = _block_index(key)
        use_b = (block is None and final_from == "b") or \
                (block is not None and block >= split)
        src = b if use_b else a
        rows = src.raw_by_key[key]
        if src is b:
            rows = rows.index_select(0, index.to(rows.device))
        if rows.shape != a.raw_by_key[key].shape:
            raise RuntimeError(
                "H3AdaLN: {} shape mismatch A={} B={} -- the two teachers "
                "are different architectures".format(
                    key, tuple(a.raw_by_key[key].shape), tuple(rows.shape)))
        merged[key] = rows.contiguous()

    a_teacher = a.meta.get("teacher", "?")
    b_teacher = b.meta.get("teacher", "?")
    if split <= 0:
        teacher = "hybrid: blocks 0-49 <- {}".format(b_teacher)
    elif split >= 50:
        teacher = "hybrid: blocks 0-49 <- {}".format(a_teacher)
    else:
        teacher = "hybrid: blocks 0-{} <- {} | blocks {}-49 <- {}".format(
            split - 1, a_teacher, split, b_teacher)

    meta = dict(a.meta)
    meta.update({
        "teacher": teacher,
        "record_mode": "merged from two recorded tables (exact per block)",
        "hybrid_split": str(split),
        "hybrid_final_from": final_from,
        "hybrid_source_a": a_teacher,
        "hybrid_source_b": b_teacher,
        # Audit trail for the bX-49 recipe: a donor recorded with the base
        # teacher's embedding is a *different* table from the donor recorded
        # as-is, and only the former matches the assembled hybrid.
        "hybrid_te_from_a": a.meta.get("time_embedder_from", ""),
        "hybrid_te_from_b": b.meta.get("time_embedder_from", ""),
        "hybrid_lora_fp_a": fp_a,
        "hybrid_lora_fp_b": fp_b,
        "t_count": str(len(a.t_values)),
        "out_dtype": a.out_dtype,
        "format": FORMAT,
    })
    # A hybrid table replaces a *bX-49* checkpoint, which is neither A nor B.
    # Inheriting A's target_base made the hybrid claim to be the table of A's
    # pruned base -- so `table=(auto)` on that base found two tables and
    # refused, and the hybrid could never be found for the file it is for.
    for _key in ("target_base", "target_name", "target_bases"):
        meta.pop(_key, None)
    # Only a weight-aware digest may be written back as `adaln_lora_fingerprint`;
    # a v1 key-name digest there would then be compared against a v2 model
    # fingerprint and refuse every apply.
    weight_fp = wa or wb
    if weight_fp:
        meta["adaln_lora_fingerprint"] = weight_fp
    else:
        meta.pop("adaln_lora_fingerprint", None)
    return AdaLNTable(a.t_values, merged, meta)


def write_table_file(path, table):
    table.finalize(required=True)
    return save_table(path, table.t_values, table.raw_by_key,
                      table.meta, out_dtype=table.out_dtype)


def read_table_file(path):
    t_values, raw, meta = load_table(path)
    return AdaLNTable(t_values, raw, meta, source=os.path.abspath(path))


def table_dir():
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "h3_exact_adaln")
    os.makedirs(d, exist_ok=True)
    return d


def _scan_dirs():
    """Places a shared table file may live: input/, output/, models/."""
    import folder_paths
    dirs = []
    for getter in ("get_input_directory", "get_output_directory"):
        try:
            dirs.append(getattr(folder_paths, getter)())
        except Exception:
            pass
    try:
        dirs.append(table_dir())
    except Exception:
        pass
    return [d for d in dirs if d]


def list_table_files():
    """Basenames for the Load node COMBO (matches what the upload widget sets)."""
    names = set()
    for d in _scan_dirs():
        if not os.path.isdir(d):
            continue
        for _root, _sub, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".safetensors"):
                    names.add(f)
    return sorted(names)


def _search_by_name(name):
    for d in _scan_dirs():
        if not os.path.isdir(d):
            continue
        direct = os.path.join(d, name)
        if os.path.isfile(direct):
            return direct
        for root, _sub, files in os.walk(d):
            if name in files:
                return os.path.join(root, name)
    return None


def find_table_file(name):
    """Resolve a filename / substring / relative path / absolute path.

    The Load node accepts free-form text: an exact filename, a partial string
    (e.g. "dmad8"), a path under input/ or output/, or an absolute path.
    """
    if not name:
        return None
    name = str(name).strip()
    if not name:
        return None
    if os.path.isabs(name):
        return name if os.path.isfile(name) else None

    names = list_table_files()
    base = os.path.basename(name)
    if base in names:
        return _search_by_name(base)
    if name in names:
        return _search_by_name(name)

    low = base.lower()
    hits = [n for n in names if low in n.lower()]
    if len(hits) == 1:
        return _search_by_name(hits[0])
    return None


def describe_available(limit=12):
    """Human-readable candidate list for Load node error messages."""
    names = list_table_files()
    if not names:
        return "(no .safetensors tables found in input/, output/ or models/h3_exact_adaln/)"
    lines = ["available table files ({}):".format(len(names))]
    for n in names[:limit]:
        p = _search_by_name(n)
        lines.append("  {}  ->  {}".format(n, p or "?"))
    if len(names) > limit:
        lines.append("  ... and {} more".format(len(names) - limit))
    return "\n".join(lines)


def list_tables():
    """Legacy: only the models/h3_exact_adaln directory."""
    try:
        return sorted(f for f in os.listdir(table_dir()) if f.endswith(".safetensors"))
    except Exception:
        return []


def describe(path):
    return read_table_file(path).summary()
