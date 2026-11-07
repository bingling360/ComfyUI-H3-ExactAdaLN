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

FORMAT = "h3_exact_adaln/v1"
BLOCK_PREFIX = "blocks."
FINAL_KEY = "final_layer"


class AdaLNTable:
    """In-memory value passed between nodes as the H3_ADALN_TABLE type."""

    def __init__(self, t_values, raw_by_key, meta, source=None):
        self.t_values = [float(t) for t in t_values]
        self.raw_by_key = dict(raw_by_key)
        self.meta = dict(meta)
        self.source = source

    @property
    def out_dtype(self):
        return self.meta.get("out_dtype", "torch.bfloat16")

    def size_bytes(self):
        return sum(v.numel() * v.element_size() for v in self.raw_by_key.values())

    def projections(self):
        return len(self.raw_by_key)

    def summary(self):
        meta = self.meta
        return "\n".join([
            "format       : {}".format(meta.get("format", FORMAT)),
            "source       : {}".format(self.source or "(in-memory, not saved yet)"),
            "teacher      : {}".format(meta.get("teacher", "?")),
            "lora tag     : {}".format(meta.get("lora_tag", "?")),
            "lora fp      : {} ({} non-AdaLN keys)".format(
                meta.get("lora_fingerprint", "?"), meta.get("lora_patch_count", "?")),
            "sigmas       : {}".format(str(meta.get("sigmas", "[]"))[:180]),
            "shift_v/a    : {} / {}".format(meta.get("shift_v", "?"), meta.get("shift_a", "?")),
            "compute dtype: {}".format(meta.get("compute_dtype", "?")),
            "t rows       : {}".format(len(self.t_values)),
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
    if meta.get("format") != FORMAT:
        raise RuntimeError("H3ExactAdaLN: unsupported table format {!r} in {}".format(
            meta.get("format"), path))
    t_values = sd["t_values"].to(torch.float32).tolist()
    raw = {k: v for k, v in sd.items() if k != "t_values"}
    return t_values, raw, meta


def write_table_file(path, table):
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
