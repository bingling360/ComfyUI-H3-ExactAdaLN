"""ComfyUI-H3-ExactAdaLN

Make a pruned MiniMax-H3 checkpoint reproduce the *full* model's AdaLN
modulation exactly, so that "pruned base + exact table + LoRA" is bitwise
identical to "full teacher + the same LoRA".

Two ways to produce the table:

  * `H3AdaLNRecord` -- read-only hooks during a real sampling run. Bitwise by
    construction, because it stores the tensors the teacher actually produced.
    **This is the exact path.**
  * `H3ExactAdaLNBake` -- offline recomputation on unloaded CPU weights. Cheap
    and needs no sampling, but cannot reproduce the kernels ComfyUI selects at
    run time (measured drift: one bf16 ulp per projection -> ~0.145 mean_abs on
    the latent after 8 steps).

Nodes (Save/Load behave like Save Image / Load Image):
  H3AdaLNRecord       full teacher MODEL -> H3_ADALN_TABLE (filled while sampling)
  H3ExactAdaLNBake    full teacher MODEL + sigmas -> H3_ADALN_TABLE (offline)
  H3ExactAdaLNSave    H3_ADALN_TABLE -> shareable .safetensors in output/
  H3ExactAdaLNLoad    upload / pick a table file -> H3_ADALN_TABLE
  H3ExactAdaLNApply   pruned MODEL + H3_ADALN_TABLE -> patched MODEL
  H3LatentCompare     verification: bitwise latent comparison
  H3LatentSave        NestedTensor-aware latent writer

See README for the exactness contract and the known limits.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # pragma: no cover - direct-file loading fallback
    import importlib.util
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "_h3_exact_adaln_nodes", os.path.join(_here, "nodes.py"),
        submodule_search_locations=[_here])
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    NODE_CLASS_MAPPINGS = _mod.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS

try:
    import logging
    from .latent_io import (EXTRA_NODE_CLASS_MAPPINGS,
                            EXTRA_NODE_DISPLAY_NAME_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(EXTRA_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(EXTRA_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _e:
    logging.getLogger(__name__).warning(
        "ComfyUI-H3-ExactAdaLN: optional latent_io nodes unavailable: %s", _e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
