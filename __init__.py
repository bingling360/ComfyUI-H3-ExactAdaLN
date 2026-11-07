"""ComfyUI-H3-ExactAdaLN

Pre-bake the full MiniMax-H3 model's AdaLN modulation (including every LoRA
delta) into an exact timestep-indexed table, then replay that table on a
pruned/curve checkpoint at generation time.

Nodes (Save/Load behave like Save Image / Load Image):
  H3ExactAdaLNBake    full teacher MODEL + sigmas -> H3_ADALN_TABLE
  H3ExactAdaLNSave    H3_ADALN_TABLE -> shareable .safetensors in output/
  H3ExactAdaLNLoad    upload / pick a table file -> H3_ADALN_TABLE
  H3ExactAdaLNApply   pruned MODEL + H3_ADALN_TABLE -> patched MODEL
  H3LatentCompare     verification: bitwise latent comparison

The bake path never parses a safetensors file by hand: it goes through
ComfyUI's ModelPatcher, so bf16 / fp16 / fp8_scaled / int8_convrot / nvfp4
checkpoints and any ComfyUI-applicable LoRA are handled by ComfyUI itself.
See README section in the workflow JSON notes for limits.
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
