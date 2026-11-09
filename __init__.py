"""ComfyUI-H3-ExactAdaLN

Make a pruned MiniMax-H3 checkpoint reproduce the *full* model's AdaLN
modulation exactly, so that "pruned base + table + LoRA" is bitwise identical
to "full teacher + the same LoRA".

Three nodes to add, two of them for daily use:

  H3 AdaLN          mode=record on the full teacher -> one sampling run writes a
                    table; mode=apply on the pruned base injects it. Nothing to
                    type: the table's name and location follow from the model.
  H3 AdaLN Hybrid   bX-49 only: blocks 0..split-1 from one table, split..49 from
                    another. A ready-made full bX-49 teacher makes this unnecessary.
  H3 Latent Compare verification: are two LATENT outputs bitwise identical?

The eight original single-purpose nodes (Record / Bake / Save / Load / Merge /
Apply / LatentSave) still load for the published workflows but are marked
deprecated, so they no longer clutter the node menu.

Two ways to produce a table, and only the first is exact:

  * record -- read-only hooks during a real sampling run. Bitwise by
    construction, because it stores the tensors the teacher actually produced.
  * bake   -- offline recomputation on unloaded CPU weights. Cheap, but cannot
    reproduce the kernels ComfyUI selects at run time (measured drift: one bf16
    ulp per projection -> ~0.145 mean_abs on the latent after 8 steps).

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

# The two nodes that matter. Imported defensively so a failure here can never
# take down the legacy nodes that published workflows still reference.
try:
    import logging
    from .h3_unified import (NODE_CLASS_MAPPINGS as SIMPLE_NODES,
                             NODE_DISPLAY_NAME_MAPPINGS as SIMPLE_NAMES,
                             register_folder)
    register_folder()
    NODE_CLASS_MAPPINGS.update(SIMPLE_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(SIMPLE_NAMES)
except Exception as _e:
    logging.getLogger(__name__).warning(
        "ComfyUI-H3-ExactAdaLN: H3 AdaLN nodes unavailable: %s", _e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
