"""ComfyUI-H3-ExactAdaLN

Make a pruned MiniMax-H3 checkpoint reproduce the *full* model's AdaLN
modulation exactly, so "pruned base + table + LoRA" is bitwise identical to
"full teacher + the same LoRA".

Exactly three nodes:

  H3 AdaLN           record on the full teacher (one sampling run writes a
                     table) / apply on the pruned base (lookup by fingerprint).
                     Nothing to type, nothing to pick.
  H3 AdaLN Hybrid    bX-49 only: merge two recordings when no full hybrid
                     teacher is available.
  H3 Latent Compare  A/B verification: are two LATENT outputs bitwise equal?

Only `record` produces an exact table: it stores the tensors the teacher
actually computed during a real sampling run. Offline recomputation was removed
(it differs by one bf16 ulp per projection, amplified to ~0.145 mean_abs after
8 steps). See README for the exactness contract and the known limits.
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .h3_unified import (NODE_CLASS_MAPPINGS as SIMPLE_NODES,
                         NODE_DISPLAY_NAME_MAPPINGS as SIMPLE_NAMES,
                         register_folder)

NODE_CLASS_MAPPINGS.update(SIMPLE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SIMPLE_NAMES)

try:
    register_folder()
except Exception as _e:  # pragma: no cover - no ComfyUI (offline test import)
    logging.getLogger(__name__).debug(
        "ComfyUI-H3-ExactAdaLN: table folder not registered: %s", _e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
