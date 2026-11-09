"""Helpers for MiniMax-H3 latents on ComfyUI 0.44.

H3 packs its video and audio latents into `comfy.nested_tensor.NestedTensor`
(a small wrapper holding a list of tensors, exposed via .unbind()). It is NOT a
torch.Tensor, so ComfyUI's stock SaveLatent (which calls .contiguous()) crashes
and any comparison code using torch.is_tensor() silently sees nothing.
"""

import logging
import os
from urllib.parse import quote

import torch


def flatten_latent(latent):
    """Return {name: torch.Tensor} from a LATENT dict, expanding NestedTensor."""
    out = {}
    if not isinstance(latent, dict):
        return out
    for key, value in latent.items():
        if torch.is_tensor(value):
            out[key] = value
        elif getattr(value, "is_nested", False) and hasattr(value, "unbind"):
            for i, part in enumerate(value.unbind()):
                out["{}__part{}".format(key, i)] = part
    return out


class H3LatentSave:
    """Save an H3 LATENT (NestedTensor aware) as a safetensors .latent file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {
                    "default": "h3_adaln/latents/verify",
                    "tooltip": "Auto-numbered file under ComfyUI/output/."}),
            },
        }

    DEPRECATED = True  # verification-only; H3 Latent Compare covers the usual case
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MiniMaxH3/ExactAdaLN"

    def save(self, latent, filename_prefix="h3_adaln/latents/verify"):
        import folder_paths
        from safetensors.torch import save_file

        prefix = str(filename_prefix).strip() or "h3_adaln/latents/verify"
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory())
        name = "{}_{:05d}.latent".format(filename, counter)
        path = os.path.join(folder, name)

        tensors = {k: v.detach().cpu().contiguous()
                   for k, v in flatten_latent(latent).items()}
        if not tensors:
            raise RuntimeError(
                "H3LatentSave: no tensors found in the LATENT input "
                "(type={}, keys={}). Expected a dict with a Tensor or "
                "NestedTensor 'samples' entry.".format(
                    type(latent).__name__,
                    sorted(latent.keys()) if isinstance(latent, dict) else "?"))

        save_file(tensors, path, metadata={
            "format": "h3_latent/v1",
            "parts": str(len(tensors)),
            "shapes": str({k: list(v.shape) for k, v in tensors.items()})[:900],
        })
        url = "/view?filename={}&subfolder={}&type=output".format(
            quote(name), quote(subfolder))
        report = "\n".join([
            "=== H3 latent saved ===",
            "path   : {}".format(path),
            "parts  : {}".format(len(tensors)),
            "shapes : {}".format({k: tuple(v.shape) for k, v in tensors.items()}),
            "download : {}".format(url),
            "",
            "Compare two files with tools/compare_latents.py",
        ])
        logging.info("%s", report)
        return {
            "ui": {
                "text": [report],
                "h3_latents": [{"filename": name, "subfolder": subfolder,
                                "type": "output", "download_url": url}],
            },
            "result": (),
        }


EXTRA_NODE_CLASS_MAPPINGS = {"H3LatentSave": H3LatentSave}
EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {"H3LatentSave": "H3 Latent Save (NestedTensor)"}
