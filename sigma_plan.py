"""Timestep planning for MiniMax-H3 exact AdaLN tables.

Mirrors comfy/ldm/minimax/model.py:671-711 so that the values baked offline
and the values the runtime asks for are the same set of distinct timesteps.

Key facts (verified against ComfyUI 0.42.5):
  * model_base passes timestep = sigma * 1000; the DiT divides it back by 1000.
  * video t  = 1 - sigma
  * audio t  = 1 - time_shift_sigma(sigma, video_shift, audio_shift)
  * cond / reference rows are pinned to max(t, 0.999) for video and
    max(t, 1.0) for audio.
  * _forward builds `unique_t` as a sorted set and feeds ONE row per distinct
    t into every adaln_proj, so a baked table is indexed by t, not by step.
"""

import math

VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0

SEGMENT_KINDS = ("text", "video", "audio", "cond", "ref_img",
                 "cond_audio", "ref_audio")


def time_shift_sigma(sigma, from_shift, to_shift):
    """Invert the shifted sigma grid and re-apply the other shift."""
    s = float(sigma)
    base = s / (from_shift + s * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def sigma_to_t_candidates(sigma, shift_v=DEFAULT_VIDEO_SHIFT,
                          shift_a=DEFAULT_AUDIO_SHIFT,
                          visual_aug=VISUAL_COND_TIMESTEP,
                          audio_aug=AUDIO_COND_TIMESTEP):
    """All distinct t values one sampling step at `sigma` can possibly use.

    Superset of model.py:681-683 `seg_t`. Baking a superset is safe: the
    runtime only reads rows it actually needs.
    """
    s = float(sigma)
    if s <= 1e-6:
        return []
    t_v = 1.0 - s
    sigma_a = time_shift_sigma(s, shift_v, shift_a)
    t_a = 1.0 - sigma_a
    return sorted({
        t_v,
        t_a,
        max(t_v, visual_aug),
        max(t_a, audio_aug),
    })


def normalise_sigmas(sigmas):
    """Accept a torch tensor, a ComfyUI SIGMAS object, or a plain sequence."""
    if sigmas is None:
        return []
    if hasattr(sigmas, "detach"):
        sigmas = sigmas.detach().cpu().flatten().tolist()
    elif hasattr(sigmas, "sigmas"):
        return normalise_sigmas(sigmas.sigmas)
    return [float(s) for s in sigmas]


def bake_t_values(sigmas, shift_v=DEFAULT_VIDEO_SHIFT,
                  shift_a=DEFAULT_AUDIO_SHIFT,
                  visual_aug=VISUAL_COND_TIMESTEP,
                  audio_aug=AUDIO_COND_TIMESTEP,
                  extra_t=None,
                  dedup_decimals=7):
    """Full t grid covered by a baked table."""
    values = set()
    for s in normalise_sigmas(sigmas):
        for t in sigma_to_t_candidates(s, shift_v, shift_a, visual_aug, audio_aug):
            values.add(round(float(t), dedup_decimals))
    for t in normalise_sigmas(extra_t):
        values.add(round(float(t), dedup_decimals))
    return sorted(values)


def runtime_unique_t(sigma_v, layout, shift_v=DEFAULT_VIDEO_SHIFT,
                     shift_a=DEFAULT_AUDIO_SHIFT,
                     visual_aug=VISUAL_COND_TIMESTEP,
                     audio_aug=AUDIO_COND_TIMESTEP):
    """Reconstruct model.py:709 `unique_t` for the current forward.

    `layout` is the PackedLayout handed to the dit block patch; only its
    `.segments` list of (start, stop, kind) is needed. Per-row denoise-mask
    timesteps are NOT recoverable here -- see apply_engine's fallback policy.
    """
    s = float(sigma_v)
    t_v = 1.0 - s
    t_a = 1.0 - time_shift_sigma(s, shift_v, shift_a)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, visual_aug),
        "ref_img": max(t_v, visual_aug),
        "cond_audio": max(t_a, audio_aug),
        "ref_audio": max(t_a, audio_aug),
    }
    kinds = set()
    segments = getattr(layout, "segments", None)
    if segments:
        for item in segments:
            if len(item) >= 3 and item[2] in seg_t:
                kinds.add(item[2])
    return sorted({t_v, t_a} | {seg_t[k] for k in kinds})


def build_index(table_t, decimals=7, tolerance=2e-5):
    """Map a float t to a table row index, tolerating fp32 round-trips."""
    coarse = {round(float(t), decimals): i for i, t in enumerate(table_t)}

    def lookup(t):
        key = round(float(t), decimals)
        if key in coarse:
            return coarse[key]
        best_i, best_d = None, None
        for i, v in enumerate(table_t):
            d = abs(float(v) - float(t))
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_d is not None and best_d <= tolerance:
            return best_i
        return None

    return lookup


def shifts_from_options(transformer_options, model):
    """model.py:672-673 honours transformer_options overrides."""
    sv = transformer_options.get(
        "minimax_h3_sigma_shift_video",
        getattr(model, "sigma_shift_video", DEFAULT_VIDEO_SHIFT))
    sa = transformer_options.get(
        "minimax_h3_sigma_shift_audio",
        getattr(model, "sigma_shift_audio", DEFAULT_AUDIO_SHIFT))
    return float(sv), float(sa)


def sigma_from_options(transformer_options, timestep):
    """Current video sigma, preferring the sampler's own sigma tensor."""
    sigmas = transformer_options.get("sigmas")
    if sigmas is not None:
        try:
            return float(sigmas.flatten()[0])
        except Exception:
            pass
    if timestep is not None:
        return float(timestep.flatten()[0]) / 1000.0
    raise RuntimeError("H3ExactAdaLN: no sigma available in transformer_options")
