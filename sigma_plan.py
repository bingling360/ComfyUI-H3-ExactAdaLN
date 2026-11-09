"""Timestep planning for MiniMax-H3 exact AdaLN tables.

Mirrors comfy/ldm/minimax/model.py:671-751 so that the values recorded/baked
and the values the runtime asks for are *the same float32 numbers*.

Key facts (verified against the shipped source):
  * model_base passes timestep = sigma * 1000; the DiT divides it back by 1000.
  * video t  = 1 - sigma
  * audio t  = 1 - time_shift_sigma(sigma, video_shift, audio_shift)
  * cond / reference rows are pinned to max(t, 0.999) for video and
    max(t, 1.0) for audio.
  * _forward builds `unique_t` as a sorted set of *floats* and feeds
    `torch.tensor(unique_t, dtype=torch.float32)` into the time embedder, so a
    baked table is indexed by the float32 image of t, not by step.

Float precision matters here and is deliberate: every arithmetic step below is
evaluated on a float32 tensor, exactly like the DiT does it. Computing the same
expression in Python float64 produces a t a few ulp away, and TimeEmbedder's
`exp(-log(10000)*k/128)` frequencies amplify that into a visibly different
`t_emb` (and therefore a different table row).
"""

import struct

import torch

VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0

SEGMENT_KINDS = ("text", "video", "audio", "cond", "ref_img",
                 "cond_audio", "ref_audio")

# Segment kind -> modality tag used by model.py:713 when it builds mod-row
# indices (video-family rows read the video slot of the 3-row t_emb block).
SEGMENT_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0,
               "cond_audio": 2, "ref_audio": 2}


def f32_image(value):
    """Round a Python float to the nearest float32, returned as a float.

    The DiT does `t_vals = torch.tensor(unique_t, dtype=torch.float32)`, so the
    float32 image of t -- not the double -- is what TimeEmbedder and the curve
    lerp actually consume. Storing the float32 image lets a table key compare
    *equal* to the runtime value instead of merely close, which is what makes
    the match rejectable instead of fuzzy.
    """
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _f32(value):
    if torch.is_tensor(value):
        return value.detach().to(torch.float32)
    return torch.tensor(float(value), dtype=torch.float32)


def time_shift_sigma_t(sigma, from_shift, to_shift):
    """float32 tensor version of model.py's `time_shift_sigma`."""
    s = _f32(sigma)
    base = s / (from_shift + s * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_sigma(sigma, from_shift, to_shift):
    """Invert the shifted sigma grid and re-apply the other shift."""
    return float(time_shift_sigma_t(sigma, from_shift, to_shift))


def sigma_to_t_pair(sigma, shift_v=DEFAULT_VIDEO_SHIFT,
                    shift_a=DEFAULT_AUDIO_SHIFT):
    """(t_video, t_audio) exactly as model.py:675-676 derives them."""
    s = _f32(sigma)
    t_v = float(1.0 - s)
    t_a = float(1.0 - time_shift_sigma_t(s, shift_v, shift_a))
    return t_v, t_a


def runtime_t_pair(sigma, shift_v=DEFAULT_VIDEO_SHIFT,
                   shift_a=DEFAULT_AUDIO_SHIFT):
    """Same, but with model.py's `clamp(min=1e-6)` applied first."""
    s = _f32(sigma).clamp(min=1e-6)
    t_v = float(1.0 - s)
    t_a = float(1.0 - time_shift_sigma_t(s, shift_v, shift_a))
    return t_v, t_a


def sigma_from_timestep(timestep):
    """model.py:674 `sigma_v = (timestep / 1000).float().clamp(min=1e-6)`."""
    if torch.is_tensor(timestep):
        return float(timestep.detach().flatten()[0]) / 1000.0
    return float(timestep) / 1000.0


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
    t_v, t_a = sigma_to_t_pair(s, shift_v, shift_a)
    return sorted({
        f32_image(t_v),
        f32_image(t_a),
        f32_image(max(t_v, visual_aug)),
        f32_image(max(t_a, audio_aug)),
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
                  extra_t=None):
    """Full t grid covered by a baked table.

    Dedup is exact on the float32 image: two doubles that round to the same
    float32 are the same row for the model, and two float32 values that differ
    stay separate even when they differ by one ulp. No decimal rounding here --
    `round(t, 7)` used to shift the baked t off the runtime t by up to 5e-8,
    which TimeEmbedder turns into a visibly different t_emb.
    """
    values = set()
    for s in normalise_sigmas(sigmas):
        for t in sigma_to_t_candidates(s, shift_v, shift_a, visual_aug, audio_aug):
            values.add(f32_image(t))
    for t in normalise_sigmas(extra_t):
        values.add(f32_image(t))
    return sorted(values)


def runtime_unique_t(sigma_v, layout, shift_v=DEFAULT_VIDEO_SHIFT,
                     shift_a=DEFAULT_AUDIO_SHIFT,
                     visual_aug=VISUAL_COND_TIMESTEP,
                     audio_aug=AUDIO_COND_TIMESTEP):
    """Reconstruct model.py:709 `unique_t` for the current forward.

    `layout` is a PackedLayout (or anything with a `.segments` list of
    (start, stop, kind)). Per-row denoise-mask timesteps are NOT recoverable
    here -- see apply_engine's fallback policy.

    This is the *fallback* path: the primary runtime mapping derives the
    timesteps from the t_emb tensor itself, which needs no reconstruction.
    """
    t_v, t_a = runtime_t_pair(sigma_v, shift_v, shift_a)
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


def build_index(table_t, decimals=9, tolerance=1e-6):
    """Map a float t to a table row index.

    Keys are the float32 image, so an exact hit is the normal case (the runtime
    feeds the same float32 value the table was recorded with). `tolerance` only
    rescues genuine fp round-trips and is deliberately tight: a loose tolerance
    silently snaps a *missing* timestep onto a neighbouring row, which is the
    failure mode that makes a wrong table look like a numerical difference.
    """
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


def missing_t_values(table_t, wanted, tolerance=1e-6):
    """Which of `wanted` have no row in `table_t` (diagnostics helper)."""
    lookup = build_index(table_t, tolerance=tolerance)
    return [float(t) for t in wanted if lookup(float(t)) is None]


def shifts_from_options(transformer_options, model):
    """model.py:672-673 honours transformer_options overrides."""
    to = transformer_options or {}
    sv = to.get("minimax_h3_sigma_shift_video",
                getattr(model, "sigma_shift_video", DEFAULT_VIDEO_SHIFT))
    sa = to.get("minimax_h3_sigma_shift_audio",
                getattr(model, "sigma_shift_audio", DEFAULT_AUDIO_SHIFT))
    return float(sv), float(sa)


def sigma_from_options(transformer_options, timestep=None):
    """Current video sigma (0..1).

    Two traps live here:
      * `transformer_options["sigmas"]` is set by comfy/samplers.py:325 to the
        *timestep* (sigma * 1000), not to sigma, and not to the schedule. The
        full schedule is under "sample_sigmas".
      * model.py:674 divides by 1000 -- do the same instead of treating the
        value as a sigma.
    """
    to = transformer_options or {}
    if timestep is not None:
        return sigma_from_timestep(timestep)
    sig = to.get("sigmas")
    if sig is not None:
        try:
            return sigma_from_timestep(sig)
        except Exception:
            pass
    raise RuntimeError("H3AdaLN: no sigma available in transformer_options")
