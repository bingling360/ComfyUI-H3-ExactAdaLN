#!/usr/bin/env python3
"""Bitwise compare two ComfyUI SaveLatent files.

The verification workflow saves A/B latents as
    output/latents/verify_a_00001_.latent
    output/latents/verify_b_00001_.latent

Usage:
    python3 compare_latents.py verify_a_00001_.latent verify_b_00001_.latent
    python3 compare_latents.py --auto /root/ComfyUI/output/latents

Exit code 0 = PASS, 1 = FAIL, 2 = usage/IO error.
"""

import glob
import os
import sys

import torch
from safetensors.torch import load_file


def pick(directory, prefix):
    hits = sorted(
        glob.glob(os.path.join(directory, "**", prefix + "*"), recursive=True),
        key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--auto":
        directory = args[1] if len(args) > 1 else os.path.join("output", "latents")
        path_a = pick(directory, "verify_a")
        path_b = pick(directory, "verify_b")
        if not path_a or not path_b:
            print("could not find verify_a / verify_b under {}".format(directory))
            return 2
    else:
        if len(args) != 2:
            print(__doc__)
            return 2
        path_a, path_b = args

    a, b = load_file(path_a), load_file(path_b)
    print("A: {} ({} bytes)".format(path_a, os.path.getsize(path_a)))
    print("B: {} ({} bytes)".format(path_b, os.path.getsize(path_b)))
    print("-" * 60)

    all_equal = True
    compared = 0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print("{}: MISSING on one side".format(key))
            all_equal = False
            continue
        x, y = a[key], b[key]
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            print("{}: non-tensor, skipped".format(key))
            continue
        if x.shape != y.shape:
            print("{}: SHAPE MISMATCH {} vs {}".format(key, tuple(x.shape), tuple(y.shape)))
            all_equal = False
            continue
        eq = torch.equal(x, y)
        delta = (x.float() - y.float()).abs()
        print("{}: equal={} max_abs={:.3e} mean_abs={:.3e} shape={}".format(
            key, eq, float(delta.max()), float(delta.mean()), tuple(x.shape)))
        all_equal = all_equal and eq
        compared += 1

    print("-" * 60)
    print("compared tensor pairs: {}".format(compared))
    if compared == 0:
        print("VERDICT: INCONCLUSIVE - no tensors in these files")
        return 2
    print("VERDICT: {}".format(
        "PASS - bitwise identical" if all_equal else "FAIL - outputs differ"))
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
