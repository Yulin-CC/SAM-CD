#!/usr/bin/env python3
"""
Inspect a SAM-CD checkpoint (.pth): top-level structure, unwrap path, load_state_dict stats.

Expects a flat or nested **SAM_CD** ``state_dict`` (as saved by train.py), not a raw Ultralytics ``FastSAM-x.pt``.

Run from repo root:
  python z-others/inspect_sam_cd_checkpoint.py runs/0-train/.../SAM_CD_best.pth
  python z-others/inspect_sam_cd_checkpoint.py weights/Levir_CD/SAM_CD_e42_....pth

Compare two checkpoints:
  python z-others/inspect_sam_cd_checkpoint.py path/a.pth --compare path/b.pth
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from models.SAM_CD import SAM_CD
from utils.checkpoint import strip_module_prefix, torch_load_compat, unpack_checkpoint_state_dict


def _describe_raw(raw, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"type: {type(raw).__name__}")
    if isinstance(raw, dict):
        print(f"top-level len: {len(raw)}")
        print(f"top-level keys (first 30): {list(raw.keys())[:30]}")
        n_tensor = sum(1 for v in raw.values() if isinstance(v, torch.Tensor))
        print(f"top-level tensor entries: {n_tensor} / {len(raw)}")


def _inspect_one(path: str, label: str) -> dict:
    raw = torch_load_compat(path, map_location="cpu")
    _describe_raw(raw, label)

    inner = unpack_checkpoint_state_dict(raw, path_hint=path)
    stripped = strip_module_prefix(inner)
    total_bytes = sum(v.numel() * v.element_size() for v in stripped.values())
    print(f"unpacked param tensors: {len(stripped)}, ~{total_bytes / 1024 / 1024:.1f} MiB")

    net = SAM_CD()
    inc = net.load_state_dict(stripped, strict=False)
    print(f"missing_keys: {len(inc.missing_keys)}")
    if inc.missing_keys:
        print(f"  first 12: {inc.missing_keys[:12]}")
        if all(k.startswith("model.") for k in inc.missing_keys):
            print(
                "  (All missing keys are under `model.*` — typical for heads-only / "
                "weights/Levir_CD release checkpoints; backbone comes from FastSAM-x.pt.)"
            )
    print(f"unexpected_keys: {len(inc.unexpected_keys)}")
    if inc.unexpected_keys:
        print(f"  first 12: {inc.unexpected_keys[:12]}")
    return stripped


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect SAM-CD checkpoint vs SAM_CD()")
    p.add_argument("chkpt_path", help="Path to .pth")
    p.add_argument("--compare", default=None, help="Optional second .pth to compare key sets")
    args = p.parse_args()

    if not os.path.isfile(args.chkpt_path):
        sys.exit(f"Not a file: {args.chkpt_path}")

    sd_a = _inspect_one(args.chkpt_path, args.chkpt_path)

    if args.compare:
        if not os.path.isfile(args.compare):
            sys.exit(f"Not a file: {args.compare}")
        sd_b = _inspect_one(args.compare, args.compare)
        keys_a, keys_b = set(sd_a), set(sd_b)
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        print("\n=== key set diff ===")
        print(f"only in first: {len(only_a)}")
        if only_a[:20]:
            print(f"  sample: {only_a[:20]}")
        print(f"only in second: {len(only_b)}")
        if only_b[:20]:
            print(f"  sample: {only_b[:20]}")


if __name__ == "__main__":
    main()
