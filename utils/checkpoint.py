"""Shared checkpoint loading helpers for SAM-CD."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Mapping, Union

import torch


def torch_load_compat(path: str, map_location: Union[str, torch.device] = "cpu") -> Any:
    """torch.load with weights_only=False when supported (PyTorch 2.4+)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _is_plain_state_dict(d: Mapping[str, Any]) -> bool:
    if not d:
        return False
    for v in d.values():
        if not isinstance(v, torch.Tensor):
            return False
    return True


def unpack_checkpoint_state_dict(raw: Any, path_hint: str = "") -> Dict[str, torch.Tensor]:
    """
    Return a flat parameter dict from torch.load output.

    Supports:
    - Plain state_dict (str keys -> Tensor)
    - ``{'state_dict': ...}`` training checkpoints
    - ``{'model': <state_dict>}`` when values are tensors
    """
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected dict checkpoint from {path_hint!r}, got {type(raw).__name__}"
        )

    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        inner = raw["state_dict"]
        if _is_plain_state_dict(inner):
            return dict(inner)

    if "model" in raw and isinstance(raw["model"], dict):
        inner = raw["model"]
        if _is_plain_state_dict(inner):
            return dict(inner)

    if _is_plain_state_dict(raw):
        return dict(raw)

    keys_preview = list(raw.keys())[:25]
    raise ValueError(
        f"Could not extract a tensor state_dict from {path_hint!r}. "
        f"Top-level keys (up to 25): {keys_preview}. "
        "This script expects a SAM_CD state_dict (e.g. SAM_CD_best.pth), not a raw Ultralytics FastSAM .pt."
    )


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Remove ``module.`` prefix from DataParallel / DDP checkpoints."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def prepare_checkpoint_for_save(
    state_dict: Mapping[str, torch.Tensor],
    mode: str = "heads_only",
    strip_dp_prefix: bool = True,
) -> "OrderedDict[str, torch.Tensor]":
    """
    Build the tensor dict to ``torch.save`` for checkpoints.

    Parameters
    ----------
    mode:
        - ``heads_only`` — drop ``model.*`` (frozen FastSAM / Ultralytics backbone). Matches the
          on-disk layout of released ``weights/Levir_CD/SAM_CD_*.pth``: small file, loads in
          ``inference.py`` with ``strict=False`` while backbone comes from local ``FastSAM-x.pt``.
        - ``full`` — save every parameter (legacy / full finetune snapshot).
    """
    sd = strip_module_prefix(state_dict) if strip_dp_prefix else OrderedDict(state_dict)
    if mode == "full":
        return OrderedDict((k, v.detach().cpu()) for k, v in sd.items())
    if mode != "heads_only":
        raise ValueError(f"Unknown checkpoint save mode: {mode!r} (use 'heads_only' or 'full')")

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in sd.items():
        if k.startswith("model."):
            continue
        out[k] = v.detach().cpu()
    return out
