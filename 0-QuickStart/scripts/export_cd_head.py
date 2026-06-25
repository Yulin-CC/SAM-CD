#!/usr/bin/env python3
"""
SAM-CD CD Head ONNX 导出（8 路特征输入 → change_logit）。

用法:
  bash 0-QuickStart/4-export.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

QUICKSTART_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICKSTART_ROOT.parent
QS_UTILS = QUICKSTART_ROOT / "utils"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QS_UTILS) not in sys.path:
    sys.path.insert(0, str(QS_UTILS))

from models.SAM_CD import SAM_CD
from utils.checkpoint import strip_module_prefix, torch_load_compat, unpack_checkpoint_state_dict
import onnx_util
from onnx_util import CDHeadExport


def resolve_path(path, base: Path = REPO_ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def merge_args_with_yaml(args: Namespace, yaml_path: str) -> Namespace:
    import yaml

    if not os.path.exists(yaml_path):
        return args
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    export_cfg = cfg.get("export", {})
    predict_cfg = cfg.get("predict", {})
    for key, val in export_cfg.items():
        if key == "cd_head_onnx_output":
            if args.output is None:
                args.output = val
            continue
        if key == "cd_head_weights":
            if args.cd_head_weights is None:
                args.cd_head_weights = val
            continue
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)
    if args.cd_head_weights is None and predict_cfg.get("chkpt_path"):
        args.cd_head_weights = predict_cfg["chkpt_path"]
    return args


def _feature_dummy_sizes(imgsz: int):
    return [
        (1, 320, imgsz // 8, imgsz // 8),
        (1, 640, imgsz // 16, imgsz // 16),
        (1, 640, imgsz // 32, imgsz // 32),
        (1, 160, imgsz // 4, imgsz // 4),
    ]


def export_to_onnx(args: Namespace) -> None:
    import torch

    ckpt_path = resolve_path(args.cd_head_weights)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CD Head 权重不存在: {ckpt_path}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = int(args.fastsam_imgsz or args.imgsz or 512)
    crop_size = int(args.crop_size or 512)

    net = SAM_CD()
    raw = torch_load_compat(str(ckpt_path), map_location="cpu")
    state = strip_module_prefix(unpack_checkpoint_state_dict(raw, path_hint=str(ckpt_path)))
    incompatible = net.load_state_dict(state, strict=False)
    cd_missing = [k for k in incompatible.missing_keys if not k.startswith("model.")]
    if cd_missing:
        raise RuntimeError(f"CD Head 权重不完整，缺失: {cd_missing[:8]}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"CD Head 存在 unexpected keys: {incompatible.unexpected_keys[:8]}")

    wrapper = CDHeadExport(net, out_h=crop_size, out_w=crop_size).eval().to(device)

    feat_shapes = _feature_dummy_sizes(imgsz)
    dummy = [torch.randn(shape, device=device) for shape in feat_shapes]
    dummy_b = [torch.randn(shape, device=device) for shape in feat_shapes]
    dummy_inputs = tuple(dummy + dummy_b)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = [
        "featA_l0", "featA_l1", "featA_l2", "featA_l3",
        "featB_l0", "featB_l1", "featB_l2", "featB_l3",
    ]
    dynamic = bool(getattr(args, "dynamic", False))
    dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
    dynamic_axes["change_logit"] = {0: "batch", 2: "height", 3: "width"}

    opset = int(getattr(args, "opset", 16))
    simplify = bool(getattr(args, "simplify", False))

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=["change_logit"],
        dynamic_axes=dynamic_axes if dynamic else None,
        opset_version=opset,
        do_constant_folding=True,
    )

    merge_external_data_to_single_file = onnx_util.merge_external_data_to_single_file

    if simplify:
        try:
            import onnx
            import onnxsim

            model_onnx = onnx.load(str(output_path), load_external_data=True)
            model_simp, check = onnxsim.simplify(model_onnx)
            assert check, "ONNX simplify check failed"
            onnx.save(model_simp, str(output_path))
            print(f"  ONNX simplified: {output_path}")
        except ImportError:
            print("  ⚠️ onnxsim 未安装，跳过 simplify")

    merge_external_data_to_single_file(output_path)

    print(f"CD Head ONNX exported to: {output_path}")
    print(f"  Inputs:  {input_names}")
    print(f"  Output:  change_logit [{1}, 1, {crop_size}, {crop_size}]")
    print(f"  Dummy feat spatial based on fastsam_imgsz={imgsz}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export SAM-CD CD Head to ONNX.")
    p.add_argument("--cd_head_weights", default=None, help="best.pth (heads_only)")
    p.add_argument("-o", "--output", default=None, help="Output ONNX path")
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "defualt.yaml"))
    p.add_argument("--imgsz", type=int, default=None, help="FastSAM 输入边长（特征 dummy 用）")
    p.add_argument("--fastsam_imgsz", type=int, default=None)
    p.add_argument("--crop_size", type=int, default=None, help="CD Head 输出 H=W")
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--no_dynamic", action="store_true")
    p.add_argument("--no_simplify", action="store_true")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = merge_args_with_yaml(args, args.config)

    if os.path.exists(args.config):
        import yaml

        exp_cfg = yaml.safe_load(open(args.config, encoding="utf-8")).get("export", {})
        if args.fastsam_imgsz is None:
            args.fastsam_imgsz = exp_cfg.get("imgsz", 512)
        if args.no_dynamic:
            args.dynamic = False
        elif "--dynamic" not in sys.argv:
            args.dynamic = bool(exp_cfg.get("cd_head_dynamic", False))
        if args.no_simplify:
            args.simplify = False
        elif "--simplify" not in sys.argv:
            args.simplify = bool(exp_cfg.get("simplify", False))

    print("=" * 56)
    print("  SAM-CD CD Head ONNX 导出")
    print(f"  权重:   {resolve_path(args.cd_head_weights)}")
    print(f"  输出:   {resolve_path(args.output) if args.output else '(yaml)'}")
    print(f"  crop_size={args.crop_size} | fastsam_imgsz={args.fastsam_imgsz or args.imgsz}")
    print("=" * 56)

    export_to_onnx(args)


if __name__ == "__main__":
    main()
