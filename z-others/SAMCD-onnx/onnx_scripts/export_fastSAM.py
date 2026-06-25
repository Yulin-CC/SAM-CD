#!/usr/bin/env python3
"""
SAM-CD 试验包：导出 FastSAM ONNX（仅 4 层特征，无 detect/proto）。

前置: 在项目根目录执行 bash install_tbr.sh

用法（在 SAMCD-onnx 根目录）:
  bash 0-export_fastSAM.sh
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from argparse import Namespace
from pathlib import Path

ONNX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ONNX_ROOT.parent.parent
for p in (REPO_ROOT, ONNX_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def resolve_path(path, base: Path = ONNX_ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def verify_samcd_tbr_installed() -> None:
    from ultralytics.nn.modules.head import Segment
    from ultralytics.nn.tasks import BaseModel

    head_src = inspect.getsource(Segment.forward)
    tasks_src = inspect.getsource(BaseModel._predict_once)
    missing = []
    if "tuple(ms_feats)" not in head_src:
        missing.append("Segment.forward export 未返回 tuple(ms_feats)")
    if "m.f = [15, 18, 21, 1]" not in tasks_src:
        missing.append("BaseModel._predict_once 缺少 Segment.m.f 补丁")
    if missing:
        raise RuntimeError(
            "未检测到 SAM-CD tbr 补丁，请先在项目根目录运行:\n  bash install_tbr.sh\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def parse_imgsz(value):
    parts = [int(part) for part in str(value).replace(",", " ").split()]
    return parts[0] if len(parts) == 1 else parts


def merge_args_with_yaml(args: Namespace, yaml_path: str) -> Namespace:
    import yaml

    if not os.path.exists(yaml_path):
        return args
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    export_cfg = cfg.get("export", {})
    for key, val in export_cfg.items():
        if key == "fastsam_onnx_output":
            if args.output is None:
                args.output = val
            continue
        if key == "sam_weights":
            if args.sam_weights is None:
                args.sam_weights = val
            continue
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)
    return args


def export_to_onnx(args: Namespace) -> None:
    weight_path = resolve_path(args.sam_weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"FastSAM 权重不存在: {weight_path}")

    verify_samcd_tbr_installed()

    import torch
    from ultralytics import YOLO

    _origin_torch_load = torch.load

    def _compat_torch_load(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _origin_torch_load(f, *args, **kwargs)

    torch.load = _compat_torch_load

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weight_path))
    raw_model = model.model
    raw_model.eval().to(device)

    for m in raw_model.modules():
        if hasattr(m, "export"):
            m.export = True
            m.format = "onnx"

    imgsz = args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
    dummy = torch.randn(1, 3, *imgsz).to(device)

    output_path = resolve_path(args.output) if args.output else weight_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_names = ["feat_l0", "feat_l1", "feat_l2", "feat_l3"]
    dynamic_axes = {
        "images": {0: "batch", 2: "height", 3: "width"},
        "feat_l0": {0: "batch", 2: "height", 3: "width"},
        "feat_l1": {0: "batch", 2: "height", 3: "width"},
        "feat_l2": {0: "batch", 2: "height", 3: "width"},
        "feat_l3": {0: "batch", 2: "height", 3: "width"},
    }

    dynamic = bool(getattr(args, "dynamic", False))
    simplify = bool(getattr(args, "simplify", False))
    opset = int(getattr(args, "opset", 16))

    torch.onnx.export(
        raw_model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic else None,
        opset_version=opset,
        do_constant_folding=True,
    )

    from onnx_scripts.onnx_util import merge_external_data_to_single_file

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

    print(f"ONNX model exported to: {output_path}")
    print(f"  Outputs: {output_names}")
    print("  feat order: [y15, y18, y21, y1] (same as PyTorch ms_feats)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export FastSAM to ONNX (4 feature maps only).")
    p.add_argument("--sam_weights", default=None, help="FastSAM .pt weights")
    p.add_argument("-o", "--output", default=None, help="Output ONNX path")
    p.add_argument("--config", default=str(ONNX_ROOT / "config" / "default.yaml"))
    p.add_argument("--imgsz", type=parse_imgsz, default=None)
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
        if args.no_dynamic:
            args.dynamic = False
        elif "--dynamic" not in sys.argv:
            args.dynamic = bool(
                exp_cfg.get("fastsam_dynamic", exp_cfg.get("dynamic", False))
            )
        if args.no_simplify:
            args.simplify = False
        elif "--simplify" not in sys.argv:
            args.simplify = bool(exp_cfg.get("simplify", False))

    print("=" * 56)
    print("  SAM-CD FastSAM ONNX 导出")
    print(f"  权重:   {resolve_path(args.sam_weights)}")
    print(f"  输出:   {resolve_path(args.output) if args.output else '(yaml)'}")
    print(f"  配置:   {args.config}")
    print(f"  imgsz={args.imgsz} | dynamic={args.dynamic} | simplify={args.simplify} | opset={args.opset}")
    print("=" * 56)

    export_to_onnx(args)


if __name__ == "__main__":
    main()
