"""ONNX 导出辅助：CD Head 包装 + 单文件合并。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDHeadExport(nn.Module):
    """
    仅导出 SAM_CD 中 Adapter + Decoder + SA + resCD + headC + segmenterC。
    输入为双时相各 4 层 FastSAM 特征（与 ms_feats 同序）。
    """

    def __init__(self, sam_cd: nn.Module, out_h: int, out_w: int):
        super().__init__()
        self.out_h = int(out_h)
        self.out_w = int(out_w)

        self.Adapter32 = sam_cd.Adapter32
        self.Adapter16 = sam_cd.Adapter16
        self.Adapter8 = sam_cd.Adapter8
        self.Adapter4 = sam_cd.Adapter4
        self.Dec2 = sam_cd.Dec2
        self.Dec1 = sam_cd.Dec1
        self.Dec0 = sam_cd.Dec0
        self.segmenter = sam_cd.segmenter
        self.SA = sam_cd.SA
        self.resCD = sam_cd.resCD
        self.headC = sam_cd.headC
        self.segmenterC = sam_cd.segmenterC

    def _decode_branch(self, feat0, feat1, feat2, feat3):
        feat_s4 = self.Adapter4(feat3)
        feat_s8 = self.Adapter8(feat0)
        feat_s16 = self.Adapter16(feat1)
        feat_s32 = self.Adapter32(feat2)
        dec_2 = self.Dec2(feat_s32, feat_s16)
        dec_1 = self.Dec1(dec_2, feat_s8)
        dec_0 = self.Dec0(dec_1, feat_s4)
        return dec_0

    def forward(
        self,
        featA0,
        featA1,
        featA2,
        featA3,
        featB0,
        featB1,
        featB2,
        featB3,
    ):
        decA_0 = self._decode_branch(featA0, featA1, featA2, featA3)
        decB_0 = self._decode_branch(featB0, featB1, featB2, featB3)
        outA = self.segmenter(decA_0)
        outB = self.segmenter(decB_0)
        attn = self.SA(torch.cat([outA, outB], dim=1))
        featC = torch.cat([decA_0, decB_0], dim=1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * attn
        outC = self.segmenterC(featC)
        return F.interpolate(
            outC, size=(self.out_h, self.out_w), mode="bilinear", align_corners=True
        )


def merge_external_data_to_single_file(onnx_path: str | Path) -> Path:
    """
    将 .onnx + .onnx.data 合并为单个自包含 .onnx 文件，并删除 .data 伴生文件。

    PyTorch 2.x 默认导出会把大权重写到 external data，推理需成对拷贝；
    合并后只需交付一个 .onnx 文件。
    """
    import onnx

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX not found: {path}")

    model = onnx.load(str(path), load_external_data=True)
    tmp_path = path.with_name(path.stem + ".__merge_tmp__.onnx")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(path)

    data_path = Path(f"{path}.data")
    if data_path.exists():
        data_path.unlink()

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Merged single-file ONNX: {path} ({size_mb:.1f} MiB)")
    return path


__all__ = ["CDHeadExport", "merge_external_data_to_single_file"]
