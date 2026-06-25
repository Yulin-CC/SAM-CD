"""
SAM-CD ONNX 两阶段 pipeline：FastSAM 四层特征 + CD Head。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


FEAT_NAMES = ("feat_l0", "feat_l1", "feat_l2", "feat_l3")
VIS_ALPHA = 0.45
VIS_HIGHLIGHT_BGR = (0, 0, 255)


def hier_feats_from_onnx_outputs(outs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    FastSAM ONNX 4 输出 → 与 PyTorch ms_feats 同序: [y15, y18, y21, y1]。
    """
    if len(outs) != 4:
        raise ValueError(f"FastSAM ONNX 期望 4 个输出，实际 {len(outs)}")
    return list(outs)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logits_to_mask(logit: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    prob = sigmoid(logit.astype(np.float32))
    if prob.ndim == 4:
        prob = prob[0, 0]
    elif prob.ndim == 3:
        prob = prob[0]
    return (prob > threshold).astype(np.uint8)


def create_crops(img_a: np.ndarray, img_b: np.ndarray, size: Tuple[int, int]):
    """与 0-QuickStart/script/inference.py 一致的滑窗裁剪。"""
    h, w = img_a.shape[:2]
    c_h, c_w = size
    if h < c_h or w < c_w:
        return [img_a], [img_b]

    h_rate = h / c_h
    w_rate = w / c_w
    rows = math.ceil(h_rate)
    cols = math.ceil(w_rate)
    stride_h = int((c_h * rows - h) / (rows - 1)) if rows > 1 else 0
    stride_w = int((c_w * cols - w) / (cols - 1)) if cols > 1 else 0

    crops_a, crops_b = [], []
    for j in range(rows):
        for i in range(cols):
            s_h = int(j * c_h - j * stride_h)
            if j == (rows - 1):
                s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i * c_w - i * stride_w)
            if i == (cols - 1):
                s_w = w - c_w
            e_w = s_w + c_w
            crops_a.append(img_a[s_h:e_h, s_w:e_w, :])
            crops_b.append(img_b[s_h:e_h, s_w:e_w, :])
    return crops_a, crops_b


def stitch_pred(patch_list: List[np.ndarray], size_stitch: Tuple[int, int]) -> np.ndarray:
    """拼接滑窗预测块。"""
    h, w = size_stitch
    ph, pw = patch_list[0].shape
    stitch_rows = math.ceil(h / ph)
    stitch_cols = math.ceil(w / pw)
    assert stitch_rows * stitch_cols == len(patch_list)

    h_overlap = int((ph * stitch_rows - h) / (stitch_rows - 1)) if stitch_rows > 1 else 0
    w_overlap = int((pw * stitch_cols - w) / (stitch_cols - 1)) if stitch_cols > 1 else 0

    stitched_img = None
    for r in range(stitch_rows):
        crop_t = math.ceil(h_overlap / 2)
        crop_b = h_overlap - crop_t
        crop_l = math.ceil(w_overlap / 2)
        crop_r = w_overlap - crop_l
        if r == 0:
            crop_t = 0
        if r == stitch_rows - 1:
            crop_b = 0
            crop_t = stitched_img.shape[0] - h if stitched_img is not None else 0

        row_patch = patch_list[r * stitch_cols][crop_t : ph - crop_b, 0 : pw - crop_r]
        for c in range(1, stitch_cols):
            if c == stitch_cols - 1:
                crop_r = 0
                crop_l = row_patch.shape[1] - w
            patch_croped = patch_list[r * stitch_cols + c][
                crop_t : ph - crop_b, crop_l : pw - crop_r
            ]
            row_patch = np.concatenate((row_patch, patch_croped), axis=1)

        if r == 0:
            stitched_img = row_patch
        else:
            stitched_img = np.concatenate((stitched_img, row_patch), axis=0)
    return stitched_img


def overlay_vis(img_bgr, mask):
    """与 0-QuickStart/scripts/inference.py 一致的可视化叠加。"""
    m = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    region = m > 127
    out = img_bgr.copy().astype(np.float32)
    if not np.any(region):
        return img_bgr.copy()
    color = np.array(VIS_HIGHLIGHT_BGR, dtype=np.float32)
    out[region] = out[region] * (1 - VIS_ALPHA) + color * VIS_ALPHA
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, VIS_HIGHLIGHT_BGR, 1)
    return out


def save_vismask(mask_uint8: np.ndarray, curr_bgr: np.ndarray, out_path: str) -> None:
    """兼容旧接口：mask → vismask PNG。"""
    import cv2
    vis = overlay_vis(curr_bgr, mask_uint8)
    cv2.imwrite(out_path, vis)
