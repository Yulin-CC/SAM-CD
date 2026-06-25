"""
ONNX 推理预处理：LetterBox + /255，对齐 Ultralytics FastSAM 推理输入。
"""

import cv2
import numpy as np


def letterbox_image(img_bgr, target_size=1024, stride=32):
    """BGR → letterboxed BGR（与 ultralytics LetterBox 行为一致）。"""
    h, w = img_bgr.shape[:2]
    r = min(target_size / h, target_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = target_size - new_unpad[0]
    dh = target_size - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr.copy()

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    if img.shape[0] % stride:
        pad_h = stride - (img.shape[0] % stride)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    if img.shape[1] % stride:
        pad_w = stride - (img.shape[1] % stride)
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    meta = {
        "scale": r,
        "orig_h": h,
        "orig_w": w,
        "input_h": img.shape[0],
        "input_w": img.shape[1],
        "pad_top": top,
        "pad_left": left,
        "new_unpad": new_unpad,
    }
    return img, meta


def preprocess(img_bgr, target_size=1024):
    """BGR image → ((1,3,H,W) float32 RGB normalized, letterbox_meta)."""
    img, meta = letterbox_image(img_bgr, target_size=target_size)
    tensor = img[..., ::-1].transpose(2, 0, 1)[None] / 255.0
    return tensor.astype(np.float32), meta


def normalize_rgb(img_rgb):
    """RGB float/uint8 → [0,1] float32 HWC。"""
    arr = img_rgb.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr
