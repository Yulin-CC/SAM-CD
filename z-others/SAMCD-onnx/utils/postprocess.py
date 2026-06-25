"""推理后处理：logit → mask PNG。"""

import numpy as np

from utils.samcd_pipeline import logits_to_mask, save_vismask


def save_prediction(mask_uint8: np.ndarray, save_path: str) -> None:
    """保存 0/1 mask 为 PNG（0/255）。"""
    from skimage import io

    io.imsave(save_path, (mask_uint8 * 255).astype(np.uint8))


__all__ = ["logits_to_mask", "save_prediction", "save_vismask"]
