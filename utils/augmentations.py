"""
SAM-CD 数据增强工具集。
"""

import math
import random
import numpy as np
import cv2


def rand_crop_CD(img1, img2, label, size):
    """随机裁剪 512×512 块。"""
    h, w = img1.shape[:2]
    c_h, c_w = size
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
        return img1, img2, label
    s_h = random.randint(0, h - c_h)
    e_h = s_h + c_h
    s_w = random.randint(0, w - c_w)
    e_w = s_w + c_w
    return (img1[s_h:e_h, s_w:e_w, :],
            img2[s_h:e_h, s_w:e_w, :],
            label[s_h:e_h, s_w:e_w])


def pos_aware_rand_crop_CD(img1, img2, label, size, pos_ratio=0.5):
    """
    以 pos_ratio 概率在包含正像素的区域附近随机裁剪，缓解正负样本不均衡。
    若 label 中无正像素，则退化为纯随机裁剪。
    """
    h, w = img1.shape[:2]
    c_h, c_w = size
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
        return img1, img2, label

    pos_coords = np.argwhere(label > 0)
    if len(pos_coords) > 0 and random.random() < pos_ratio:
        anchor_y, anchor_x = pos_coords[random.randint(0, len(pos_coords) - 1)]
        offset_y = random.randint(-c_h // 4, c_h // 4)
        offset_x = random.randint(-c_w // 4, c_w // 4)
        center_y = anchor_y + offset_y
        center_x = anchor_x + offset_x
        s_h = center_y - c_h // 2
        s_w = center_x - c_w // 2
    else:
        s_h = random.randint(0, h - c_h)
        s_w = random.randint(0, w - c_w)

    s_h = max(0, min(s_h, h - c_h))
    s_w = max(0, min(s_w, w - c_w))
    e_h = s_h + c_h
    e_w = s_w + c_w
    return (img1[s_h:e_h, s_w:e_w, :],
            img2[s_h:e_h, s_w:e_w, :],
            label[s_h:e_h, s_w:e_w])


def rand_flip_CD(img1, img2, label, p=0.75):
    """
    随机水平/垂直/对角翻转（A/B/label 同步）。
    p: 触发翻转的概率（0.0~1.0），方向在三种翻转中均匀随机。
    """
    if random.random() >= p:
        return img1, img2, label
    direction = random.randint(0, 2)
    if direction == 0:
        return (np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(),
                np.flip(label, axis=0).copy())
    elif direction == 1:
        return (np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(),
                np.flip(label, axis=1).copy())
    else:
        return (img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(),
                label[::-1, ::-1].copy())


def random_scale_CD(img1, img2, label, p=0.5, scale_limit=0.1, shift_limit=0.125, rotate_limit=45):
    """
    随机平移 + 缩放 + 旋转（ShiftScaleRotate 等价实现），A/B/label 使用同一变换。
    在训练时裁剪后进行，50% 概率触发。
    """
    if random.random() >= p:
        return img1, img2, label

    h, w = img1.shape[:2]
    center = (w / 2, h / 2)

    angle = random.uniform(-rotate_limit, rotate_limit)
    scale = random.uniform(1.0 - scale_limit, 1.0 + scale_limit)
    dx = random.uniform(-shift_limit, shift_limit) * w
    dy = random.uniform(-shift_limit, shift_limit) * h

    mat = cv2.getRotationMatrix2D(center, angle, scale)
    mat[:, 2] += [dx, dy]

    flags_img = cv2.INTER_LINEAR
    flags_lbl = cv2.INTER_NEAREST
    border_val = (0, 0, 0)

    img1_out = cv2.warpAffine(img1, mat, (w, h), flags=flags_img, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    img2_out = cv2.warpAffine(img2, mat, (w, h), flags=flags_img, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    label_out = cv2.warpAffine(label, mat, (w, h), flags=flags_lbl, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return img1_out, img2_out, label_out
