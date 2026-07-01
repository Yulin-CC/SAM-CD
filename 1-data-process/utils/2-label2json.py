#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: Label PNG mask → Labelme JSON 反向转换。
#   读取 label/ 目录下的二值 mask，通过轮廓检测提取 polygon 坐标，
#   生成对应的 labelme JSON 文件到 json/ 目录。
#
# @Command: python 6-labels2json.py --Path /path/to/dataset
# @Command: python 6-labels2json.py --Path /path/to/dataset --label-filter change
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ==============================
# 接口配置
# ==============================
DATA_DIR = "/path/to/dataset"
LABEL_FILTER = ""   # 例: "change"；空=默认 label "change"
LABEL_NAME = "change"  # 生成的 JSON 中 shape label
POLY_APPROX_EPS_RATIO = 0.005  # 轮廓近似精度比例
MAX_CONTOUR_AREA = 1e6    # 忽略过大的轮廓（可能是噪点）


def check_layout(data_dir: str) -> bool:
    """检查 label/ 和 json/ 目录。"""
    label_dir = os.path.join(data_dir, "label")
    if not os.path.isdir(label_dir):
        print(f"  [缺少] 无 label/：{label_dir}")
        return False
    # json/ 目录不存在则创建
    return True


def mask_to_polygons(mask: np.ndarray, label: str = "change") -> list:
    """从二值 mask 提取 polygon 轮廓，返回 labelme shapes 列表。"""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = mask.shape
    shapes = []
    total_pixels = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:  # 忽略太小（<2x2 像素）的噪点
            continue

        # 轮廓近似简化
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, POLY_APPROX_EPS_RATIO * peri, True)

        # 限制最大顶点数
        if len(approx) > 200:
            approx = cv2.approxPolyDP(
                approx, POLY_APPROX_EPS_RATIO * peri * 2, True
            )

        pts = approx.reshape(-1, 2).astype(np.float64).tolist()
        if len(pts) < 3:
            continue

        shapes.append({
            "label": label,
            "points": pts,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })
        total_pixels += area

    print(f"    提取 {len(shapes)} 个多边形，共 {total_pixels} 像素")
    return shapes


def labels_to_json(label_dir: str, json_dir: str, label_filter: str) -> dict:
    """批量 label → JSON。"""
    label_files = sorted(
        f for f in os.listdir(label_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not label_files:
        print("  未发现 label 文件。")
        return {"total": 0, "ok": 0}

    ok = 0
    for fname in tqdm(label_files, desc="  label → JSON"):
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, fname)

        # 读取 mask
        img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [跳过] 无法读取: {fname}")
            continue

        # 二值化
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 提取多边形
        label = LABEL_NAME if not label_filter else label_filter
        shapes = mask_to_polygons(binary, label=label)

        # 构造 JSON
        h, w = binary.shape
        json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": stem + ".jpg",
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
            "change_detection": True,
            "imagePathA": stem + ".jpg",
            "imagePathB": stem + ".jpg",
            "maskPath": stem + ".png",
        }

        # 写入 JSON
        json_path = os.path.join(json_dir, stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        ok += 1

    return {"total": len(label_files), "ok": ok}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label mask → Labelme JSON 反向转换")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="数据目录")
    parser.add_argument("--label-name", default=LABEL_NAME, help="生成的 JSON 中 shape 的 label 名称")
    return parser


def main(argv=None):
    global LABEL_NAME
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    LABEL_NAME = args.label_name

    print(f"数据目录：{data_dir}")
    print(f"Label 名称：{LABEL_NAME}")
    print("-" * 50)

    step, total = 1, 3

    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    if not check_layout(data_dir):
        print("  目录结构不完整，退出。")
        return

    print(f"[{step}/{total}] label → JSON...")
    step += 1
    label_dir = os.path.join(data_dir, "label")
    json_dir = os.path.join(data_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    stats = labels_to_json(label_dir, json_dir, LABEL_FILTER)
    print(f"  成功 {stats['ok']} / {stats['total']} → {json_dir}")

    print(f"[{step}/{total}] 完成")
    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
