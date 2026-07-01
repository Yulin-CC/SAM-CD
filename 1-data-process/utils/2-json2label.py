"""
# @Author: 算法组
# @Date: 2026-05-18
# @Description: Labelme JSON 转二值 mask：读取 GEOAI 目录 json/ 中 polygon 标注，
#   输出至同目录 label/（与 A/B 样本 stem 同名 .png）。
# @Filter: LABEL_FILTER 为空表示保留全部类别；否则仅绘制指定 label（逗号分隔）。
# @Compat: Windows 含中文路径用 imencode + tofile 写 png。
# @Command: python 1-labels2mask.py
"""

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

#==============================#
# 接口配置
#==============================#
DATA_DIR = r"D:\0-data\1-ChangeDetect\cache\黑龙镇\GEOAI-((ChangeD_Fuxingshuiku))-Selfcollect-2605-(D)(OL)"
LABEL_FILTER = ""   # 例: "change" 或 "change,building"；空=全部类别
#==============================#


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Windows 含中文路径写图。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        ok, enc = cv2.imencode(".png", img)
    else:
        ok, enc = cv2.imencode(".png", img)
    if not ok or enc is None:
        return False
    try:
        enc.tofile(path)
    except OSError:
        return False
    return True


def parse_label_filter(s: str) -> set | None:
    """解析类别过滤字符串。"""
    if not s or not s.strip():
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def check_geoai_layout(data_dir: str) -> bool:
    """检查 json/ 目录是否存在。"""
    json_dir = os.path.join(data_dir, "json")
    if not os.path.isdir(json_dir):
        print(f"  [缺少] 无 json/：{json_dir}")
        return False
    return True


def json_to_binary_mask(json_path: str, out_path: str, label_filter: set | None = None) -> bool:
    """将单个 Labelme JSON 转为二值 mask PNG。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    h = int(data.get("imageHeight", 0))
    w = int(data.get("imageWidth", 0))
    if h <= 0 or w <= 0:
        print(f"  [跳过] 无效尺寸: {os.path.basename(json_path)}")
        return False

    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "polygon")
        label = shape.get("label", "")
        points = shape.get("points", [])

        if label_filter and label not in label_filter:
            continue
        if shape_type != "polygon" or len(points) < 3:
            continue

        pts = np.array(points, dtype=np.float32).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], color=255)

    return imwrite_unicode(out_path, mask)


def convert_all(data_dir: str, label_filter: set | None = None) -> dict:
    """批量 json → label/mask。"""
    json_dir = os.path.join(data_dir, "json")
    label_dir = os.path.join(data_dir, "label")
    os.makedirs(label_dir, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(json_dir) if f.lower().endswith(".json")
    )
    if not json_files:
        print("  未发现 json 文件。")
        return {"total": 0, "ok": 0}

    ok = 0
    for fname in tqdm(json_files, desc="  json → mask"):
        stem = os.path.splitext(fname)[0]
        src = os.path.join(json_dir, fname)
        dst = os.path.join(label_dir, stem + ".png")
        if json_to_binary_mask(src, dst, label_filter=label_filter):
            ok += 1

    return {"total": len(json_files), "ok": ok}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Labelme JSON 转二值 mask")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="GEOAI 数据目录")
    parser.add_argument("--label-filter", default=LABEL_FILTER, help="保留的 label，逗号分隔；空=全部")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    label_filter_str = args.label_filter
    print(f"数据目录：{data_dir}")
    print(f"类别过滤：{label_filter_str or '(全部)'}")
    print("-" * 50)

    step, total = 1, 3

    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    if not check_geoai_layout(data_dir):
        print("  目录结构不完整，退出。")
        return

    print(f"[{step}/{total}] 转换 mask...")
    step += 1
    label_filter = parse_label_filter(label_filter_str)
    stats = convert_all(data_dir, label_filter=label_filter)
    print(f"  成功 {stats['ok']} / {stats['total']} → {os.path.join(data_dir, 'label')}")

    print(f"[{step}/{total}] 完成")
    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
