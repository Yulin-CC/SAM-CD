"""
# @Author: 算法组
# @Date: 2026-05-29
# @Description: 变化检测图像自适应切片，同步处理 label / json。
#   自动计算步长使所有 tile 恰好铺满图像，不溢出、不补边。
#   若存在 label/ 目录则同步裁切 label mask；
#   若存在 json/ 目录则同步生成 tile 级 JSON（过滤/偏移多边形坐标）。
# @Naming : 图像原名_s001.jpg, 图像原名_s002.jpg ...
# @Command: python 3-image_tiling.py
"""

import os
import math
import json
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

#==============================#
# 接口配置
#==============================#
INPUT_DIR = r"D:\0-data\1-ChangeDetect\cahce-add\GEOAI-((ChangeD_Luoboxiang))-Selfcollect-2605-(D)(OL)"
OUTPUT_DIR = r"D:\0-data\1-ChangeDetect\cahce-add\GEOAI-((ChangeD_Luoboxiang))-Selfcollect-2605-(D)(OL)\tiles_1024"

TILE_SIZE = 1024          # 切片尺寸（正方形）
SUBS = ["A", "B"]         # 基础子目录（label / json 自动检测）
#==============================#

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _compute_layout(length: int) -> tuple[int, float]:
    """计算单方向上的 tile 数量及自适应步长。"""
    n = max(1, math.ceil(length / TILE_SIZE))
    stride = (length - TILE_SIZE) / (n - 1) if n > 1 else float(TILE_SIZE)
    return n, stride


def _iter_tile_positions(w: int, h: int):
    """生成器：为图像尺寸 (w, h) 产出 (全局索引, x, y) 三元组，步长自适应。"""
    cols, stride_x = _compute_layout(w)
    rows, stride_y = _compute_layout(h)
    idx = 0
    for row in range(rows):
        y = round(row * stride_y)
        y = min(y, h - TILE_SIZE)
        for col in range(cols):
            x = round(col * stride_x)
            x = min(x, w - TILE_SIZE)
            idx += 1
            yield idx, x, y


def _polygon_overlaps(points, x, y) -> bool:
    """判断多边形是否与矩形 [x, x+TILE_SIZE] x [y, y+TILE_SIZE] 重叠。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    poly_xmin, poly_xmax = min(xs), max(xs)
    poly_ymin, poly_ymax = min(ys), max(ys)
    tile_xmin, tile_xmax = x, x + TILE_SIZE
    tile_ymin, tile_ymax = y, y + TILE_SIZE
    return not (poly_xmax <= tile_xmin or poly_xmin >= tile_xmax or
                poly_ymax <= tile_ymin or poly_ymin >= tile_ymax)


def _offset_and_clip_polygon(points, x, y) -> list | None:
    """平移多边形坐标到 tile 局部坐标，用 Sutherland-Hodgman 裁剪到 tile 内。
    若有效顶点 < 3 则返回 None。
    """
    x_min, x_max = 0.0, float(TILE_SIZE)
    y_min, y_max = 0.0, float(TILE_SIZE)

    offset_pts = [[p[0] - x, p[1] - y] for p in points]

    # Sutherland-Hodgman：逐边裁剪
    def _clip_by_edge(pts, is_x: bool, bound: float, keep_ge: bool):
        """沿一条裁剪边裁剪多边形。keep_ge=True: 保留 >= bound; False: 保留 <= bound"""
        if not pts:
            return []
        output = []
        n = len(pts)
        for i in range(n):
            curr = pts[i]
            prev = pts[(i - 1 + n) % n]
            c_val = curr[0] if is_x else curr[1]
            p_val = prev[0] if is_x else prev[1]

            c_inside = c_val >= bound if keep_ge else c_val <= bound
            p_inside = p_val >= bound if keep_ge else p_val <= bound

            if c_inside:
                if not p_inside:
                    t = (bound - p_val) / (c_val - p_val) if c_val != p_val else 0.0
                    ix = prev[0] + t * (curr[0] - prev[0])
                    iy = prev[1] + t * (curr[1] - prev[1])
                    output.append([ix, iy])
                output.append(curr)
            elif p_inside:
                t = (bound - p_val) / (c_val - p_val) if c_val != p_val else 0.0
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                output.append([ix, iy])
        return output

    # 依次裁剪：左(x>=0), 右(x<=1024), 下(y>=0), 上(y<=1024)
    clipped = offset_pts
    clipped = _clip_by_edge(clipped, True, x_min, True)   # x >= 0
    clipped = _clip_by_edge(clipped, True, x_max, False)  # x <= 1024
    clipped = _clip_by_edge(clipped, False, y_min, True)  # y >= 0
    clipped = _clip_by_edge(clipped, False, y_max, False) # y <= 1024

    # 去重（相邻点距离过近）
    deduped = []
    for pt in clipped:
        if not deduped or (abs(pt[0] - deduped[-1][0]) > 0.5 or
                           abs(pt[1] - deduped[-1][1]) > 0.5):
            deduped.append(pt)

    # 首尾相近，去掉尾点
    if len(deduped) >= 2:
        dx = abs(deduped[0][0] - deduped[-1][0])
        dy = abs(deduped[0][1] - deduped[-1][1])
        if dx < 0.5 and dy < 0.5:
            deduped = deduped[:-1]

    return deduped if len(deduped) >= 3 else None


# ========================
# 核心处理函数
# ========================

def _make_tile_stem(orig_name: str, idx: int) -> str:
    """生成 tile 基础名（不含扩展名）。"""
    stem, _ = os.path.splitext(orig_name)
    return f"{stem}_s{idx:03d}"


def crop_image(img: np.ndarray, x: int, y: int, save_path: str):
    """裁剪一个 tile 并保存为 JPEG。"""
    tile = img[y: y + TILE_SIZE, x: x + TILE_SIZE]
    cv2.imwrite(save_path, tile, [cv2.IMWRITE_JPEG_QUALITY, 95])


def crop_label(mask: np.ndarray, x: int, y: int, save_path: str):
    """裁剪一个 label tile 并保存为 PNG。"""
    tile = mask[y: y + TILE_SIZE, x: x + TILE_SIZE]
    cv2.imwrite(save_path, tile)


def make_tile_json(
    orig_json: dict,
    x: int, y: int,
    tile_name: str,
    tile_mask_name: str | None,
    out_path: str,
):
    """根据原始 JSON 生成对应 tile 的 JSON（过滤/平移多边形）。"""
    new_shapes = []
    for shape in orig_json.get("shapes", []):
        points = shape.get("points", [])
        if not points:
            continue
        if not _polygon_overlaps(points, x, y):
            continue

        new_pts = _offset_and_clip_polygon(points, x, y)
        if new_pts is None:
            continue

        new_shape = dict(shape)
        new_shape["points"] = new_pts
        new_shapes.append(new_shape)

    tile_json = {
        "version": orig_json.get("version", "0.1.4"),
        "flags": orig_json.get("flags", {}),
        "shapes": new_shapes,
        "imagePath": tile_name,
        "imageData": None,
        "imageHeight": TILE_SIZE,
        "imageWidth": TILE_SIZE,
        "change_detection": orig_json.get("change_detection", True),
        "imagePathA": tile_name,
        "imagePathB": tile_name,
        "maskPath": tile_mask_name or "",
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tile_json, f, indent=2, ensure_ascii=False)


# ========================
# 主流程
# ========================

def main():
    # 检测可选目录
    has_label = os.path.isdir(join(INPUT_DIR, "label"))
    has_json = os.path.isdir(join(INPUT_DIR, "json"))
    active_subs = list(SUBS)
    if has_label:
        active_subs.append("label")
    if has_json:
        active_subs.append("json")

    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"切片尺寸：{TILE_SIZE}x{TILE_SIZE}  |  自适应步长（铺满、不溢出）")
    print(f"基础子目录：{SUBS}")
    label_status = "[有，同步裁剪]" if has_label else "[无，跳过]"
    json_status = "[有，同步裁剪]" if has_json else "[无，跳过]"
    print(f"label/  : {label_status}")
    print(f"json/   : {json_status}")
    print("-" * 50)

    step, total = 1, 5

    # [1/5] 检查目录
    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    for sub in SUBS:
        src = join(INPUT_DIR, sub)
        if not os.path.isdir(src):
            print(f"  [错误] 缺少必要目录 {src}")
            return
        files = [f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS)]
        print(f"  {sub}/  → {len(files)} 张图像")
    if has_label:
        lf = [f for f in os.listdir(join(INPUT_DIR, "label")) if f.lower().endswith(IMAGE_EXTS)]
        print(f"  label/ → {len(lf)} 张 mask")
    if has_json:
        jf = [f for f in os.listdir(join(INPUT_DIR, "json")) if f.lower().endswith(".json")]
        print(f"  json/ → {len(jf)} 个标注文件")

    # [2/5] 创建输出目录
    print(f"[{step}/{total}] 创建输出目录...")
    step += 1
    for sub in active_subs:
        os.makedirs(join(OUTPUT_DIR, sub), exist_ok=True)

    # [3/5] 读取 A 目录文件列表作为基准
    print(f"[{step}/{total}] 扫描图像列表...")
    step += 1
    a_dir = join(INPUT_DIR, "A")
    img_names = sorted([f for f in os.listdir(a_dir) if f.lower().endswith(IMAGE_EXTS)])
    print(f"  基准（A/）共 {len(img_names)} 张图像")

    # [4/5] 逐张切片
    print(f"[{step}/{total}] 开始切片...")
    step += 1
    total_tiles = 0

    for fname in tqdm(img_names, desc="切片进度", unit="img"):
        # ---- 读取原始图 ----
        img_a = cv2.imread(join(INPUT_DIR, "A", fname))
        img_b = cv2.imread(join(INPUT_DIR, "B", fname))
        if img_a is None or img_b is None:
            print(f"  [跳过] 无法读取 A/B: {fname}")
            continue
        h, w = img_a.shape[:2]

        # ---- 读取 label（可选） ----
        mask = None
        if has_label:
            mask_path = join(INPUT_DIR, "label", fname)
            if not os.path.isfile(mask_path):
                stem, _ = os.path.splitext(fname)
                for ext in IMAGE_EXTS:
                    mp = join(INPUT_DIR, "label", stem + ext)
                    if os.path.isfile(mp):
                        mask_path = mp
                        break
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # ---- 读取 JSON（可选） ----
        jdata = None
        if has_json:
            json_path = join(INPUT_DIR, "json", os.path.splitext(fname)[0] + ".json")
            if os.path.isfile(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    jdata = json.load(f)

        # ---- 遍历每个 tile 位置 ----
        for idx, x, y in _iter_tile_positions(w, h):
            stem = _make_tile_stem(fname, idx)

            # A
            crop_image(img_a, x, y, join(OUTPUT_DIR, "A", stem + ".jpg"))
            # B
            crop_image(img_b, x, y, join(OUTPUT_DIR, "B", stem + ".jpg"))
            # label
            if mask is not None:
                crop_label(mask, x, y, join(OUTPUT_DIR, "label", stem + ".png"))
            # json
            if jdata is not None:
                mask_name = stem + ".png" if mask is not None else None
                make_tile_json(jdata, x, y, stem + ".jpg", mask_name,
                               join(OUTPUT_DIR, "json", stem + ".json"))

            total_tiles += 1

    # [5/5] 汇总
    print(f"\n[{step}/{total}] 完成")
    print(f"  总 tile 数：{total_tiles}")
    for sub in active_subs:
        out_dir = join(OUTPUT_DIR, sub)
        if os.path.isdir(out_dir):
            cnt = len([f for f in os.listdir(out_dir) if not f.startswith(".")])
            print(f"  {sub}/ → {cnt} 个文件")
    print(f"  输出路径：{OUTPUT_DIR}")
    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
