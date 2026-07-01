#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: 从 bcd_map 连通分量裁剪变化区域子图，支持外扩比例。
#              crop_mask_vismask=true：左 vismask（A+红色 mask）| 右 B
#              crop_mask_vismask=false：左 A 原图 | 右 B 原图（无红色高亮）
#
# 每个 mask 独立裁剪（每个连通分量一张图），输出命名：原图名_序号.jpg
# 例如 01.png 有 3 个变化区域 → 01_01.jpg, 01_02.jpg, 01_03.jpg
#
# 输入目录结构（推理输出）：
#   pred_dir/
#     ├── bcd_map/      — 二值变化 mask (0/255)
#     ├── vismask/      — 可视化叠加图（crop_mask_vismask=true 时需要）
#     └── register_B/   — 配准后 B（register 开启时 inference 写出，与 vismask 同坐标系）
#
# 输入（test_dir）：
#   imgA_dir/           — 时相 A（crop_mask_vismask=false 时用于左图）
#   imgB_dir/           — 时相 B 原图（仅用于配准；裁剪优先读 pred_dir/register_B/）
#
# 输出：
#   output_dir/
#     └── compare_crop/ — 左右拼接对比图
#
# Usage:
#   python crop_mask.py --pred_dir ./pred --output_dir ./crop --imgA_dir ./test/A --imgB_dir ./test/B
#   python crop_mask.py --pred_dir ./pred --output_dir ./crop --vismask --imgB_dir ./test/B
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils.register as reg

#==============================#
# 默认配置（写死，不通过 config 传递）
#==============================#
DEFAULT_PRED_DIR = None       # --pred_dir 参数（从 config 读取）
DEFAULT_OUTPUT_DIR = None     # --output_dir 参数（从 config 读取）
DEFAULT_EXPAND = 0.4          # 外扩比例（0.0 ~ 1.0），与 config predict.crop_mask_expand 对齐
DEFAULT_USE_VISMASK = False     # 是否使用 vismask 红色高亮；false 则拼接 A|B 原图
DEFAULT_CROP_A = False        # 是否额外单独保存 A 裁剪（--crop_a，仅 vismask 模式有意义）
DEFAULT_IMG_A_DIR = None      # --imgA_dir 参数（test_dir/A）
DEFAULT_IMG_B_DIR = None      # --imgB_dir 参数（test_dir/B）
DEFAULT_WORKERS = 4           # 并行线程数
DEFAULT_MIN_AREA = 1219       # 最小连通分量面积（像素），约 0.01% 图像面积，过滤噪声
COMPARE_GAP = 2               # 左右拼接间隔（像素，白线分隔）
#==============================#


def expand_bbox(x, y, w, h, expand_ratio, full_w, full_h):
    """
    根据外扩比例扩展 bbox。
    expand_ratio=0.2 表示在每边扩展 bbox 宽高各 20%。
    边界自动截断。
    """
    if expand_ratio <= 0:
        return x, y, w, h

    # 每边扩展量 = bbox 尺寸的 expand_ratio / 2
    dx = int(w * expand_ratio / 2)
    dy = int(h * expand_ratio / 2)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(full_w, x + w + dx)
    y2 = min(full_h, y + h + dy)

    return x1, y1, x2 - x1, y2 - y1


def resolve_img_b_path(fname, pred_dir, img_b_dir):
    """优先 pred_dir/register_B（配准后），否则回退 test_dir/B。"""
    aligned = os.path.join(pred_dir, "register_B", fname)
    if os.path.isfile(aligned):
        return aligned
    return resolve_img_path(fname, img_b_dir)


def _warp_b_to_a_full(img_a_bgr, img_b_bgr, scales, max_iter):
    """与 inference._register_core 一致：将 B warp 到 A 画布。"""
    h, w = img_a_bgr.shape[:2]
    motion = reg.motion_type(reg.DEFAULT_MOTION)
    W, _info = reg.register_pair(
        img_a_bgr, img_b_bgr, motion, True, scales, None, None, max_iter=max_iter,
    )
    if W is None:
        return img_b_bgr
    return cv2.warpAffine(
        img_b_bgr, W, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _align_b_worker(task):
    fname, path_a, path_b, out_path, scales, max_iter = task
    img_a = reg.imread_unicode(path_a)
    img_b = reg.imread_unicode(path_b)
    if img_a is None or img_b is None:
        return fname, False, "读取失败"
    b_full = _warp_b_to_a_full(img_a, img_b, scales, max_iter)
    ok = reg.imwrite_unicode(out_path, b_full)
    return fname, ok, "" if ok else "写入失败"


def ensure_aligned_b_dir(pred_dir, img_a_dir, img_b_dir, fnames, reg_cfg, workers=4):
    """补写 pred_dir/register_B/（与 inference 配准参数一致）。"""
    register_b_dir = os.path.join(pred_dir, "register_B")
    os.makedirs(register_b_dir, exist_ok=True)

    scales = [float(x) for x in reg_cfg.get("scales", reg.DEFAULT_SCALES)]
    max_iter = int(reg_cfg.get("max_iter", reg.DEFAULT_MAX_ITER))
    missing = [f for f in fnames if not os.path.isfile(os.path.join(register_b_dir, f))]
    if not missing:
        print(f"  配准 B: pred_dir/register_B/ 已就绪 ({len(fnames)} 张)")
        return register_b_dir

    print(f"  配准 B: 未找到 register_B/，正在配准 {len(missing)} 张 -> {register_b_dir}")
    print(f"           scales={scales}, max_iter={max_iter}, workers={workers}")
    tasks = [
        (fn, os.path.join(img_a_dir, fn), os.path.join(img_b_dir, fn),
         os.path.join(register_b_dir, fn), scales, max_iter)
        for fn in missing
    ]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_align_b_worker, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="配准 B", unit="张"):
                fname, ok, err = fut.result()
                if not ok:
                    print(f"  [warn] {fname}: {err}")
    else:
        for t in tqdm(tasks, desc="配准 B", unit="张"):
            fname, ok, err = _align_b_worker(t)
            if not ok:
                print(f"  [warn] {fname}: {err}")
    return register_b_dir


def load_register_cfg(config_path):
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("predict", {}).get("register", {}) or {}


def resolve_img_path(fname, img_dir):
    """从目录解析与 bcd_map 同名的图像路径。"""
    if not img_dir or not os.path.isdir(img_dir):
        return None
    path = os.path.join(img_dir, fname)
    return path if os.path.isfile(path) else None


def align_image_to_ref(img, ref_h, ref_w):
    """将 img 缩放到与参考图相同尺寸（A/B 与 mask 坐标对齐）。"""
    h, w = img.shape[:2]
    if h == ref_h and w == ref_w:
        return img
    return cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)


def concat_compare(crop_left, crop_right, gap=COMPARE_GAP):
    """左右拼接对比图，中间白线分隔。"""
    if gap > 0:
        sep = np.full((crop_left.shape[0], gap, 3), 255, dtype=np.uint8)
        return np.concatenate([crop_left, sep, crop_right], axis=1)
    return np.concatenate([crop_left, crop_right], axis=1)


def _crop_single(args):
    """单文件裁剪 worker（用于 ProcessPoolExecutor）。

    对每个连通分量单独裁剪，输出命名：原图名_序号.jpg
    例如 01.png → 01_01.jpg, 01_02.jpg, ...
    """
    (fname, bcd_map_path, vismask_path, img_a_path, img_b_path,
     expand_ratio, output_dir_path, output_imga_dir, use_vismask, min_area) = args

    # 1. 读取 bcd_map
    mask = cv2.imread(bcd_map_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return fname, "skip", "无法读取 bcd_map"

    # 2. 查找所有连通分量 (connected components)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_labels <= 1:
        return fname, "skip", "无变化区域"

    full_h, full_w = mask.shape[:2]

    # 3. 读取左/右图（各只读一次）
    vismask = None
    img_a = None
    img_b = None

    if use_vismask:
        if not vismask_path:
            return fname, "skip", "vismask 模式缺少 vismask"
        vismask = cv2.imread(vismask_path, cv2.IMREAD_COLOR)
        if vismask is None:
            return fname, "skip", "无法读取 vismask"
        ref_h, ref_w = vismask.shape[:2]
        if img_b_path:
            img_b = cv2.imread(img_b_path, cv2.IMREAD_COLOR)
            if img_b is not None:
                img_b = align_image_to_ref(img_b, ref_h, ref_w)
    else:
        if not img_a_path or not img_b_path:
            return fname, "skip", "原图模式缺少 A 或 B"
        img_a = cv2.imread(img_a_path, cv2.IMREAD_COLOR)
        img_b = cv2.imread(img_b_path, cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            return fname, "skip", "无法读取 A 或 B 原图"
        img_a = align_image_to_ref(img_a, full_h, full_w)
        img_b = align_image_to_ref(img_b, full_h, full_w)
        ref_h, ref_w = full_h, full_w

    if use_vismask and img_b is None:
        return fname, "skip", "vismask 模式缺少 B 图"

    has_pair = img_b is not None
    result = {"fname": fname, "status": "ok", "components": 0, "details": [], "has_pair": has_pair}
    saved_count = 0
    details = []

    # 跳过 label=0 (背景)，遍历每个前景连通分量
    for label_idx in range(1, n_labels):
        cx, cy, cw, ch, carea = stats[label_idx]

        if min_area > 0 and carea < min_area:
            continue

        x1, y1, crop_w, crop_h = expand_bbox(cx, cy, cw, ch, expand_ratio, full_w, full_h)

        if use_vismask:
            crop_left = vismask[y1:y1 + crop_h, x1:x1 + crop_w]
        else:
            crop_left = img_a[y1:y1 + crop_h, x1:x1 + crop_w]

        crop_b = img_b[y1:y1 + crop_h, x1:x1 + crop_w]
        crop_out = concat_compare(crop_left, crop_b)

        stem, _ext = os.path.splitext(fname)
        seq = f"{saved_count + 1:02d}"
        out_ext = ".jpg"
        out_path = os.path.join(output_dir_path, f"{stem}_{seq}{out_ext}")
        cv2.imwrite(out_path, crop_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1

        if use_vismask and output_imga_dir and img_a_path:
            if img_a is None:
                img_a = cv2.imread(img_a_path, cv2.IMREAD_COLOR)
                if img_a is not None:
                    img_a = align_image_to_ref(img_a, ref_h, ref_w)
            if img_a is not None:
                crop_a = img_a[y1:y1 + crop_h, x1:x1 + crop_w]
                out_imga = os.path.join(output_imga_dir, f"{stem}_{seq}{out_ext}")
                cv2.imwrite(out_imga, crop_a, [cv2.IMWRITE_JPEG_QUALITY, 95])

        mode_tag = "vismask|B" if use_vismask else "A|B"
        details.append({
            "fname": f"{stem}_{seq}{out_ext}",
            "bbox": [int(cx), int(cy), int(cw), int(ch)],
            "crop": [int(x1), int(y1), int(crop_w), int(crop_h)],
            "mode": mode_tag,
        })

    result["components"] = saved_count
    result["details"] = details

    return fname, result, ""


def crop_mask(
    pred_dir,
    output_dir,
    expand_ratio=0.4,
    img_a_dir=None,
    img_b_dir=None,
    use_vismask=False,
    workers=4,
    min_area=0,
    save_crop_a=False,
    config_path=None,
):
    """
    主处理流程。

    Args:
        pred_dir:      推理输出目录，包含 bcd_map/
        output_dir:    裁剪输出目录
        expand_ratio:  外扩比例 (0.0~1.0)
        img_a_dir:     时相 A 目录（原图模式必需）
        img_b_dir:     时相 B 目录
        use_vismask:   True=左 vismask|右 B；False=左 A 原图|右 B 原图
        workers:       并行进程数
        save_crop_a:   vismask 模式下额外单独保存 A 裁剪
    """
    bcd_map_dir = os.path.join(pred_dir, "bcd_map")
    vismask_dir = os.path.join(pred_dir, "vismask")
    output_compare_dir = os.path.join(
        output_dir, "vismask_crop" if use_vismask else "compare_crop"
    )
    output_imga_dir = os.path.join(output_dir, "imgA_crop")

    has_img_a = img_a_dir is not None and os.path.isdir(img_a_dir)
    has_img_b = img_b_dir is not None and os.path.isdir(img_b_dir)
    save_extra_a = save_crop_a and use_vismask and has_img_a

    if not os.path.isdir(bcd_map_dir):
        print(f"  [错误] 目录不存在: {bcd_map_dir}")
        return 0
    if use_vismask and not os.path.isdir(vismask_dir):
        print(f"  [错误] vismask 模式需要目录: {vismask_dir}")
        return 0
    if not use_vismask and not has_img_a:
        print("  [错误] 原图模式需要 imgA_dir")
        return 0
    if not has_img_b:
        print("  [错误] 需要 imgB_dir")
        return 0

    reg_cfg = load_register_cfg(config_path)
    register_enable = bool(reg_cfg.get("enable", False))
    if register_enable and not has_img_a:
        print("  [错误] register.enable=true 时需要 imgA_dir 以生成配准 B (pred_dir/register_B/)")
        return 0

    os.makedirs(output_compare_dir, exist_ok=True)
    if save_extra_a:
        os.makedirs(output_imga_dir, exist_ok=True)

    bcd_files = sorted([f for f in os.listdir(bcd_map_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    vis_files = set()
    if use_vismask:
        vis_files = set(f for f in os.listdir(vismask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    a_files = set()
    if has_img_a:
        a_files = set(f for f in os.listdir(img_a_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    b_files = set(f for f in os.listdir(img_b_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))

    if use_vismask:
        valid_list = [f for f in bcd_files if f in vis_files and f in b_files]
        if not valid_list:
            print("  [警告] 未找到匹配的 bcd_map + vismask + B 文件")
            return 0
    else:
        valid_list = [f for f in bcd_files if f in a_files and f in b_files]
        if not valid_list:
            print("  [警告] 未找到匹配的 bcd_map + A + B 文件")
            return 0

    if register_enable:
        reg_workers = int(reg_cfg.get("workers", workers))
        ensure_aligned_b_dir(pred_dir, img_a_dir, img_b_dir, valid_list, reg_cfg, reg_workers)

    aligned_n = sum(
        1 for f in valid_list if os.path.isfile(os.path.join(pred_dir, "register_B", f))
    )
    if register_enable or aligned_n:
        print(f"  B 图源: pred_dir/register_B/ ({aligned_n}/{len(valid_list)} 张)")
    else:
        print("  B 图源: test_dir/B 原图（未配准，仅适用于 A/B 已对齐数据）")
    if register_enable and aligned_n < len(valid_list):
        print("  [警告] 部分 B 未能配准，将回退原图，对比可能不准")

    mode_label = "vismask|B" if use_vismask else "A|B"
    print(f"  基准文件数: {len(valid_list)}")
    print(f"  对比模式: {mode_label}")
    print(f"  外扩比例: {expand_ratio:.1%}")
    print(f"  imgA_dir={img_a_dir}, imgB_dir={img_b_dir}")
    if save_extra_a:
        print(f"  额外输出 A 裁剪: {output_imga_dir}")

    tasks = [
        (
            fname,
            os.path.join(bcd_map_dir, fname),
            os.path.join(vismask_dir, fname) if use_vismask else None,
            resolve_img_path(fname, img_a_dir),
            resolve_img_b_path(fname, pred_dir, img_b_dir),
            expand_ratio,
            output_compare_dir,
            output_imga_dir if save_extra_a else None,
            use_vismask,
            min_area,
        )
        for fname in valid_list
    ]

    ok_count = 0
    skip_count = 0
    total_components = 0
    all_details = []
    start_time = time.perf_counter()

    if workers > 1 and len(tasks) > 4:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_crop_single, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="裁剪进度", unit="张"):
                fname, result, err = fut.result()
                if isinstance(result, dict):
                    ok_count += 1
                    total_components += result.get("components", 0)
                    all_details.extend(result.get("details", []))
                else:
                    skip_count += 1
                    if err:
                        print(f"  [skip] {fname}: {err}")
    else:
        for t in tqdm(tasks, desc="裁剪进度", unit="张"):
            fname, result, err = _crop_single(t)
            if isinstance(result, dict):
                ok_count += 1
                total_components += result.get("components", 0)
                all_details.extend(result.get("details", []))
            else:
                skip_count += 1
                if err:
                    print(f"  [skip] {fname}: {err}")

    all_details.sort(key=lambda d: d["fname"])

    elapsed = time.perf_counter() - start_time
    print(f"\n  Done: {ok_count} processed, {skip_count} skipped")
    print(f"     共裁剪 {total_components} 个连通分量（{mode_label}）")
    for d in all_details:
        print(f"     [OK] {d['fname']} ({d.get('mode', mode_label)})  bbox={d['bbox']}  crop={d['crop']}")
    print(f"     Time: {elapsed:.2f}s ({ok_count / max(elapsed, 0.001):.1f} imgs/s)")
    print(f"     Output: {output_compare_dir}")
    if save_extra_a:
        print(f"     Output: {output_imga_dir}")

    return ok_count


def main():
    parser = argparse.ArgumentParser(description="Crop changed regions and save side-by-side compare")
    parser.add_argument("--pred_dir", default=DEFAULT_PRED_DIR,
                        help="Inference output dir (contains bcd_map/)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Crop output dir (contains compare_crop/)")
    parser.add_argument("--expand", type=float, default=DEFAULT_EXPAND,
                        help="Expansion ratio 0.0~1.0 (default 0.4)")
    parser.add_argument("--vismask", action="store_true", default=DEFAULT_USE_VISMASK,
                        help="Use vismask (A+red highlight)|B; default is raw A|B")
    parser.add_argument("--imgA_dir", default=DEFAULT_IMG_A_DIR,
                        help="Image A directory (test_dir/A; required when --vismask is off)")
    parser.add_argument("--imgB_dir", default=DEFAULT_IMG_B_DIR,
                        help="Image B source directory (test_dir/B; warp source when register on)")
    parser.add_argument("--config", default=None,
                        help="config/default.yaml (read predict.register for aligned B)")
    parser.add_argument("--crop_a", action="store_true", default=DEFAULT_CROP_A,
                        help="Also save separate A crops to imgA_crop/ (vismask mode only)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel workers (default 4)")
    parser.add_argument("--min_area", type=int, default=DEFAULT_MIN_AREA,
                        help="Minimum connected component area in pixels (default=1219)")

    args = parser.parse_args()

    if not args.pred_dir:
        print("[Error] Please specify --pred_dir")
        sys.exit(1)
    if not args.output_dir:
        args.output_dir = os.path.join(args.pred_dir, "crop_output")
    if not (0.0 <= args.expand <= 1.0):
        print("[Error] --expand must be between 0.0 and 1.0")
        sys.exit(1)
    if not args.vismask and not args.imgA_dir:
        print("[Error] 原图模式需要 --imgA_dir")
        sys.exit(1)
    if not args.imgB_dir:
        print("[Error] Please specify --imgB_dir")
        sys.exit(1)
    reg_cfg = load_register_cfg(args.config)
    if bool(reg_cfg.get("enable", False)) and not args.imgA_dir:
        print("[Error] register.enable=true 时需要 --imgA_dir")
        sys.exit(1)

    mode_label = "vismask|B" if args.vismask else "A|B"
    print(f"Input:  {args.pred_dir}")
    print(f"Output: {args.output_dir}")
    crop_a_msg = ", save_crop_a=true" if args.crop_a else ""
    print(f"Args: mode={mode_label}, expand={args.expand:.1%}, workers={args.workers}, "
          f"min_area={args.min_area}, imgA_dir={args.imgA_dir}, imgB_dir={args.imgB_dir}{crop_a_msg}")
    print("-" * 50)

    crop_mask(
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        expand_ratio=args.expand,
        img_a_dir=args.imgA_dir,
        img_b_dir=args.imgB_dir,
        use_vismask=args.vismask,
        workers=args.workers,
        min_area=args.min_area,
        save_crop_a=args.crop_a,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
