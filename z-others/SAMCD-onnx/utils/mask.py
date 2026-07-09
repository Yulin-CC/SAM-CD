"""
# @Description: 将 label mask 半透明叠加到 A 或 B 原图上，高亮变化区域，输出至 mask/。
#   输入: GEOAI 目录（含 A/ 或 B/、label/）
#   输出: 同目录下 mask/（与原图同名）
# @Switch: 改 IMAGE_DIR 为 "A" 或 "B"，或命令行 --image-dir A|B
# @Command: python z-rm_mask.py
"""

import os
from os.path import join

import argparse
import cv2
import numpy as np
from tqdm import tqdm

# ==============================#
# 接口配置
# ==============================#
DATA_DIR = "path/to/test/dir"
IMAGE_DIR = "B"                 # 原图目录：改 "A" 或 "B" 即可切换叠加底图
# ==============================#
ALPHA = 0.45                    # 叠加透明度（0~1，越大越显眼）
HIGHLIGHT_BGR = (0, 0, 255)     # 高亮颜色（BGR），默认红色
# ==============================#
DRAW_CONTOUR = True             # 是否在边界描边
CONTOUR_THICKNESS = 1
# ==============================#

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LABEL_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    ok, enc = cv2.imencode(ext, img)
    if not ok or enc is None:
        return False
    enc.tofile(path)
    return True


def _label_path(label_dir: str, img_name: str) -> str | None:
    """A 常为 .JPG，label 多为同名 .png。"""
    direct = join(label_dir, img_name)
    if os.path.isfile(direct):
        return direct
    stem, _ = os.path.splitext(img_name)
    for suf in LABEL_EXTS:
        p = join(label_dir, stem + suf)
        if os.path.isfile(p):
            return p
    return None


def overlay_label_on_image(
    img: np.ndarray,
    label: np.ndarray,
    alpha: float = ALPHA,
    color: tuple[int, int, int] = HIGHLIGHT_BGR,
    draw_contour: bool = DRAW_CONTOUR,
    contour_thickness: int = CONTOUR_THICKNESS,
) -> np.ndarray:
    """将二值 label 半透明叠加到 BGR 图像上。"""
    if label.ndim == 3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    if label.shape[:2] != img.shape[:2]:
        label = cv2.resize(label, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    region = label > 0
    if not np.any(region):
        return img.copy()

    out = img.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[region] = (1.0 - alpha) * out[region] + alpha * color_arr

    if draw_contour:
        binary = region.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = out.astype(np.uint8)
        cv2.drawContours(out, contours, -1, color, contour_thickness)
        return out

    return out.astype(np.uint8)


def list_images(data_dir: str, image_dir: str) -> list[str]:
    src_dir = join(data_dir, image_dir)
    names = []
    for fname in os.listdir(src_dir):
        low = fname.lower()
        if any(low.endswith(ext) for ext in IMAGE_EXTS):
            names.append(fname)
    return sorted(names)


def process_one(
    data_dir: str,
    image_dir: str,
    fname: str,
    pred_dir: str,
    out_dir: str,
    alpha: float,
    color: tuple[int, int, int],
    draw_contour: bool,
) -> str:
    """处理单张图，返回状态: ok / skip_no_label / skip_read / fail_write。"""
    path_img = join(data_dir, image_dir, fname)
    path_lbl = _label_path(pred_dir, fname)
    if path_lbl is None:
        return "skip_no_label"

    img = imread_unicode(path_img, cv2.IMREAD_COLOR)
    lbl = imread_unicode(path_lbl, cv2.IMREAD_GRAYSCALE)
    if img is None or lbl is None:
        return "skip_read"

    vis = overlay_label_on_image(
        img, lbl, alpha=alpha, color=color, draw_contour=draw_contour
    )
    out_path = join(out_dir, fname)
    if imwrite_unicode(out_path, vis):
        return "ok"
    return "fail_write"


def run(
    data_dir: str,
    pred_dir: str,
    out_dir: str,
    image_dir: str = IMAGE_DIR,
    alpha: float = ALPHA,
    color: tuple[int, int, int] = HIGHLIGHT_BGR,
    draw_contour: bool = DRAW_CONTOUR,
) -> dict:
    image_dir = image_dir.upper()
    if image_dir not in ("A", "B"):
        raise ValueError(f'image_dir 须为 "A" 或 "B"，当前: {image_dir}')
    if not os.path.isdir(join(data_dir, image_dir)):
        raise FileNotFoundError(f"缺少 {image_dir}/ 目录: {data_dir}")
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"缺少推理结果目录: {pred_dir}")

    os.makedirs(out_dir, exist_ok=True)

    stats = {"ok": 0, "skip_no_label": 0, "skip_read": 0, "fail_write": 0, "total": 0}
    names = list_images(data_dir, image_dir)
    stats["total"] = len(names)

    for fname in tqdm(names, desc=f"label → {image_dir} 叠加"):
        status = process_one(
            data_dir, image_dir, fname, pred_dir, out_dir, alpha, color, draw_contour
        )
        stats[status] = stats.get(status, 0) + 1

    return stats


def parse_opt():
    parser = argparse.ArgumentParser(description="将推理 mask 叠加到 A 或 B 并输出可视化图")
    parser.add_argument("--data-dir", default=DATA_DIR, help="GEOAI 数据集根目录")
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="推理 mask 目录（默认: data-dir/repro/1-ChangeDetect_SAMCD-2605-v1.0）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="可视化输出目录（默认: data-dir/mask）",
    )
    parser.add_argument(
        "--image-dir",
        default=IMAGE_DIR,
        choices=("A", "B"),
        help=f'原图子目录，默认 {IMAGE_DIR}（与顶部 IMAGE_DIR 一致）',
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help="叠加透明度 0~1")
    parser.add_argument(
        "--color",
        default="0,0,255",
        help="高亮颜色 BGR，逗号分隔，如 0,255,0 为绿色",
    )
    parser.add_argument("--no-contour", action="store_true", help="不绘制边界描边")
    return parser.parse_args()


def _parse_color(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("color 须为三个整数，如 0,0,255")
    return tuple(parts)


if __name__ == "__main__":
    opt = parse_opt()
    color = _parse_color(opt.color)
    image_dir = opt.image_dir.upper()
    pred_dir = opt.pred_dir or join(opt.data_dir, "repro/1-ChangeDetect_SAMCD-2605-v1.0")
    out_dir = opt.out_dir or join(opt.data_dir, "mask")
    print(f"数据目录: {opt.data_dir}")
    print(f"推理 mask: {pred_dir}")
    print(f"底图目录: {image_dir}/")
    print(f"输出目录: {out_dir}")
    print(f"叠加: alpha={opt.alpha}, color(BGR)={color}, contour={not opt.no_contour}")

    stats = run(
        opt.data_dir,
        pred_dir=pred_dir,
        out_dir=out_dir,
        image_dir=image_dir,
        alpha=opt.alpha,
        color=color,
        draw_contour=not opt.no_contour,
    )
    print(
        f"完成: {stats['ok']}/{stats['total']} → {out_dir}\n"
        f"  无 label: {stats['skip_no_label']}, 读取失败: {stats['skip_read']}, "
        f"写入失败: {stats['fail_write']}"
    )
