"""
# @Author: 算法组
# @Date: 2026-06-22
# @Description: 双时相 ECC 配准库（金字塔 + ORB 初值），供 inference 与批量配准脚本共用
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_MOTION = 'AFFINE'
DEFAULT_SCALES = [0.125, 0.25, 0.5, 1.0]
DEFAULT_MAX_ITER = 2000
DEFAULT_EPS = 1e-6
DEFAULT_CROP_MARGIN = 1.0


def imread_unicode(path: str | os.PathLike) -> np.ndarray | None:
    """Windows 下含中文路径时 cv2.imread 易失败，改用 fromfile + imdecode。"""
    path = os.fspath(path)
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str | os.PathLike, img: np.ndarray) -> bool:
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == '.png':
        ok, enc = cv2.imencode('.png', img)
    else:
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok or enc is None:
        return False
    try:
        enc.tofile(path)
    except OSError:
        return False
    return True


def motion_type(name: str) -> int:
    n = name.upper().replace('-', '_')
    if n in ('EUCLIDEAN', 'Euclidean'):
        return cv2.MOTION_EUCLIDEAN
    if n in ('AFFINE', 'Affine'):
        return cv2.MOTION_AFFINE
    raise ValueError(f'未知 motion: {name}，使用 EUCLIDEAN 或 AFFINE')


def _affine_b_to_a_inverse_warp(M_fwd: np.ndarray) -> np.ndarray | None:
    R = M_fwd[:, :2].astype(np.float64)
    t = M_fwd[:, 2:3].astype(np.float64)
    det = np.linalg.det(R)
    if abs(det) < 1e-9:
        return None
    Rinv = np.linalg.inv(R)
    W = np.zeros((2, 3), dtype=np.float32)
    W[:, :2] = Rinv.astype(np.float32)
    W[:, 2] = (-Rinv @ t).flatten().astype(np.float32)
    return W


def orb_initial_inverse_warp(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    max_features: int = 4000,
    ransac_thresh: float = 3.0,
) -> np.ndarray | None:
    orb = cv2.ORB_create(nfeatures=max_features)
    kpa, da = orb.detectAndCompute(gray_a, None)
    kpb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(kpa) < 4 or len(kpb) < 4:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(db, da)
    if len(matches) < 12:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    src_pts = np.float32([kpb[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpa[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M_fwd, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh, confidence=0.99,
    )
    if M_fwd is None:
        return None
    inl = inliers.ravel() if inliers is not None else None
    if inl is None or inl.sum() < 6:
        return None
    return _affine_b_to_a_inverse_warp(M_fwd)


def _upscale_inverse_warp(W: np.ndarray, scale_ratio: float) -> np.ndarray:
    out = W.astype(np.float64).copy()
    out[0, 2] *= scale_ratio
    out[1, 2] *= scale_ratio
    return out.astype(np.float32)


def pyramid_find_transform_ecc(
    template: np.ndarray,
    image: np.ndarray,
    motion: int,
    criteria: tuple,
    warp_init: np.ndarray | None,
    scales: list[float],
    gauss_ksize: int = 0,
) -> tuple[np.ndarray, float]:
    h, w = template.shape[:2]
    W = warp_init
    if W is None:
        W = np.eye(2, 3, dtype=np.float32)

    last_cc = -1.0
    for si, s in enumerate(scales):
        tw = max(32, int(round(w * s)))
        th = max(32, int(round(h * s)))
        t_s = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        i_s = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        if gauss_ksize and gauss_ksize >= 3:
            t_s = cv2.GaussianBlur(t_s, (gauss_ksize, gauss_ksize), 0)
            i_s = cv2.GaussianBlur(i_s, (gauss_ksize, gauss_ksize), 0)

        if si > 0:
            ratio = scales[si] / scales[si - 1]
            W = _upscale_inverse_warp(W, ratio)
        cc, W = cv2.findTransformECC(
            t_s, i_s, W, motion, criteria, inputMask=None, gaussFiltSize=5,
        )
        last_cc = float(cc)

    return W, last_cc


def ncc_after_warp(
    gray_a: np.ndarray, gray_b: np.ndarray, warp_matrix: np.ndarray, h: int, w: int,
) -> float:
    b_w = cv2.warpAffine(
        gray_b, warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    a = gray_a.astype(np.float64)
    b = b_w.astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def _compute_forward_warp(W: np.ndarray) -> np.ndarray:
    R = W[:, :2].astype(np.float64)
    t = W[:, 2:3].astype(np.float64)
    R_inv = np.linalg.inv(R)
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = R_inv
    M[:, 2] = (-R_inv @ t).flatten()
    return M.astype(np.float32)


def _loose_footprint_aabb_in_a(
    W: np.ndarray, h_a: int, w_a: int, h_b: int, w_b: int,
) -> tuple[int, int, int, int] | None:
    M_fwd = _compute_forward_warp(W)
    corners_b = np.array(
        [[0, 0, 1], [w_b - 1, 0, 1], [w_b - 1, h_b - 1, 1], [0, h_b - 1, 1]],
        dtype=np.float32,
    ).T
    pts_a = (M_fwd @ corners_b).T
    x_min = max(0, int(np.floor(pts_a[:, 0].min())))
    x_max = min(w_a - 1, int(np.ceil(pts_a[:, 0].max())))
    y_min = max(0, int(np.floor(pts_a[:, 1].min())))
    y_max = min(h_a - 1, int(np.ceil(pts_a[:, 1].max())))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _crop_rect_samples_only_inside_b(
    W: np.ndarray, x: int, y: int, cw: int, ch: int,
    w_b: int, h_b: int, margin: float,
) -> bool:
    if cw < 1 or ch < 1:
        return False
    Wd = W.astype(np.float64)
    corners = np.array(
        [[x, y, 1.0], [x + cw - 1, y, 1.0], [x + cw - 1, y + ch - 1, 1.0], [x, y + ch - 1, 1.0]],
        dtype=np.float64,
    )
    pb = (Wd @ corners.T).T
    lo = float(margin)
    x_hi = float(w_b) - 1.0 - margin
    y_hi = float(h_b) - 1.0 - margin
    if x_hi < lo or y_hi < lo:
        return False
    return bool(
        np.all(pb[:, 0] >= lo) and np.all(pb[:, 0] <= x_hi)
        and np.all(pb[:, 1] >= lo) and np.all(pb[:, 1] <= y_hi)
    )


def _shrink_aabb_symmetric_until_inside_b(
    loose: tuple[int, int, int, int], W: np.ndarray,
    w_a: int, h_a: int, w_b: int, h_b: int, margin: float,
) -> tuple[int, int, int, int] | None:
    x0, y0, w0, h0 = loose
    max_k = min(w0, h0) // 2
    for k in range(0, max_k + 1):
        cw, ch = w0 - 2 * k, h0 - 2 * k
        if cw < 1 or ch < 1:
            break
        x, y = x0 + k, y0 + k
        if x < 0 or y < 0 or x + cw > w_a or y + ch > h_a:
            continue
        if _crop_rect_samples_only_inside_b(W, x, y, cw, ch, w_b, h_b, margin):
            return (x, y, cw, ch)
    return None


def compute_crop_roi(
    W: np.ndarray, h_a: int, w_a: int, h_b: int, w_b: int, inside_b_margin: float,
) -> tuple[int, int, int, int] | None:
    loose = _loose_footprint_aabb_in_a(W, h_a, w_a, h_b, w_b)
    if loose is None:
        return None
    return _shrink_aabb_symmetric_until_inside_b(loose, W, w_a, h_a, w_b, h_b, inside_b_margin)


def register_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    motion: int,
    use_orb_init: bool,
    scales: list[float],
    min_ecc_cc: float | None,
    min_ncc: float | None,
    max_iter: int = DEFAULT_MAX_ITER,
    eps: float = DEFAULT_EPS,
) -> tuple[np.ndarray | None, dict]:
    h, w = img_a.shape[:2]
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iter), float(eps))
    s0 = scales[0]
    tw0 = max(32, int(round(w * s0)))
    th0 = max(32, int(round(h * s0)))
    ga0 = cv2.resize(gray_a, (tw0, th0), interpolation=cv2.INTER_AREA)
    gb0 = cv2.resize(gray_b, (tw0, th0), interpolation=cv2.INTER_AREA)
    warp_init = orb_initial_inverse_warp(ga0, gb0) if use_orb_init else None

    info: dict = {'orb_init': warp_init is not None}
    try:
        W, ecc_cc = pyramid_find_transform_ecc(
            gray_a, gray_b, motion, criteria, warp_init, scales, gauss_ksize=0,
        )
    except cv2.error as e:
        info['error'] = str(e)
        return None, info

    info['ecc_correlation'] = ecc_cc
    ncc = info['ncc'] = ncc_after_warp(gray_a, gray_b, W, h, w)

    if min_ecc_cc is not None and ecc_cc < min_ecc_cc:
        info['reject'] = 'ecc_cc_below_threshold'
        return None, info
    if min_ncc is not None and ncc < min_ncc:
        info['reject'] = 'ncc_below_threshold'
        return None, info

    return W, info


def parse_scales(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    scales = [float(p) for p in parts]
    if not scales or scales[-1] != 1.0:
        raise ValueError('scales 须为非空列表且最后一级须为 1.0，例如 0.25,0.5,1.0')
    return scales


def register_batch(
    a_dir: str,
    b_dir: str,
    out_b_dir: str,
    warp_dir: str,
    motion_name: str,
    scales_str: str,
    use_orb_init: bool,
    min_ecc_cc: float | None,
    min_ncc: float | None,
    crop_border: bool = True,
    crop_margin: float = DEFAULT_CROP_MARGIN,
    log_json: str | None = None,
) -> list[dict]:
    os.makedirs(warp_dir, exist_ok=True)

    if crop_border:
        out_a_crop = os.path.join(os.path.dirname(out_b_dir), 'A_cropped')
        out_b_crop = os.path.join(os.path.dirname(out_b_dir), 'B_cropped')
        os.makedirs(out_a_crop, exist_ok=True)
        os.makedirs(out_b_crop, exist_ok=True)
        os.makedirs(out_b_dir, exist_ok=True)
    else:
        out_a_crop = out_b_crop = None
        os.makedirs(out_b_dir, exist_ok=True)

    motion = motion_type(motion_name)
    scales = parse_scales(scales_str)

    files = sorted(os.listdir(a_dir))
    print(
        f'共 {len(files)} 个文件（过滤后成对处理），motion={motion_name}, scales={scales}, ORB初值={use_orb_init}\n'
    )

    log_lines: list[dict] = []

    for fname in tqdm(files):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path_a = os.path.join(a_dir, fname)
        path_b = os.path.join(b_dir, fname)
        if not os.path.isfile(path_b):
            tqdm.write(f'[Skip] B 中无对应文件: {fname}')
            continue

        img_a = imread_unicode(path_a)
        img_b = imread_unicode(path_b)
        if img_a is None or img_b is None:
            tqdm.write(f'[Skip] 读取失败: {fname}')
            continue

        stem = os.path.splitext(fname)[0]
        warp_path = os.path.join(warp_dir, stem + '.npy')
        meta_path = os.path.join(warp_dir, stem + '.json')

        W, info = register_pair(
            img_a, img_b, motion, use_orb_init, scales, min_ecc_cc, min_ncc,
        )

        rec = {'file': fname, **{k: v for k, v in info.items() if k != 'error'}}
        if W is None:
            tqdm.write(f'[Warn] 配准未采用变换 {fname}: {info}')
            fallback_dir = out_b_crop if crop_border else out_b_dir
            imwrite_unicode(os.path.join(fallback_dir, fname), img_b)
            rec['used_fallback'] = True
            rec['warp_saved'] = False
        else:
            h, w = img_a.shape[:2]
            h_b, w_b = img_b.shape[:2]

            border_mode = cv2.BORDER_CONSTANT if crop_border else cv2.BORDER_REFLECT
            border_val = (0, 0, 0) if crop_border else None
            warp_kw: dict = {
                'flags': cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                'borderMode': border_mode,
            }
            if border_val is not None:
                warp_kw['borderValue'] = border_val
            img_b_aligned = cv2.warpAffine(img_b, W, (w, h), **warp_kw)
            np.save(warp_path, W.astype(np.float32))

            meta = {
                'file': fname,
                'motion': motion_name,
                'scales': scales,
                'ecc_correlation': info.get('ecc_correlation'),
                'ncc': info.get('ncc'),
                'orb_init': info.get('orb_init'),
            }

            if crop_border:
                roi = compute_crop_roi(W, h, w, h_b, w_b, crop_margin)
                if roi is not None:
                    x, y, cw, ch = roi
                    imwrite_unicode(os.path.join(out_a_crop, fname), img_a[y:y + ch, x:x + cw])
                    imwrite_unicode(os.path.join(out_b_crop, fname), img_b_aligned[y:y + ch, x:x + cw])
                    meta['crop_roi'] = {'x': int(x), 'y': int(y), 'w': int(cw), 'h': int(ch)}
                else:
                    tqdm.write(
                        f'[Warn] {fname} 无满足「无反射」的裁剪框，已写入整幅 B_registered'
                    )
                    imwrite_unicode(os.path.join(out_b_dir, fname), img_b_aligned)
                    meta['crop_roi'] = None
                    meta['crop_note'] = 'no_mirror_safe_roi_fallback_full'
            else:
                imwrite_unicode(os.path.join(out_b_dir, fname), img_b_aligned)

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            rec['used_fallback'] = False
            rec['warp_saved'] = True

        log_lines.append(rec)

    if log_json:
        with open(log_json, 'w', encoding='utf-8') as f:
            for row in log_lines:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'\n完成。对齐 B: {out_b_dir}\nwarp 与 meta: {warp_dir}')
    if log_json:
        print(f'日志: {log_json}')
    return log_lines
