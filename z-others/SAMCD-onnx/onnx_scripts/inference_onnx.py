#!/usr/bin/env python3
"""
SAM-CD ONNX Runtime 两阶段推理（FastSAM ONNX + CD Head ONNX）。
Pipeline 与 0-QuickStart/scripts/inference.py 对齐：配准 / 滑窗 / TTA / 计时 / vismask。
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage import io
from skimage.util import img_as_ubyte
from tqdm import tqdm

warnings.filterwarnings("ignore", message="is a low contrast image")

ONNX_ROOT = Path(__file__).resolve().parents[1]
_script_dir = str(Path(__file__).resolve().parent)
for p in (_script_dir, str(ONNX_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.preprocess import preprocess
from utils.samcd_pipeline import create_crops, stitch_pred
from utils import register as reg

VIS_ALPHA = 0.45
VIS_HIGHLIGHT_BGR = (0, 0, 255)
REGISTER_MODE = 'crop_pasteback'


def resolve_config_path(config_arg: str) -> str:
    p = Path(config_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = ONNX_ROOT / config_arg
    if cand.exists():
        return str(cand.resolve())
    if p.exists():
        return str(p.resolve())
    return str((ONNX_ROOT / 'config' / 'default.yaml').resolve())


def resolve_path(path, base: Path = ONNX_ROOT) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())


class PredOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--config', default=str(ONNX_ROOT / 'config' / 'default.yaml'))
        parser.add_argument('--fastsam_onnx', default=None)
        parser.add_argument('--cd_head_onnx', default=None)
        parser.add_argument('--test_dir', '--dataset', dest='test_dir', default=None)
        parser.add_argument('--pred_dir', '-o', '--out_dir', dest='pred_dir', default=None)
        parser.add_argument('--crop_size', nargs=2, type=int, default=None)
        parser.add_argument('--fastsam_imgsz', type=int, default=None)
        parser.add_argument('--TTA', '--tta', dest='TTA', default=None)
        parser.add_argument('--threshold', type=float, default=None)
        parser.add_argument('--inference_sliding_crop', default=None)
        parser.add_argument('--dev_id', type=int, default=None)
        parser.add_argument('--time', action='store_true', default=None, help='打印分阶段计时')
        self.initialized = True
        return parser

    def gather_options(self):
        import argparse
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        return parser.parse_args()

    def parse(self):
        return self.gather_options()


class SamCdOnnxRunner:
    def __init__(self, fastsam_path: str, cd_head_path: str, device_prefer_cuda: bool = True):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if device_prefer_cuda
            else ['CPUExecutionProvider']
        )
        self.fastsam = ort.InferenceSession(fastsam_path, opts, providers=providers)
        self.cd_head = ort.InferenceSession(cd_head_path, opts, providers=providers)
        self.fs_in = self.fastsam.get_inputs()[0].name
        self.cd_inputs = [inp.name for inp in self.cd_head.get_inputs()]

    def encode(self, tensor_nchw: np.ndarray):
        from utils.samcd_pipeline import hier_feats_from_onnx_outputs
        outs = self.fastsam.run(None, {self.fs_in: tensor_nchw})
        return hier_feats_from_onnx_outputs(outs)

    def predict_logits(self, feats_a, feats_b):
        if len(self.cd_inputs) != 8:
            raise RuntimeError(f'CD Head ONNX 期望 8 个输入，实际 {len(self.cd_inputs)}')
        feed = {}
        for i, name in enumerate(self.cd_inputs):
            feed[name] = feats_a[i] if i < 4 else feats_b[i - 4]
        return self.cd_head.run(None, feed)[0]

    def predict_patch_logits(self, img_a_rgb, img_b_rgb, fastsam_imgsz: int):
        """HWC RGB [0,1] patch → raw logits (before sigmoid)."""
        def _once(a_rgb, b_rgb):
            a_inp, _ = preprocess((a_rgb * 255).astype(np.uint8)[..., ::-1], target_size=fastsam_imgsz)
            b_inp, _ = preprocess((b_rgb * 255).astype(np.uint8)[..., ::-1], target_size=fastsam_imgsz)
            fa = self.encode(a_inp)
            fb = self.encode(b_inp)
            return self.predict_logits(fa, fb).astype(np.float32)

        return _once(img_a_rgb, img_b_rgb)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def infer_patch(runner: SamCdOnnxRunner, img_a, img_b, opt, gpu_timing=None):
    logit = runner.predict_patch_logits(img_a, img_b, opt.fastsam_imgsz)
    if gpu_timing is not None:
        t0 = time.perf_counter()
    output = _sigmoid(logit)
    if gpu_timing is not None:
        gpu_timing['base'] = gpu_timing.get('base', 0.0) + time.perf_counter() - t0

    if opt.TTA:
        if gpu_timing is not None:
            t0 = time.perf_counter()
        for axis in ([0], [1], [0, 1]):
            av = np.flip(img_a, axis=axis).copy()
            bv = np.flip(img_b, axis=axis).copy()
            lv = runner.predict_patch_logits(av, bv, opt.fastsam_imgsz)
            flip_axes = tuple(2 + a for a in axis)
            output += _sigmoid(np.flip(lv, axis=flip_axes))
        if gpu_timing is not None:
            gpu_timing['tta'] = gpu_timing.get('tta', 0.0) + time.perf_counter() - t0
        output = output / 4.0

    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
    return output > opt.threshold


def run_infer(runner, opt, img_a, img_b, gpu_timing=None):
    h, w = img_a.shape[:2]
    is_large = h > opt.crop_size[0] or w > opt.crop_size[1]
    if is_large and opt.inference_sliding_crop:
        img_a_crops, img_b_crops = create_crops(img_a, img_b, opt.crop_size)
        preds = [
            infer_patch(runner, img_a_crops[i], img_b_crops[i], opt, gpu_timing)
            for i in range(len(img_a_crops))
        ]
        return stitch_pred([p.astype(np.uint8) for p in preds], size_stitch=(h, w)).astype(bool)
    if is_large:
        resize_wh = (opt.crop_size[1], opt.crop_size[0])
        img_a_rs = cv2.resize(img_a, resize_wh, interpolation=cv2.INTER_LINEAR)
        img_b_rs = cv2.resize(img_b, resize_wh, interpolation=cv2.INTER_LINEAR)
        pred_rs = infer_patch(runner, img_a_rs, img_b_rs, opt, gpu_timing)
        return cv2.resize(pred_rs.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return infer_patch(runner, img_a, img_b, opt, gpu_timing)


def _register_core(img_a_bgr, img_b_bgr, scales, crop_margin, max_iter):
    h, w = img_a_bgr.shape[:2]
    h_b, w_b = img_b_bgr.shape[:2]
    motion = reg.motion_type(reg.DEFAULT_MOTION)

    W, info = reg.register_pair(
        img_a_bgr, img_b_bgr, motion, True, scales, None, None, max_iter=max_iter,
    )
    pack_info = {k: v for k, v in info.items() if k != 'error'}
    pack_info['fallback'] = W is None

    if W is None:
        return {
            'img_a': img_a_bgr, 'img_b': img_b_bgr, 'a_full': img_a_bgr, 'b_full': img_b_bgr,
            'roi': None, 'info': pack_info,
        }

    b_full = cv2.warpAffine(
        img_b_bgr, W, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    roi = reg.compute_crop_roi(W, h, w, h_b, w_b, float(crop_margin))
    if roi is None:
        pack_info['crop_note'] = 'no_roi'
        return {
            'img_a': img_a_bgr, 'img_b': b_full, 'a_full': img_a_bgr, 'b_full': b_full,
            'roi': None, 'info': pack_info,
        }

    x, y, cw, ch = roi
    return {
        'img_a': img_a_bgr[y:y + ch, x:x + cw],
        'img_b': b_full[y:y + ch, x:x + cw],
        'a_full': img_a_bgr,
        'b_full': b_full,
        'roi': {'x': int(x), 'y': int(y), 'w': int(cw), 'h': int(ch)},
        'info': pack_info,
    }


def _register_worker(task):
    fname, path_a, path_b, scales, crop_margin, max_iter = task
    img_a_bgr = reg.imread_unicode(path_a)
    img_b_bgr = reg.imread_unicode(path_b)
    pack = _register_core(img_a_bgr, img_b_bgr, scales, crop_margin, max_iter)
    return fname, pack, img_a_bgr.shape[0], img_a_bgr.shape[1]


def parallel_register(tasks, workers):
    results = {}
    if workers <= 1:
        for task in tqdm(tasks, desc='register'):
            fname, pack, fh, fw = _register_worker(task)
            results[fname] = (pack, fh, fw)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_register_worker, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f'register x{workers}'):
            fname, pack, fh, fw = fut.result()
            results[fname] = (pack, fh, fw)
    return results


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def crop_bgr_to_infer_rgb(img_bgr):
    ok, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok or enc is None:
        return bgr_to_rgb(img_bgr)
    return io.imread(BytesIO(enc.tobytes()))


def normalize_image(im):
    return (im / 255).astype(np.float32)


def paste_mask(full_h, full_w, crop_pred, roi):
    if roi is None:
        return crop_pred.astype(np.uint8) * 255
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    x, y, cw, ch = roi['x'], roi['y'], roi['w'], roi['h']
    pm = crop_pred.astype(bool) if crop_pred.dtype != bool else crop_pred
    full[y:y + ch, x:x + cw] = pm.astype(np.uint8) * 255
    return full


def overlay_vis(img_bgr, mask):
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


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _empty_timing_stats():
    return {
        'model_load': 0.0, 'register_batch': 0.0, 'infer_total': 0.0,
        'tta_total': 0.0, 'infer_base_total': 0.0,
        'per_register': [], 'per_infer': [], 'per_post': [], 'per_total': [], 'per_tta': [],
    }


def _record_gpu_timing(stats, gpu_timing):
    if not gpu_timing:
        return
    tta_t = gpu_timing.get('tta', 0.0)
    base_t = gpu_timing.get('base', 0.0)
    stats['tta_total'] += tta_t
    stats['infer_base_total'] += base_t
    stats['per_tta'].append(tta_t)


def _print_tta_timing(stats, n):
    tta_total = stats.get('tta_total', 0.0)
    if tta_total <= 0:
        return
    base_total = stats.get('infer_base_total', 0.0)
    print(f'     · TTA 额外耗时:     {tta_total:.2f} s  (均摊 {tta_total / n:.2f} s/张, 3×flip)')
    print(f'     · 基础前向耗时:     {base_total:.2f} s  (均摊 {base_total / n:.2f} s/张)')


def print_timing(stats, register_mode, n_processed):
    print('\n⏱  计时统计')
    print(f'   - 模型加载时间:       {stats["model_load"]:.2f} s')
    reg_batch = stats.get('register_batch', 0.0)
    infer_total = stats['infer_total']
    n = max(n_processed, 1)
    if reg_batch > 0:
        print(f'   - 配准总时间:         {reg_batch:.2f} s  ({n_processed} 对, {register_mode}, 并行)')
        print(f'   - 推理总时间:         {infer_total:.2f} s  ({n_processed} 张)')
        pipeline = reg_batch + infer_total
        print(f'   - 流水线总时间:       {pipeline:.2f} s  (配准+推理, 不含模型加载)')
        print(f'   - 平均单图总时间:     {pipeline / n:.2f} s')
        print(f'     · register (均摊):  {_avg(stats["per_register"]):.2f} s')
        print(f'     · infer+预处理:     {_avg(stats["per_infer"]):.2f} s  (SAM-CD ONNX 滑窗等)')
        _print_tta_timing(stats, n)
        print(f'     · postprocess:      {_avg(stats["per_post"]):.2f} s  (贴回 + vismask + 保存)')
    else:
        print(f'   - 全部推理时间:       {infer_total:.2f} s  ({n_processed} 张)')
        print(f'   - 平均单图总时间:     {_avg(stats["per_total"]):.2f} s')
        print(f'     · inference:        {_avg(stats["per_infer"]):.2f} s')
        _print_tta_timing(stats, n)
        print(f'     · postprocess:      {_avg(stats["per_post"]):.2f} s')


def predict_plain(runner, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, enable_time):
    stats = _empty_timing_stats()
    t_infer_all = time.perf_counter()
    for it in tqdm(valid_list, desc='Inference'):
        t_img = time.perf_counter()
        bgr_a = reg.imread_unicode(os.path.join(img_a_dir, it))
        bgr_b = reg.imread_unicode(os.path.join(img_b_dir, it))
        if bgr_a is None or bgr_b is None:
            continue
        img_a = normalize_image(bgr_to_rgb(bgr_a))
        img_b = normalize_image(bgr_to_rgb(bgr_b))

        gpu_timing = {'base': 0.0, 'tta': 0.0} if enable_time else None
        t_stage = time.perf_counter()
        pred = run_infer(runner, opt, img_a, img_b, gpu_timing)
        infer_t = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        io.imsave(os.path.join(bcd_map_dir, it), img_as_ubyte(pred))
        post_t = time.perf_counter() - t_stage

        if enable_time:
            _record_gpu_timing(stats, gpu_timing)
            stats['per_infer'].append(infer_t)
            stats['per_post'].append(post_t)
            stats['per_total'].append(time.perf_counter() - t_img)

    stats['infer_total'] = time.perf_counter() - t_infer_all
    return stats


def predict_with_register(runner, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, vis_dir, reg_cfg, enable_time):
    scales = [float(x) for x in reg_cfg.get('scales', reg.DEFAULT_SCALES)]
    max_iter = int(reg_cfg.get('max_iter', reg.DEFAULT_MAX_ITER))
    crop_margin = reg.DEFAULT_CROP_MARGIN
    workers = int(reg_cfg.get('workers', 4))

    tasks = [
        (fn, os.path.join(img_a_dir, fn), os.path.join(img_b_dir, fn),
         scales, crop_margin, max_iter)
        for fn in valid_list
    ]

    stats = _empty_timing_stats()

    t_reg = time.perf_counter()
    reg_results = parallel_register(tasks, workers)
    stats['register_batch'] = time.perf_counter() - t_reg

    t_infer_all = time.perf_counter()
    reg_share = stats['register_batch'] / max(len(valid_list), 1)
    for fname in tqdm(valid_list, desc='infer'):
        t_img = time.perf_counter()
        pack, full_h, full_w = reg_results[fname]
        img_a_rgb = normalize_image(crop_bgr_to_infer_rgb(pack['img_a']))
        img_b_rgb = normalize_image(crop_bgr_to_infer_rgb(pack['img_b']))

        gpu_timing = {'base': 0.0, 'tta': 0.0} if enable_time else None
        t_stage = time.perf_counter()
        pred_crop = run_infer(runner, opt, img_a_rgb, img_b_rgb, gpu_timing)
        infer_t = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        pred_full = paste_mask(full_h, full_w, pred_crop, pack['roi'])
        reg.imwrite_unicode(os.path.join(bcd_map_dir, fname), pred_full)
        vis = overlay_vis(pack['a_full'], pred_full)
        reg.imwrite_unicode(os.path.join(vis_dir, fname), vis)
        post_t = time.perf_counter() - t_stage

        if enable_time:
            _record_gpu_timing(stats, gpu_timing)
            loop_t = time.perf_counter() - t_img
            stats['per_register'].append(reg_share)
            stats['per_infer'].append(loop_t - post_t)
            stats['per_post'].append(post_t)
            stats['per_total'].append(reg_share + loop_t)

    stats['infer_total'] = time.perf_counter() - t_infer_all
    return stats


def _parse_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in ['1', 'true', 'yes', 'y']


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    config_path = resolve_config_path(opt.config)
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    pred_cfg = cfg.get('predict', {})
    reg_cfg = pred_cfg.get('register', {}) or {}

    if opt.crop_size is None:
        crop = pred_cfg.get('crop_size', [512, 512])
        if isinstance(crop, (list, tuple)):
            opt.crop_size = (int(crop[0]), int(crop[1]))
        else:
            opt.crop_size = (int(crop), int(crop))
    else:
        opt.crop_size = (int(opt.crop_size[0]), int(opt.crop_size[1]))

    if opt.fastsam_imgsz is None:
        opt.fastsam_imgsz = int(pred_cfg.get('fastsam_imgsz', opt.crop_size[0]))
    else:
        opt.fastsam_imgsz = int(opt.fastsam_imgsz)

    if opt.TTA is None:
        opt.TTA = bool(pred_cfg.get('tta', False))
    else:
        opt.TTA = _parse_bool(opt.TTA)

    if opt.threshold is None:
        opt.threshold = float(pred_cfg.get('threshold', 0.5))

    if opt.inference_sliding_crop is None:
        opt.inference_sliding_crop = bool(pred_cfg.get('inference_sliding_crop', True))
    else:
        opt.inference_sliding_crop = _parse_bool(opt.inference_sliding_crop)

    if opt.test_dir is None:
        opt.test_dir = pred_cfg.get('test_dir') or pred_cfg.get('dataset')
    if opt.pred_dir is None:
        opt.pred_dir = pred_cfg.get('pred_dir') or pred_cfg.get('out_dir')

    fastsam_onnx = resolve_path(opt.fastsam_onnx or pred_cfg.get('fastsam_onnx'))
    cd_head_onnx = resolve_path(opt.cd_head_onnx or pred_cfg.get('cd_head_onnx'))

    if opt.dev_id is None:
        opt.dev_id = int(pred_cfg.get('dev_id', 0))
    enable_time = bool(pred_cfg.get('time', False)) if opt.time is None else bool(opt.time)

    register_enable = bool(reg_cfg.get('enable', False))

    test_dir = resolve_path(opt.test_dir)
    pred_dir = resolve_path(opt.pred_dir)

    for label, path in [
        ('FastSAM ONNX', fastsam_onnx),
        ('CD Head ONNX', cd_head_onnx),
        ('测试集', test_dir),
    ]:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f'缺少 {label}: {path}')

    if opt.fastsam_imgsz != opt.crop_size[0]:
        raise ValueError(
            f'fastsam_imgsz({opt.fastsam_imgsz}) 须与 crop_size[0]({opt.crop_size[0]}) 一致；'
            'PyTorch/ONNX 在 tensor 输入下不做 letterbox 放大，否则特征尺度错误。'
        )

    img_a_dir = os.path.join(test_dir, 'A')
    img_b_dir = os.path.join(test_dir, 'B')
    valid_list = sorted(
        f for f in os.listdir(img_a_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_register' not in f
    )

    bcd_map_dir = os.path.join(pred_dir, 'bcd_map')
    vis_dir = os.path.join(pred_dir, 'vismask')
    os.makedirs(bcd_map_dir, exist_ok=True)
    if register_enable:
        os.makedirs(vis_dir, exist_ok=True)

    print(f'Prediction masks -> {bcd_map_dir}')
    if register_enable:
        print(f'Register: enable=True, mode={REGISTER_MODE}, workers={reg_cfg.get("workers", 4)}, '
              f'scales={reg_cfg.get("scales", reg.DEFAULT_SCALES)}, max_iter={reg_cfg.get("max_iter", reg.DEFAULT_MAX_ITER)}')
        print(f'Vismask -> {vis_dir}')
    mode = 'sliding crop' if opt.inference_sliding_crop else 'resize'
    print(f'Inference mode: {mode}, crop_size={opt.crop_size}, threshold={opt.threshold}, TTA={opt.TTA}')
    print(f'FastSAM ONNX: {fastsam_onnx}')
    print(f'CD Head ONNX: {cd_head_onnx}')

    import torch
    t_load = time.perf_counter()
    runner = SamCdOnnxRunner(fastsam_onnx, cd_head_onnx, device_prefer_cuda=torch.cuda.is_available())
    model_load_time = time.perf_counter() - t_load

    if register_enable:
        stats = predict_with_register(
            runner, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, vis_dir, reg_cfg, enable_time,
        )
    else:
        stats = predict_plain(
            runner, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, enable_time,
        )

    stats['model_load'] = model_load_time
    if enable_time:
        print_timing(stats, REGISTER_MODE if register_enable else 'none', len(valid_list))

    print(f'\n✅ 处理完成，共 {len(valid_list)} 对')
    print(f'   - 二值变化图: bcd_map/')
    if register_enable:
        print(f'   - 可视化叠加: vismask/')
    print(f'Total time: {time.time() - begin_time:.2f}s')


if __name__ == '__main__':
    main()
