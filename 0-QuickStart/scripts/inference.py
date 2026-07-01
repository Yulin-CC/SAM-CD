import math
import os
import sys
import time
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO

import cv2
import numpy as np
import torch
import yaml
from skimage import io
from skimage.util import img_as_ubyte
from torch.nn import functional as F
from torchvision.transforms import functional as transF
from tqdm import tqdm

warnings.filterwarnings("ignore", message="is a low contrast image")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from models.SAM_CD import SAM_CD as Net

NET_NAME = 'SAM_CD'
DATA_NAME = 'Levir_CD'

from utils import datasets as Data
from utils.checkpoint import strip_module_prefix, torch_load_compat, unpack_checkpoint_state_dict
from utils import register as reg


VIS_ALPHA = 0.45
VIS_HIGHLIGHT_BGR = (0, 0, 255)
REGISTER_MODE = 'crop_pasteback'


class PredOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config', 'default.yaml'))
        parser.add_argument('--crop_size', nargs=2, type=int, default=None)
        parser.add_argument('--TTA', default=None)
        parser.add_argument('--threshold', type=float, default=None)
        parser.add_argument('--inference_sliding_crop', default=None)
        parser.add_argument('--test_dir', default=None)
        parser.add_argument('--pred_dir', default=None)
        parser.add_argument('--chkpt_path', default=None)
        parser.add_argument('--dev_id', type=int, default=None)
        parser.add_argument('--time', action='store_true', default=None, help='打印分阶段计时')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        return parser.parse_args()

    def parse(self):
        return self.gather_options()


def create_crops(imgA, imgB, size):
    imgA_crops, imgB_crops = [], []
    h, w = imgA.shape[0], imgA.shape[1]
    c_h, c_w = size[0], size[1]
    if h < c_h or w < c_w:
        print(f'Cannot crop area {size} from image with size ({h}, {w})')
        return 1
    rows = math.ceil(h / c_h)
    cols = math.ceil(w / c_w)
    stride_h = int((c_h * rows - h) / (rows - 1))
    stride_w = int((c_w * cols - w) / (cols - 1))
    for j in range(rows):
        for i in range(cols):
            s_h = int(j * c_h - j * stride_h)
            if j == (rows - 1):
                s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i * c_w - i * stride_w)
            if i == (cols - 1):
                s_w = w - c_w
            e_w = s_w + c_w
            imgA_crops.append(imgA[s_h:e_h, s_w:e_w, :])
            imgB_crops.append(imgB[s_h:e_h, s_w:e_w, :])
    return imgA_crops, imgB_crops


def stitch_pred(patch_list, size_stitch):
    H, W = size_stitch
    h, w = patch_list[0].shape
    stitch_rows = math.ceil(H / h)
    stitch_cols = math.ceil(W / w)
    assert stitch_rows * stitch_cols == len(patch_list)
    h_overlap = int((h * stitch_rows - H) / (stitch_rows - 1))
    w_overlap = int((w * stitch_cols - W) / (stitch_cols - 1))
    stitched_img = None
    for r in range(stitch_rows):
        crop_t = math.ceil(h_overlap / 2)
        crop_b = h_overlap - crop_t
        crop_l = math.ceil(w_overlap / 2)
        crop_r = w_overlap - crop_l
        if r == 0:
            crop_t = 0
        if r == stitch_rows - 1:
            crop_b = 0
            crop_t = stitched_img.shape[0] - H
        stitched_r = patch_list[r * stitch_cols][crop_t:h - crop_b, 0:w - crop_r]
        for c in range(1, stitch_cols):
            if c == stitch_cols - 1:
                crop_r = 0
                crop_l = stitched_r.shape[1] - W
            patch_croped = patch_list[r * stitch_cols + c][crop_t:h - crop_b, crop_l:w - crop_r]
            stitched_r = np.concatenate((stitched_r, patch_croped), axis=1)
        stitched_img = stitched_r if r == 0 else np.concatenate((stitched_img, stitched_r), axis=0)
    return stitched_img


def _pad_to_stride(tensor, stride=32):
    """padding 使 H/W 能被 stride 整除，返回 (padded_tensor, pad_h, pad_w)。"""
    h, w = tensor.shape[2:]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
    return F.pad(tensor, (0, pad_w, 0, pad_h)), pad_h, pad_w

def infer_patch(net, img_a, img_b, opt, gpu_timing=None):
    device = torch.device('cuda', int(opt.dev_id))
    tensor_a = transF.to_tensor(img_a).unsqueeze(0).to(device).float()
    tensor_b = transF.to_tensor(img_b).unsqueeze(0).to(device).float()
    # FastSAM 要求输入 H/W 能被 32 整除，配准裁剪后的 ROI 可能不满足
    tensor_a, ph_a, pw_a = _pad_to_stride(tensor_a)
    tensor_b, ph_b, pw_b = _pad_to_stride(tensor_b)
    if gpu_timing is not None:
        t0 = time.perf_counter()
    output, _, _ = net(tensor_a, tensor_b)
    # 去除 padding 恢复原始尺寸
    if ph_a or pw_a:
        output = output[:, :, :-ph_a, :-pw_a]
    if gpu_timing is not None:
        _sync_cuda(opt.dev_id)
        gpu_timing['base'] = gpu_timing.get('base', 0.0) + time.perf_counter() - t0
    output = F.sigmoid(output)
    if opt.TTA:
        if gpu_timing is not None:
            t0 = time.perf_counter()
        for flip_dim in ([2], [3], [2, 3]):
            ta = torch.flip(tensor_a, flip_dim)
            tb = torch.flip(tensor_b, flip_dim)
            out_f, _, _ = net(ta, tb)
            out_f = torch.flip(out_f, flip_dim)
            output += F.sigmoid(out_f)
        if gpu_timing is not None:
            _sync_cuda(opt.dev_id)
            gpu_timing['tta'] = gpu_timing.get('tta', 0.0) + time.perf_counter() - t0
        output = output / 4.0
    return output.cpu().detach().numpy().squeeze() > opt.threshold


def run_infer(net, opt, img_a, img_b, gpu_timing=None):
    h, w = img_a.shape[:2]
    cs = opt.crop_size[0]                 # 模型输入尺寸（如 512）
    is_large = h > cs or w > cs
    with torch.no_grad():
        if is_large and opt.inference_sliding_crop:
            img_a_crops, img_b_crops = create_crops(img_a, img_b, opt.crop_size)
            preds = [
                infer_patch(net, img_a_crops[i], img_b_crops[i], opt, gpu_timing)
                for i in range(len(img_a_crops))
            ]
            return stitch_pred(preds, size_stitch=img_a.shape[:-1])

        if is_large:
            tile_size = cs * 2             # 中间 tile 尺寸（如 1024）
            if h <= tile_size and w <= tile_size:
                # 中小图：直接 resize 到 crop_size 推理
                resize_wh = (cs, cs)
                img_a_rs = cv2.resize(img_a, resize_wh, interpolation=cv2.INTER_LINEAR)
                img_b_rs = cv2.resize(img_b, resize_wh, interpolation=cv2.INTER_LINEAR)
                pred_rs = infer_patch(net, img_a_rs, img_b_rs, opt, gpu_timing)
                return cv2.resize(pred_rs.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                # 超大图：先滑窗到 tile_size，每块 resize 到 crop_size 推理，再拼回
                tiles = create_crops(img_a, img_b, (tile_size, tile_size))
                if isinstance(tiles, int):
                    # 图片某一维小于 tile_size，回退到直接 resize
                    resize_wh = (cs, cs)
                    img_a_rs = cv2.resize(img_a, resize_wh, interpolation=cv2.INTER_LINEAR)
                    img_b_rs = cv2.resize(img_b, resize_wh, interpolation=cv2.INTER_LINEAR)
                    pred_rs = infer_patch(net, img_a_rs, img_b_rs, opt, gpu_timing)
                    return cv2.resize(pred_rs.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                img_a_tiles, img_b_tiles = tiles
                pred_tiles = []
                for ta, tb in zip(img_a_tiles, img_b_tiles):
                    resize_wh = (cs, cs)
                    ta_rs = cv2.resize(ta, resize_wh, interpolation=cv2.INTER_LINEAR)
                    tb_rs = cv2.resize(tb, resize_wh, interpolation=cv2.INTER_LINEAR)
                    pred = infer_patch(net, ta_rs, tb_rs, opt, gpu_timing)
                    pred = cv2.resize(pred.astype(np.uint8), (tile_size, tile_size),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                    pred_tiles.append(pred)
                return stitch_pred(pred_tiles, size_stitch=(h, w))
        return infer_patch(net, img_a, img_b, opt, gpu_timing)


def _sync_cuda(dev_id):
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device('cuda', int(dev_id)))


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
    """与 A_cropped/B_cropped 落盘再 io.imread 一致（JPEG Q=95），避免内存裁剪与磁盘流水线偏差。"""
    ok, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok or enc is None:
        return bgr_to_rgb(img_bgr)
    return io.imread(BytesIO(enc.tobytes()))


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
    print(f'     · TTA 额外 GPU:     {tta_total:.2f} s  (均摊 {tta_total / n:.2f} s/张, 3×flip)')
    print(f'     · 基础前向 GPU:     {base_total:.2f} s  (均摊 {base_total / n:.2f} s/张)')


def print_timing(stats, register_mode, n_processed, dev_id):
    print('\n⏱  计时统计')
    if torch.cuda.is_available():
        print('   (GPU 段已含 torch.cuda.synchronize，为真实设备耗时)')
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
        print(f'     · infer+预处理:     {_avg(stats["per_infer"]):.2f} s  (SAM-CD 滑窗等)')
        _print_tta_timing(stats, n)
        print(f'     · postprocess:      {_avg(stats["per_post"]):.2f} s  (贴回 + vismask + 保存)')
    else:
        print(f'   - 全部推理时间:       {infer_total:.2f} s  ({n_processed} 张)')
        print(f'   - 平均单图总时间:     {_avg(stats["per_total"]):.2f} s')
        print(f'     · inference:        {_avg(stats["per_infer"]):.2f} s')
        _print_tta_timing(stats, n)
        print(f'     · postprocess:      {_avg(stats["per_post"]):.2f} s')


def predict_plain(net, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, enable_time):
    stats = _empty_timing_stats()
    t_infer_all = time.perf_counter()
    for it in tqdm(valid_list, desc='Inference'):
        t_img = time.perf_counter()
        img_a = Data.normalize_image(io.imread(os.path.join(img_a_dir, it)))
        img_b = Data.normalize_image(io.imread(os.path.join(img_b_dir, it)))

        gpu_timing = {'base': 0.0, 'tta': 0.0} if enable_time else None
        t_stage = time.perf_counter()
        pred = run_infer(net, opt, img_a, img_b, gpu_timing)
        if gpu_timing is None:
            _sync_cuda(opt.dev_id)
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


def predict_with_register(net, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, vis_dir, register_b_dir, reg_cfg, enable_time):
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
        img_a_rgb = Data.normalize_image(crop_bgr_to_infer_rgb(pack['img_a']))
        img_b_rgb = Data.normalize_image(crop_bgr_to_infer_rgb(pack['img_b']))

        gpu_timing = {'base': 0.0, 'tta': 0.0} if enable_time else None
        t_stage = time.perf_counter()
        pred_crop = run_infer(net, opt, img_a_rgb, img_b_rgb, gpu_timing)
        if gpu_timing is None:
            _sync_cuda(opt.dev_id)
        infer_t = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        pred_full = paste_mask(full_h, full_w, pred_crop, pack['roi'])
        reg.imwrite_unicode(os.path.join(bcd_map_dir, fname), pred_full)
        vis = overlay_vis(pack['a_full'], pred_full)
        reg.imwrite_unicode(os.path.join(vis_dir, fname), vis)
        reg.imwrite_unicode(os.path.join(register_b_dir, fname), pack['b_full'])
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


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    with open(opt.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    pred_cfg = cfg.get('predict', {})
    reg_cfg = pred_cfg.get('register', {}) or {}
    dataset_cfg = cfg.get('dataset', {})
    if dataset_cfg.get('root'):
        Data.root = dataset_cfg['root']

    if opt.crop_size is None:
        crop = pred_cfg.get('crop_size', [512, 512])
        opt.crop_size = (int(crop[0]), int(crop[1]))
    else:
        opt.crop_size = (int(opt.crop_size[0]), int(opt.crop_size[1]))
    if opt.TTA is None:
        opt.TTA = bool(pred_cfg.get('tta', False))
    else:
        opt.TTA = str(opt.TTA).lower() in ['1', 'true', 'yes', 'y']
    if opt.threshold is None:
        opt.threshold = float(pred_cfg.get('threshold', 0.5))
    if opt.inference_sliding_crop is None:
        opt.inference_sliding_crop = bool(pred_cfg.get('inference_sliding_crop', True))
    else:
        opt.inference_sliding_crop = str(opt.inference_sliding_crop).lower() in ['1', 'true', 'yes', 'y']
    if opt.test_dir is None:
        opt.test_dir = pred_cfg.get('test_dir', os.path.join(Data.root, 'test'))
    if opt.pred_dir is None:
        opt.pred_dir = pred_cfg.get('pred_dir', os.path.join(PROJECT_ROOT, 'eval', DATA_NAME, NET_NAME))
    if opt.chkpt_path is None:
        opt.chkpt_path = pred_cfg.get('chkpt_path', os.path.join(PROJECT_ROOT, 'checkpoints', DATA_NAME, 'xxx.pth'))
    if opt.dev_id is None:
        opt.dev_id = int(pred_cfg.get('dev_id', 0))
    enable_time = bool(pred_cfg.get('time', False)) if opt.time is None else bool(opt.time)

    register_enable = bool(reg_cfg.get('enable', False))

    img_a_dir = os.path.join(opt.test_dir, 'A')
    img_b_dir = os.path.join(opt.test_dir, 'B')
    valid_list = sorted(
        f for f in os.listdir(img_a_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_register' not in f
    )

    bcd_map_dir = os.path.join(opt.pred_dir, 'bcd_map')
    vis_dir = os.path.join(opt.pred_dir, 'vismask')
    register_b_dir = os.path.join(opt.pred_dir, 'register_B')
    os.makedirs(bcd_map_dir, exist_ok=True)
    if register_enable:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(register_b_dir, exist_ok=True)

    print(f'Prediction masks -> {bcd_map_dir}')
    if register_enable:
        print(f'Register: enable=True, mode={REGISTER_MODE}, workers={reg_cfg.get("workers", 4)}, '
              f'scales={reg_cfg.get("scales", reg.DEFAULT_SCALES)}, max_iter={reg_cfg.get("max_iter", reg.DEFAULT_MAX_ITER)}')
        print(f'Vismask -> {vis_dir}')
        print(f'Register B -> {register_b_dir}')
    mode = 'sliding crop' if opt.inference_sliding_crop else 'resize'
    print(f'Inference mode: {mode}, crop_size={opt.crop_size}, threshold={opt.threshold}, TTA={opt.TTA}')

    t_load = time.perf_counter()
    net = Net()
    raw_ckpt = torch_load_compat(opt.chkpt_path, map_location='cpu')
    state_dict = unpack_checkpoint_state_dict(raw_ckpt, path_hint=opt.chkpt_path)
    incompatible = net.load_state_dict(strip_module_prefix(state_dict), strict=False)
    if incompatible.missing_keys:
        print(f'Warning: missing_keys when loading checkpoint: {len(incompatible.missing_keys)} '
              f'(first 12: {incompatible.missing_keys[:12]})')
    net.to(torch.device('cuda', int(opt.dev_id))).eval()
    model_load_time = time.perf_counter() - t_load

    if register_enable:
        stats = predict_with_register(
            net, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, vis_dir, register_b_dir, reg_cfg, enable_time
        )
    else:
        stats = predict_plain(net, opt, valid_list, img_a_dir, img_b_dir, bcd_map_dir, enable_time)

    stats['model_load'] = model_load_time
    if enable_time:
        print_timing(stats, REGISTER_MODE if register_enable else 'none', len(valid_list), opt.dev_id)

    print(f'\n✅ 处理完成，共 {len(valid_list)} 对')
    print(f'   - 二值变化图: bcd_map/')
    if register_enable:
        print(f'   - 可视化叠加: vismask/')
        print(f'   - 配准后 B 图: register_B/')
    print(f'Total time: {time.time() - begin_time:.2f}s')


if __name__ == '__main__':
    main()
