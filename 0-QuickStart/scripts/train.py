import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import time
import random
import argparse
import shutil
import glob
import contextlib
import numpy as np
import yaml
import torch
import torch.autograd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import cv2
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torchvision.utils import make_grid

working_path = PROJECT_ROOT
os.environ.setdefault('YOLO_VERBOSE', 'False')


class _Tee:
    """同时写到 stdout 和 results.txt 的流包装器（所有 print 自动同步）。"""
    def __init__(self, filepath, mode='w'):
        self._file = open(filepath, mode, encoding='utf-8', buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


@contextlib.contextmanager
def suppress_third_party_output():
    """Suppress noisy stdout/stderr emitted inside third-party model forward calls."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)

from utils.loss import LatentSimilarity, weighted_BCE_logits, BinaryDiceLoss, binary_focal_loss
from utils.metric_tool import ConfuseMatrixMeter
from utils.utils import (
    AverageMeter,
    BinaryMetricsTracker,
    apply_miou_mf1,
    make_patch_metrics_meters,
    metric_mode_label,
    metrics_from_patch_meters,
    metrics_from_tracker,
    update_cm_from_preds,
    update_patch_metrics,
)

###################### Data and Model ########################
from models.SAM_CD import SAM_CD as Net
NET_NAME = 'SAM_CD'

from utils import datasets as RS
from utils.checkpoint import (
    prepare_checkpoint_for_save,
    strip_module_prefix,
    torch_load_compat,
    unpack_checkpoint_state_dict,
)
DATA_NAME = 'Levir_CD'
###################### Data and Model ########################

args = {}
writer = None


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config', 'default.yaml'), help='yaml config path')
    parser.add_argument('--project', default=None, help='override project name')
    parser.add_argument('--data_root', default=None, help='override dataset root')
    parser.add_argument('--epochs', type=int, default=None, help='override epochs')
    parser.add_argument('--batch', type=int, default=None, help='override train/val batch size')
    parser.add_argument('--lr', type=float, default=None, help='override learning rate')
    parser.add_argument('--load_path', default=None, help='override checkpoint path for finetuning')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint_last.pth (restore optimizer + epoch)')
    return parser.parse_args()


def apply_config(cfg, cli, local_rank):
    train_cfg = cfg.get('train', {})
    dataset_cfg = cfg.get('dataset', {})

    dataset_root = cli.data_root if cli.data_root else dataset_cfg.get('root')
    if dataset_root:
        if dataset_root.endswith(('.yaml', '.yml')):
            with open(dataset_root, 'r', encoding='utf-8') as f:
                ds_cfg = yaml.safe_load(f) or {}
            RS.list_files = {}
            for k in ('train', 'val'):
                if k in ds_cfg and isinstance(ds_cfg[k], list) and ds_cfg[k]:
                    RS.list_files[k] = ds_cfg[k]
        else:
            RS.root = dataset_root

    project = cli.project if cli.project else train_cfg.get('project', 'default')
    run_base_dir = train_cfg.get('run_base_dir', os.path.join(working_path, 'runs', '0-train'))
    run_dir = os.path.join(run_base_dir, project)

    pred_dir_cfg = train_cfg.get('pred_dir')
    chkpt_dir_cfg = train_cfg.get('chkpt_dir')
    log_dir_cfg = train_cfg.get('log_dir')

    batch_size = cli.batch if cli.batch is not None else train_cfg.get('train_batch_size', 4)
    args.update({
        'train_batch_size': batch_size,
        'val_batch_size': cli.batch if cli.batch is not None else train_cfg.get('val_batch_size', 4),
        'lr': cli.lr if cli.lr is not None else train_cfg.get('lr', 0.1),
        'epochs': cli.epochs if cli.epochs is not None else train_cfg.get('epochs', 50),
        'dev_id': local_rank,
        'weight_decay': train_cfg.get('weight_decay', 5e-4),
        'momentum': train_cfg.get('momentum', 0.9),
        'print_freq': train_cfg.get('print_freq', 50),
        'predict_step': train_cfg.get('predict_step', 5),
        'crop_size': train_cfg.get('crop_size', 512),
        'num_workers': train_cfg.get('num_workers', 4),
        'project': project,
        'run_dir': run_dir,
        'pred_dir': pred_dir_cfg if pred_dir_cfg else os.path.join(run_dir, 'vis'),
        'chkpt_dir': chkpt_dir_cfg if chkpt_dir_cfg else os.path.join(run_dir, 'checkpoint'),
        'log_dir': log_dir_cfg if log_dir_cfg else os.path.join(run_dir, 'checkpoint'),
        'load_path': cli.load_path if cli.load_path else train_cfg.get('load_path'),
        'train_random_crop': train_cfg.get('train_random_crop', True),
        'train_crop_nums': train_cfg.get('train_crop_nums', 10),
        'train_random_flip': float(train_cfg.get('train_random_flip', 0.75)),
        'train_random_scale': float(train_cfg.get('train_random_scale', 0.0)),
        'train_pos_aware_crop': train_cfg.get('train_pos_aware_crop', False),
        'train_pos_aware_crop_ratio': train_cfg.get('train_pos_aware_crop_ratio', 0.5),
        'train_loss_type': train_cfg.get('train_loss_type', 'bce'),
        'val_random_flip': train_cfg.get('val_random_flip', False),
        'patch_average': train_cfg.get('patch_average', False),
    })

    if dist.get_rank() == 0:
        for out_dir in [args['log_dir'], args['chkpt_dir'], args['pred_dir']]:
            os.makedirs(out_dir, exist_ok=True)
        _snapshot_configs(args['run_dir'], cli.config)
    dist.barrier()


def _snapshot_configs(run_dir, cfg_path):
    """训练开始前把 config + data/*.yaml 快照到 run_dir/config/"""
    dst = os.path.join(run_dir, 'config')
    os.makedirs(dst, exist_ok=True)
    # 主配置文件
    shutil.copy2(cfg_path, dst)
    # data/ 目录下所有 yaml
    for p in glob.glob(os.path.join(working_path, 'data', '*.yaml')):
        shutil.copy2(p, dst)
    print(f'[Config snapshot] saved to {dst}')


def load_checkpoint_if_needed(net):
    load_path = args.get('load_path')
    if not load_path:
        if dist.get_rank() == 0:
            print('No load_path provided. Train from current initialization.')
        return
    if not os.path.exists(load_path):
        if dist.get_rank() == 0:
            print(f'load_path does not exist: {load_path}. Skip loading.')
        return

    checkpoint = torch_load_compat(load_path, map_location='cpu')
    try:
        state_dict = unpack_checkpoint_state_dict(checkpoint, path_hint=load_path)
    except (ValueError, TypeError):
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
    cleaned_state = strip_module_prefix(state_dict)
    incompatible = net.load_state_dict(cleaned_state, strict=False)
    if dist.get_rank() == 0:
        print(f'Loaded finetune checkpoint from: {load_path}')
        if incompatible.missing_keys:
            print(f'Missing keys when loading checkpoint: {len(incompatible.missing_keys)}')
        if incompatible.unexpected_keys:
            print(f'Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)}')


def main():
    global writer

    # torchrun 自动设置 LOCAL_RANK / RANK / WORLD_SIZE 环境变量
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    cli = parse_cli()
    cfg = load_config(cli.config)
    apply_config(cfg, cli, local_rank)

    rank = dist.get_rank()
    _tee = None
    if rank == 0:
        writer = SummaryWriter(args['log_dir'])
        results_path = os.path.join(args['chkpt_dir'], 'results.txt')
        _tee = _Tee(results_path, mode='w')
        print('[Logging] stdout -> %s' % results_path)

    net = Net().to(local_rank)
    load_checkpoint_if_needed(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_set = RS.RS(
        'train',
        random_crop=args['train_random_crop'],
        crop_nums=args['train_crop_nums'],
        crop_size=args['crop_size'],
        random_flip=args['train_random_flip'],
        random_scale=args['train_random_scale'],
        pos_aware_crop=args['train_pos_aware_crop'],
        pos_aware_crop_ratio=args['train_pos_aware_crop_ratio']
    )
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=args['num_workers'],
        sampler=train_sampler,
        persistent_workers=args['num_workers'] > 0,
        pin_memory=True,
    )

    # 验证仅在 rank0 进行，避免指标汇总的复杂性
    val_loader = None
    if rank == 0:
        val_set = RS.RS(
            'val',
            sliding_crop=True if args['train_random_crop'] else False,
            crop_size=args['crop_size'],
            random_flip=args['val_random_flip']
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args['val_batch_size'],
            num_workers=args['num_workers'],
            shuffle=False,
            persistent_workers=args['num_workers'] > 0,
            pin_memory=True,
        )

    # 正负样本混合可视化（懒加载兼容，随机采样 8 张）
    if rank == 0:
        _cs = args['crop_size']
        def _resize(arr, is_label=False):
            interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
            return cv2.resize(arr, (_cs, _cs), interpolation=interp)
        selected = random.sample(range(len(train_set)), min(8, len(train_set)))
        random.shuffle(selected)
        imgs_A, imgs_B, labels_np = [], [], []
        for si in selected:
            a, b, l = train_set[si]
            imgs_A.append(a)
            imgs_B.append(b)
            labels_np.append(l.numpy() if hasattr(l, 'numpy') else l)
        imgs_A = torch.stack(imgs_A)
        imgs_B = torch.stack(imgs_B)
        labels_np = np.stack(labels_np)
        save_batch_viz(imgs_A, imgs_B, labels_np,
                       os.path.join(args['pred_dir'], 'train_vis.jpg'))
        print(f'train_vis.jpg saved: {len(selected)} samples')

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        args['lr'],
        weight_decay=args['weight_decay'],
        momentum=args['momentum'],
        nesterov=True
    )

    # 断点续训：从 checkpoint_last.pth 恢复（支持新旧两种格式）
    resume_state = {}
    if cli.resume:
        ckpt_path = os.path.join(args['chkpt_dir'], 'checkpoint_last.pth')
        if not os.path.isfile(ckpt_path):
            if rank == 0:
                print(f'[Resume] checkpoint not found: {ckpt_path}, starting from scratch.')
        else:
            ckpt = torch_load_compat(ckpt_path, map_location='cpu')
            # 优先恢复模型权重（兼容旧格式 flat dict 和新格式嵌套 dict）
            if isinstance(ckpt, dict):
                if 'model' in ckpt and isinstance(ckpt['model'], dict):
                    sd = strip_module_prefix(ckpt['model'])
                else:
                    sd = strip_module_prefix(ckpt)
                net.load_state_dict(sd, strict=False)
                if rank == 0:
                    print(f'[Resume] loaded model weights from {ckpt_path}')

            # 恢复 optimizer + epoch 信息（新格式才有）
            if isinstance(ckpt, dict) and 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                resume_state = {
                    'start_epoch': ckpt.get('epoch', 0) + 1,
                    'resume_bestF': ckpt.get('bestF', -1.0),
                    'resume_best_metrics': ckpt.get('best_metrics', None),
                    'resume_bestaccT': ckpt.get('bestaccT', 0.0),
                    'resume_bestEpoch': ckpt.get('bestEpoch', -1),
                }
                if rank == 0:
                    print(f'[Resume] restored optimizer, continuing from epoch {resume_state["start_epoch"]}')
            else:
                if rank == 0:
                    print('[Resume] ⚠️  checkpoint 未保存 optimizer，学习率将从初始值 {:.6f} 重新开始'.format(args['lr']))

    train(train_loader, train_sampler, net, optimizer, val_loader, **resume_state)

    if rank == 0:
        writer.close()
        if _tee is not None:
            _tee.close()

    dist.destroy_process_group()


def _process_batch_metrics(preds, labels, patch_average, meters, tracker, cm_meter):
    """Update per-class metrics (patch/micro) and global confusion matrix (mIoU/mF1)."""
    batch_acc = AverageMeter()
    for pred, label in zip(preds, labels):
        update_cm_from_preds(cm_meter, pred, label)
        if patch_average:
            acc, _, _ = update_patch_metrics(meters, pred, label)
            batch_acc.update(acc)
        else:
            tracker.update(pred, label)
    if patch_average:
        return batch_acc.avg
    acc1, _, _, _, _ = tracker.cls1_metrics()
    return acc1


def _get_metrics(patch_average, meters, tracker, cm_meter):
    if patch_average:
        metrics = metrics_from_patch_meters(meters)
    else:
        metrics = metrics_from_tracker(tracker)
    return apply_miou_mf1(metrics, cm_meter)


def _print_cd_metrics(metrics, prefix='', mode_label=''):
    tag = f' [{mode_label}]' if mode_label else ''
    print(f'{prefix}cls_0{tag}：Iou=%.5f, F1=%.5f, Pre=%.5f, Rec=%.5f' % (
        metrics['iou0'], metrics['f10'], metrics['pre0'], metrics['rec0']))
    print(f'{prefix}cls_1{tag}：Iou=%.5f, F1=%.5f, Pre=%.5f, Rec=%.5f' % (
        metrics['iou1'], metrics['f11'], metrics['pre1'], metrics['rec1']))
    print(f'{prefix}aver{tag}： Iou=%.5f, F1=%.5f, Acc=%.4f' % (
        metrics['aver_iou'], metrics['aver_f1'], metrics['aver_acc']))


def _format_epoch_summary(metrics, mode_label=''):
    return ('[%s] Acc=%.4f, mF1=%.5f, mIou=%.5f | [cls_1] F1=%.5f, Iou=%.5f, Pre=%.5f, Rec=%.5f' % (
        mode_label,
        metrics['acc'], metrics['aver_f1'], metrics['aver_iou'],
        metrics['f11'], metrics['iou1'], metrics['pre1'], metrics['rec1']))


def train(train_loader, train_sampler, net, optimizer, val_loader,
          start_epoch=0, resume_bestF=-1.0, resume_best_metrics=None,
          resume_bestaccT=0.0, resume_bestEpoch=-1):
    rank = dist.get_rank()
    bestF = resume_bestF
    best_metrics = resume_best_metrics
    bestaccT = resume_bestaccT
    bestEpoch = resume_bestEpoch

    chkpt_dir = args['chkpt_dir']
    path_ckpt_last = os.path.join(chkpt_dir, 'checkpoint_last.pth')
    path_best = os.path.join(chkpt_dir, 'best.pth')

    curr_epoch = start_epoch
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    device = torch.device('cuda', int(args['dev_id']))
    criterion_sem = LatentSimilarity(T=3.0).to(device)
    scaler = GradScaler('cuda')
    loss_type = args.get('train_loss_type', 'bce')

    def compute_loss_bn(outputs, labels):
        if loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(outputs, labels)
        elif loss_type == 'weighted_bce':
            # 根据当前 batch 正负像素数动态加权
            return weighted_BCE_logits(outputs, labels, weight_pos=0.5, weight_neg=0.5)
        elif loss_type == 'dice':
            probs = F.sigmoid(outputs)
            return BinaryDiceLoss(smooth=1, p=2, reduction='mean')(probs, labels)
        elif loss_type == 'focal':
            return binary_focal_loss(outputs, labels, alpha=0.7, gamma=3.0)
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

    _SEP = '-' * 77
    patch_average = args.get('patch_average', False)
    mode_label = metric_mode_label(patch_average)

    if rank == 0 and start_epoch > 0:
        print(f'[Resume] continuing from epoch {start_epoch}, bestF={resume_bestF:.5f} @ epoch {resume_bestEpoch}')
    while True:
        torch.cuda.empty_cache()
        train_sampler.set_epoch(curr_epoch)  # 保证每个 epoch 的 shuffle 不同
        net.train()
        if rank == 0:
            print(f'[epoch {curr_epoch}]  metric_mode={mode_label}')
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()
        patch_meters = make_patch_metrics_meters() if patch_average else None
        micro_tracker = BinaryMetricsTracker() if not patch_average else None
        cm_meter = ConfuseMatrixMeter(n_class=2)
        cm_meter.clear()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, labels = data

            imgs_A = imgs_A.to(device, non_blocking=True).float()
            imgs_B = imgs_B.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad()
            with suppress_third_party_output(), autocast('cuda'):
                outputs, outA, outB = net(imgs_A, imgs_B)
                loss_bn = compute_loss_bn(outputs, labels)
                loss_t = criterion_sem(outA, outB, labels)
                loss = loss_bn + loss_t
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            batch_acc = _process_batch_metrics(
                preds, labels, patch_average, patch_meters, micro_tracker, cm_meter)
            acc_meter.update(batch_acc)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            is_first_iter = i == 0
            is_last_iter = (i + 1) == len(train_loader)
            if rank == 0 and (is_first_iter or (i + 1) % args['print_freq'] == 0 or is_last_iter):
                train_m = _get_metrics(patch_average, patch_meters, micro_tracker, cm_meter)
                print('Is_training: True [%s] | [iter %d / %d] [lr %.6f] [loss %.4f] '
                      '[Acc %.4f] [Iou %.5f] [F1 %.5f]' % (
                    mode_label, i + 1, len(train_loader), optimizer.param_groups[0]['lr'],
                    train_loss.val, train_m['acc'], train_m['iou1'], train_m['f11']))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        # rank0 跑验证，其余进程在 barrier 等待
        if rank == 0:
            print(_SEP)
            print('Validating...')
            val_m, val_loss, vis_data = validate(val_loader, net, curr_epoch)
            val_F = val_m['f11']
            sd = strip_module_prefix(net.module.state_dict())
            torch.save({
                'model': {k: v.detach().cpu() for k, v in sd.items()},
                'optimizer': optimizer.state_dict(),
                'epoch': curr_epoch,
                'bestF': bestF,
                'bestEpoch': bestEpoch,
                'bestaccT': bestaccT,
                'best_metrics': best_metrics,
            }, path_ckpt_last)
            if acc_meter.avg > bestaccT:
                bestaccT = acc_meter.avg

            # 首次验证或 cls_1 F1 创新高时记为 best
            is_best = (best_metrics is None) or (val_F > bestF)
            if is_best:
                bestF = val_F
                best_metrics = val_m
                bestEpoch = curr_epoch
                torch.save(
                    prepare_checkpoint_for_save(sd, mode='heads_only', strip_dp_prefix=False),
                    path_best,
                )
                print('⭐[epoch %d/%d %.1fs]  New best! %s' % (
                    curr_epoch, args['epochs'], time.time() - begin_time,
                    _format_epoch_summary(val_m, mode_label)))
            else:
                ref_m = best_metrics if best_metrics is not None else val_m
                print('[epoch %d/%d %.1fs]  best @ epoch %d %s' % (
                    curr_epoch, args['epochs'], time.time() - begin_time,
                    bestEpoch, _format_epoch_summary(ref_m, mode_label)))

            # 保存可视化：每 predict_step epoch 一次，或 new best
            if curr_epoch % args['predict_step'] == 0 \
                    or curr_epoch == args['epochs'] - 1 \
                    or is_best:
                if vis_data is not None:
                    vA, vB, vL, vP = vis_data
                    save_batch_viz(vA, vB, vL,
                                   os.path.join(args['pred_dir'], f'val_vis-{curr_epoch}.jpg'),
                                   preds=vP)

        dist.barrier(device_ids=[int(args['dev_id'])])
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            if rank == 0:
                print(_SEP)
                print('Training finished.')
                print('Best checkpoint: %s' % path_best)
                if best_metrics is not None:
                    print('Best @ epoch %d %s' % (
                        bestEpoch, _format_epoch_summary(best_metrics, mode_label)))
            return


def _collect_vis_samples(imgs_A, imgs_B, labels_np, preds, vis_pos, vis_neg, max_vis=8):
    """流式收集可视化样本：优先正样本，最多 max_vis 条。"""
    imgs_A = imgs_A.cpu()
    imgs_B = imgs_B.cpu()
    for i in range(imgs_A.shape[0]):
        label_i = labels_np[i]
        pred_i = preds[i]
        if label_i.max() > 0 and len(vis_pos['A']) < max_vis:
            bucket = vis_pos
        elif label_i.max() == 0 and len(vis_neg['A']) < max_vis:
            bucket = vis_neg
        else:
            continue
        bucket['A'].append(imgs_A[i:i + 1])
        bucket['B'].append(imgs_B[i:i + 1])
        bucket['L'].append(label_i)
        bucket['P'].append(pred_i)


def _pack_vis_data(vis_pos, vis_neg, max_vis=8):
    """将流式缓存的正/负样本拼成 save_batch_viz 所需格式。"""
    need_neg = max(0, max_vis - len(vis_pos['A']))
    a_list = vis_pos['A'] + vis_neg['A'][:need_neg]
    b_list = vis_pos['B'] + vis_neg['B'][:need_neg]
    l_list = vis_pos['L'] + vis_neg['L'][:need_neg]
    p_list = vis_pos['P'] + vis_neg['P'][:need_neg]
    if not a_list:
        return None
    return (
        torch.cat(a_list, dim=0),
        torch.cat(b_list, dim=0),
        np.stack(l_list, axis=0),
        np.stack(p_list, axis=0),
    )


def validate(val_loader, net, curr_epoch):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()
    device = torch.device('cuda', int(args['dev_id']))
    patch_average = args.get('patch_average', False)
    mode_label = metric_mode_label(patch_average)

    val_loss = AverageMeter()
    patch_meters = make_patch_metrics_meters() if patch_average else None
    micro_tracker = BinaryMetricsTracker() if not patch_average else None
    cm_meter = ConfuseMatrixMeter(n_class=2)

    max_vis = 8
    vis_pos = {'A': [], 'B': [], 'L': [], 'P': []}
    vis_neg = {'A': [], 'B': [], 'L': [], 'P': []}
    n_batches = len(val_loader)
    log_step = max(1, n_batches // 10)

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad(), suppress_third_party_output():
            outputs, outA, outB = net(imgs_A, imgs_B)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels_np = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        _process_batch_metrics(
            preds, labels_np, patch_average, patch_meters, micro_tracker, cm_meter)

        _collect_vis_samples(imgs_A, imgs_B, labels_np, preds, vis_pos, vis_neg, max_vis)

        if (vi + 1) % log_step == 0 or (vi + 1) == n_batches:
            print(f'  Validating {vi + 1}/{n_batches} batches...', flush=True)

        del imgs_A, imgs_B, labels, outputs, outA, outB

    vis_data = _pack_vis_data(vis_pos, vis_neg, max_vis)
    del vis_pos, vis_neg

    _SEP = '-' * 77
    val_m = _get_metrics(patch_average, patch_meters, micro_tracker, cm_meter)
    print('Is_val: True [%s] |  [loss %.4f] [Acc %.4f]' % (
        mode_label, val_loss.average(), val_m['acc']))
    print(_SEP)
    _print_cd_metrics(val_m, mode_label=mode_label)
    print(_SEP)

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', val_m['acc'], curr_epoch)
    writer.add_scalar('val_F1_cls1', val_m['f11'], curr_epoch)
    writer.add_scalar('val_IoU_cls1', val_m['iou1'], curr_epoch)
    writer.add_scalar('val_mF1', val_m['aver_f1'], curr_epoch)
    writer.add_scalar('val_mIoU', val_m['aver_iou'], curr_epoch)

    return val_m, val_loss.avg, vis_data


def adjust_lr(optimizer, curr_iter, all_iter, cfg_args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = cfg_args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def save_batch_viz(imgs_A, imgs_B, labels, save_path, preds=None, max_samples=8):
    """
    BIT_CD-style batch visualization for change detection.
    Layout (top to bottom): A grid | B grid | Pred grid | GT grid
    Each row arranges batch samples horizontally via make_grid.
    """
    B = min(imgs_A.shape[0], max_samples)
    nrow = B

    imgs_A = imgs_A[:B].cpu().float().clamp(0, 1)
    imgs_B = imgs_B[:B].cpu().float().clamp(0, 1)

    gt_list = []
    for i in range(B):
        if hasattr(labels, 'cpu'):
            gt_slice = labels[i].cpu().numpy()
        else:
            gt_slice = labels[i]
        if gt_slice.ndim == 3:
            gt_slice = gt_slice[0]
        gt = (gt_slice > 0.5).astype(np.float32)
        gt_list.append(torch.from_numpy(gt).unsqueeze(0))
    gt_vis = torch.stack(gt_list, dim=0).repeat(1, 3, 1, 1)

    vis_a = make_grid(imgs_A, nrow=nrow, padding=2, pad_value=1.0)
    vis_b = make_grid(imgs_B, nrow=nrow, padding=2, pad_value=1.0)
    vis_gt = make_grid(gt_vis, nrow=nrow, padding=2, pad_value=1.0)

    if preds is not None:
        pred_list = []
        for i in range(B):
            pr = (preds[i, 0] > 0.5).astype(np.float32)
            pred_list.append(torch.from_numpy(pr).unsqueeze(0))
        pred_vis = torch.stack(pred_list, dim=0).repeat(1, 3, 1, 1)
        vis_pred = make_grid(pred_vis, nrow=nrow, padding=2, pad_value=1.0)
        grids = [vis_a, vis_b, vis_pred, vis_gt]
    else:
        grids = [vis_a, vis_b, vis_gt]

    vis = torch.cat(grids, dim=1)
    vis_np = vis.permute(1, 2, 0).numpy().clip(0, 1)
    io.imsave(save_path, (vis_np * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
