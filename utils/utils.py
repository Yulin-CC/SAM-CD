import os
import random
import numpy as np
from scipy import stats

def read_idtxt(path):
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np_input.unsqueeze(0)        
    assert len(np_output.shape) == expected_dims
    return np_output

def binary_confusion(pred, label, threshold=0.5):
    """Return TP, FP, FN, TN for one prediction map (float prob or bool)."""
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= threshold)
    label = (label >= threshold)
    tp = float((pred * label).sum())
    fp = float((pred * (1 - label)).sum())
    fn = float(((1 - pred) * label).sum())
    tn = float(((1 - pred) * (1 - label)).sum())
    return tp, fp, fn, tn


def confusion_to_metrics(tp, fp, fn, tn):
    """Derive acc / precision / recall / F1 / IoU from confusion counts."""
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    f1 = 0.0
    if acc > 0.999 and tp == 0:
        precision = 1.0
        recall = 1.0
        iou = 1.0
    if precision > 0 and recall > 0:
        f1 = stats.hmean([precision, recall])
    return acc, precision, recall, f1, iou


def binary_accuracy(pred, label):
    tp, fp, fn, tn = binary_confusion(pred, label)
    return confusion_to_metrics(tp, fp, fn, tn)


class BinaryMetricsTracker:
    """Accumulate confusion counts for micro (all-pixel) metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp1 = self.fp1 = self.fn1 = self.tn1 = 0.0
        self.tp0 = self.fp0 = self.fn0 = self.tn0 = 0.0

    def update(self, pred, label, threshold=0.5):
        tp, fp, fn, tn = binary_confusion(pred, label, threshold)
        self.tp1 += tp
        self.fp1 += fp
        self.fn1 += fn
        self.tn1 += tn
        tp, fp, fn, tn = binary_confusion(1 - pred, 1 - label, threshold)
        self.tp0 += tp
        self.fp0 += fp
        self.fn0 += fn
        self.tn0 += tn

    def cls1_metrics(self):
        return confusion_to_metrics(self.tp1, self.fp1, self.fn1, self.tn1)

    def cls0_metrics(self):
        return confusion_to_metrics(self.tp0, self.fp0, self.fn0, self.tn0)


def make_patch_metrics_meters():
    """AverageMeter set for per-patch metric averaging."""
    return {
        'acc': AverageMeter(),
        'pre0': AverageMeter(), 'rec0': AverageMeter(),
        'f10': AverageMeter(), 'iou0': AverageMeter(),
        'pre1': AverageMeter(), 'rec1': AverageMeter(),
        'f11': AverageMeter(), 'iou1': AverageMeter(),
    }


def update_patch_metrics(meters, pred, label):
    acc, pre1, rec1, f1_1, iou1 = binary_accuracy(pred, label)
    _, pre0, rec0, f1_0, iou0 = binary_accuracy(1 - pred, 1 - label)
    meters['acc'].update(acc)
    meters['pre0'].update(pre0)
    meters['rec0'].update(rec0)
    meters['f10'].update(f1_0)
    meters['iou0'].update(iou0)
    meters['pre1'].update(pre1)
    meters['rec1'].update(rec1)
    meters['f11'].update(f1_1)
    meters['iou1'].update(iou1)
    return acc, f1_1, iou1


def metrics_from_patch_meters(meters):
    """Return dict of aggregated patch-average metrics (per-class lines only)."""
    return {
        'acc': meters['acc'].avg,
        'pre0': meters['pre0'].avg, 'rec0': meters['rec0'].avg,
        'f10': meters['f10'].avg, 'iou0': meters['iou0'].avg,
        'pre1': meters['pre1'].avg, 'rec1': meters['rec1'].avg,
        'f11': meters['f11'].avg, 'iou1': meters['iou1'].avg,
    }


def metrics_from_tracker(tracker):
    """Return dict of micro (global) per-class metrics from accumulated confusion."""
    acc1, pre1, rec1, f11, iou1 = tracker.cls1_metrics()
    _, pre0, rec0, f10, iou0 = tracker.cls0_metrics()
    return {
        'acc': acc1,
        'pre0': pre0, 'rec0': rec0,
        'f10': f10, 'iou0': iou0,
        'pre1': pre1, 'rec1': rec1,
        'f11': f11, 'iou1': iou1,
    }


def pred_label_to_class_maps(pred, label, threshold=0.5):
    """Binary prob/GT maps -> uint8 class indices {0, 1} for confusion matrix."""
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pr = (pred >= threshold).astype(np.uint8)
    gt = (label >= threshold).astype(np.uint8)
    return pr, gt


def update_cm_from_preds(cm_meter, pred, label, threshold=0.5):
    """Accumulate one sample into global 2-class confusion matrix."""
    pr, gt = pred_label_to_class_maps(pred, label, threshold)
    cm_meter.update_cm(pr=pr, gt=gt)


def apply_miou_mf1(metrics, cm_meter):
    """Fill aver_* fields with eval.py-style mIoU / mF1 / pixel Acc."""
    scores = cm_meter.get_scores()
    metrics['aver_iou'] = float(scores['miou'])
    metrics['aver_f1'] = float(scores['mf1'])
    metrics['aver_acc'] = float(scores['acc'])
    return metrics


def metric_mode_label(patch_average):
    return 'patch_avg' if patch_average else 'micro'

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)
    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    
    union_hist = pred_hist + lab_hist - TP_hist
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)

    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist, union_hist)