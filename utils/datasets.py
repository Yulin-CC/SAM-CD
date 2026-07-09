import os
import math
import random
import numpy as np
import cv2
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F

from utils.augmentations import rand_crop_CD, pos_aware_rand_crop_CD, rand_flip_CD, random_scale_CD

num_classes = 1
MEAN = np.array([123.675, 116.28, 103.53])
STD  = np.array([58.395, 57.12, 57.375])
root = 'path/to/dataset/root'
list_files = {}  # e.g. {'train': '/path/train.txt', 'val': '/path/val.txt'}

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    #im = (im - MEAN) / STD
    im = im/255
    return im.astype(np.float32)

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap

def Index2Color(pred):
    pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    return pred

def sliding_crop_CD(imgs1, imgs2, labels, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img1, img2, label in zip(imgs1, imgs2, labels):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' %len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels

def _resolve_label_path(label_dir, img_basename):
    """列表项与 A/B 文件名一致；mask 常为 stem.png 等与影像扩展名不同的情况。"""
    p = os.path.join(label_dir, img_basename)
    if os.path.isfile(p):
        return p
    stem, _ = os.path.splitext(img_basename)
    for suf in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF'):
        alt = os.path.join(label_dir, stem + suf)
        if os.path.isfile(alt):
            return alt
    return p

def read_RSimages(mode, read_list=False):
    assert mode in ['train', 'val', 'test']
    data_A, data_B, labels = [], [], []
    if mode in list_files and list_files[mode]:
        list_paths = list_files[mode]
        if isinstance(list_paths, str):
            list_paths = [list_paths]
        for list_path in list_paths:
            base_dir = os.path.dirname(list_path)
            img_A_dir = os.path.join(base_dir, 'A')
            img_B_dir = os.path.join(base_dir, 'B')
            label_dir = os.path.join(base_dir, 'label')
            with open(list_path, 'r') as f:
                data_list = [l.strip() for l in f if l.strip()]
            for idx, it in enumerate(data_list):
                if it.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_A_path = os.path.join(img_A_dir, it)
                    img_B_path = os.path.join(img_B_dir, it)
                    label_path = _resolve_label_path(label_dir, it)
                    if not os.path.isfile(label_path):
                        print('[Skip] no label file for:', it)
                        continue
                    img_A = io.imread(img_A_path)
                    img_A = normalize_image(img_A)
                    img_B = io.imread(img_B_path)
                    img_B = normalize_image(img_B)
                    label = Color2Index(io.imread(label_path))
                    data_A.append(img_A)
                    data_B.append(img_B)
                    labels.append(label)
                if not idx%50: print('%d/%d images loaded.'%(idx, len(data_list)))
        if data_A:
            print(data_A[0].shape)
        print(str(len(data_A)) + ' ' + mode + ' images loaded.')
        return data_A, data_B, labels
    else:
        img_A_dir = os.path.join(root, mode, 'A')
        img_B_dir = os.path.join(root, mode, 'B')
        label_dir = os.path.join(root, mode, 'label')
        if mode == 'train' and read_list:
            list_path = os.path.join(root, mode + '0.4_info.txt')
            with open(list_path, 'r') as f:
                data_list = [item.rstrip() for item in f]
        else:
            data_list = os.listdir(img_A_dir)
        for idx, it in enumerate(data_list):
            if it.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_A_path = os.path.join(img_A_dir, it)
                img_B_path = os.path.join(img_B_dir, it)
                label_path = _resolve_label_path(label_dir, it)
                if not os.path.isfile(label_path):
                    print('[Skip] no label file for:', it)
                    continue
                
                img_A = io.imread(img_A_path)
                img_A = normalize_image(img_A)
                img_B = io.imread(img_B_path)
                img_B = normalize_image(img_B)
                label = Color2Index(io.imread(label_path))
                
                data_A.append(img_A)
                data_B.append(img_B)
                labels.append(label)
            #if idx>10: break    
            if not idx%50: print('%d/%d images loaded.'%(idx, len(data_list)))
        print(data_A[0].shape)
        print(str(len(data_A)) + ' ' + mode + ' images loaded.')   
        return data_A, data_B, labels

class RS(data.Dataset):
    """懒加载：只存路径，__getitem__ 时再读单张图，不再一次性加载全量到内存。"""

    def __init__(self, mode, random_crop=False, crop_nums=6, sliding_crop=False, crop_size=512,
                 random_flip=0.0, random_scale=0.0, pos_aware_crop=False, pos_aware_crop_ratio=0.5):
        self.random_flip = random_flip
        self.random_scale = random_scale
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        self.pos_aware_crop = pos_aware_crop
        self.pos_aware_crop_ratio = pos_aware_crop_ratio
        self.sliding_crop = sliding_crop

        # 只收集路径，不加载图片
        self.samples = self._collect_paths(mode)

        # 简单 IO 缓存：__getitem__ 连续请求同一张图时避免重复读盘
        self._cache_idx = -1
        self._cache = None

        if sliding_crop and not random_crop:
            # 预计算滑窗坐标，展开为 (sample_idx, y1, y2, x1, x2) 扁平列表
            self._tiles = []
            for sidx, (ap, bp, lp) in enumerate(self.samples):
                h, w = self._get_img_shape(ap)
                if h < crop_size or w < crop_size:
                    self._tiles.append((sidx, 0, h, 0, w))
                    continue
                coords = self._sliding_coords(h, w, crop_size, crop_size)
                for y1, y2, x1, x2 in coords:
                    self._tiles.append((sidx, y1, y2, x1, x2))
            self.len = len(self._tiles)
        elif self.random_crop:
            self.len = crop_nums * len(self.samples)
        else:
            self.len = len(self.samples)

    @staticmethod
    def _get_img_shape(path: str) -> tuple:
        """快速获取图片尺寸（不保留像素数据）。"""
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img.shape[:2]

    @staticmethod
    def _sliding_coords(h, w, c_h, c_w):
        """生成滑窗坐标列表 [(y1, y2, x1, x2), ...]。"""
        import math
        rows = math.ceil(h / c_h)
        cols = math.ceil(w / c_w)
        stride_h = int((c_h * rows - h) / (rows - 1)) if rows > 1 else 0
        stride_w = int((c_w * cols - w) / (cols - 1)) if cols > 1 else 0
        coords = []
        for j in range(rows):
            for i in range(cols):
                y1 = int(j * c_h - j * stride_h)
                if j == rows - 1:
                    y1 = h - c_h
                x1 = int(i * c_w - i * stride_w)
                if i == cols - 1:
                    x1 = w - c_w
                coords.append((y1, y1 + c_h, x1, x1 + c_w))
        return coords

    def _collect_paths(self, mode):
        """收集 (path_a, path_b, path_label) 三元组列表，不加载像素数据。"""
        assert mode in ['train', 'val', 'test']

        if mode in list_files and list_files[mode]:
            list_paths = list_files[mode]
            if isinstance(list_paths, str):
                list_paths = [list_paths]
            entries = []
            for list_path in list_paths:
                base_dir = os.path.dirname(list_path)
                a_dir = os.path.join(base_dir, 'A')
                b_dir = os.path.join(base_dir, 'B')
                l_dir = os.path.join(base_dir, 'label')
                with open(list_path, 'r') as f:
                    names = [l.strip() for l in f if l.strip()]
                for name in names:
                    if not name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    lp = _resolve_label_path(l_dir, name)
                    if not os.path.isfile(lp):
                        print(f'[Skip] no label: {name}')
                        continue
                    entries.append((os.path.join(a_dir, name),
                                    os.path.join(b_dir, name), lp))
                print(f'{len(entries)} samples from {list_path}')
            return entries

        # 默认 root / {mode} / {A,B,label}
        a_dir = os.path.join(root, mode, 'A')
        b_dir = os.path.join(root, mode, 'B')
        l_dir = os.path.join(root, mode, 'label')
        names = [f for f in os.listdir(a_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        entries = []
        for name in names:
            lp = _resolve_label_path(l_dir, name)
            if not os.path.isfile(lp):
                continue
            entries.append((os.path.join(a_dir, name),
                            os.path.join(b_dir, name), lp))
        print(f'{len(entries)} {mode} samples collected.')
        return entries

    def _load_triplet(self, idx):
        """加载第 idx 个三元组 (A, B, label)，归一化后返回。
        带简单缓存：相同 idx 连续调用时跳过重复 IO。
        """
        if idx == self._cache_idx and self._cache is not None:
            return self._cache
        ap, bp, lp = self.samples[idx]
        a = io.imread(ap)
        b = io.imread(bp)
        l = Color2Index(io.imread(lp))
        self._cache = (normalize_image(a), normalize_image(b), l)
        self._cache_idx = idx
        return self._cache

    def __getitem__(self, idx):
        if hasattr(self, '_tiles'):
            # 滑窗模式：从预计算的 tiles 中取一个
            sidx, y1, y2, x1, x2 = self._tiles[idx]
            data_a, data_b, label = self._load_triplet(sidx)
            data_a = data_a[y1:y2, x1:x2]
            if label.ndim == 2:
                label = label[y1:y2, x1:x2]
            else:
                label = label[y1:y2, x1:x2]
            data_b = data_b[y1:y2, x1:x2]
        elif self.random_crop:
            sidx = idx // self.crop_nums
            data_a, data_b, label = self._load_triplet(sidx)
            if self.pos_aware_crop:
                data_a, data_b, label = pos_aware_rand_crop_CD(
                    data_a, data_b, label, [self.crop_size, self.crop_size],
                    pos_ratio=self.pos_aware_crop_ratio)
            else:
                data_a, data_b, label = rand_crop_CD(
                    data_a, data_b, label, [self.crop_size, self.crop_size])
        else:
            # resize 模式：尺寸不同才缩放，相同则跳过避免多余拷贝
            data_a, data_b, label = self._load_triplet(idx)
            if data_a.shape[0] != self.crop_size or data_a.shape[1] != self.crop_size:
                dsize = (self.crop_size, self.crop_size)
                data_a = cv2.resize(data_a, dsize, interpolation=cv2.INTER_LINEAR)
                data_b = cv2.resize(data_b, dsize, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)

        if self.random_scale > 0:
            data_a, data_b, label = random_scale_CD(data_a, data_b, label, p=self.random_scale)
        if self.random_flip > 0:
            data_a, data_b, label = rand_flip_CD(data_a, data_b, label, p=self.random_flip)
        return F.to_tensor(data_a), F.to_tensor(data_b), label

    def __len__(self):
        return self.len

