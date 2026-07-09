"""
# @Author: 算法组
# @Date: 2026-05-18
# @Description: 变化检测数据集划分：GEOAI 目录含 A/B/label，按比例生成 train.txt / val.txt，
#   或复制到 for_training/{train,val}/{A,B,label}。
# @Mode  : txt=仅写列表；dir=复制子目录结构。
# @Split : 默认 9:1（SPLIT_RATIO=0.9），PART_RATIO=1.0 表示使用全部成对样本。
# @Command: python 2-dataSet_division.py
"""

import argparse
import os
import random
import shutil
from os.path import join

from tqdm import tqdm

#==============================#
# 接口配置
#==============================#
DATA_DIR = "path/to/dataset/dir"
MODE = "txt"           # txt 或 dir
SPLIT_RATIO = 0.9      # 训练集占比
PART_RATIO = 1.0       # 使用样本比例（1.0=全部）
RANDOM_SEED = 42       # 固定随机种子，便于复现
#==============================#

LABEL_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def check_geoai_layout(path: str) -> bool:
    """检查 A/B/label 目录。"""
    ok = True
    for sub in ("A", "B", "label"):
        if not os.path.isdir(join(path, sub)):
            print(f"  [缺少] 无 {sub}/")
            ok = False
    return ok


def _label_path(label_dir: str, img_name: str) -> str | None:
    """A/B 常为 .JPG，label 多为同名 .png，返回实际 label 文件路径。"""
    if os.path.isfile(join(label_dir, img_name)):
        return join(label_dir, img_name)
    stem, _ = os.path.splitext(img_name)
    for suf in LABEL_EXTS:
        p = join(label_dir, stem + suf)
        if os.path.isfile(p):
            return p
    return None


def list_paired_images(path: str) -> list:
    """列出 A/B/label 均存在的图像文件名（以 A 目录文件名为准）。"""
    a_dir = join(path, "A")
    b_dir = join(path, "B")
    label_dir = join(path, "label")
    imgs = []
    for fname in os.listdir(a_dir):
        low = fname.lower()
        if not any(low.endswith(ext) for ext in IMAGE_EXTS):
            continue
        if not os.path.isfile(join(b_dir, fname)):
            continue
        if _label_path(label_dir, fname) is None:
            continue
        imgs.append(fname)
    return imgs


def split_train_val(imgs: list, split_ratio: float, part_ratio: float) -> tuple[list, list]:
    """打乱后按比例截取，再划分 train / val。"""
    imgs = list(imgs)
    random.shuffle(imgs)

    n = int(len(imgs) * part_ratio)
    imgs = imgs[:n] if n > 0 else imgs

    if len(imgs) <= 1:
        return imgs, []

    k = int(len(imgs) * split_ratio)
    k = max(1, min(k, len(imgs) - 1))
    return imgs[:k], imgs[k:]


def write_split_txt(path: str, train_set: list, val_set: list):
    """写入 train.txt / val.txt。"""
    with open(join(path, "train.txt"), "w", encoding="utf-8", newline="\n") as f:
        for name in train_set:
            f.write(name + "\n")
    with open(join(path, "val.txt"), "w", encoding="utf-8", newline="\n") as f:
        for name in val_set:
            f.write(name + "\n")


def create_txt(path: str, split_ratio: float, part_ratio: float) -> dict:
    """划分并写入 train.txt / val.txt。"""
    imgs = list_paired_images(path)
    train_set, val_set = split_train_val(imgs, split_ratio, part_ratio)
    write_split_txt(path, train_set, val_set)
    return {"total": len(imgs), "train": len(train_set), "val": len(val_set)}


def _copy_sample(src_root: str, dst_sub: str, img_name: str):
    """复制 A、B、label 到 for_training/{train|val}/。"""
    for sub in ("A", "B"):
        shutil.copy2(join(src_root, sub, img_name), join(dst_sub, sub, img_name))
    label_src = _label_path(join(src_root, "label"), img_name)
    label_name = os.path.basename(label_src)
    shutil.copy2(label_src, join(dst_sub, "label", label_name))


def create_dir(path: str, split_ratio: float, part_ratio: float) -> dict:
    """划分并复制到 for_training/train、for_training/val。"""
    imgs = list_paired_images(path)
    train_set, val_set = split_train_val(imgs, split_ratio, part_ratio)

    base = join(path, "for_training")
    for sp in ("train", "val"):
        for sub in ("A", "B", "label"):
            os.makedirs(join(base, sp, sub), exist_ok=True)

    for img in tqdm(train_set, desc="  复制 train"):
        _copy_sample(path, join(base, "train"), img)
    for img in tqdm(val_set, desc="  复制 val"):
        _copy_sample(path, join(base, "val"), img)

    return {"total": len(imgs), "train": len(train_set), "val": len(val_set)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="变化检测数据集划分 train/val")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="GEOAI 数据目录")
    parser.add_argument("--mode", default=MODE, choices=("txt", "dir"), help="txt=写列表；dir=复制子目录")
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO, help="训练集占比")
    parser.add_argument("--part-ratio", type=float, default=PART_RATIO, help="使用样本比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    mode = args.mode
    split_ratio = args.split_ratio
    part_ratio = args.part_ratio
    random_seed = args.seed

    print(f"数据目录：{data_dir}")
    print(f"划分模式：{mode}  |  train 占比 {split_ratio}  |  样本比例 {part_ratio}")
    print("-" * 50)

    if random_seed is not None:
        random.seed(random_seed)

    step, total = 1, 3

    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    if not check_geoai_layout(data_dir):
        print("  目录结构不完整，退出。")
        return
    pairs = list_paired_images(data_dir)
    print(f"  成对样本：{len(pairs)}")

    print(f"[{step}/{total}] 划分数据集...")
    step += 1
    if mode == "txt":
        stats = create_txt(data_dir, split_ratio, part_ratio)
    elif mode == "dir":
        stats = create_dir(data_dir, split_ratio, part_ratio)
    else:
        print(f'  [错误] MODE 应为 "txt" 或 "dir"，当前: {mode}')
        return
    print(f"  train / val: {stats['train']} / {stats['val']}  (共 {stats['total']})")

    print(f"[{step}/{total}] 完成")
    if mode == "txt":
        print(f"  train.txt → {join(data_dir, 'train.txt')}")
        print(f"  val.txt   → {join(data_dir, 'val.txt')}")
    else:
        print(f"  for_training → {join(data_dir, 'for_training')}")

    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
