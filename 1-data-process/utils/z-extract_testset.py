"""
# @Author: 算法组
# @Date: 2026-05-29
# @Description: 从母路径下各 GEOAI 子目录的 train.txt 中按比例抽取样本，
#   移动至测试集目录（A/B/json/label），并更新各子目录 train.txt。
# @Move  : 默认移动（非复制）。
# @Command: python z-extract_testset.py --parent-dir <母路径> --dst-dir <测试集目录>
"""

import argparse
import math
import os
import random
import shutil

#==============================#
# 接口配置
#==============================#
PARENT_DIR = "/home/yulin/0-data/1-ChangeDetct"
DST_DIR = "/home/yulin/0-data/TestSet/1-ChangeDetect/testset-2605"
EXTRACT_RATIO = 0.004   # 0.1%
RANDOM_SEED = 42
#==============================#

SUBDIRS = ("A", "B", "json", "label")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def read_split_file(path: str) -> list:
    if not os.path.isfile(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def write_split_file(path: str, lines: list):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def check_geoai_layout(root_dir: str, name: str) -> bool:
    ok = True
    for sub in SUBDIRS:
        path = os.path.join(root_dir, sub)
        if not os.path.isdir(path):
            print(f"  [缺少] {name} 无 {sub}/")
            ok = False
    return ok


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def _resolve_file(folder: str, name: str, allowed_exts: tuple[str, ...] | None = None) -> str | None:
    """在目录中按文件名或 stem 解析实际文件（扩展名大小写不敏感）。"""
    if not os.path.isdir(folder):
        return None
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    stem, ext = os.path.splitext(name)
    ext_lower = ext.lower()
    for fn in os.listdir(folder):
        s, e = os.path.splitext(fn)
        if s != stem:
            continue
        if allowed_exts is not None and e.lower() not in {x.lower() for x in allowed_exts}:
            continue
        if ext_lower and e.lower() != ext_lower:
            continue
        return os.path.join(folder, fn)
    if allowed_exts:
        for fn in os.listdir(folder):
            s, e = os.path.splitext(fn)
            if s == stem and e.lower() in {x.lower() for x in allowed_exts}:
                return os.path.join(folder, fn)
    return None


def _label_path(label_dir: str, img_name: str) -> str | None:
    found = _resolve_file(label_dir, img_name, LABEL_EXTS)
    if found:
        return found
    stem = _stem(img_name)
    for ext in LABEL_EXTS:
        p = os.path.join(label_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _dst_has_stem(dst_dir: str, stem: str) -> bool:
    a_dir = os.path.join(dst_dir, "A")
    if not os.path.isdir(a_dir):
        return False
    for fn in os.listdir(a_dir):
        if _stem(fn) == stem:
            return True
    return False


def _move_file(src: str, dst: str):
    if not os.path.isfile(src):
        if os.path.isfile(dst):
            return
        raise FileNotFoundError(f"缺少文件: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def _move_sample(src_dir: str, dst_dir: str, img_name: str):
    stem = _stem(img_name)
    for sub in ("A", "B"):
        src = _resolve_file(os.path.join(src_dir, sub), img_name, IMAGE_EXTS)
        if src is None:
            raise FileNotFoundError(
                f"缺少 {sub}: {os.path.join(src_dir, sub, img_name)}"
            )
        dst = os.path.join(dst_dir, sub, os.path.basename(src))
        _move_file(src, dst)

    json_src = _resolve_file(os.path.join(src_dir, "json"), stem + ".json", (".json",))
    if json_src is None:
        raise FileNotFoundError(f"缺少 json: {os.path.join(src_dir, 'json', stem)}.json")
    dst_json = os.path.join(dst_dir, "json", os.path.basename(json_src))
    _move_file(json_src, dst_json)

    label_src = _label_path(os.path.join(src_dir, "label"), img_name)
    if label_src is None:
        raise FileNotFoundError(
            f"缺少 label: {os.path.join(src_dir, 'label', stem)}.*"
        )
    dst_label = os.path.join(dst_dir, "label", os.path.basename(label_src))
    _move_file(label_src, dst_label)


def sample_count(n_train: int, ratio: float) -> int:
    """按比例计算抽取数量；ratio>0 且训练集非空时至少抽 1 条。"""
    if n_train <= 0 or ratio <= 0:
        return 0
    k = int(math.floor(n_train * ratio))
    if k < 1:
        k = 1
    return min(k, n_train)


def shuffled_train_pool(train_lines: list, seed: int | None) -> list:
    pool = list(train_lines)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool


def discover_datasets(parent_dir: str) -> list[str]:
    """返回母路径下含 train.txt 且具备 GEOAI 结构的子目录路径列表。"""
    if not os.path.isdir(parent_dir):
        return []
    datasets = []
    for name in sorted(os.listdir(parent_dir)):
        sub = os.path.join(parent_dir, name)
        if not os.path.isdir(sub):
            continue
        if not os.path.isfile(os.path.join(sub, "train.txt")):
            continue
        if not check_geoai_layout(sub, name):
            continue
        datasets.append(sub)
    return datasets


def extract_from_dataset(
    src_dir: str,
    dst_dir: str,
    ratio: float,
    seed: int | None,
    dst_test_lines: list,
) -> dict:
    train_path = os.path.join(src_dir, "train.txt")
    train_lines = read_split_file(train_path)
    if not train_lines:
        print("  train.txt 为空，跳过。")
        return {"picked": 0, "moved": 0, "remain": 0, "skipped": 0}

    target_k = sample_count(len(train_lines), ratio)
    pool = shuffled_train_pool(train_lines, seed)
    moved_names: list[str] = []

    for sub in SUBDIRS:
        os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)

    moved, skipped = 0, 0
    for img_name in pool:
        if moved >= target_k:
            break
        stem = _stem(img_name)
        if _dst_has_stem(dst_dir, stem):
            print(f"  [冲突] 测试集已有同名 stem，跳过: {img_name}")
            skipped += 1
            continue
        try:
            _move_sample(src_dir, dst_dir, img_name)
        except FileNotFoundError as e:
            print(f"  [错误] {img_name}: {e}")
            skipped += 1
            continue
        moved_names.append(img_name)
        dst_test_lines.append(img_name)
        moved += 1

    moved_set = set(moved_names)
    remain = [x for x in train_lines if x not in moved_set]
    write_split_file(train_path, remain)

    return {
        "target": target_k,
        "moved": moved,
        "remain": len(remain),
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从母路径各子目录 train.txt 抽取测试集并移动至目标目录"
    )
    parser.add_argument(
        "--parent-dir",
        "--Path",
        "--path",
        dest="parent_dir",
        default=PARENT_DIR,
        help="母路径（遍历其下一级子文件夹）",
    )
    parser.add_argument(
        "--dst-dir",
        default=DST_DIR,
        help="测试集输出目录（含 A/B/json/label）",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=EXTRACT_RATIO,
        help="抽取比例，默认 0.001 即 0.1%%",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="随机种子；传 -1 表示不固定",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    parent_dir = args.parent_dir
    dst_dir = args.dst_dir
    ratio = args.ratio
    seed = None if args.seed is not None and args.seed < 0 else args.seed

    print(f"母路径：{parent_dir}")
    print(f"测试集目录：{dst_dir}")
    print(f"抽取比例：{ratio} ({ratio * 100:.4g}%)")
    if seed is not None:
        print(f"随机种子：{seed}")
    print("-" * 50)

    datasets = discover_datasets(parent_dir)
    if not datasets:
        print("未发现可用的 GEOAI 子目录（需含 train.txt 与 A/B/json/label）。")
        return

    print(f"共 {len(datasets)} 个子目录待处理：")
    for d in datasets:
        n = len(read_split_file(os.path.join(d, "train.txt")))
        print(f"  - {os.path.basename(d)}  (train: {n})")
    print("-" * 50)

    dst_test_lines = read_split_file(os.path.join(dst_dir, "test.txt"))
    total_moved = total_skipped = 0

    for src_dir in datasets:
        name = os.path.basename(src_dir)
        n_train = len(read_split_file(os.path.join(src_dir, "train.txt")))
        k = sample_count(n_train, ratio)
        print(f"[{name}] 计划抽取 {k} / {n_train}")
        stats = extract_from_dataset(src_dir, dst_dir, ratio, seed, dst_test_lines)
        print(
            f"  已移动 {stats['moved']}，剩余 train {stats['remain']}，"
            f"跳过 {stats['skipped']}"
        )
        total_moved += stats["moved"]
        total_skipped += stats["skipped"]

    if total_moved > 0 or dst_test_lines:
        write_split_file(os.path.join(dst_dir, "test.txt"), sorted(set(dst_test_lines)))

    print("-" * 50)
    print(f"合计移动 {total_moved} 组，跳过 {total_skipped}")
    if os.path.isdir(dst_dir):
        for sub in SUBDIRS:
            sub_path = os.path.join(dst_dir, sub)
            n = len(os.listdir(sub_path)) if os.path.isdir(sub_path) else 0
            print(f"  测试集 {sub}/: {n}")
        print(f"  test.txt: {len(read_split_file(os.path.join(dst_dir, 'test.txt')))}")
    print("完成！")


if __name__ == "__main__":
    main()
