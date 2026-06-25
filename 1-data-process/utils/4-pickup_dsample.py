"""
# @Author: 算法组
# @Date: 2026-05-18
# @Description: 正负样本分离：从正样本 GEOAI 目录提取 json 中 shapes 为空的样本，
#   移动至负样本 disample 目录（A/B/json/label 一并迁移），并同步更新 train.txt / val.txt。
# @Move  : 默认移动（非复制），正样本目录仅保留有 polygon 标注的样本。
# @Format: 负样本 json 统一为 basename 形式的 imagePath / imagePathA / imagePathB。
# @Command: python 2-pickup_dsample.py
"""

import argparse
import json
import os
import shutil

#==============================#
# 接口配置
#==============================#
POS_DIR = r"D:\0-data\1-ChangeDetect\cache\黑龙镇\GEOAI-((ChangeD_Fuxingshuiku))-Selfcollect-2605-(D)(OL)"
NEG_DIR = r"D:\0-data\1-ChangeDetect\cache\黑龙镇\GEOAI-((ChangeD_disample_Fuxingshuiku))-Selfcollect-2605-(D)(OL)"
#==============================#

SUBDIRS = ("A", "B", "json", "label")
EXT_MAP = {"A": ".JPG", "B": ".JPG", "json": ".json", "label": ".png"}


def _list_jpg_names(folder: str) -> set:
    """返回目录下 JPG 文件名集合（大写扩展名）。"""
    if not os.path.isdir(folder):
        return set()
    return {
        f for f in os.listdir(folder)
        if f.upper().endswith(".JPG")
    }


def read_split_file(path: str) -> list:
    """读取 train.txt / val.txt，保留非空行顺序。"""
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
    """写入 train.txt / val.txt。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")


def collect_empty_stems(json_dir: str) -> list:
    """收集 shapes 为空的样本 stem 列表。"""
    stems = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.lower().endswith(".json"):
            continue
        json_path = os.path.join(json_dir, fname)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("shapes"):
            stems.append(os.path.splitext(fname)[0])
    return stems


def normalize_json(data: dict, jpg_name: str) -> dict:
    """负样本 json 格式统一（与 disample 参考集一致）。"""
    data["shapes"] = []
    data["imagePath"] = jpg_name
    data["imagePathA"] = jpg_name
    data["imagePathB"] = jpg_name
    data["change_detection"] = True
    return data


def check_geoai_layout(root_dir: str, name: str) -> bool:
    """检查目录是否具备 A/B/json/label 结构。"""
    ok = True
    for sub in SUBDIRS:
        path = os.path.join(root_dir, sub)
        if not os.path.isdir(path):
            print(f"  [缺少] {name} 无 {sub}/")
            ok = False
    return ok


def pickup_empty_samples(pos_dir: str, neg_dir: str) -> dict:
    """
    将正样本中 shapes 为空的样本移至负样本目录。
    返回统计信息 dict。
    """
    pos_json = os.path.join(pos_dir, "json")
    empty_stems = collect_empty_stems(pos_json)

    if not empty_stems:
        print("  未发现空标注样本，跳过移动。")
        return {"moved": 0, "neg_train": 0, "neg_val": 0, "pos_train": 0, "pos_val": 0}

    train_lines = read_split_file(os.path.join(pos_dir, "train.txt"))
    val_lines = read_split_file(os.path.join(pos_dir, "val.txt"))
    train_set = set(train_lines)
    val_set = set(val_lines)

    for sub in SUBDIRS:
        os.makedirs(os.path.join(neg_dir, sub), exist_ok=True)

    neg_train, neg_val = [], []
    moved = 0
    not_in_split = []

    for stem in empty_stems:
        jpg_name = stem + EXT_MAP["A"]

        #------------#
        # 移动 A/B/json/label
        #------------#
        for sub in SUBDIRS:
            ext = EXT_MAP[sub]
            src = os.path.join(pos_dir, sub, stem + ext)
            dst = os.path.join(neg_dir, sub, stem + ext)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"缺少文件: {src}")
            if os.path.exists(dst):
                raise FileExistsError(f"目标已存在，请先清空负样本目录: {dst}")
            shutil.move(src, dst)

        #------------#
        # 统一负样本 json 格式
        #------------#
        json_path = os.path.join(neg_dir, "json", stem + ".json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data = normalize_json(data, jpg_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

        if jpg_name in train_set:
            neg_train.append(jpg_name)
        elif jpg_name in val_set:
            neg_val.append(jpg_name)
        else:
            not_in_split.append(jpg_name)
            neg_train.append(jpg_name)

        moved += 1

    if not_in_split:
        print(f"  [提示] {len(not_in_split)} 个样本不在 train/val 中，已归入负样本 train.txt")

    #------------#
    # 重建 train.txt / val.txt
    #------------#
    pos_train, pos_val = [], []
    for jpg_name in sorted(_list_jpg_names(os.path.join(pos_dir, "A"))):
        if jpg_name in train_set:
            pos_train.append(jpg_name)
        elif jpg_name in val_set:
            pos_val.append(jpg_name)

    write_split_file(os.path.join(neg_dir, "train.txt"), neg_train)
    write_split_file(os.path.join(neg_dir, "val.txt"), neg_val)
    write_split_file(os.path.join(pos_dir, "train.txt"), pos_train)
    write_split_file(os.path.join(pos_dir, "val.txt"), pos_val)

    return {
        "moved": moved,
        "neg_train": len(neg_train),
        "neg_val": len(neg_val),
        "pos_train": len(pos_train),
        "pos_val": len(pos_val),
    }


def count_dir_files(root_dir: str, sub: str) -> int:
    path = os.path.join(root_dir, sub)
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def summarize_dataset(root_dir: str, title: str):
    """打印目录样本统计。"""
    json_dir = os.path.join(root_dir, "json")
    annotated = 0
    empty = 0
    if os.path.isdir(json_dir):
        for fname in os.listdir(json_dir):
            if not fname.lower().endswith(".json"):
                continue
            with open(os.path.join(json_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            if data.get("shapes"):
                annotated += 1
            else:
                empty += 1

    n_a = count_dir_files(root_dir, "A")
    n_train = len(read_split_file(os.path.join(root_dir, "train.txt")))
    n_val = len(read_split_file(os.path.join(root_dir, "val.txt")))
    print(f"  {title}")
    print(f"    A/B/json/label: {n_a} / {count_dir_files(root_dir, 'B')} / "
          f"{count_dir_files(root_dir, 'json')} / {count_dir_files(root_dir, 'label')}")
    print(f"    有标注 / 空标注: {annotated} / {empty}")
    print(f"    train / val: {n_train} / {n_val}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分离空标注负样本至 disample 目录")
    parser.add_argument("--Path", "--path", dest="pos_dir", default=POS_DIR, help="正样本 GEOAI 目录")
    parser.add_argument("--neg-dir", default=NEG_DIR, help="负样本 disample 目录")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pos_dir = args.pos_dir
    neg_dir = args.neg_dir

    print(f"正样本目录：{pos_dir}")
    print(f"负样本目录：{neg_dir}")
    print("-" * 50)

    step, total = 1, 4

    print(f"[{step}/{total}] 检查目录结构...")
    step += 1
    if not check_geoai_layout(pos_dir, "正样本"):
        print("  正样本目录结构不完整，退出。")
        return
    os.makedirs(neg_dir, exist_ok=True)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(neg_dir, sub), exist_ok=True)

    print(f"[{step}/{total}] 扫描空标注样本...")
    step += 1
    empty_count = len(collect_empty_stems(os.path.join(pos_dir, "json")))
    print(f"  共 {empty_count} 个空标注样本待分离")

    print(f"[{step}/{total}] 移动至负样本目录...")
    step += 1
    try:
        stats = pickup_empty_samples(pos_dir, neg_dir)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"  [错误] {e}")
        return
    print(f"  已移动 {stats['moved']} 组")
    print(f"  正样本 train/val: {stats['pos_train']} / {stats['pos_val']}")
    print(f"  负样本 train/val: {stats['neg_train']} / {stats['neg_val']}")

    print(f"[{step}/{total}] 统计结果...")
    summarize_dataset(pos_dir, "正样本")
    summarize_dataset(neg_dir, "负样本")

    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
