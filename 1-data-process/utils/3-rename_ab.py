"""
A/B/json/label 成对重命名：A/B -> .jpg，label -> .png，json -> .json。

@Command: python rename_ab.py
@Command: python rename_ab.py --dry-run
"""

import argparse
import os

#==============================#
# 接口配置
#==============================#
DATA_DIR = r"D:\0-data\TestSet\1-ChangeDetect\new\z-dsample"
PREFIX = "Test-dsample"
DATE_TAG = "260630"       # 日期标识
START_INDEX = 1           # 起始序号
DIGITS = 3                # 序号位数，如 3 -> 001
#==============================#

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = IMAGE_EXTS
SUBDIRS = ("A", "B", "json", "label")


def stem_map(folder: str, exts: set[str]) -> dict[str, str]:
    """stem -> filename"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            out[stem] = fn
    return out


def list_paired_stems(data_dir: str) -> list[str]:
    """返回 A/B 均存在，且 json/label（若存在）也齐全的 stem，按 stem 排序。"""
    a_map = stem_map(os.path.join(data_dir, "A"), IMAGE_EXTS)
    b_map = stem_map(os.path.join(data_dir, "B"), IMAGE_EXTS)
    stems = set(a_map) & set(b_map)

    json_dir = os.path.join(data_dir, "json")
    if os.path.isdir(json_dir):
        json_map = stem_map(json_dir, {".json"})
        stems &= set(json_map)

    label_dir = os.path.join(data_dir, "label")
    if os.path.isdir(label_dir):
        label_map = stem_map(label_dir, LABEL_EXTS)
        stems &= set(label_map)

    return sorted(stems)


def find_file(folder: str, stem: str, exts: set[str]) -> str | None:
    if not os.path.isdir(folder):
        return None
    for fn in os.listdir(folder):
        s, ext = os.path.splitext(fn)
        if s == stem and ext.lower() in exts:
            return fn
    return None


def target_name(sub: str, index: int) -> str:
    """按子目录类型生成新文件名：A/B .jpg，label .png，json .json。"""
    stem = f"{PREFIX}-{DATE_TAG}-{index:0{DIGITS}d}"
    if sub == "json":
        return stem + ".json"
    if sub == "label":
        return stem + ".png"
    return stem + ".jpg"


def active_subdirs(data_dir: str) -> tuple[str, ...]:
    return tuple(s for s in SUBDIRS if os.path.isdir(os.path.join(data_dir, s)))


def rename_pairs(data_dir: str, dry_run: bool = False) -> None:
    stems = list_paired_stems(data_dir)
    if not stems:
        print("未找到完整成对样本（A/B 必填，json/label 若存在则也需齐全）。")
        return

    subs = active_subdirs(data_dir)
    temp_prefix = "__tmp_rename__"
    pairs: list[tuple[str, dict[str, str], dict[str, str]]] = []

    for i, stem in enumerate(stems, start=START_INDEX):
        old_names: dict[str, str] = {}
        new_names: dict[str, str] = {}
        for sub in subs:
            if sub == "json":
                old_fn = find_file(os.path.join(data_dir, sub), stem, {".json"})
            elif sub == "label":
                old_fn = find_file(os.path.join(data_dir, sub), stem, LABEL_EXTS)
            else:
                old_fn = find_file(os.path.join(data_dir, sub), stem, IMAGE_EXTS)
            if old_fn is None:
                raise FileNotFoundError(f"{sub}/{stem} 不存在")
            old_names[sub] = old_fn
            new_names[sub] = target_name(sub, i)
        pairs.append((stem, old_names, new_names))

    print(f"数据目录：{data_dir}")
    print(f"处理子目录：{', '.join(subs)}")
    print(f"命名格式：{PREFIX}-{DATE_TAG}-{'0' * (DIGITS - 1)}x.jpg / .png / .json")
    print(f"成对样本：{len(pairs)}")
    print("-" * 50)

    if not dry_run:
        for stem, old_names, new_names in pairs:
            for sub in subs:
                folder = os.path.join(data_dir, sub)
                old_path = os.path.join(folder, old_names[sub])
                temp_path = os.path.join(folder, temp_prefix + new_names[sub])
                os.rename(old_path, temp_path)

        for _, _, new_names in pairs:
            for sub in subs:
                folder = os.path.join(data_dir, sub)
                temp_path = os.path.join(folder, temp_prefix + new_names[sub])
                final_path = os.path.join(folder, new_names[sub])
                os.rename(temp_path, final_path)

    shown = min(5, len(pairs))
    print(f"\n示例（前 {shown} 组）：")
    for stem, old_names, new_names in pairs[:shown]:
        print(f"  {stem}")
        for sub in subs:
            print(f"    {sub}/{old_names[sub]} -> {sub}/{new_names[sub]}")
    if len(pairs) > shown:
        print(f"  ... 共 {len(pairs)} 组")

    action = "预览完成" if dry_run else "重命名完成"
    print(f"\n{action}。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B/json/label 成对重命名")
    parser.add_argument("--path", dest="data_dir", default=DATA_DIR, help="含 A/B 的数据目录")
    parser.add_argument("--prefix", default=PREFIX, help="文件名前缀")
    parser.add_argument("--date", dest="date_tag", default=DATE_TAG, help="日期标识")
    parser.add_argument("--start", dest="start_index", type=int, default=START_INDEX, help="起始序号")
    parser.add_argument("--digits", type=int, default=DIGITS, help="序号位数")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际重命名")
    return parser


def main(argv=None):
    global PREFIX, DATE_TAG, START_INDEX, DIGITS
    args = build_parser().parse_args(argv)

    PREFIX = args.prefix
    DATE_TAG = args.date_tag
    START_INDEX = args.start_index
    DIGITS = args.digits

    rename_pairs(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
