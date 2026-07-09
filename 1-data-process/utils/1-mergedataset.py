"""
# @Description: GEOAI 数据集整理：
#   1) JSON 中含指定 label（如 f）的样本，删除其 A/B/json/label 全部文件；
#   2) A/B/json/label 严格对齐，删除未配对文件；
#   3) 同步 train.txt / val.txt，仅保留 A/B 均存在的样本。
# @Command: python 1-mergedataset.py
"""

import argparse
import json
import os

#==============================#
# 接口配置
#==============================#
DATA_DIR = r"path/to/dataset/dir"
EXCLUDE_LABELS = "f"  # 逗号分隔，如 "f,error"；空字符串表示不处理 JSON
#==============================#

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = (".png", ".jpg", ".jpeg")


def parse_labels(s: str) -> set[str]:
    if not s or not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def list_images(folder: str) -> dict[str, str]:
    """stem -> filename"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in IMAGE_EXTS:
            out[stem] = fn
    return out


def find_label(label_dir: str, stem: str) -> str | None:
    for ext in LABEL_EXTS:
        p = os.path.join(label_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def find_stems_with_excluded_labels(json_dir: str, exclude: set[str]) -> set[str]:
    """若 JSON 的 shapes 中含 exclude 类别，返回该样本 stem。"""
    bad = set()
    if not exclude or not os.path.isdir(json_dir):
        return bad
    for fn in os.listdir(json_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, fn)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if shape.get("label", "") in exclude:
                bad.add(os.path.splitext(fn)[0])
                break
    return bad


def delete_stems(data_dir: str, stems: set[str]) -> list[str]:
    """删除指定 stem 在 A/B/json/label 中的全部文件。"""
    removed = []
    if not stems:
        return removed

    for sub in ("A", "B"):
        folder = os.path.join(data_dir, sub)
        for stem, fn in list_images(folder).items():
            if stem in stems:
                os.remove(os.path.join(folder, fn))
                removed.append(f"{sub}/{fn}")

    json_dir = os.path.join(data_dir, "json")
    if os.path.isdir(json_dir):
        for fn in os.listdir(json_dir):
            if not fn.lower().endswith(".json"):
                continue
            if os.path.splitext(fn)[0] in stems:
                os.remove(os.path.join(json_dir, fn))
                removed.append(f"json/{fn}")

    label_dir = os.path.join(data_dir, "label")
    if os.path.isdir(label_dir):
        for fn in os.listdir(label_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in LABEL_EXTS:
                continue
            if stem in stems:
                os.remove(os.path.join(label_dir, fn))
                removed.append(f"label/{fn}")

    return removed


def valid_pairs(data_dir: str) -> set[str]:
    """A/B 同名且 label（及 json，若存在）均有的 stem 集合。"""
    a_map = list_images(os.path.join(data_dir, "A"))
    b_map = list_images(os.path.join(data_dir, "B"))
    stems = {s for s in a_map if s in b_map and a_map[s] == b_map[s]}

    label_dir = os.path.join(data_dir, "label")
    if os.path.isdir(label_dir):
        stems = {s for s in stems if find_label(label_dir, s)}

    json_dir = os.path.join(data_dir, "json")
    if os.path.isdir(json_dir):
        json_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(json_dir)
            if f.lower().endswith(".json")
        }
        stems &= json_stems

    return stems


def delete_unpaired(data_dir: str, keep: set[str]) -> list[str]:
    """删除 A/B/json/label 中不在 keep 内的文件。"""
    removed = []

    for sub, is_image in (("A", True), ("B", True)):
        folder = os.path.join(data_dir, sub)
        for stem, fn in list_images(folder).items():
            if stem not in keep:
                os.remove(os.path.join(folder, fn))
                removed.append(f"{sub}/{fn}")

    json_dir = os.path.join(data_dir, "json")
    if os.path.isdir(json_dir):
        for fn in os.listdir(json_dir):
            if not fn.lower().endswith(".json"):
                continue
            if os.path.splitext(fn)[0] not in keep:
                os.remove(os.path.join(json_dir, fn))
                removed.append(f"json/{fn}")

    label_dir = os.path.join(data_dir, "label")
    if os.path.isdir(label_dir):
        for fn in os.listdir(label_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in LABEL_EXTS:
                continue
            if stem not in keep:
                os.remove(os.path.join(label_dir, fn))
                removed.append(f"label/{fn}")

    return removed


def sync_split_list(path: str, valid_names: set[str]) -> tuple[int, int]:
    """重写 train.txt / val.txt，仅保留 valid_names 中的条目。"""
    if not os.path.isfile(path):
        return 0, 0
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    kept = [ln for ln in lines if ln in valid_names]
    removed = len(lines) - len(kept)
    with open(path, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")
    return len(kept), removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GEOAI 数据集整理：JSON 清理 + A/B 对齐 + 同步列表")
    parser.add_argument("--Path", "--path", dest="data_dir", default=DATA_DIR, help="GEOAI 数据目录")
    parser.add_argument("--exclude-labels", default=EXCLUDE_LABELS,
                        help="含这些 label 的样本整组删除（A/B/json/label），逗号分隔")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    exclude = parse_labels(args.exclude_labels)
    print(f"数据目录：{data_dir}")
    print(f"整组删除 label：{exclude or '(不处理)'}")
    print("-" * 50)

    json_dir = os.path.join(data_dir, "json")
    bad_stems = find_stems_with_excluded_labels(json_dir, exclude)
    removed_bad = delete_stems(data_dir, bad_stems)
    print(f"[1] 含排除 label 的样本：删除 {len(bad_stems)} 组，共 {len(removed_bad)} 个文件")
    for p in removed_bad[:10]:
        print(f"      - {p}")
    if len(removed_bad) > 10:
        print(f"      ... 共 {len(removed_bad)} 个")

    keep = valid_pairs(data_dir)
    removed = delete_unpaired(data_dir, keep)
    print(f"[2] 对齐清理：保留 {len(keep)} 对，删除 {len(removed)} 个文件")
    for p in removed[:20]:
        print(f"      - {p}")
    if len(removed) > 20:
        print(f"      ... 共 {len(removed)} 个")

    a_map = list_images(os.path.join(data_dir, "A"))
    valid_names = {a_map[s] for s in keep if s in a_map}

    for name in ("train.txt", "val.txt"):
        path = os.path.join(data_dir, name)
        n_kept, n_rm = sync_split_list(path, valid_names)
        if os.path.isfile(path):
            print(f"[3] {name}：保留 {n_kept} 条，移除 {n_rm} 条")

    print("-" * 50)
    print("完成！")


if __name__ == "__main__":
    main()
