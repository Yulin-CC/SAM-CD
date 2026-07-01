#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-06-30
# @Description: 数据集一致性校验器 — 输出结构化 JSON 报告，供 Agent/CLI 使用。
#
# 检查项：
#   1. 目录结构：A/ B/ json/ label/ 是否存在
#   2. 文件计数：各目录文件数
#   3. A/B 对齐：同名同文件判断
#   4. json 对齐：与 A/B stem 匹配
#   5. label 对齐：与 A/B stem 匹配
#   6. 训练集划分：train.txt / val.txt 是否存在、覆盖率
#   7. JSON 内容：格式是否正确、空标注
#
# 输出格式：
#   {
#     "status": "pass" | "fail",
#     "dataset": "dataset_name",
#     "path": "/abs/path",
#     "counts": {"A": N, "B": N, "json": N, "label": N},
#     "checks": {
#       "dir_structure": {"pass": true/false, "missing": [...], "detail": "..."},
#       "a_b_match": {"pass": true/false, "matching": N, "a_only": N, "b_only": N, "detail": "..."},
#       "json_match": {"pass": true/false, "matching": N, "json_only": N, "missing": N, "detail": "..."},
#       "label_match": {"pass": true/false, "matching": N, "label_only": N, "missing": N, "detail": "..."},
#       "split": {"pass": true/false, "has_train": bool, "has_val": bool, "train_count": N, "val_count": N, "coverage": float, "detail": "..."},
#       "json_content": {"pass": true/false, "total": N, "empty_shapes": N, "bad_format": N, "detail": "..."}
#     },
#     "actions_needed": [
#       {"action": "align" | "json2label" | "label2json" | "rename" | "split", "priority": 1-5, "reason": "描述"}
#     ]
#   }
#
# Usage:
#   python validate_dataset.py /path/to/dataset
#   python validate_dataset.py /path/to/dataset --json   # 仅输出 JSON
#   python validate_dataset.py /path/to/dataset --strict # 严格模式
"""

import argparse
import json
import os
import sys


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".png", ".jpg", ".jpeg"}
JSON_EXT = {".json"}


def list_files(folder: str, exts: set) -> dict:
    """返回目录 {stem: filename}。"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in {e.lower() for e in exts}:
            out[stem] = fn
    return out


def read_split(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def validate(data_dir: str, strict: bool = False) -> dict:
    """校验数据集一致性，返回结构化结果。"""
    data_dir = os.path.abspath(data_dir)
    name = os.path.basename(data_dir)

    result = {
        "status": "pass",
        "dataset": name,
        "path": data_dir,
        "counts": {},
        "checks": {},
        "actions_needed": []
    }

    # 1. 目录结构
    dir_missing = []
    for sub in ("A", "B", "json", "label"):
        if not os.path.isdir(os.path.join(data_dir, sub)):
            dir_missing.append(sub)
    result["checks"]["dir_structure"] = {
        "pass": len(dir_missing) == 0,
        "missing": dir_missing,
        "detail": f"缺失目录: {', '.join(dir_missing)}" if dir_missing else "所有目录存在"
    }
    if dir_missing:
        result["status"] = "fail"

    # 2. 文件计数
    a_files = list_files(os.path.join(data_dir, "A"), IMAGE_EXTS)
    b_files = list_files(os.path.join(data_dir, "B"), IMAGE_EXTS)
    json_files = list_files(os.path.join(data_dir, "json"), JSON_EXT)
    label_files = list_files(os.path.join(data_dir, "label"), LABEL_EXTS)

    n_a, n_b, n_j, n_l = len(a_files), len(b_files), len(json_files), len(label_files)
    result["counts"] = {"A": n_a, "B": n_b, "json": n_j, "label": n_l}

    # 3. A/B 对齐
    ab_match = {s for s in a_files if s in b_files and a_files[s] == b_files[s]}
    a_only = set(a_files.keys()) - ab_match
    b_only = set(b_files.keys()) - ab_match
    ab_ok = len(a_only) == 0 and len(b_only) == 0
    result["checks"]["a_b_match"] = {
        "pass": ab_ok,
        "matching": len(ab_match),
        "a_only": len(a_only),
        "b_only": len(b_only),
        "detail": f"A: {n_a}, B: {n_b}, 匹配: {len(ab_match)}"
    }
    if not ab_ok:
        result["status"] = "fail"
        result["actions_needed"].append({
            "action": "align",
            "priority": 1,
            "reason": f"A/B 未对齐 — A-only: {len(a_only)}, B-only: {len(b_only)}"
        })

    # 4. json 对齐
    ab_stems = set(ab_match)
    json_only = set(json_files.keys()) - ab_stems
    ab_not_json = ab_stems - set(json_files.keys())
    json_ok = len(json_only) == 0 and len(ab_not_json) == 0
    result["checks"]["json_match"] = {
        "pass": json_ok,
        "matching": len(set(json_files.keys()) & ab_stems),
        "json_only": len(json_only),
        "missing": len(ab_not_json),
        "detail": f"json: {n_j}, 匹配: {len(set(json_files.keys()) & ab_stems)}"
    }
    if not json_ok:
        result["status"] = "fail"

    # 5. label 对齐
    ab_not_label = ab_stems - set(label_files.keys())
    label_only = set(label_files.keys()) - ab_stems
    label_ok = len(label_only) == 0 and len(ab_not_label) == 0
    result["checks"]["label_match"] = {
        "pass": label_ok,
        "matching": len(set(label_files.keys()) & ab_stems),
        "label_only": len(label_only),
        "missing": len(ab_not_label),
        "detail": f"label: {n_l}, 匹配: {len(set(label_files.keys()) & ab_stems)}"
    }
    if not label_ok:
        result["status"] = "fail"

    # 6. 训练集划分
    train_set = read_split(os.path.join(data_dir, "train.txt"))
    val_set = read_split(os.path.join(data_dir, "val.txt"))
    all_filenames = set(a_files.values())

    has_train = len(train_set) > 0
    has_val = len(val_set) > 0
    overlap = train_set & val_set
    missing_from_split = all_filenames - (train_set | val_set)
    extra_in_split = (train_set | val_set) - all_filenames
    coverage = len(train_set | val_set) / len(all_filenames) * 100 if all_filenames else 0

    split_ok = has_train and has_val and len(overlap) == 0 and len(missing_from_split) == 0
    result["checks"]["split"] = {
        "pass": split_ok,
        "has_train": has_train,
        "has_val": has_val,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "coverage": round(coverage, 1),
        "overlap": len(overlap),
        "missing": len(missing_from_split),
        "detail": f"train: {len(train_set)}, val: {len(val_set)}, 覆盖率: {coverage:.1f}%"
    }
    if not split_ok:
        result["status"] = "fail"
        if not has_train or not has_val:
            result["actions_needed"].append({
                "action": "split",
                "priority": 5,
                "reason": f"缺失 train/val 或覆盖率不足 ({coverage:.1f}%)"
            })

    # 7. JSON 内容
    json_dir = os.path.join(data_dir, "json")
    empty_shapes = 0
    bad_json = 0
    total_json = 0
    if os.path.isdir(json_dir):
        for fn in os.listdir(json_dir):
            if not fn.endswith(".json"):
                continue
            total_json += 1
            try:
                with open(os.path.join(json_dir, fn), encoding="utf-8") as f:
                    data = json.load(f)
                if not data.get("shapes"):
                    empty_shapes += 1
            except (json.JSONDecodeError, Exception):
                bad_json += 1

    json_ok = bad_json == 0
    result["checks"]["json_content"] = {
        "pass": json_ok,
        "total": total_json,
        "empty_shapes": empty_shapes,
        "bad_format": bad_json,
        "detail": f"总 JSON: {total_json}, 空标注: {empty_shapes}, 格式错误: {bad_json}"
    }
    if bad_json > 0:
        result["status"] = "fail"

    # 8. 标签转换需求
    has_json = n_j > 0
    has_label = n_l > 0
    if has_json and not has_label:
        result["actions_needed"].append({
            "action": "json2label",
            "priority": 2,
            "reason": f"有 json/ ({n_j}) 但无 label/，需要 JSON → mask 转换"
        })
    elif has_label and not has_json:
        result["actions_needed"].append({
            "action": "label2json",
            "priority": 2,
            "reason": f"有 label/ ({n_l}) 但无 json/，需要 label → JSON 转换"
        })

    # 9. 重命名需求（如果文件名不符合 Train-{prefix}-{date}-{seq}.jpg 格式）
    if n_a > 0:
        import re
        pattern = re.compile(r'^[A-Za-z0-9]+-\d{6}-\d{3,}\.(jpg|png|jpeg|bmp|tif)$', re.IGNORECASE)
        bad_names = [f for f in a_files.values() if not pattern.match(f)]
        if bad_names:
            result["actions_needed"].append({
                "action": "rename",
                "priority": 3,
                "reason": f"{len(bad_names)} 个文件未按要求命名（示例: {bad_names[:3]}）"
            })

    # 10. 排序 actions
    result["actions_needed"].sort(key=lambda x: x["priority"])

    return result


def print_report(result: dict, strict: bool = False):
    """打印人类可读报告。"""
    print(f"\n{'=' * 60}")
    print(f"  数据集校验: {result['dataset']}")
    print(f"  路径: {result['path']}")
    print(f"{'=' * 60}")

    counts = result['counts']
    print(f"\n  文件计数: A={counts['A']}, B={counts['B']}, json={counts['json']}, label={counts['label']}")

    for check_name, check_data in result['checks'].items():
        icon = "✅" if check_data['pass'] else "❌"
        print(f"\n  [{icon}] {check_name}: {check_data['detail']}")

    if result['actions_needed']:
        print(f"\n{'─' * 60}")
        print(f"  需要执行的操作:")
        for i, action in enumerate(result['actions_needed'], 1):
            icon = "🔧" if action['action'] != 'rename' else "📝"
            print(f"    [{i}] {icon} {action['action']}: {action['reason']}")
    else:
        print(f"\n  ✅ 无需整理 — 数据集完整")

    status_icon = "✅" if result['status'] == 'pass' else "❌"
    print(f"\n{'─' * 60}")
    print(f"  状态: {status_icon} {'通过' if result['status'] == 'pass' else '未通过'}")
    print(f"{'=' * 60}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="数据集一致性校验器")
    parser.add_argument("dataset", help="数据集目录路径")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON")
    parser.add_argument("--strict", action="store_true", help="严格模式（对齐不一致即报错）")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print(f"[错误] 目录不存在: {args.dataset}")
        sys.exit(1)

    result = validate(args.dataset, strict=args.strict)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(result, strict=args.strict)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    sys.exit(0 if result['status'] == 'pass' else 1)


if __name__ == "__main__":
    import json  # 导入 json 模块（之前忘了）
    main()
