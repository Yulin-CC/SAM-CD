"""
# @Author: 算法组
# @Date: 2026-05-14
# @Description: 双时相配准 CLI：遍历 ROOT_DIR 下所有含 A/B 的子文件夹，批量 ECC 配准落盘
# @Command: python 1-data-process/utils/0-2-register.py
# @Output: B_registered/ + warp_meta/（CROP_BORDER=True 时额外生成 A_cropped/ + B_cropped/）
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.register import register_batch

#==============================#
# 接口配置
#==============================#
ROOT_DIR = r"D:\0-data\1-ChangeDetect\cache\数据中心文件-罗波乡小学"
LOG_JSON = None
CROP_BORDER = True
CROP_INSIDE_B_MARGIN = 1.0

MOTION = "AFFINE"
SCALES = "0.125,0.25,0.5,1.0"
USE_ORB_INIT = True
MIN_ECC_CC = None
MIN_NCC = None
#==============================#


def find_ab_folders(root_dir: str) -> list[tuple[str, str, str, str, str]]:
    pairs = []
    for entry in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, entry)
        if not os.path.isdir(sub):
            continue
        a_dir = os.path.join(sub, 'A')
        b_dir = os.path.join(sub, 'B')
        if os.path.isdir(a_dir) and os.path.isdir(b_dir):
            out_b = os.path.join(sub, 'B_registered')
            warp = os.path.join(sub, 'warp_meta')
            pairs.append((entry, a_dir, b_dir, out_b, warp))
    return pairs


def main():
    print(f'根目录：{ROOT_DIR}')
    print('-' * 50)

    step, total = 1, 3

    print(f'[{step}/{total}] 扫描含 A/B 的子文件夹...')
    step += 1
    pairs = find_ab_folders(ROOT_DIR)
    if not pairs:
        print('  未找到含 A/B 的子文件夹，退出。')
        return
    print(f'  共 {len(pairs)} 个：')
    for name, *_ in pairs:
        print(f'    {name}')

    print(f'[{step}/{total}] 批量 ECC 配准...')
    step += 1
    total_log = []
    for idx, (name, a_dir, b_dir, out_b, warp) in enumerate(pairs, 1):
        print(f'\n  [{idx}/{len(pairs)}] {name}')
        print(f'    A → {a_dir}')
        print(f'    B → {b_dir}')
        log_lines = register_batch(
            a_dir, b_dir, out_b, warp,
            MOTION, SCALES, USE_ORB_INIT, MIN_ECC_CC, MIN_NCC,
            crop_border=CROP_BORDER, crop_margin=CROP_INSIDE_B_MARGIN, log_json=None,
        )
        total_log.extend(log_lines)

    print(f'\n[{step}/{total}] 保存日志...')
    if LOG_JSON:
        with open(LOG_JSON, 'w', encoding='utf-8') as f:
            for row in total_log:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'  总日志已保存：{LOG_JSON}')
    else:
        print('  未配置 LOG_JSON，跳过')

    print('-' * 50)
    print('完成！')


if __name__ == '__main__':
    main()
