#!/usr/bin/env python3
"""
# @Author: 算法组
# @Date: 2026-05-12
# @Description: 从完整 SAM_CD 权重导出仅 CD 头（去掉冻结 model.*），与 weights/Levir_CD 发布格式一致
# @Command: python z-others/export_heads_checkpoint.py [input_pth] [output_pth]
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from utils.checkpoint import prepare_checkpoint_for_save, torch_load_compat, unpack_checkpoint_state_dict


#--------------------------------------#
# 默认输入输出（相对仓库根）
#--------------------------------------#
_DEFAULT_REL_IN = os.path.join("runs", "0-train", "SAM-CD-2605-test", "checkpoint_last.pth")
_DEFAULT_IN = os.path.join(REPO, _DEFAULT_REL_IN)
_DEFAULT_OUT = os.path.join(REPO, os.path.dirname(_DEFAULT_REL_IN), "best_heads.pth")


#--------------------------------------#
# 参数解析
#--------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(
        description="导出 heads-only checkpoint（不含冻结的 model.*）。",
        epilog="无参数时默认：仓库内 checkpoint_last.pth（全量）-> 同目录 best_heads.pth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input_pth",
        nargs="?",
        default=_DEFAULT_IN,
        help="完整 SAM_CD .pth；默认：%s" % _DEFAULT_REL_IN,
    )
    p.add_argument(
        "output_pth",
        nargs="?",
        default=_DEFAULT_OUT,
        help="输出 .pth；默认：与输入同目录 best_heads.pth",
    )
    return p.parse_args()


#--------------------------------------#
# 导出逻辑
#--------------------------------------#
def export_heads(input_pth: str, output_pth: str) -> int:
    inp = os.path.abspath(input_pth)
    outp = os.path.abspath(output_pth)

    #-------------#
    # 校验输入
    #-------------#
    if not os.path.isfile(inp):
        print("Input not found: %s" % inp)
        return 1

    #-------------#
    # 解包并去掉 model.*
    #-------------#
    raw = torch_load_compat(inp, map_location="cpu")
    sd = unpack_checkpoint_state_dict(raw, path_hint=inp)
    out = prepare_checkpoint_for_save(sd, mode="heads_only", strip_dp_prefix=True)

    #-------------#
    # 写出文件
    #-------------#
    out_dir = os.path.dirname(outp)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, outp)
    print("Wrote %d tensors -> %s" % (len(out), outp))
    return 0


#--------------------------------------#
# 主调
#--------------------------------------#
def main():
    args = parse_args()
    return export_heads(args.input_pth, args.output_pth)


if __name__ == "__main__":
    raise SystemExit(main())
