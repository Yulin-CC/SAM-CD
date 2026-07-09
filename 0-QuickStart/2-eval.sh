#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-05-08 14:43:00
 # @LastEditTime: 2026-05-08
 # @Description: 需要先执行 1-inference.sh 脚本，生成预测结果目录
###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
 
#--------------------------------------#
# 需要修改的值
#--------------------------------------#
config_path="config/default.yaml"      # 统一配置文件路径（相对项目根目录）
GT_dir="path/to/GT/dir"                # Ground Truth 标签目录
pred_dir="path/to/pred/dir"            # 预测结果目录
#--------------------------------------#


#---------------#
# 运行评估程序
#---------------#
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/scripts/eval.py" \
  --config "$config_path" \
  --GT_dir "$GT_dir" \
  --pred_dir "$pred_dir"
