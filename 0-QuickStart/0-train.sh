#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-05-08 14:43:00
 # @LastEditTime: 2026-05-12
 # @Description:
###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
dev_id="0"                            # 使用的 GPU ID
#--------------------------------------#
project="[ChangeDetect]-SAMCD-2606-v1.2"      # 训练任务名
#--------------------------------------#
dataset="data/0-ChangeD.yaml"          # 数据集路径（yaml 或根目录）
#--------------------------------------#
load_path=""                           # 可选：微调初始化权重路径；留空则从头训练
resume=0                               # 1=从上次 checkpoint_last.pth 断点续训（恢复优化器+epoch）
background=1                           # 1=后台运行（nohup）；0=前台盯着看
config_path="config/default.yaml"      # 统一配置文件路径（相对项目根目录）
#--------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh  # 虚拟环境切换实例化
conda activate changeD


#---------------#
# 运行训练程序
#---------------#
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$dev_id"
_resume_flag=""
[ "$resume" = "1" ] && _resume_flag="--resume"

if [ "$background" = "1" ]; then
  log_file="runs/0-train/$project/train.log"
  mkdir -p "$(dirname "$log_file")"
  nohup torchrun --nproc_per_node=1 --master_port=29501 "$SCRIPT_DIR/scripts/train.py" \
    --config    "$config_path"               \
    --project   "$project"                   \
    --data_root "$dataset"                   \
    --load_path "$load_path"                 \
    $_resume_flag > "$log_file" 2>&1 &
  echo "✅ 后台训练已启动，PID: $!"
  echo "   日志: $log_file"
  echo "   查看: tail -f $log_file"
  echo "   停止: kill $!"
else
  torchrun --nproc_per_node=1 --master_port=29501 "$SCRIPT_DIR/scripts/train.py" \
    --config    "$config_path"               \
    --project   "$project"                   \
    --data_root "$dataset"                   \
    --load_path "$load_path"                 \
    $_resume_flag
fi
