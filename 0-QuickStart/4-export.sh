#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-08
 # @Description: SAM-CD ONNX 导出（FastSAM 4 特征 + CD Head）
 #               前置: bash install_tbr.sh
 #               须在 0-QuickStart 目录下执行
###


WORK_DIR=$(pwd)

#--------------#
# 需要修改的值
#----------------------------------------------------------------------#
devices=0                                                              # GPU 设备 ID
#----------------------------------------------------------------------#
project="SAMCD-2606-v1.2"                               # 训练任务名（CD Head 权重路径）
#----------------------------------------------------------------------#
sam_weights="./weights/FastSAM-x.pt"                                   # FastSAM PyTorch 权重
fastsam_onnx_output="./weights/FastSAM-x.onnx"                         # FastSAM ONNX 输出
#----------------------------------------------------------------------#
cd_head_weights="./runs/0-train/$project/checkpoint/best.pth"          # CD Head PyTorch 权重
cd_head_onnx_output="./runs/0-train/$project/checkpoint/best.onnx"                     # CD Head ONNX 输出
#----------------------------------------------------------------------#
config_file="./config/default.yaml"                                    # 配置文件（相对项目根目录）
#----------------------------------------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate changeD


#---------------#
# 运行导出程序（切换到项目根目录）
#---------------#
cd "$WORK_DIR"/..

if [ ! -f "$cd_head_weights" ]; then
  echo "❌ 缺少 CD Head 权重: $cd_head_weights"
  exit 1
fi

# echo "========== [1/2] FastSAM ONNX（4 特征输出）=========="
# CUDA_VISIBLE_DEVICES=$devices python ./0-QuickStart/scripts/export_fastSAM.py \
#       --sam_weights  "$sam_weights"           \
#       -o             "$fastsam_onnx_output"   \
#       --config       "$config_file"           \
#       || exit $?

echo ""
echo "========== [2/2] CD Head ONNX =========="
CUDA_VISIBLE_DEVICES=$devices python ./0-QuickStart/scripts/export_cd_head.py \
      --cd_head_weights "$cd_head_weights"      \
      -o                  "$cd_head_onnx_output" \
      --config            "$config_file"          \
      || exit $?

echo ""
echo "✅ 全部导出完成"
echo "   - FastSAM:  $fastsam_onnx_output"
echo "   - CD Head:  $cd_head_onnx_output"
