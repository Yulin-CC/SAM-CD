#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-05-12
 # @Description: 清洗好的数据进行训练验证集的划分，准备训练
 # @Update: 2026-05-12
###

#---------------#
# 需要修改的值
#-----------------#
Path='/home/yulin/0-data/1-ChangeDetct/GEOAI-((ChangeD_Zhongke_dsample02))-Selfcollect-2606-(D)(OL)'              # 清洗好的数据路径
part_ratio=1.0                                                                                                              # 使用样本比例（1.0=全部）

#-------------------------------------------------------------------------------------#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="${SCRIPT_DIR}/utils"

#-------------------#
# 切换到虚拟环境
#-------------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate changeD

#-------------------#
# 整理数据集（可选）
#-------------------#
python "${UTIL_DIR}/1-mergedataset.py" --Path "$Path"

#-------------------#
# 标签转换 label2mask
#-------------------#
python "${UTIL_DIR}/2-labels2mask.py" --Path "$Path"

#-------------------#
# 划分训练验证集
#-------------------#
python "${UTIL_DIR}/3-dataSet_division.py" --Path "$Path" --mode "txt" --part-ratio "$part_ratio"

#-------------------#
# 筛选负样本
#-------------------#
python "${UTIL_DIR}/4-pickup_dsample.py" --Path "$Path"


#-------------------#
# 从 disample 提取正样本（可选）
#-------------------#
# python "${UTIL_DIR}/5-extract_positive.py" --Path "$Path" --src-dir "$NegPath"
