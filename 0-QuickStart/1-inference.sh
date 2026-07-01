#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-05-08 14:43:00
 # @LastEditTime: 2026-05-08
 # @Description:
###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
dev_id=0                                                                    # GPU 设备 ID
#---------------------------------------------------------------------------#
project="1-ChangeDetect_SAMCD-2606-v1.2"                                    # 项目名称
#---------------------------------------------------------------------------#
test_dir="/home/yulin/0-data/TestSet/1-ChangeDetect/0-Feedback/260623"      # 测试集路径（含 A/ B/ 子目录）
#---------------------------------------------------------------------------#
chkpt_path="runs/0-train/$project/checkpoint/best.pth"            # 模型权重（与 train 保存的 best 一致）
#---------------------------------------------------------------------------#
config_path="config/default.yaml"                                            # 统一配置文件路径（相对项目根目录）
#---------------------------------------------------------------------------#


#---------------#
# 切换到虚拟环境
#---------------#
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh  # 虚拟环境切换实例化 (本地服务器的 annaconda 所在的位置)
conda activate changeD                                  # 切换到 changeD 虚拟环境 (实际的虚拟环境的路径)


#---------------#
# 运行推理程序
#---------------#
cd "$PROJECT_ROOT"
run_dir=$test_dir/"repro"/$project             # 推理输出根目录
mkdir -p "$run_dir/bcd_map" "$run_dir/vismask"
python "$SCRIPT_DIR/scripts/inference.py" \
  --config "$config_path" \
  --chkpt_path "$chkpt_path" \
  --test_dir "$test_dir" \
  --pred_dir "$run_dir" \
  --dev_id "$dev_id"

#---------------#
# 可视化高亮结果（register.enable=true 时 inference 已输出 vismask，跳过 mask.py）
#---------------#
vismask_on=$(python - <<PY
import yaml, sys
with open("$PROJECT_ROOT/$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
v = cfg.get("predict", {}).get("vismask", True)
print(v)
PY
)
reg_on=$(python - <<PY
import yaml, sys
with open("$PROJECT_ROOT/$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("predict", {}).get("register", {}).get("enable", False))
PY
)
if [ "$vismask_on" = "False" ] || [ "$vismask_on" = "false" ]; then
  echo "vismask=false：跳过可视化生成"
elif [ "$reg_on" = "True" ]; then
  echo "register.enable=true：vismask 已由 inference.py 生成"
else
  vismask_dir=$run_dir/"vismask"
  python "$PROJECT_ROOT/1-data-process/utils/mask.py" \
    --data-dir "$test_dir" \
    --pred-dir "$run_dir/bcd_map" \
    --out-dir "$vismask_dir" \
    --image-dir A
fi

#---------------#
# crop_mask：裁剪 vismask 变化区域子图（config 控制）
#---------------#
cm_enable=$(python - <<PY
import yaml
with open("$PROJECT_ROOT/$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("predict", {}).get("crop_mask", False))
PY
)
if [ "$cm_enable" = "True" ] || [ "$cm_enable" = "true" ]; then
  cm_expand=$(python - <<PY
import yaml
with open("$PROJECT_ROOT/$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("predict", {}).get("crop_mask_expand", 0.4))
PY
)
  cm_vismask=$(python - <<PY
import yaml
with open("$PROJECT_ROOT/$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("predict", {}).get("crop_mask_vismask", False))
PY
)
  cm_args=(--pred_dir "$run_dir" --output_dir "$run_dir/crop"
    --expand "$cm_expand" --imgA_dir "$test_dir/A" --imgB_dir "$test_dir/B"
    --config "$PROJECT_ROOT/$config_path")
  if [ "$cm_vismask" = "True" ] || [ "$cm_vismask" = "true" ]; then
    cm_args+=(--vismask)
  fi
  echo ""
  echo "🔪 运行 crop_mask 后处理 (expand=$cm_expand, vismask=$cm_vismask)..."
  python "$PROJECT_ROOT/0-QuickStart/utils/crop_mask.py" "${cm_args[@]}"
  echo "✅ crop_mask 完成"
else
  echo "[crop_mask] 已跳过（predict.crop_mask=false）"
fi
