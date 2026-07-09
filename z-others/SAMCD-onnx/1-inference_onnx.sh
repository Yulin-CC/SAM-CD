#!/bin/bash
###
 # @Author: 算法组 蔡雨霖
 # @Date: 2026-06-08
 # @Description: SAM-CD ONNX 独立推理包入口（FastSAM + CD Head + crop_pasteback 配准）
###
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#--------------------------------------#
# 需要修改的值
#--------------------------------------#
dev_id=0
#---------------------------------------------------------------------------#
test_dir="/path/to/your/dataset"          # 测试集路径（须含 A/ B/ 子目录）
#---------------------------------------------------------------------------#
out_dir=""                                # 留空则输出到 {test_dir}/repro/onnx/
#---------------------------------------------------------------------------#
fastsam_onnx="$SCRIPT_DIR/weights/FastSAM-x.onnx"
cd_head_onnx="$SCRIPT_DIR/weights/best_CDHead.onnx"
config_path="$SCRIPT_DIR/config/default.yaml"
#---------------------------------------------------------------------------#


cd "$SCRIPT_DIR"
if [ -z "$out_dir" ]; then
  out_dir="$test_dir/repro/onnx"
fi
mkdir -p "$out_dir/bcd_map" "$out_dir/vismask"

if [ ! -f "$fastsam_onnx" ]; then
  echo "❌ 缺少 FastSAM ONNX: $fastsam_onnx"
  exit 1
fi
if [ ! -f "$cd_head_onnx" ]; then
  echo "❌ 缺少 CD Head ONNX: $cd_head_onnx"
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=$dev_id python "$SCRIPT_DIR/onnx_scripts/inference_onnx.py" \
  --config "$config_path" \
  --fastsam_onnx "$fastsam_onnx" \
  --cd_head_onnx "$cd_head_onnx" \
  --test_dir "$test_dir" \
  --pred_dir "$out_dir" \
  --dev_id "$dev_id"

reg_on=$(python - <<PY
import yaml
with open("$config_path", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("predict", {}).get("register", {}).get("enable", False))
PY
)
if [ "$reg_on" = "True" ]; then
  echo "register.enable=true：vismask 已由 inference_onnx.py 生成"
else
  python "$SCRIPT_DIR/utils/mask.py" \
    --data-dir "$test_dir" \
    --pred-dir "$out_dir/bcd_map" \
    --out-dir "$out_dir/vismask" \
    --image-dir A
fi

echo ""
echo "✅ 推理完成，结果: $out_dir"
