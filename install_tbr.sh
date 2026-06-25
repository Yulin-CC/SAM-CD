#!/bin/bash
###
 # @Description: 将 utils/tbr/ 补丁安装到当前 conda 环境的 ultralytics 包
 #               安装一次即可支持 SAM-CD 训练/推理 + FastSAM 四层特征 ONNX 导出
###
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "❌ 请先执行: conda activate changeD"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TBR_DIR="$SCRIPT_DIR/utils/tbr"
ULTRA_DIR="$(python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))")"

echo "SAM-CD tbr 补丁安装"
echo "  源目录:   $TBR_DIR"
echo "  目标包:   $ULTRA_DIR"
echo ""

for f in head.py predictor.py tasks.py; do
    if [ ! -f "$TBR_DIR/$f" ]; then
        echo "❌ 缺少补丁文件: $TBR_DIR/$f"
        exit 1
    fi
done

cp "$TBR_DIR/head.py"      "$ULTRA_DIR/nn/modules/head.py"
cp "$TBR_DIR/predictor.py" "$ULTRA_DIR/engine/predictor.py"
cp "$TBR_DIR/tasks.py"     "$ULTRA_DIR/nn/tasks.py"

echo "✅ 已安装:"
echo "   head.py      → $ULTRA_DIR/nn/modules/head.py"
echo "   predictor.py → $ULTRA_DIR/engine/predictor.py"
echo "   tasks.py     → $ULTRA_DIR/nn/tasks.py"
echo ""
echo "验证补丁..."
python -c "
from ultralytics.nn.modules.head import Segment, SAM_CD_TBR_PATCH_VERSION
from ultralytics.nn.tasks import BaseModel
import inspect

head_src = inspect.getsource(Segment.forward)
tasks_src = inspect.getsource(BaseModel._predict_once)
pred_src = open('$ULTRA_DIR/engine/predictor.py', encoding='utf-8').read()

missing = []
if 'SAM_CD_TBR_PATCH' not in open('$TBR_DIR/head.py', encoding='utf-8').read():
    missing.append('head.py 缺少 SAM_CD_TBR_PATCH 标记')
if 'tuple(ms_feats)' not in head_src:
    missing.append('Segment.forward export 未返回 tuple(ms_feats)')
if 'm.f = [15, 18, 21, 1]' not in tasks_src:
    missing.append('BaseModel._predict_once 缺少 Segment.m.f 补丁')
if 'SAM-CD: when model returns multi-scale features' not in pred_src:
    missing.append('predictor.py 缺少 SAM-CD 特征直通逻辑')

if missing:
    raise SystemExit('补丁验证失败:\n  - ' + '\n  - '.join(missing))

print(f'✅ SAM-CD tbr 补丁验证通过 (v{SAM_CD_TBR_PATCH_VERSION})')
print('   推理: 四层特征 [y15,y18,y21,y1] → 跳过 FastSAM postprocess')
print('   导出: ONNX 4 输出 feat_l0~feat_l3')
"
