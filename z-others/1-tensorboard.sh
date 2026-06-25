#!/bin/bash
# 启动 TensorBoard 查看训练日志

# 检查 pkg_resources，缺少则安装 setuptools
python -c "import pkg_resources" 2>/dev/null || {
    echo ">>> 缺少 pkg_resources，正在安装 setuptools ..."
    pip install setuptools -q
}

cd "$(dirname "$0")/.."

LOGDIR="runs/0-train/1-ChangeDetect-2605-02/tb"
PORT=6006

echo ">>> TensorBoard logdir: $LOGDIR"
echo ">>> 浏览器打开: http://localhost:$PORT"
echo ""

tensorboard --logdir "$LOGDIR" --port $PORT --bind_all
