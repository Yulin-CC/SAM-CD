# FastSAM ultralytics 补丁说明

原版 FastSAM **无法直接访问多尺度中间特征**。SAM-CD 通过 `utils/tbr/` 补丁改造 ultralytics，在推理时提取四层特征 `[y15, y18, y21, y1]`，供 Adapter + CD Head 使用。

> **不要手动改 site-packages**。请使用项目根目录的一键安装脚本。

## 一键安装（推荐）

```bash
conda activate changeD
pip install -r z-others/requirements.txt   # ultralytics == 8.4.56
bash install_tbr.sh                        # 将 utils/tbr/ 覆盖到当前环境的 ultralytics
```

安装成功后会看到：

```
✅ SAM-CD tbr 补丁验证通过 (v1.0)
   推理: 四层特征 [y15,y18,y21,y1] → 跳过 FastSAM postprocess
   导出: ONNX 4 输出 feat_l0~feat_l3
```

## 补丁内容

| 文件 | 作用 |
|------|------|
| `utils/tbr/tasks.py` | Segment head 接入四层输入 `m.f = [15, 18, 21, 1]`，保留 stride-4 浅层特征 |
| `utils/tbr/head.py` | 截获 `ms_feats`；推理返回特征列表；**export 仅输出 4 层特征** |
| `utils/tbr/predictor.py` | 检测到多尺度特征时跳过 NMS/mask postprocess |

## 四层特征顺序

与 PyTorch `run_encoder()` / 后续 ONNX 导出一致：

| 索引 | 输出名 | 来源层 | 通道 | stride |
|------|--------|--------|------|--------|
| `[0]` | `feat_l0` | y[15] | 320 | 8 |
| `[1]` | `feat_l1` | y[18] | 640 | 16 |
| `[2]` | `feat_l2` | y[21] | 640 | 32 |
| `[3]` | `feat_l3` | y[1] | 160 | 4 |

## 重装 / 升级 ultralytics 后

`pip install -U ultralytics` 会覆盖补丁，需重新执行：

```bash
bash install_tbr.sh
```
