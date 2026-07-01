# SAM-CD 🚀

基于 [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) 视觉编码器的遥感图像变化检测框架。

> 原论文：[Adapting Segment Anything Model for Change Detection in HR Remote Sensing Images](https://ieeexplore.ieee.org/document/10443350)（IEEE TGRS 2024）
> 学习笔记：https://www.wolai.com/bvidXuVuXUaTKzkzMchGft

---

## 更新日志
- [x] 2026-06-08 初始版本代码完成
- [x] 2026-06-23 完善了数据集构建流程，训练数据加载模式，并支持图像配准 pipeline
- [x] 2026-06-30 优化了 onnx 导出的代码，适配 TensorRT 的转换
- [x] 2026-07-01 1.新增数据集处理 Agent skills; 2.新增裁剪拼接功能

---

## 项目结构

```
SAM-CD
├── 0-QuickStart                # 快速启动脚本（训练 / 推理 / 评估 / ONNX 导出）
│   ├── 0-train.sh
│   ├── 1-inference.sh
│   ├── 2-eval.sh
│   ├── 4-export.sh
│   ├── scripts                 # Python 入口（train / inference / eval / export）
│   └── utils
│       └── onnx_util.py        # ONNX 导出辅助（CDHeadExport + 单文件合并）
├── 1-data-process              
│   ├── utils
│   └── 4-create_trainset.sh    # 数据集处理脚本，包括整理，标签转换和train/val划分等
├── config
│   └── defualt.yaml            # 统一配置文件（所有参数默认值）
├── models
│   ├── SAM_CD.py               # 主模型（FastSAM encoder + CD head）
│   └── FastSAM                 # FastSAM 本地封装（含兼容性修改）
├── utils
│   └── tbr                     # ultralytics 四层特征补丁（install_tbr.sh 安装）
├── weights                     # 权重文件目录 (FastSAM-x.pt 需自行下载) 
├── z-others/SAMCD-onnx         # ONNX Runtime 推理试验包（可选）
└── install_tbr.sh              # 一键打 ultralytics 补丁
```

---

## 0 环境安装

```bash
conda create -n samcd python=3.12
conda activate samcd
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r z-others/requirements.txt
bash install_tbr.sh    # 一键打 FastSAM 四层特征补丁（必做）
```

- `install_tbr.sh` 会将 `utils/tbr/` 覆盖到当前 conda 环境的 `ultralytics` 包，使 FastSAM 在推理时输出四层中间特征（而非分割 mask）。补丁说明见 `models/FastSAM/README.md`。

- 升级或重装 `ultralytics` 后需重新执行 `bash install_tbr.sh`。

---

## 1 数据准备

数据集目录约定如下：

```
YOUR_DATA_DIR
│   ├── A           # 时相 A 图像（.png）
│   ├── B           # 时相 B 图像（.png，与 A 同名）
│   ├── label       # 变化标签（.png，0=不变 / 1=变化）
│   ├── train.txt
│   └── val.txt     
```

在 `config/defual.yaml` 中填写数据集根目录：

```yaml
dataset:
  root: "/your/data/root"
```

---

## 2 配置说明

所有参数集中在 `config/defual.yaml`，分四个命名空间：

| 命名空间 | 说明 |
|---|---|
| `dataset` | 数据集根目录 |
| `train` | 训练超参、输出目录、增强配置 |
| `predict` | 推理权重路径、测试集路径、TTA 等 |
| `eval` | GT 目录、预测结果目录 |
| `export` | ONNX 导出路径、`crop_size`、动态轴等 |

`0-QuickStart/` 脚本顶部变量会**覆盖** yaml 中对应字段，优先级：  
**脚本变量 > config/defual.yaml > 代码默认值**

---

## 3 快速开始

所有操作都在 `0-QuickStart/` 目录下进行，**只需修改脚本顶部变量**：

### 3.1 训练

```bash
# 编辑 0-QuickStart/0-train.sh 顶部：
# data_root、project、epochs、batch、lr、dev_id
bash 0-QuickStart/0-train.sh
```

输出保存到 `runs/0-train/$project/`（权重 + 验证图 + TensorBoard 日志）。

### 3.2 推理

```bash
# 编辑 0-QuickStart/1-inference.sh 顶部：
# chkpt_path、test_dir、dev_id
bash 0-QuickStart/1-inference.sh
```

推理结果默认写入 `$test_dir/repro/$project/`：

| 子目录 | 内容 |
|---|---|
| `bcd_map/` | 二值变化 mask |
| `vismask/` | 叠加 B 时相的可视化图 |

### 3.3 评估

```bash
# 编辑 0-QuickStart/2-eval.sh 顶部：
# GT_dir、pred_dir
bash 0-QuickStart/2-eval.sh
```

### 3.4 ONNX 导出

将训练好的 CD Head（及可选 FastSAM）导出为 ONNX，供部署或 `z-others/SAMCD-onnx` 推理：

```bash
# 编辑 0-QuickStart/4-export.sh 顶部：
# project、cd_head_weights、cd_head_onnx_output
bash 0-QuickStart/4-export.sh
```

导出逻辑在 `0-QuickStart/scripts/export_*.py`，共用 `0-QuickStart/utils/onnx_util.py`：

- `CDHeadExport`：可 trace 的 CD Head 包装
- `merge_external_data_to_single_file`：合并 `.onnx` + `.onnx.data` 为单文件

相关默认参数见 `config/defualt.yaml` 的 `export` 段。

---

## 4 附录

### 4.1 预训练权重

[FastSAM-x](https://pan.baidu.com/s/18KzBmOTENjByoWWR17zdiQ?pwd=0000)  
[Levir_CD](https://pan.baidu.com/s/1V25TFGL5V05ZB5ttFXFSEA?pwd=SMCD) (pswd: SMCD)

---

### 4.2 公共数据集

- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [WHU-CD](https://gpcv.whu.edu.cn/data/building_dataset.html)
- [UAV-CD](https://pan.baidu.com/share/init?surl=H5lO1HwZKfBpv3kx0cPf3A&pwd=jr5l)
- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [S2Looking](https://github.com/S2Looking/Dataset)
- [SYSU-CD](https://github.com/liumency/SYSU-CD)

---
