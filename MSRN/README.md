# MSRN

流程：高光谱 TIFF → Reconstruction → RGB → Swin V2-B → 场景类别。LPN 预测像素权重，用于强度和光谱重建损失；训练交替执行元学习更新与重建／分类更新。

## 目录

```text
MSRN/
├── train.py                   # 训练入口
├── evaluate.py                # 评估入口
├── MSRN/
│   ├── config.py              # 配置读取与校验
│   ├── data.py                # TIFF 数据集和归一化
│   ├── models.py              # Reconstruction、LPN 及元学习层
│   ├── losses.py              # 原损失公式与可微参数更新
│   ├── engine.py              # 两阶段训练步骤
│   └── runtime.py             # 分类器、设备与评估指标
├── configs/datasets.json      # 四个数据集的显式配置
├── tests/test_core.py         # 无外部数据的 CPU 回归检查
├── .github/workflows/tests.yml
├── requirements.txt
├── .gitattributes
└── .gitignore
```

仅包含源码、配置、文档和测试。数据、训练权重、临时权重、实验日志、生成图片、绘图脚本和重复实验版本不随仓库发布。

## 安装

建议 Python 3.12。在本仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

需要 GPU 训练时，根据自己的 CUDA 环境安装相匹配的 PyTorch 与 torchvision。仅用 CPU 时也可执行训练和评估，但完整模型计算量较大。依赖范围用于安装兼容版本，不是精确锁定的复现实验环境。

## 数据准备

训练集和测试集分别使用类别子目录，类别按文件夹名称排序：

```text
data/HSC-XMS/1_9/
├── train/
│   ├── class_01/*.tif
│   ├── class_02/*.tif
│   └── ...
└── test/
    ├── class_01/*.tif
    ├── class_02/*.tif
    └── ...
```

`--data-root` 直接指向 `train` 或 `test`，不是它们的父目录。支持 `.tif` 和 `.tiff`，每个文件是一幅三维高光谱图像。

| 数据集 | 输入通道 | 类别数 | RGB 索引 | TIFF 布局 | 归一化 |
| --- | ---: | ---: | --- | --- | --- |
| HSC-XMS | 150 | 9 | 140, 117, 67 | CHW | 每幅图像逐波段 min-max |
| HSC-ZHS | 32 | 12 | 13, 7, 5 | CHW | 除以 5000 |
| HSRS-SC | 48 | 5 | 19, 13, 7 | CHW | 除以 4000 |
| OHS | 32 | 9 | 13, 7, 5 | HWC | 除以 1000 |

**原文件夹缺少 `conf/settings.py`，因此无法确认其 TIFF 波段筛选表。** 上表的通道、类别、RGB 索引、布局和归一化方式来自现有训练／加载代码；`band_indices: null` 是整理版明确采用的假设：文件已含所需的全部波段，且顺序正确。输入通道不匹配会报错，不会自动裁掉多余波段。用于原实验复现前，应核对真实数据的波段顺序。

如原始 TIFF 需要选取波段，在 `configs/datasets.json` 为相应数据集填写完整的 `band_indices` 列表，长度必须等于 `channels`。所有索引均从 **0** 开始；`band_indices` 指向原始 TIFF，`rgb_bands` 指向选取后的输入通道。自定义配置文件通过 `--config path/to/datasets.json` 指定，格式与默认文件一致。

输入统一返回 `float32`、CHW、[0, 1]。每个空间维度至少为 8。同一批次的图像尺寸必须一致；不同尺寸可用 `--image-size 256` 统一缩放。默认保持原始分辨率，原脚本声明的 256 尺寸实际上未用于加载，因此本框架也不默认缩放。

## 训练

```bash
python train.py --dataset HSC-XMS --data-root data/HSC-XMS/1_9/train --output-dir runs/hsc-xms
```

主要默认值保持原脚本设置：51 个 epoch、batch size 20、学习率 `1e-4`、每 20 个 epoch 衰减至 0.2 倍、前 10 个 epoch 冻结分类器骨干，每轮最多 200 对元学习批次。强度／光谱／梯度损失系数分别为 1.0／0.2／0.1。

- `--device auto` 自动选择 CUDA 或 CPU，也可指定 `cpu`、`cuda:0` 等。
- 默认使用 ImageNet 预训练 Swin V2-B，首次使用需要下载权重；`--no-pretrained` 使用随机初始化，适合离线流程检查。
- `--meta-steps` 表示 support/query 批次对数。启用时每轮至少需要两个批次；设为 0 会禁用元学习阶段，仅用于调试或消融。
- `--base-ch 128 --lpn-dim 64` 对应保留的原模型结构。缩小这两个值会改变模型和权重尺寸，只适合开发验证或新实验。
- `--num-workers 0` 是跨平台默认值；入口带有主程序保护，支持 Windows 多进程加载。
- 已有训练产物的输出目录会被拒绝，避免覆盖；不指定输出目录时自动创建带时间戳的新目录。

每个训练目录包含：

```text
config.json    # 参数、类别顺序、归一化与模型宽度
metrics.jsonl  # 每轮损失
latest.pt      # 最近一次保存的 R、C、LPN 权重和推理所需配置
```

默认每 5 个 epoch 以及最后一轮保存 `latest.pt`。这是供评估使用的模型包，不包含优化器状态；此最小框架没有断点续训入口。

## 评估

```bash
python evaluate.py --data-root data/HSC-XMS/1_9/test --checkpoint runs/hsc-xms/latest.pt --output-dir results/hsc-xms
```

```bash
python evaluate.py --data-root data/HSC-XMS/1_9/test --legacy-model-dir model --dataset HSC-XMS --output-dir results/legacy
```


## 验证

```bash
python -m unittest discover -s tests -v
python train.py --help
python evaluate.py --help
```


