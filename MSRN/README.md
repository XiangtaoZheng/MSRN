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

## 安装

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据准备

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

## 训练

```bash
python train.py --dataset HSC-XMS --data-root data/HSC-XMS/1_9/train --output-dir runs/hsc-xms
```

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

