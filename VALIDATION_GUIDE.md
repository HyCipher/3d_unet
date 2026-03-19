# 3D UNet 模型验证指南

## 概述
当前项目里与验证相关的 3 个脚本：
1. `validate.py`: 对单个 `.pth` 模型做完整验证（推荐）
2. `train_3d_unet.py`: 训练时周期性验证并记录历史
3. `plot_validation.py`: 绘制训练/验证曲线

---

## 方法 1: 使用 `validate.py` 验证单个模型

### 最小命令
```bash
python validate.py
```

默认参数等价于：
```bash
python validate.py \
  --model ./models/unet_3d_best.pth \
  --val-img-dir data/validation/images \
  --val-label-dir data/validation/labels \
  --patch-size 16 512 512 \
  --stride 8 256 256 \
  --threshold 0.5 \
  --loss-type dicefocal \
  --save-results \
  --visualize
```

### 可用参数
- `--model`: 模型路径
- `--val-img-dir`: 验证图像目录
- `--val-label-dir`: 验证标签目录
- `--patch-size D H W`: 滑窗 patch 大小
- `--stride D H W`: 滑窗步长
- `--threshold`: 二值化阈值
- `--loss-type`: `none|bce|focal|dicefocal`
- `--save-results` / `--no-save-results`: 是否保存 `pred_*.tif` 和 `prob_*.tif`
- `--visualize` / `--no-visualize`: 是否保存 6 图可视化
- `--plot-curves`: 生成 PR/ROC 曲线图

### 输出文件
1. `validation_report_<model_name>.txt`
2. `validation_results/pred_*.tif`
3. `validation_results/prob_*.tif`
4. `validation_visualization.png`
5. `validation_curves_<model_name>.png`（仅 `--plot-curves`）

### `validation_visualization.png` 内容
固定 6 个子图：
- 原始图像（中间切片）
- 标签
- 预测
- 概率图
- 标签叠加
- 预测叠加

---

## 方法 2: 训练过程中自动验证（`train_3d_unet.py`）

运行：
```bash
python train_3d_unet.py
```

主要行为：
- 从头训练时，epoch 1 会额外验证一次
- 导入预训练模型时，不做 epoch 1 的额外验证
- 每 `VALIDATE_EVERY` 个 epoch 验证一次
- 记录 `train_loss` 和 `val_loss`

### 历史文件与防覆写
当已有同名文件时，训练脚本会自动改为时间戳文件名，避免覆写：
- `training_loss_YYYYMMDD_HHMMSS.json`
- `validation_history_YYYYMMDD_HHMMSS.json`

无同名文件时，仍使用默认：
- `training_loss.json`
- `validation_history.json`

### JSON 字段（当前版本）
`training_loss*.json`:
```json
[
  {"epoch": 1, "train_loss": 0.8042},
  {"epoch": 2, "train_loss": 0.7918}
]
```

`validation_history*.json`（单条示例）:
```json
{
  "epoch": 10,
  "train": null,
  "validation": {
    "dice": 0.2047,
    "iou": 0.1140,
    "f1": 0.2047,
    "precision": 0.1563,
    "recall": 0.2966,
    "specificity": 0.9988,
    "loss": 0.6201
  },
  "train_loss": 0.5723,
  "val_loss": 0.6201
}
```

---

## 方法 3: 绘图脚本（`plot_validation.py`）

支持命令：
```bash
python plot_validation.py allmetrics
python plot_validation.py loss
python plot_validation.py losscompare
python plot_validation.py table
```

输入文件说明：
- `allmetrics` / `loss` / `losscompare`: 读取 `validation_history*.json` 与 `training_loss*.json`
- `table`: 当前实现读取固定文件名 `validation_report.txt`

如果你当前是 `validation_report_<model_name>.txt`，可先临时复制：
```bash
cp validation_report_unet_3d_best.txt validation_report.txt
python plot_validation.py table
```

---

## 推荐流程
```bash
# 1) 训练
python train_3d_unet.py

# 2) 画曲线
python plot_validation.py loss
python plot_validation.py losscompare
python plot_validation.py allmetrics

# 3) 详细验证最佳模型
python validate.py --model ./models/unet_3d_best.pth --plot-curves

# 4) 查看报告
cat validation_report_unet_3d_best.txt
```

---

## 常见问题

### Q1: GPU 内存不足怎么办？
减小 `--patch-size` 和 `--stride`，例如：
```bash
python validate.py --patch-size 8 256 256 --stride 4 128 128
```

### Q2: 想提高 Recall？
降低阈值，例如：
```bash
python validate.py --threshold 0.3
```

### Q3: 想提高 Precision？
提高阈值，例如：
```bash
python validate.py --threshold 0.7
```

### Q4: 需要关闭结果文件保存？
```bash
python validate.py --no-save-results --no-visualize
```

---

## 依赖
```bash
pip install torch torchvision
pip install scikit-learn scipy numpy matplotlib tifffile
```
