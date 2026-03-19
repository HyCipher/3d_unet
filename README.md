# 3D UNet Usage Guide

This document explains how to use the training, validation, and plotting scripts in this project.

## 1. Go to the project directory

```bash
cd /BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet
```

## 2. Train the model

```bash
python train_3d_unet.py
```

The training script will:
- Save model checkpoints in `models/`
- Automatically record training loss in `training_loss*.json`
- Automatically record validation metrics in `validation_history*.json`
- Use timestamped JSON filenames when the default files already exist (to avoid overwriting)

## 3. Evaluate a single model

Simplest command (uses default arguments):

```bash
python validate.py
```

Default model path:
- `./models/unet_3d_best.pth`

Common custom example:

```bash
python validate.py \
  --model ./models/Focal_Dice_100/unet_3d_best.pth \
  --val-img-dir data/validation/images \
  --val-label-dir data/validation/labels \
  --patch-size 16 512 512 \
  --stride 8 256 256 \
  --threshold 0.5 \
  --loss-type dicefocal \
  --plot-curves
```

Optional arguments:
- `--loss-type`: `none` / `bce` / `focal` / `dicefocal`
- `--save-results` or `--no-save-results`
- `--visualize` or `--no-visualize`
- `--plot-curves`

## 4. Validation output files

After running `validate.py`, the script usually generates:
- `validation_report_<model_name>.txt`: validation summary report
- `validation_results/pred_*.tif`: binary prediction files
- `validation_results/prob_*.tif`: probability maps
- `validation_visualization.png`: 6-panel visualization (image, label, prediction, probability map, label overlay, prediction overlay)
- `validation_curves_<model_name>.png`: PR/ROC curves (only when `--plot-curves` is enabled)

## 5. Plot training/validation curves

```bash
python plot_validation.py loss
python plot_validation.py losscompare
python plot_validation.py allmetrics
python plot_validation.py table
```

Mode descriptions:
- `loss`: training loss curve
- `losscompare`: training loss vs validation loss
- `allmetrics`: Dice/IoU/F1/Precision/Recall/Specificity curves
- `table`: read summary table from `validation_report.txt`

If your report file is named `validation_report_<model_name>.txt`, copy it first:

```bash
cp validation_report_unet_3d_best.txt validation_report.txt
python plot_validation.py table
```

## 6. Common issues

### 6.1 GPU out of memory
Use smaller patch size and stride:

```bash
python validate.py --patch-size 8 256 256 --stride 4 128 128
```

### 6.2 Need higher Recall
Lower the threshold:

```bash
python validate.py --threshold 0.3
```

### 6.3 Need higher Precision
Raise the threshold:

```bash
python validate.py --threshold 0.7
```

## 7. Dependencies

```bash
pip install torch torchvision
pip install scikit-learn scipy numpy matplotlib tifffile
```
