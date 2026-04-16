# 3D UNet Usage Guide

This document explains how to use the training, validation, and plotting scripts in this project.

## 1. Train the model

```bash
python train_3d_unet.py
```

The training script will:

- Save model checkpoints in `models/`
- Use `DiceFocalLoss(alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=1.0)` by default
- Use `Adam(lr=1e-4)` and `ReduceLROnPlateau` driven by validation Dice
- Save `./models/unet_3d_best.pth` based on the best validation Dice
- Automatically record training loss in `training_loss*.json`
- Automatically record validation metrics in `validation_history*.json`
- Use timestamped JSON filenames when the default files already exist (to avoid overwriting)

Current validation behavior inside training:

- Validate every 10 epochs
- Also run validation at epoch 1 when training from scratch
- By default only validate the first validation volume during training for speed (`MAX_VAL_VOLUMES = 1` in `train_3d_unet.py`)
- Use validation patch size `16 512 512` and stride `8 256 256`

If you want more stable validation curves during training, edit `train_3d_unet.py` and set:

```python
MAX_VAL_VOLUMES = None
```

This will validate all volumes, but training will be slower.

## 2. Evaluate a single model

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

Enable wandb upload (metrics + validation slice images):

```bash
python validate.py --wandb
```

Example with explicit wandb settings:

```bash
python validate.py \
  --model ./models/unet_3d_best.pth \
  --plot-curves \
  --wandb \
  --wandb-project c_elegans_3d_unet_validation \
  --wandb-run-name validate_unet_3d_best
```

Optional arguments:

- `--loss-type`: `none` / `bce` / `focal` / `dicefocal`
- `--save-results` or `--no-save-results`
- `--visualize` or `--no-visualize`
- `--plot-curves`
- `--wandb`: upload validation metrics and images to wandb
- `--wandb-project`: wandb project name (default: `c_elegans_3d_unet_validation`)
- `--wandb-run-name`: optional run name (default: `validate_<model_name>`)

Notes:

- `validate.py` can compute validation loss with `bce`, `focal`, or `dicefocal`, but model selection during training is based on validation Dice, not validation loss
- `Dice+Focal` loss is usually less smooth than BCE, so validation loss may plateau or fluctuate even when Dice is still improving

## 3. Validation output files

After running `validate.py`, the script usually generates:

- `validation_report_<model_name>.txt`: validation summary report
- `validation_results/pred_*.tif`: binary prediction files
- `validation_results/prob_*.tif`: probability maps
- `validation_visualization.png`: 6-panel visualization (image, label, prediction, probability map, label overlay, prediction overlay)
- `validation_curves_<model_name>.png`: PR/ROC curves (only when `--plot-curves` is enabled)

When `--wandb` is enabled, `validate.py` also uploads:

- Per-sample metrics
- Center-slice images for each sample (original / label / prediction / probability)
- Summary PNGs (`validation_visualization.png` and PR/ROC curves if generated)
- Dataset-level mean validation metrics

## 4. Plot training/validation curves

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

Interpretation note:

- `losscompare` mixes training patch loss and validation-time sliding-window patch loss; for checkpoint selection, prefer Dice/IoU/F1 over loss alone

If your report file is named `validation_report_<model_name>.txt`, copy it first:

```bash
cp validation_report_unet_3d_best.txt validation_report.txt
python plot_validation.py table
```

## 5. Common issues

### 5.1 GPU out of memory

Use smaller patch size and stride:

```bash
python validate.py --patch-size 8 256 256 --stride 4 128 128
```

### 5.2 Need higher Recall

Lower the threshold:

```bash
python validate.py --threshold 0.3
```

### 5.3 Need higher Precision

Raise the threshold:

```bash
python validate.py --threshold 0.7
```

### 5.4 Validation loss does not keep decreasing with Dice+Focal

This is common and does not automatically mean training failed.

Check these points first:

- Look at validation Dice/IoU/F1, not validation loss alone
- If training uses `MAX_VAL_VOLUMES = 1`, validation curves can be noisy; use `None` for full validation
- Dice+Focal is more sensitive to learning rate than BCE; if needed, reduce the initial LR from `1e-4` to `3e-5`
- Validation loss is averaged over sliding-window patches, while Dice is computed on the reconstructed full volume, so they are related but not identical objectives

### 5.5 wandb logging checklist

- Install `wandb` and run `wandb login` once
- Run validation with `--wandb`
- If run creation fails in restricted environments, set `WANDB_MODE=offline` and sync later

## 6. Dependencies

```bash
pip install torch torchvision
pip install scikit-learn scipy numpy matplotlib tifffile wandb
```
