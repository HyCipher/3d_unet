# 3D UNet Usage Guide

This document explains how to use the training, evaluation, and inference scripts in this project.

## Project structure

```
train_3d_unet_model.py       # Training entry point
evaluator.py                 # Standalone evaluation (PyTorch)
evaluator_onnx.py            # Standalone evaluation (ONNX)
auto_train_then_evaluate.py  # Train then auto-evaluate in one command
infer.py                     # Inference on new volumes (no labels required)
config/
  tra_config.py              # Training hyperparameters
  val_config.py              # Validation/evaluation defaults
augmentations/               # Data augmentation modules
dataset/                     # Dataset / dataloader
evaluation/                  # Inference, I/O, postprocessing, PR curves
losses/                      # Loss functions (BCE, Focal, DiceFocal)
models/                      # UNet architecture
training/                    # Training loop helpers
utils/                       # wandb logging utilities
validate/                    # Metrics, evaluators, reporting
```

## 0. Install dependencies

Create or activate your Python environment first, then install the project dependencies:

```bash
pip install -r requirements.txt
```

If you plan to use wandb logging, log in once after installation:

```bash
wandb login
```

## 1. Train the model

```bash
python train_3d_unet_model.py
```

All training hyperparameters are centralized in `config/tra_config.py`:

| Parameter | Default | Description |
|---|---|---|
| `num_epochs` | 50 | Total training epochs |
| `loss_type` | `bce` | `bce` / `focal` / `dicefocal` |
| `dice_weight` | 0.8 | Weight for Dice term (dicefocal only) |
| `focal_weight` | 1.0 | Weight for Focal term (dicefocal only) |
| `val_patch_size` | `(8, 512, 512)` | Patch size used during in-training validation |
| `val_stride` | `(4, 256, 256)` | Stride used during in-training validation |
| `val_threshold` | 0.1 | Binarization threshold for in-training validation |
| `validate_every` | 10 | Validate every N epochs |
| `max_val_volumes` | `None` | Limit volumes validated per run (set integer for speed) |
| `dust_remove_min_size` | 64 | Remove connected components smaller than this (voxels) |
| `pos_weight_cap` | 10.0 | Cap on BCE pos_weight to prevent instability |
| `grad_clip_norm` | 1.0 | Gradient clipping threshold |
| `disable_aug_last_epochs` | 8 | Disable augmentation in the final N epochs |
| `eval_train_set` | `False` | Also evaluate training set each validation step |

The training script will:

- Save the best checkpoint as `./model_results/run_<timestamp>/unet_3d_best.pth` (selected by validation Dice)
- Use `Adam` + `ReduceLROnPlateau(mode=max, factor=0.5, patience=3, min_lr=1e-6)`
- Auto-compute `pos_weight` from training labels (capped by `pos_weight_cap`)

## 2. Evaluate a model (PyTorch)

All evaluation defaults are centralized in `config/val_config.py`. Edit that file to set the
model path, patch size, threshold, and wandb settings, then simply run:

```bash
python evaluator.py
```

`evaluator.py` will:

- Run sliding-window inference on all validation volumes
- Compute Dice / IoU / F1 / Precision / Recall / Specificity / Accuracy per sample and as mean
- Optionally upload metrics, per-sample visualizations, PR/ROC curves, and summary tables to wandb
- Save prediction and probability `.tif` files to `validation_results/` when `save_results = True`

Key fields in `config/val_config.py`:

| Field | Default | Description |
|---|---|---|
| `model_path` | `./model_results/run_.../unet_3d_best.pth` | Checkpoint to evaluate |
| `patch_size` | `(8, 512, 512)` | Sliding-window patch size |
| `stride` | `(4, 256, 256)` | Sliding-window stride |
| `threshold` | 0.1 | Binarization threshold |
| `dust_remove_min_size` | 128 | Remove small components (voxels) |
| `loss_type` | `bce` | `bce` / `focal` / `dicefocal` / `none` |
| `save_results` | `True` | Save pred/prob tif files |
| `wandb` | `True` | Enable wandb logging |
| `wandb_project` | `c_elegans_3d_unet_validation` | wandb project name |
| `wandb_run_name` | _(set in config)_ | wandb run name |

## 3. Evaluate a model (ONNX)

```bash
python evaluator_onnx.py
```

Uses the same `config/val_config.py` defaults. Replace `model_path` with a `.onnx` file path.
Runs on CPU by default; switches to CUDA if `onnxruntime-gpu` is installed and a GPU is detected.

## 4. Train then auto-evaluate in one command

```bash
python auto_train_then_evaluate.py
```

This runs Step 1 (training) and immediately Step 2 (evaluation of the best checkpoint) using
the defaults in `config/val_config.py`. Optional CLI overrides:

```bash
python auto_train_then_evaluate.py \
  --model-path ./model_results/run_XYZ/unet_3d_best.pth \
  --threshold 0.1 \
  --loss-type bce \
  --save-results \
  --eval-wandb
```

| Flag | Description |
|---|---|
| `--model-path` | Explicit checkpoint; if omitted uses latest `run_*/unet_3d_best.pth` |
| `--val-img-dir` | Override validation image directory |
| `--val-label-dir` | Override validation label directory |
| `--threshold` | Override binarization threshold |
| `--loss-type` | Override loss type for validation loss calculation |
| `--save-results` / `--no-save-results` | Toggle saving tif outputs |
| `--eval-wandb` / `--no-eval-wandb` | Toggle wandb logging for the eval stage |

## 5. Inference on new volumes (no labels)

Edit the `__main__` block in `infer.py` to set your paths, then:

```bash
python infer.py
```

Outputs:
- `result_seg.tif`: binary segmentation (uint8, values 0/255, shape H×W×Z)
- `prob_map.tif`: probability map (float32, shape H×W×Z)

Key parameters inside `infer.py`:

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | `(16, 512, 512)` | Patch size for sliding-window inference |
| `stride` | `(4, 128, 128)` | Stride for sliding-window inference |
| `threshold` | 0.1 | Binarization threshold |

## 6. Evaluation output files

| File | Description |
|---|---|
| `validation_results/pred_*.tif` | Binary predictions (uint8) |
| `validation_results/prob_*.tif` | Probability maps (float32) |

When `wandb = True`, `evaluator.py` also uploads:

- Per-sample metrics table
- Center-slice images (original / label / prediction / probability) per sample
- PR curve, ROC curve, and F1-threshold curve
- Dataset-level mean metrics summary table

## 7. Common issues

### 7.1 GPU out of memory

Reduce patch size and stride in `config/val_config.py` (or `config/tra_config.py` for training):

```python
"patch_size": (8, 256, 256),
"stride":     (4, 128, 128),
```

### 7.2 Need higher Recall

Lower the threshold in `config/val_config.py`:

```python
"threshold": 0.05,
```

### 7.3 Need higher Precision

Raise the threshold:

```python
"threshold": 0.3,
```

### 7.4 Noisy validation curves during training

Set `max_val_volumes` to `None` in `config/tra_config.py` to validate all volumes (slower but more stable).
Also consider increasing `validate_every` if validation is too frequent.

### 7.5 wandb logging checklist

- Install `wandb` and run `wandb login` once
- Set `"wandb": True` in `config/val_config.py`
- If run creation fails in restricted environments, set `WANDB_MODE=offline` and sync later

## 8. Dependencies

```
numpy
scipy
matplotlib
tifffile
torch
torchvision
wandb
onnxruntime
```

Install all at once:

```bash
pip install -r requirements.txt
```
