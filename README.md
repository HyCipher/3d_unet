# C. elegans Synapse Detection with 3D Separable U-Net

This project is designed for synapse detection on 3D C. elegans volumes.
It uses a separable cascaded 3D U-Net (`SepUNet`) for voxel-level binary segmentation and supports the full workflow: training, validation, and inference.

## 1. What This Project Does

Main goals:

- Read 3D TIFF volumes (stored as H, W, Z by default; auto-transposed to Z, H, W internally)
- Train a separable cascaded 3D U-Net (`SepUNet`) to detect synapse regions
- Run sliding-window validation and inference on full volumes
- Report standard segmentation metrics (Dice, IoU, F1, Precision, Recall, Specificity, Accuracy)
- Export trained models as `.pth` checkpoints and `.onnx` artifacts
- Optionally log experiments with Weights and Biases (wandb)

## 2. Project Structure

```text
3d_unet/
├── config/
│   ├── tra_config.py                 # Training hyperparameters
│   └── val_config.py                 # Standalone validation settings
├── data/
│   ├── training/
│   │   ├── images/                   # Training images (*.tif)
│   │   └── labels/                   # Training labels (*.tif)
│   └── validation/
│       ├── images/                   # Validation images (*.tif)
│       └── labels/                   # Validation labels (*.tif)
├── models/
│   ├── detect_sep.py                 # SepUNet — active model (separable cascaded 3D U-Net)
│   └── model_3d_origin.py            # Reference standard 3D U-Net
├── dataset/
│   ├── dataset.py                    # Tif3DPatchDataset with priority patch sampling
│   └── hard_sampling.py              # Hard-negative / hard-positive coordinate builders
├── training/
│   ├── axis_utils.py                 # Shared Z/H/W axis normalization utilities
│   ├── build_train_dataset.py        # Dataset factory
│   ├── build_optimizer.py            # Optimizer factory
│   ├── epoch.py                      # Single training epoch
│   ├── val_epoch.py                  # Lightweight in-training validation (Dice)
│   ├── init_model.py                 # Model initialization and checkpoint loading
│   ├── pth_export.py                 # .pth checkpoint saving (periodic / best / interrupted)
│   └── onnx_export.py                # ONNX export (best / final, opset 18)
├── augmentations/                    # 3D data augmentation pipeline
├── evaluation/
│   ├── inference.py                  # Sliding-window inference + connected-component stats
│   ├── postprocessing.py             # Small connected-component removal (dust removal)
│   └── ...                           # Visualization, loss factory, PR curve helpers
├── losses/                           # BCE / Focal / DiceFocal loss builders
├── validate/
│   └── metrics.py                    # Dice, IoU, Precision/Recall/F1/Specificity/Accuracy
├── utils/                            # wandb helpers
├── train_3d_unet_model.py            # Training entry point
├── evaluator.py                      # PyTorch validation entry point
├── evaluator_onnx.py                 # ONNX validation entry point
├── infer.py                          # Single-volume inference entry point
├── build_hard_negative_dataset.py    # Hard-negative/positive dataset builder
└── requirements.txt
```

## 3. Model: SepUNet

`SepUNet` (in `models/detect_sep.py`) replaces standard 3×3×3 convolutions with a cascaded separable design:

1. `conv_xy` — 1×3×3 convolution (spatial features)
2. `conv_z`  — 3×1×1 convolution (depth features, receives same input as conv_xy)
3. Concatenate → 1×1×1 fusion → GroupNorm → ReLU

Downsampling uses `MaxPool(1,2,2)` — **Z is never pooled**, which preserves the thin-slab depth dimension across all 4 encoder levels.

## 4. Patch Sampling Strategy

Each training patch is drawn with a priority policy controlled by four ratios (must sum ≤ 1.0):

| Ratio | Center selection |
|---|---|
| `pos_sample_ratio` | Random foreground (GT > 0) voxel |
| `edge_sample_ratio` | GT voxel touching background (6-neighborhood boundary) |
| `hard_negative_sample_ratio` | Predicted connected component with **zero GT overlap** |
| `hard_positive_sample_ratio` | GT connected component with **zero prediction overlap** |
| remainder | Fully random crop |

Hard-negative and hard-positive coordinates are computed at dataset load time from saved probability maps (see Section 7).

## 5. How To Use

Run all commands from the project root `C_elegans_UNet/3d_unet`.

### 5.1 Install Dependencies

```bash
pip install -r requirements.txt
wandb login   # optional, for experiment tracking
```

### 5.2 Prepare Data

```text
data/
├── training/
│   ├── images/*.tif
│   └── labels/*.tif
└── validation/
    ├── images/*.tif
    └── labels/*.tif
```

- Image and label files are matched by sorted filename order
- Volumes are auto-transposed from H, W, Z → Z, H, W
- Label voxels > 0 are treated as foreground (synapse)

### 5.3 Train the Model

```bash
python train_3d_unet_model.py
```

Training saves the following artifacts under `model_results/<run_name>/`:

| File | Trigger |
|---|---|
| `unet_3d_epoch_N.pth` | Every `save_every` epochs |
| `unet_3d_best.pth` | New best validation Dice |
| `unet_3d_best.onnx` | New best validation Dice |
| `unet_3d_interrupted.pth` | Ctrl-C / KeyboardInterrupt |
| `unet_3d_final.onnx` | Training end (any exit path) |

Key parameters in `config/tra_config.py`:

| Parameter | Default | Notes |
|---|---|---|
| `patch_size` | (8, 512, 512) | Z is not pooled; keep Z ≥ 8 |
| `batch_size` | 2 | Reduce if out of memory |
| `num_epochs` | 50 | |
| `save_every` | 10 | Periodic checkpoint interval |
| `loss_type` | `bce` | `bce` / `focal` / `dicefocal` |
| `pos_weight_cap` | 10.0 | Cap on auto-computed neg/pos BCE weight |
| `grad_clip_norm` | 1.0 | Gradient clipping |
| `val_threshold` | 0.1 | Sigmoid threshold for binary prediction |
| `dust_remove_min_size` | 128 | Remove connected components smaller than this (voxels) |
| `disable_aug_last_epochs` | 8 | Disable augmentation in final N epochs |

### 5.4 Validate (PyTorch)

Set `model_path`, `val_img_dir`, `val_label_dir`, `patch_size`, `stride`, `threshold` in `config/val_config.py`, then:

```bash
python evaluator.py
```

When `save_results=True`, outputs go to:
- `validation_results/pred/*.tif`
- `validation_results/probs/*.tif`

### 5.5 Validate (ONNX)

Point `model_path` in `config/val_config.py` to an `.onnx` file, then:

```bash
python evaluator_onnx.py
```

### 5.6 Single-Volume Inference

Edit the `__main__` block in `infer.py` (`img_path`, `model_path`, `save_path`), then:

```bash
python infer.py
```

Outputs: `result_seg.tif` (binary mask), `prob_map.tif` (probability map).

## 6. Memory Estimate

With `batch_size=2`, `patch_size=(8,512,512)`, float32:

| Component | ~Size |
|---|---|
| Skip connections (L0–L3) | ~480 MB |
| L0 backward activations (peak) | ~1.5 GB |
| Weights + gradients + Adam | ~70 MB |
| **Total peak** | **~2–4 GB** |

To reduce memory: lower `patch_size` (e.g., `(8,256,256)`) or enable mixed precision (AMP).

## 7. Hard-Negative / Hard-Positive Dataset

Connected-component definitions used by the sampler:

- **Hard negative** — a predicted connected component that has **no voxel overlap** with any GT synapse. The model should not have predicted this region.
- **Hard positive** — a GT synapse connected component that has **no overlap** with any predicted component. The model completely missed this region.

To build the hard-sampling dataset from saved probability maps:

```bash
python build_hard_negative_dataset.py \
    --pred-dir path/to/prob_maps \
    --train-img-dir data/training/images \
    --train-label-dir data/training/labels \
    --output-dir data/hard_sampling \
    --threshold 0.5 \
    --enable-hard-negative
```

Then enable in `config/tra_config.py`:

```python
"hard_negative_enable": True,
"hard_negative_dir": "data/hard_sampling/hard_negative_masks",
"hard_negative_sample_ratio": 0.2,
"hard_positive_sample_ratio": 0.1,
```

## 8. Practical Tuning Tips

- **Out of memory** — reduce `patch_size` or `batch_size`
- **Low recall** — lower `val_threshold` (e.g., 0.05) or raise `pos_weight_cap`
- **Low precision** — raise `val_threshold` (e.g., 0.2–0.3); enable hard-negative sampling
- **Many false-positive blobs** — increase `dust_remove_min_size`
- **Unstable training** — lower learning rate or reduce `pos_weight_cap` and `grad_clip_norm`

## 9. One-Line Summary

End-to-end 3D separable U-Net pipeline for C. elegans synapse detection: training with priority patch sampling, validation-based best-model export (.pth + .onnx), and hard-negative/hard-positive mining from connected-component overlap.
