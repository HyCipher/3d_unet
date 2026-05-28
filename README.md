# 3D UNet Usage Guide

This README covers how to train, evaluate, and run inference with this repository, plus which parameters you can tune.

## 1. Project overview

Main scripts:

```
train_3d_unet_model.py       # Training entry point
evaluator.py                 # Standalone evaluation (PyTorch)
evaluator_onnx.py            # Standalone evaluation (ONNX Runtime)
infer.py                     # Inference on unlabeled volumes
build_hard_negative_dataset.py  # Build hard-negative masks from probability maps
```

Config files:

```
config/tra_config.py         # Training config
config/val_config.py         # Evaluation config
```

## 2. Installation

Create or activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you use wandb:

```bash
wandb login
```

## 3. Data layout

Default training/validation paths are configured in `config/tra_config.py`.

Expected layout:

```
data/
  training/
    images/*.tif
    labels/*.tif
  validation/
    images/*.tif
    labels/*.tif
  training_hn/
    hard_negative_masks/*.tif    # optional, used when hard negatives are enabled
```

Notes:

- Image/label files must match one-to-one by sorted filename order.
- Volumes are read as 3D TIFF and internally converted from `(H, W, Z)` to `(Z, H, W)`.

## 4. Train

Run training:

```bash
python train_3d_unet_model.py
```

Training outputs:

- Epoch checkpoints: `model_results/run_<timestamp>/unet_3d_epoch_*.pth`
- Best checkpoint (by validation Dice): `model_results/run_<timestamp>/unet_3d_best.pth`

### 4.1 Train config: `config/tra_config.py`

#### Paths and runtime

| Parameter | Default | Description |
|---|---|---|
| `project` | `c_elegans_3d_unet` | wandb project name used by training logger |
| `architecture` | `3D UNet` | Metadata field for logging |
| `train_img_dir` | `data/training/images` | Training images |
| `train_label_dir` | `data/training/labels` | Training labels |
| `val_img_dir` | `data/validation/images` | Validation images |
| `val_label_dir` | `data/validation/labels` | Validation labels |
| `batch_size` | 2 | Batch size |
| `num_workers` | 4 | DataLoader worker processes |
| `num_epochs` | 50 | Number of epochs |

DataLoader behavior in `train_3d_unet_model.py`:

- `pin_memory` is enabled only when CUDA is available.
- `persistent_workers` is enabled only when `num_workers > 0`.

#### Patch sampling and augmentation

| Parameter | Default | Description |
|---|---|---|
| `patch_size` | `(8, 512, 512)` | Training patch size `(Z, H, W)` |
| `patches_per_volume` | 500 | Random patches sampled per training volume |
| `val_patches_per_volume` | 50 | Random patches sampled per validation volume during in-training eval |
| `disable_aug_last_epochs` | 8 | Disable augmentation in final N epochs |

#### Hard-negative sampling

| Parameter | Default | Description |
|---|---|---|
| `hard_negative_enable` | `True` | Enable hard-negative masks |
| `hard_negative_dir` | `data/training_hn/hard_negative_masks` | Hard-negative mask directory |
| `hard_negative_sample_ratio` | 0.5 | Sampling ratio for hard-negative-centered patches |
| `hardest_negative` | `False` | Use eroded hard-negative core if available |
| `hardest_negative_erosion_iters` | 1 | Erosion iterations for hardest-negative mode |

The sampling ratios used by the dataset must satisfy:

`pos_sample_ratio + edge_sample_ratio + hard_negative_sample_ratio <= 1.0`

with defaults from dataset builder:

- `pos_sample_ratio`: `0.4`
- `edge_sample_ratio`: `0.1`

#### Optimization and loss

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | `adam` | `adam` / `adamw` / `sgd` |
| `weight_decay` | 0.0 | Weight decay for optimizer |
| `momentum` | 0.9 | Only used when `optimizer = sgd` |
| `loss_type` | `bce` | `bce` / `focal` / `dicefocal` |
| `dice_weight` | 0.8 | Dice term weight in `dicefocal` |
| `focal_weight` | 1.0 | Focal term weight in `dicefocal` |
| `pos_weight_cap` | 10.0 | Cap for auto-computed BCE positive class weight |
| `grad_clip_norm` | 1.0 | Gradient clipping threshold (0 to disable) |

Scheduler in training script:

- `ReduceLROnPlateau(mode="max", factor=0.5, patience=3, min_lr=1e-6)`
- Stepped by validation Dice.

#### In-training validation

| Parameter | Default | Description |
|---|---|---|
| `validate_every` | 10 | Run validation every N epochs |
| `eval_train_set` | `False` | Also evaluate training set each validation step |
| `max_val_volumes` | `None` | Limit number of validation volumes per validation pass |
| `val_patch_size` | `(8, 512, 512)` | Sliding-window patch size for in-training validation |
| `val_stride` | `(4, 256, 256)` | Sliding-window stride for in-training validation |
| `val_threshold` | 0.1 | Probability threshold for in-training metrics |
| `dust_remove_min_size` | 128 | Remove connected components smaller than this |

## 5. Build hard-negative masks (optional)

You can generate hard-negative masks from saved probability maps:

```bash
python build_hard_negative_dataset.py \
  --pred-dir validation_results/probs \
  --train-img-dir data/training/images \
  --train-label-dir data/training/labels \
  --output-dir data/training_hn \
  --threshold 0.8 \
  --enable-hard-negative
```

Useful flags:

- `--hardest-negative`
- `--erosion-iters`
- `--pos-ratio`
- `--edge-ratio`
- `--hard-negative-ratio`
- `--pred-prefix`, `--img-token`, `--label-token`

The script writes `hard_negative_config.json` into `--output-dir`.

## 6. Evaluate (PyTorch)

Set parameters in `config/val_config.py`, then run:

```bash
python evaluator.py
```

Key evaluation config fields:

| Field | Default | Description |
|---|---|---|
| `model_path` | `./model_results/run_.../unet_3d_best.pth` | Checkpoint path |
| `val_img_dir` | `data/validation/images` | Validation images |
| `val_label_dir` | `data/validation/labels` | Validation labels |
| `patch_size` | `(8, 512, 512)` | Sliding-window patch size |
| `stride` | `(4, 256, 256)` | Sliding-window stride |
| `threshold` | 0.1 | Binarization threshold |
| `dust_remove_min_size` | 128 | Remove small connected components |
| `eval_affinity` | `True` | Enable affinity evaluation |
| `affinity_offsets` | `[(1,0,0), (0,1,0), (0,0,1)]` | Neighbor offsets for affinity metrics |
| `loss_type` | `bce` | `bce` / `focal` / `dicefocal` |
| `save_results` | `True` | Save prediction and probability TIFF files |
| `wandb` | `True` | Enable wandb logging |
| `wandb_project` | `c_elegans_3d_unet_validation` | wandb project |
| `wandb_run_name` | set in config | wandb run name |

Output files:

- `validation_results/pred/*.tif`
- `validation_results/probs/*.tif`

## 7. Evaluate (ONNX)

```bash
python evaluator_onnx.py
```

Set `model_path` in `config/val_config.py` to an `.onnx` model.

## 8. Inference on new volumes

Edit input/output and model path in the `__main__` block of `infer.py`, then run:

```bash
python infer.py
```

Typical outputs:

- `result_seg.tif` (binary segmentation)
- `prob_map.tif` (probability map)

## 9. Practical tuning tips

- Out-of-memory during training or validation:
  reduce `patch_size` and increase overlap later only if needed.
- Higher recall:
  lower `val_threshold` (or validation `threshold`) to values like `0.05`.
- Higher precision:
  raise threshold to values like `0.2` to `0.3`.
- Noisy validation curves:
  set `max_val_volumes=None` and/or increase `validate_every`.
- Unstable training:
  lower learning rate, reduce `hard_negative_sample_ratio`, or increase `pos_weight_cap` carefully.

## 10. Dependencies

Main libraries:

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

Install with:

```bash
pip install -r requirements.txt
```
