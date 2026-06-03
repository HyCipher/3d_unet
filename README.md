# C. elegans Synapse Detection with 3D U-Net

This project is designed for synapse detection on 3D C. elegans volumes.
It uses a 3D U-Net for voxel-level binary segmentation and supports the full workflow: training, validation, and inference.

## 1. What This Project Does

Main goals:

- Read 3D TIFF volumes (stored as H, W, Z by default)
- Train a 3D U-Net to detect synapse regions
- Run sliding-window validation and inference on full volumes
- Report standard segmentation metrics (Dice, IoU, F1, Precision, Recall, etc.)
- Optionally log experiments with Weights and Biases (wandb)

Typical use cases:

- Automatic synapse detection in C. elegans neuroimaging data
- Supervised 3D segmentation with paired image/label volumes

## 2. Project Structure

Key folders and scripts:

```text
3d_unet/
├── config/
│   ├── tra_config.py                 # Training configuration
│   └── val_config.py                 # Validation configuration
├── data/
│   ├── training/
│   │   ├── images/                   # Training images (*.tif)
│   │   └── labels/                   # Training labels (*.tif)
│   └── validation/
│       ├── images/                   # Validation images (*.tif)
│       └── labels/                   # Validation labels (*.tif)
├── models/
│   └── detect.py                     # 3D U-Net model definition
├── training/                         # Training pipeline and optimizer utilities
├── evaluation/                       # Sliding-window inference, visualization, evaluation tools
├── validate/                         # Metric computation
├── train_3d_unet_model.py            # Training entry point
├── evaluator.py                      # PyTorch validation entry point
├── evaluator_onnx.py                 # ONNX validation entry point
├── infer.py                          # Inference entry point for a single volume
├── build_hard_negative_dataset.py    # Optional: hard-negative dataset builder
└── requirements.txt
```

## 3. How To Use

Run the following commands from the project root: C_elegans_UNet/3d_unet.

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

If you want experiment tracking:

```bash
wandb login
```

### 3.2 Prepare Data

Default training/validation paths are defined in config/tra_config.py.

Recommended layout:

```text
data/
├── training/
│   ├── images/*.tif
│   └── labels/*.tif
└── validation/
    ├── images/*.tif
    └── labels/*.tif
```

Notes:

- Image and label files must match one-to-one (after sorted filename pairing)
- Volumes are internally converted from H, W, Z to Z, H, W
- Label voxels greater than 0 are treated as foreground (synapse)

### 3.3 Train the Model

```bash
python train_3d_unet_model.py
```

Training outputs:

- Periodic checkpoints: model_results/run_timestamp/unet_3d_epoch_xx.pth
- Interrupted training checkpoint: model_results/run_timestamp/unet_3d_interrupted.pth

Commonly tuned parameters in config/tra_config.py:

- train_img_dir, train_label_dir, val_img_dir, val_label_dir
- patch_size, batch_size, num_epochs
- loss_type (bce / focal / dicefocal)
- val_threshold, dust_remove_min_size
- disable_aug_last_epochs, grad_clip_norm

### 3.4 Validate the Model (PyTorch)

First set these fields in config/val_config.py:

- model_path (checkpoint to evaluate)
- val_img_dir, val_label_dir
- patch_size, stride, threshold

Then run:

```bash
python evaluator.py
```

Validation outputs (when save_results=True):

- validation_results/pred/*.tif
- validation_results/probs/*.tif

### 3.5 Validate with ONNX (Optional)

Set model_path in config/val_config.py to an .onnx file, then run:

```bash
python evaluator_onnx.py
```

### 3.6 Run Inference on New Data

Edit the __main__ section in infer.py:

- img_path: input 3D TIFF file
- model_path: trained .pth checkpoint
- save_path: output segmentation path

Run:

```bash
python infer.py
```

Default outputs:

- result_seg.tif: binary segmentation
- prob_map.tif: probability map

## 4. Practical Tuning Tips

- Out of memory: reduce patch_size or batch_size
- Low recall: lower threshold (for example, threshold=0.05)
- Low precision: increase threshold (for example, 0.2 to 0.3)
- Unstable training: lower learning rate or tune pos_weight_cap and grad_clip_norm

## 5. One-Line Summary

This is an end-to-end 3D U-Net pipeline for C. elegans synapse detection, covering data preparation, training, validation, and inference.
