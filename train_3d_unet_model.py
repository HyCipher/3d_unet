import os
import glob
import json
from datetime import datetime
import wandb
import tifffile as tiff
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from detect import UNet  # ← 你的3D UNet


# =========================
# Data Augmentation Functions
# =========================
def random_flip_3d(img, label, prob=0.5):
    """随机翻转（沿各个轴）"""
    # Flip along Z axis
    if np.random.rand() < prob:
        img = np.flip(img, axis=0).copy()
        label = np.flip(label, axis=0).copy()
    
    # Flip along Y axis  
    if np.random.rand() < prob:
        img = np.flip(img, axis=1).copy()
        label = np.flip(label, axis=1).copy()
    
    # Flip along X axis
    if np.random.rand() < prob:
        img = np.flip(img, axis=2).copy()
        label = np.flip(label, axis=2).copy()
    
    return img, label


def random_rotation_90_3d(img, label, prob=0.5):
    """随机90度旋转"""
    if np.random.rand() < prob:
        # 在XY平面旋转（常见做法）
        k = np.random.randint(1, 4)  # 1,2,3 对应 90,180,270度
        img = np.rot90(img, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(1, 2)).copy()
    
    return img, label


def random_gaussian_noise(img, label, prob=0.3, std=0.01):
    """添加高斯噪声"""
    if np.random.rand() < prob:
        noise = np.random.normal(0, std, img.shape)
        img = img + noise
    
    return img, label


def apply_augmentation(img, label, augment=True):
    """统一的增强接口"""
    if not augment:
        return img, label
    
    # 应用增强（可选择性启用）
    img, label = random_flip_3d(img, label, prob=0.5)
    img, label = random_rotation_90_3d(img, label, prob=0.3)
    img, label = random_gaussian_noise(img, label, prob=0.2, std=0.01)
    
    return img, label


# =========================
# Focal Loss Implementation
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        inputs: logits from model (not sigmoid applied)
        targets: binary labels (0 or 1)
        """
        targets = targets.float()
        sigmoid_inputs = torch.sigmoid(inputs)

        # Cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Focal term: (1 - p_t)^gamma
        p_t = torch.where(targets == 1, sigmoid_inputs, 1 - sigmoid_inputs)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()
    
    
def dice_loss(inputs, targets, smooth=1e-6):
    """Compute Dice loss (1 - Dice coefficient) using logits as input."""
    # apply sigmoid to logits
    inputs = torch.sigmoid(inputs)
    targets = targets.float()
    inputs_flat = inputs.contiguous().view(inputs.shape[0], -1)
    targets_flat = targets.contiguous().view(targets.shape[0], -1)

    intersection = (inputs_flat * targets_flat).sum(dim=1)
    union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class DiceFocalLoss(nn.Module):
    """Combination of Dice loss and Focal loss."""
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, inputs, targets):
        d_loss = dice_loss(inputs, targets)
        f_loss = self.focal(inputs, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss
    
    
# =========================
# Validation Functions
# =========================
def dice_coefficient(pred, gt, smooth=1e-6):
    """计算Dice系数"""
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)
    return dice


def sliding_window_inference_val(
    volume,
    model,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    device="cuda"
):
    """滑动窗口推理（用于验证）"""
    model.eval()
    Z, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((Z, H, W), dtype=np.float32)
    count_map = np.zeros((Z, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, Z, sd):
            z0 = min(z, Z - pd)
            for y in range(0, H - ph + 1, sh):
                y0 = min(y, H - ph)
                for x in range(0, W - pw + 1, sw):
                    x0 = min(x, W - pw)

                    patch = volume[z0:z0+pd, y0:y0+ph, x0:x0+pw]
                    patch = (patch - patch.mean()) / (patch.std() + 1e-8)

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                    patch = patch.to(device)

                    pred = model(patch)
                    pred = torch.sigmoid(pred)
                    pred = pred.cpu().numpy()[0, 0]

                    output[z0:z0+pd, y0:y0+ph, x0:x0+pw] += pred
                    count_map[z0:z0+pd, y0:y0+ph, x0:x0+pw] += 1

    output /= np.maximum(count_map, 1e-8)
    return output


def validate_with_full_metrics(model, dataset, device, patch_size=(8, 512, 512), stride=(2, 64, 64), threshold=0.5, criterion=None):
    """计算完整指标：Dice, IoU, F1, Precision, Recall, Specificity（可选 val loss）"""
    model.eval()
    dice_scores = []
    iou_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    loss_values = []

    total_volumes = len(dataset.volumes)
    print(f"Validation start: {total_volumes} volumes, patch={patch_size}, stride={stride}")

    with torch.no_grad():
        for idx in range(total_volumes):
            vol = dataset.volumes[idx]
            lab = dataset.labels[idx]

            print(f"  validating volume {idx+1}/{total_volumes} ...")

            # 计算 val loss：在滑动窗口每个 patch 上算 loss 再平均
            if criterion is not None:
                pd_s, ph_s, pw_s = patch_size
                sd_s, sh_s, sw_s = stride
                Z, H, W = vol.shape
                patch_losses = []
                for z in range(0, Z, sd_s):
                    z0 = min(z, Z - pd_s)
                    for y in range(0, H - ph_s + 1, sh_s):
                        y0 = min(y, H - ph_s)
                        for x in range(0, W - pw_s + 1, sw_s):
                            x0 = min(x, W - pw_s)
                            xp = vol[z0:z0+pd_s, y0:y0+ph_s, x0:x0+pw_s].copy()
                            yp = lab[z0:z0+pd_s, y0:y0+ph_s, x0:x0+pw_s].copy()
                            xp = (xp - xp.mean()) / (xp.std() + 1e-8)
                            yp = (yp > 0).astype(np.float32)
                            xt = torch.from_numpy(xp).unsqueeze(0).unsqueeze(0).float().to(device)
                            yt = torch.from_numpy(yp).unsqueeze(0).unsqueeze(0).float().to(device)
                            logits = model(xt)
                            patch_losses.append(criterion(logits, yt).item())
                loss_values.append(float(np.mean(patch_losses)))

            prob_map = sliding_window_inference_val(
                vol, model, patch_size=patch_size, stride=stride, device=device
            )

            pred_seg = (prob_map > threshold).astype(np.uint8)
            gt_seg = (lab > 0).astype(np.uint8)

            # Dice
            dice = dice_coefficient(pred_seg, gt_seg)
            dice_scores.append(dice)

            # IoU
            iou = iou_score(pred_seg, gt_seg)
            iou_scores.append(iou)

            # Precision, Recall, F1, Specificity
            tp = np.sum(pred_seg * gt_seg)
            fp = np.sum(pred_seg * (1 - gt_seg))
            fn = np.sum((1 - pred_seg) * gt_seg)
            tn = np.sum((1 - pred_seg) * (1 - gt_seg))

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            specificity = tn / (tn + fp + 1e-6)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            specificity_scores.append(specificity)

    # 返回平均值
    result = {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'f1': np.mean(f1_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'specificity': np.mean(specificity_scores)
    }
    if loss_values:
        result['loss'] = float(np.mean(loss_values))
    return result


def save_validation_history(history, history_path="validation_history.json"):
    """Persist validation history after each evaluation so curves can be plotted later."""
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def save_training_loss_history(history, history_path="training_loss.json"):
    """Persist per-epoch training loss for plotting and comparison."""
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def iou_score(pred, gt, smooth=1e-6):
    """计算IoU"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)


# =========================
# Dataset：3D tif → 3D patch
# =========================
class Tif3DPatchDataset(Dataset):
    def __init__(self, img_dir, label_dir, patch_size=(4, 128, 128), patches_per_volume=200, augment=True):
        """
        patch_size: (depth, height, width)
        augment: whether to apply data augmentation
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
        assert len(self.img_paths) == len(self.label_paths)

        self.patch_size = patch_size
        self.augment = augment
        self.volumes = []
        self.labels = []

        for ip, lp in zip(self.img_paths, self.label_paths):
            vol = tiff.imread(ip).astype(np.float32)
            lab = tiff.imread(lp).astype(np.float32)

            # (H,W,Z) → (Z,H,W)
            vol = np.transpose(vol, (2, 0, 1))
            lab = np.transpose(lab, (2, 0, 1))

            assert vol.shape == lab.shape

            self.volumes.append(vol)
            self.labels.append(lab)

        self.num_volumes = len(self.volumes)
        self.patches_per_volume = patches_per_volume

        # 预计算正样本坐标（避免每次 __getitem__ 重新扫描整个 volume）
        self.pos_coords = []
        for lab in self.labels:
            coords = np.argwhere(lab > 0)
            self.pos_coords.append(coords)
        print(f"Dataset: {self.num_volumes} volumes, "
              f"pos coords cached: {[len(c) for c in self.pos_coords]}")

    def __len__(self):
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx):
        vid = idx // self.patches_per_volume
        vol = self.volumes[vid]
        lab = self.labels[vid]

        d, h, w = vol.shape
        pd, ph, pw = self.patch_size

        # 随机裁剪3D patch
        if np.random.rand() < 0.8:
            pos = self.pos_coords[vid]   # ← 直接用预计算的坐标
            if len(pos) > 0:
                zc, yc, xc = pos[np.random.randint(len(pos))]
                z = np.clip(zc - pd//2, 0, d - pd)
                y = np.clip(yc - ph//2, 0, h - ph)
                x = np.clip(xc - pw//2, 0, w - pw)
            else:
                z = np.random.randint(0, d - pd + 1)
                y = np.random.randint(0, h - ph + 1)
                x = np.random.randint(0, w - pw + 1)
        else:
            # 纯随机背景
            z = np.random.randint(0, d - pd + 1)
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)

        x_patch = vol[z:z+pd, y:y+ph, x:x+pw]
        y_patch = lab[z:z+pd, y:y+ph, x:x+pw]

        # normalize image
        # x_patch = (x_patch - x_patch.min()) / (x_patch.max() - x_patch.min() + 1e-8)
        x_patch = (x_patch - x_patch.mean()) / (x_patch.std() + 1e-8)
        
        # label 二值化
        y_patch = (y_patch > 0).astype(np.float32)
        
        # ========== Data Augmentation ==========
        x_patch, y_patch = apply_augmentation(x_patch, y_patch, augment=self.augment)

        # 转成 tensor [C,D,H,W]
        x_patch = torch.from_numpy(x_patch).unsqueeze(0)
        y_patch = torch.from_numpy(y_patch).unsqueeze(0)

        return x_patch.float(), y_patch.float()


# =========================
# Training
# =========================
def init_model_and_lr(device, pretrained_path="./models/unet_3d_best.pth"):
    """Create model and optionally load a pretrained checkpoint."""
    model = UNet().to(device)
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pre-trained model from {pretrained_path}")
        return model, 1e-4, True

    print("No pre-trained model found, starting from scratch")
    return model, 1e-4, False


def get_control_panel():
    """Centralized training/validation hyperparameters."""
    return {
        "validate_every": 10,
        "eval_train_set": True,
        "max_val_volumes": None,
        "val_patch_size": (16, 512, 512),
        "val_stride": (8, 256, 256),
        "val_threshold": 0.1,
        "dice_weight": 0.8,
        "focal_weight": 1.0,
        "num_epochs": 50,
        "loss_type": "bce",  # "bce" | "focal" | "dicefocal"
    }


def build_criterion(loss_type, dice_weight, focal_weight):
    """Build loss function from LOSS_TYPE."""
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    if loss_type == "dicefocal":
        return DiceFocalLoss(
            alpha=0.25,
            gamma=2.0,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
        )
    raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")


def create_optimizer_and_scheduler(model, lr):
    """Create optimizer and LR scheduler."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    return optimizer, scheduler


def resolve_history_paths():
    """Avoid overwriting history files across different runs."""
    validation_history_path = "validation_history.json"
    if os.path.exists(validation_history_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_history_path = f"validation_history_{timestamp}.json"
        print(
            "Detected existing validation_history.json; "
            f"saving current run to {validation_history_path}"
        )

    training_loss_path = "training_loss.json"
    if os.path.exists(training_loss_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_loss_path = f"training_loss_{timestamp}.json"
        print(
            "Detected existing training_loss.json; "
            f"saving current run to {training_loss_path}"
        )

    return validation_history_path, training_loss_path


def build_wandb_config(loader, lr, controls):
    """Build wandb config from runtime values."""
    config = {
        "architecture": "3D UNet",
        "epochs": controls["num_epochs"],
        "batch_size": loader.batch_size,
        "learning_rate": lr,
        "patch_size": loader.dataset.patch_size,
        "val_patch_size": controls["val_patch_size"],
        "val_stride": controls["val_stride"],
        "val_threshold": controls["val_threshold"],
        "loss_function": controls["loss_type"],
        "validate_every": controls["validate_every"],
        "eval_train_set": controls["eval_train_set"],
    }
    if controls["loss_type"] == "dicefocal":
        config["dice_weight"] = controls["dice_weight"]
        config["focal_weight"] = controls["focal_weight"]
    return config


def run_sanity_check(model, dataset, device):
    """Quick prediction sanity check on one training patch."""
    model.eval()
    with torch.no_grad():
        x_test, y_test = dataset[0]
        x_test = x_test.unsqueeze(0).to(device)
        y_test = y_test.to(device)

        pred_test = torch.sigmoid(model(x_test))
        print("Sanity check:")
        print(
            " pred mean:", pred_test.mean().item(),
            " pred max:", pred_test.max().item(),
            " gt mean:", y_test.mean().item(),
            " gt max:", y_test.max().item(),
        )


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return avg loss."""
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate_with_optional_limit(model, dataset, device, controls, criterion):
    """Evaluate validation dataset with optional volume cap for speed."""
    original_volumes = dataset.volumes
    original_labels = dataset.labels
    max_val_volumes = controls["max_val_volumes"]

    if max_val_volumes is not None:
        dataset.volumes = dataset.volumes[:max_val_volumes]
        dataset.labels = dataset.labels[:max_val_volumes]

    metrics = validate_with_full_metrics(
        model,
        dataset,
        device,
        patch_size=controls["val_patch_size"],
        stride=controls["val_stride"],
        threshold=controls["val_threshold"],
        criterion=criterion,
    )

    dataset.volumes = original_volumes
    dataset.labels = original_labels
    return metrics


def maybe_evaluate_train_set(model, train_eval_dataset, device, controls, criterion):
    """Optionally evaluate training set to monitor overfitting."""
    if not controls["eval_train_set"]:
        return None

    return validate_with_full_metrics(
        model,
        train_eval_dataset,
        device,
        patch_size=controls["val_patch_size"],
        stride=controls["val_stride"],
        threshold=controls["val_threshold"],
        criterion=criterion,
    )


def print_metrics(train_metrics, val_metrics):
    """Console-friendly metrics output."""
    if train_metrics is not None:
        print(
            "Train Metrics -> "
            f"Dice: {train_metrics['dice']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"Specificity: {train_metrics['specificity']:.4f}"
        )

    if "loss" in val_metrics:
        print(f"Validation Loss: {val_metrics['loss']:.4f}")

    print(
        "Validation Metrics -> "
        f"Dice: {val_metrics['dice']:.4f}, "
        f"IoU: {val_metrics['iou']:.4f}, "
        f"F1: {val_metrics['f1']:.4f}, "
        f"Precision: {val_metrics['precision']:.4f}, "
        f"Recall: {val_metrics['recall']:.4f}, "
        f"Specificity: {val_metrics['specificity']:.4f}"
    )


def log_validation_to_wandb(train_metrics, val_metrics, epoch):
    """Send validation metrics to wandb."""
    payload = {
        "train_dice": train_metrics.get("train_dice"),
        "train_iou": train_metrics.get("train_iou"),
        "train_f1": train_metrics.get("train_f1"),
        "train_precision": train_metrics.get("train_precision"),
        "train_recall": train_metrics.get("train_recall"),
        "train_specificity": train_metrics.get("train_specificity"),
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_specificity": val_metrics["specificity"],
    }
    
    if "loss" in val_metrics:
        payload["val_loss"] = val_metrics["loss"]
    wandb.log(payload)


def save_epoch_model(model, epoch):
    """Save periodic epoch checkpoint."""
    model_path = f"./models/unet_3d_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def maybe_save_best_model(model, val_metrics, best_val_dice):
    """Save best model by validation dice and return updated best score."""
    if val_metrics["dice"] > best_val_dice:
        best_val_dice = val_metrics["dice"]
        torch.save(model.state_dict(), "./models/unet_3d_best.pth")
        print(f"Best model saved! (Dice: {best_val_dice:.4f})")
    return best_val_dice


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Tif3DPatchDataset(
        img_dir="data/training/images",
        label_dir="data/training/labels",
        patch_size=(8, 512, 512),
        patches_per_volume=200,
        augment=True  # ← Enable augmentation for training
    )

    x, y = dataset[0]
    print("x shape:", x.shape)
    print("label mean:", y.mean().item())
    print("label max:", y.max().item())

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True   # ← 避免每 epoch 重建 worker 进程
    )

    # Create validation dataset
    val_dataset = Tif3DPatchDataset(
        img_dir="data/validation/images",
        label_dir="data/validation/labels",
        patch_size=(16, 512, 512),
        patches_per_volume=50,  # 用更少的patch验证以加快速度
        augment=False  # ← Disable augmentation for validation
    )

    # Create a dataset object for evaluating training-set performance
    # (用于检测过拟合：训练集 dice vs 验证集 dice)
    train_eval_dataset = Tif3DPatchDataset(
        img_dir="data/training/images",
        label_dir="data/training/labels",
        patch_size=(16, 512, 512),
        patches_per_volume=50,
        augment=False
    )

    model, lr, loaded_pretrained = init_model_and_lr(device)
    controls = get_control_panel()
    criterion = build_criterion(
        controls["loss_type"],
        controls["dice_weight"],
        controls["focal_weight"],
    )
    optimizer, scheduler = create_optimizer_and_scheduler(model, lr)

    # For tracking best validation score
    best_val_dice = 0.0
    val_history = []
    training_loss_history = []

    validation_history_path, training_loss_path = resolve_history_paths()
    wandb_config = build_wandb_config(loader, lr, controls)

    wandb.init(
        project="c_elegans_3d_unet",
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=wandb_config
    )
    try:
        for epoch in range(controls["num_epochs"]):
            avg_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{controls['num_epochs']}]  "
                f"Loss: {avg_epoch_loss:.4f}  LR: {current_lr:.2e}"
            )

            # Save per-epoch training loss so it can be plotted independently.
            training_loss_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss
            })
            save_training_loss_history(training_loss_history, training_loss_path)
            wandb.log({"train_loss": avg_epoch_loss, "epoch": epoch + 1})
            run_sanity_check(model, dataset, device)
                
            # 每隔 VALIDATE_EVERY 个 epoch 进行验证并保存模型
            # 从头训练时 epoch 1 额外做一次，导入预训练模型时则不需要
            if (epoch + 1) % controls["validate_every"] == 0 or (epoch == 0 and not loaded_pretrained):
                train_metrics = maybe_evaluate_train_set(
                    model,
                    train_eval_dataset,
                    device,
                    controls,
                    criterion,
                )

                val_metrics = evaluate_with_optional_limit(
                    model,
                    val_dataset,
                    device,
                    controls,
                    criterion,
                )

                history_item = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "train_loss": avg_epoch_loss,
                    "val_loss": val_metrics.get('loss')
                }
                val_history.append(history_item)
                save_validation_history(val_history, validation_history_path)

                print_metrics(train_metrics, val_metrics)
                log_validation_to_wandb(val_metrics, epoch + 1)

                scheduler.step(val_metrics['dice'])
                print(f"Scheduler updated by validation Dice; next LR: {optimizer.param_groups[0]['lr']:.2e}")

                save_epoch_model(model, epoch + 1)
                best_val_dice = maybe_save_best_model(model, val_metrics, best_val_dice)

        if val_history:
            save_validation_history(val_history, validation_history_path)
            print(f"Validation history saved to: {validation_history_path}")

        if training_loss_history:
            save_training_loss_history(training_loss_history, training_loss_path)
            print(f"Training loss history saved to: {training_loss_path}")
        wandb.finish()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        torch.save(model.state_dict(), "./models/unet_3d_interrupted.pth")
        print("Model saved as: unet_3d_interrupted.pth")
        
        # Save validation history
        if val_history:
            save_validation_history(val_history, validation_history_path)
            print(f"Validation history saved to: {validation_history_path}")

        if training_loss_history:
            save_training_loss_history(training_loss_history, training_loss_path)
            print(f"Training loss history saved to: {training_loss_path}")
        wandb.finish()


if __name__ == "__main__":
    train()
