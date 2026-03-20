import os
import glob
import json
from datetime import datetime
from turtle import pd
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

    model = UNet().to(device)

    # 加载预训练模型（如果存在）
    pretrained_path = "./models/unet_3d_best.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pre-trained model from {pretrained_path}")
        # 继续训练时降低学习率
        lr = 1e-4
        loaded_pretrained = True
    else:
        print("No pre-trained model found, starting from scratch")
        lr = 1e-4
        loaded_pretrained = False

    # ====== 损失函数设置 ======
    # 1. 二元交叉熵损失（不考虑类别不平衡时的基础选择）
    # criterion = nn.BCEWithLogitsLoss()
    # 2. Focal Loss（推荐用于小目标/高度不平衡）
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # 3. 组合 Dice Loss + Focal Loss（推荐用于小目标/高度不平衡），可以根据需要调整 dice_weight 和 focal_weight 的比例
    criterion = DiceFocalLoss(alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=1.0)
    
    # 学习权重调整（如果使用组合损失，可以适当调整权重）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # For tracking best validation score
    best_val_dice = 0.0
    val_history = []
    training_loss_history = []

    # Avoid overwriting existing validation history from previous runs.
    validation_history_path = "validation_history.json"
    if os.path.exists(validation_history_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_history_path = f"validation_history_{timestamp}.json"
        print(
            "Detected existing validation_history.json; "
            f"saving current run to {validation_history_path}"
        )

    # Avoid overwriting existing training loss history from previous runs.
    training_loss_path = "training_loss.json"
    if os.path.exists(training_loss_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_loss_path = f"training_loss_{timestamp}.json"
        print(
            "Detected existing training_loss.json; "
            f"saving current run to {training_loss_path}"
        )

    # ====== Validation speed controls ======
    VALIDATE_EVERY = 10         # 每隔多少 epoch 进行一次验证（过于频繁会显著增加训练时间）
    EVAL_TRAIN_SET = False      # True 会额外评估训练集，速度明显变慢
    MAX_VAL_VOLUMES = 1         # 每次最多验证多少个 volume
    VAL_PATCH_SIZE = (16, 512, 512)
    VAL_STRIDE = (8, 256, 256)  # 比 (4,128,128) 快很多

    num_epochs = 100
    try:
        for epoch in range(num_epochs):
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

            avg_epoch_loss = epoch_loss / len(loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_epoch_loss:.4f}  LR: {current_lr:.2e}")

            # Save per-epoch training loss so it can be plotted independently.
            training_loss_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss
            })
            save_training_loss_history(training_loss_history, training_loss_path)
            
            # ====== Sanity check ======
            model.eval()
            with torch.no_grad():
                x_test, y_test = dataset[0]   # 直接拿训练集里的一个 patch
                x_test = x_test.unsqueeze(0).to(device)  # [1,1,D,H,W]
                y_test = y_test.to(device)

                pred_test = model(x_test)
                pred_test = torch.sigmoid(pred_test)

                print("Sanity check:")
                print(
                    " pred mean:", pred_test.mean().item(),
                    " pred max:", pred_test.max().item(),
                    " gt mean:", y_test.mean().item(),
                    " gt max:", y_test.max().item()
                )
                
            # 每隔 VALIDATE_EVERY 个 epoch 进行验证并保存模型
            # 从头训练时 epoch 1 额外做一次，导入预训练模型时则不需要
            if (epoch + 1) % VALIDATE_EVERY == 0 or (epoch == 0 and not loaded_pretrained):
                if EVAL_TRAIN_SET:
                    train_metrics = validate_with_full_metrics(
                        model,
                        train_eval_dataset,
                        device,
                        patch_size=VAL_PATCH_SIZE,
                        stride=VAL_STRIDE,
                        threshold=0.5,
                        criterion=criterion
                    )
                else:
                    train_metrics = None

                # 只取前 MAX_VAL_VOLUMES 个 volume 做快速验证
                original_val_volumes = val_dataset.volumes
                original_val_labels = val_dataset.labels
                if MAX_VAL_VOLUMES is not None:
                    val_dataset.volumes = val_dataset.volumes[:MAX_VAL_VOLUMES]
                    val_dataset.labels = val_dataset.labels[:MAX_VAL_VOLUMES]

                val_metrics = validate_with_full_metrics(
                    model,
                    val_dataset,
                    device,
                    patch_size=VAL_PATCH_SIZE,
                    stride=VAL_STRIDE,
                    threshold=0.5,
                    criterion=criterion
                )

                val_dataset.volumes = original_val_volumes
                val_dataset.labels = original_val_labels

                history_item = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": val_metrics,
                    "train_loss": avg_epoch_loss,
                    "val_loss": val_metrics.get('loss')
                }
                val_history.append(history_item)
                save_validation_history(val_history, validation_history_path)

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

                if 'loss' in val_metrics:
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

                scheduler.step(val_metrics['dice'])
                print(f"Scheduler updated by validation Dice; next LR: {optimizer.param_groups[0]['lr']:.2e}")

                # Save model every 10 epochs
                model_path = f"./models/unet_3d_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")
                
                # Save best model
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    torch.save(model.state_dict(), "./models/unet_3d_best.pth")
                    print(f"Best model saved! (Dice: {best_val_dice:.4f})")

        if val_history:
            save_validation_history(val_history, validation_history_path)
            print(f"Validation history saved to: {validation_history_path}")

        if training_loss_history:
            save_training_loss_history(training_loss_history, training_loss_path)
            print(f"Training loss history saved to: {training_loss_path}")
            
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
        


if __name__ == "__main__":
    train()
