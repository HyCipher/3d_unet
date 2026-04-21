import os
import glob
import numpy as np
import torch
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from config import get_control_panel
from nets.detect import UNet  
from augmentations import apply_augmentation  
from losses import build_criterion  
from validate.evaluators import (  
    evaluate_with_optional_limit,
    maybe_evaluate_train_set,
)
from validate.reporting import print_metrics  
from tracking import (
    build_wandb_config,
    finish_wandb_run,
    init_wandb_run,
    log_train_loss,
    log_validation_to_wandb,
)


# =========================
# Dataset: 3D tif -> 3D patch
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

            # (H,W,Z) -> (Z,H,W)
            vol = np.transpose(vol, (2, 0, 1))
            lab = np.transpose(lab, (2, 0, 1))

            assert vol.shape == lab.shape

            self.volumes.append(vol)
            self.labels.append(lab)

        self.num_volumes = len(self.volumes)
        self.patches_per_volume = patches_per_volume

        # Cache positive coordinates to avoid scanning the full volume every __getitem__ call.
        self.pos_coords = []
        for lab in self.labels:
            coords = np.argwhere(lab > 0)
            self.pos_coords.append(coords)
        print(
            f"Dataset: {self.num_volumes} volumes, "
            f"pos coords cached: {[len(c) for c in self.pos_coords]}"
        )

    def __len__(self):
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx):
        vid = idx // self.patches_per_volume
        vol = self.volumes[vid]
        lab = self.labels[vid]

        d, h, w = vol.shape
        pd, ph, pw = self.patch_size

        # Random crop 3D patch
        if np.random.rand() < 0.8:
            pos = self.pos_coords[vid]
            if len(pos) > 0:
                zc, yc, xc = pos[np.random.randint(len(pos))]
                z = np.clip(zc - pd // 2, 0, d - pd)
                y = np.clip(yc - ph // 2, 0, h - ph)
                x = np.clip(xc - pw // 2, 0, w - pw)
            else:
                z = np.random.randint(0, d - pd + 1)
                y = np.random.randint(0, h - ph + 1)
                x = np.random.randint(0, w - pw + 1)
        else:
            z = np.random.randint(0, d - pd + 1)
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)

        x_patch = vol[z : z + pd, y : y + ph, x : x + pw]
        y_patch = lab[z : z + pd, y : y + ph, x : x + pw]

        x_patch = (x_patch - x_patch.mean()) / (x_patch.std() + 1e-8)
        y_patch = (y_patch > 0).astype(np.float32)

        x_patch, y_patch = apply_augmentation(x_patch, y_patch, augment=self.augment)

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
        torch.save(model.state_dict(), f"./models/3d_unet_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        print(f"Best model saved! (Dice: {best_val_dice:.4f})")
    return best_val_dice


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Tif3DPatchDataset(
        img_dir="data/training/images",
        label_dir="data/training/labels",
        patch_size=(8, 512, 512),
        patches_per_volume=200,
        augment=True,
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
        persistent_workers=True,
    )

    val_dataset = Tif3DPatchDataset(
        img_dir="data/validation/images",
        label_dir="data/validation/labels",
        patch_size=(8, 512, 512),
        patches_per_volume=50,
        augment=False,
    )

    train_eval_dataset = Tif3DPatchDataset(
        img_dir="data/training/images",
        label_dir="data/training/labels",
        patch_size=(8, 512, 512),
        patches_per_volume=50,
        augment=False,
    )

    model, lr, loaded_pretrained = init_model_and_lr(device)
    controls = get_control_panel()
    criterion = build_criterion(
        controls["loss_type"],
        controls["dice_weight"],
        controls["focal_weight"],
    )
    optimizer, scheduler = create_optimizer_and_scheduler(model, lr)

    best_val_dice = 0.0

    wandb_config = build_wandb_config(loader, lr, controls)

    init_wandb_run(project="c_elegans_3d_unet", config=wandb_config)
    try:
        for epoch in range(controls["num_epochs"]):
            avg_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{controls['num_epochs']}]  "
                f"Loss: {avg_epoch_loss:.4f}  LR: {current_lr:.2e}"
            )

            log_train_loss(epoch=epoch + 1, train_loss=avg_epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                run_sanity_check(model, dataset, device)

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

                print_metrics(train_metrics, val_metrics)
                log_validation_to_wandb(train_metrics, val_metrics, epoch + 1)

                scheduler.step(val_metrics["dice"])
                print(f"Scheduler updated by validation Dice; next LR: {optimizer.param_groups[0]['lr']:.2e}")

                save_epoch_model(model, epoch + 1)
                best_val_dice = maybe_save_best_model(model, val_metrics, best_val_dice)

        finish_wandb_run()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        torch.save(model.state_dict(), "./models/unet_3d_interrupted.pth")
        print("Model saved as: unet_3d_interrupted.pth")
        finish_wandb_run()


if __name__ == "__main__":
    train()
