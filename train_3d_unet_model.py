import os
import torch
from torch.utils.data import DataLoader
from config.tra_config import tra_hyper
from losses import build_criterion
from validate.evaluators import (  
    evaluate_with_optional_limit,
    maybe_evaluate_train_set,
)
from validate.reporting import print_metrics  
from utils import (
    build_wandb_config,
    build_aug_wandb_config,
    init_wandb_run,
    log_training_loss,
    log_validation_to_wandb,
    finish_wandb_run,
)
from dataset import Tif3DPatchDataset
from training import *


# =========================
# Training
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controls = tra_hyper()

    dataset = build_train_dataset(controls, augment=True)

    loader = DataLoader(
        dataset,
        batch_size=controls["batch_size"],
        shuffle=True,
        num_workers=controls["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = Tif3DPatchDataset(
        img_dir=controls["val_img_dir"],
        label_dir=controls["val_label_dir"],
        patch_size=controls["patch_size"],
        patches_per_volume=controls["val_patches_per_volume"],
        augment=False,
    )

    train_eval_dataset = build_train_dataset(controls, augment=False)

    model, lr, loaded_pretrained = init_model_and_lr(device)

    # Compute pos_weight from training labels to handle class imbalance in BCE.
    # Raw ratio (neg/pos) is very large for this dataset, so cap it for stability.
    total_voxels = sum(lab.size for lab in dataset.labels)
    pos_voxels = sum((lab > 0).sum() for lab in dataset.labels)
    neg_voxels = total_voxels - pos_voxels
    raw_pos_weight = float(neg_voxels) / float(pos_voxels + 1e-8)
    pos_weight_cap = float(controls.get("pos_weight_cap", 30.0))
    pos_weight = min(raw_pos_weight, pos_weight_cap)
    print(
        f"Class balance — pos: {pos_voxels}, neg: {neg_voxels}, "
        f"raw_pos_weight: {raw_pos_weight:.1f}, capped_pos_weight: {pos_weight:.1f}"
    )

    criterion = build_criterion(
        controls["loss_type"],
        controls["dice_weight"],
        controls["focal_weight"],
        pos_weight=pos_weight,
    ).to(device)

    optimizer = build_optimizer(model, controls, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    
    sample_input = dataset[0][0].unsqueeze(0).to(device)

    # Track best validation dice to save best model checkpoint
    best_val_dice = 0.0

    # Initialize wandb run before training loop
    wandb_config = build_wandb_config(loader, lr, controls)
    wandb_config.update(build_aug_wandb_config())
    run_name = init_wandb_run(project=wandb_config["project"], config=wandb_config)
    
    try:
        for epoch in range(controls["num_epochs"]):
            # Disable augmentation in the final epochs to reduce late-stage noise.
            disable_aug_last_epochs = int(controls.get("disable_aug_last_epochs", 0))
            aug_enabled = (epoch + 1) <= (controls["num_epochs"] - disable_aug_last_epochs)
            dataset.augment = aug_enabled

            avg_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, device,
                                             grad_clip_norm=float(controls.get("grad_clip_norm", 0.0)))
            
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{controls['num_epochs']}]  "
                f"Loss: {avg_epoch_loss:.4f}  LR: {current_lr:.2e}  "
                f"Augment: {'ON' if dataset.augment else 'OFF'}"
            )

            log_training_loss(epoch=epoch + 1, train_loss=avg_epoch_loss)

            if (epoch + 1) % controls["validate_every"] == 0 or (epoch == 0 and not loaded_pretrained):
                # Evaluate on train set for sanity check (optional, can be disabled in config)
                train_metrics = maybe_evaluate_train_set(
                    model,
                    train_eval_dataset,
                    device,
                    controls,
                    criterion,
                )

                # Evaluate on validation set with optional volume limit for faster feedback
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

                # Save periodic epoch checkpoint and best model by validation Dice
                os.makedirs(f"./model_results/{run_name}", exist_ok=True)
                model_path = f"./model_results/{run_name}/unet_3d_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")


                # Save best model
                best_val_dice = save_best_model(model, val_metrics, best_val_dice, sample_input, run_name)
                
        finish_wandb_run()

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        os.makedirs(f"./model_results/{run_name}", exist_ok=True)
        torch.save(model.state_dict(), f"./model_results/{run_name}/unet_3d_interrupted.pth")
        print(f"Model saved as: ./model_results/{run_name}/unet_3d_interrupted.pth")
        finish_wandb_run()


if __name__ == "__main__":
    train()
