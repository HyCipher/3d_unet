import os
import torch


def save_epoch_checkpoint(model, run_name, epoch):
    """Save periodic epoch checkpoint."""
    os.makedirs(f"./model_results/{run_name}", exist_ok=True)
    model_path = f"./model_results/{run_name}/unet_3d_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def save_best_model(model, run_name, val_dice):
    """Unconditionally save best.pth. Caller decides when to invoke."""
    os.makedirs(f"./model_results/{run_name}", exist_ok=True)
    model_path = f"./model_results/{run_name}/unet_3d_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved! (Dice: {val_dice:.4f})")


def save_interrupted_checkpoint(model, run_name):
    """Save interrupted training checkpoint."""
    os.makedirs(f"./model_results/{run_name}", exist_ok=True)
    model_path = f"./model_results/{run_name}/unet_3d_interrupted.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as: {model_path}")