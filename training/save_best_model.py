import os
import torch


def save_best_model(model, val_metrics, best_val_dice, sample_input):
    """Save best model by validation dice and return updated best score."""
    if val_metrics["dice"] > best_val_dice:
        best_val_dice = val_metrics["dice"]
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), "./models/unet_3d_best.pth")
        model.eval()
        torch.onnx.export(model, sample_input, "./models/unet_3d_best.onnx")
        model.train()
        print(f"Best model saved! (Dice: {best_val_dice:.4f})")
    return best_val_dice