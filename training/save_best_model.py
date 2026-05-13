import os
import torch


def save_best_model(model, val_metrics, best_val_dice, sample_input):
    """Save best model by validation dice and return updated best score."""
    if val_metrics["dice"] > best_val_dice:
        model.eval()
        best_val_dice = val_metrics["dice"]
        os.makedirs("./model_results", exist_ok=True)
        torch.save(model.state_dict(), "./model_results/unet_3d_best.pth")
        torch.onnx.export(model, sample_input, "./model_results/unet_3d_best.onnx", opset_version=17)
        model.train()
        print(f"Best model saved! (Dice: {best_val_dice:.4f})")
    return best_val_dice