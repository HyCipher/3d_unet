import os
import torch
import wandb


def save_best_model(model, val_metrics, best_val_dice, sample_input, run_name):
    """Save best model by validation dice and return updated best score."""
    if val_metrics["dice"] > best_val_dice:
        model.eval()
        best_val_dice = val_metrics["dice"]
        os.makedirs(f"./model_results/{run_name}", exist_ok=True)
        pth_path = f"./model_results/{run_name}/unet_3d_best.pth"
        onnx_path = f"./model_results/{run_name}/unet_3d_best.onnx"
        torch.save(model.state_dict(), pth_path)
        torch.onnx.export(model, sample_input, onnx_path)

        if wandb.run is not None:
            artifact = wandb.Artifact(name="unet_3d_best_onnx", type="model")
            artifact.add_file(onnx_path)
            wandb.run.log_artifact(artifact)

        model.train()
        print(f"Best model saved! (Dice: {best_val_dice:.4f})")
    return best_val_dice