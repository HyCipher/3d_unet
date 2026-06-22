import os
import torch
import wandb


def _do_export(model, patch_size, onnx_path):
    """Core ONNX export. Returns True on success, False on failure."""
    try:
        model.eval()
        device = next(model.parameters()).device
        sample_input = torch.randn((1, 1, *patch_size), device=device)
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            )
        return True
    except Exception as e:
        print(f"[ONNX export] Failed to export {onnx_path}: {e}")
        return False
    finally:
        model.train()


def export_final_onnx(model, patch_size, run_name):
    """Export final model state to ONNX. Called when training ends in any way."""
    os.makedirs(f"./model_results/{run_name}", exist_ok=True)
    onnx_path = f"./model_results/{run_name}/unet_3d_final.onnx"

    if _do_export(model, patch_size, onnx_path):
        print(f"Final ONNX exported: {onnx_path}")

        if wandb.run is not None:
            try:
                artifact = wandb.Artifact(name="unet_3d_final_onnx", type="model")
                artifact.add_file(onnx_path)
                wandb.run.log_artifact(artifact)
            except Exception as e:
                print(f"[ONNX export] wandb artifact upload failed: {e}")


def export_best_onnx(model, patch_size, run_name, val_dice):
    """Unconditionally export best.onnx. Caller decides when to invoke."""
    os.makedirs(f"./model_results/{run_name}", exist_ok=True)
    onnx_path = f"./model_results/{run_name}/unet_3d_best.onnx"

    if _do_export(model, patch_size, onnx_path):
        print(f"Best ONNX exported (Dice: {val_dice:.4f}): {onnx_path}")

        if wandb.run is not None:
            try:
                artifact = wandb.Artifact(name="unet_3d_best_onnx", type="model")
                artifact.add_file(onnx_path)
                wandb.run.log_artifact(artifact)
            except Exception as e:
                print(f"[ONNX export] wandb artifact upload failed: {e}")
