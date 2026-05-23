from datetime import datetime
import wandb


# ── Run Lifecycle ──────────────────────────────────────────────────────────────

def build_wandb_config(loader, lr, controls):
    """Build wandb config from runtime values."""
    config = {
        "project": "c_elegans_3d_unet",
        "architecture": controls["architecture"],
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
        "pos_weight_cap": controls["pos_weight_cap"],
        "grad_clip_norm": controls["grad_clip_norm"],
    }
    if controls["loss_type"] == "dicefocal":
        config["dice_weight"] = controls["dice_weight"]
        config["focal_weight"] = controls["focal_weight"]
    return config


def init_wandb_run(project, config):
    """Initialize a wandb run with a timestamped name."""
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=project,
        name=run_name,
        config=config,
        # settings=wandb.Settings(silent=True, console="off"),
    )
    return run_name


def finish_wandb_run():
    """Close current wandb run."""
    wandb.finish()
