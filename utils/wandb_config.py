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
        "save_every": controls["save_every"],
        "eval_train_set": controls["eval_train_set"],
        "pos_weight_cap": controls["pos_weight_cap"],
        "grad_clip_norm": controls["grad_clip_norm"], 
    }
    if controls["loss_type"] == "dicefocal":
        config["dice_weight"] = controls["dice_weight"]
        config["focal_weight"] = controls["focal_weight"]
    return config