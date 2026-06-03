import wandb


# ── Training Logging ───────────────────────────────────────────────────────────

def log_training_loss(epoch, train_loss):
    """Log train loss for one epoch."""
    wandb.log({"train_loss": train_loss, "epoch": epoch})
