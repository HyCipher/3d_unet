"""Utils package.

Provides functionality for recording training progress and logging metrics.

Submodules:
    wandb_run      — Run lifecycle (build config, init, finish)
    wandb_metrics  — Training scalar metrics
    wandb_curves   — PR/ROC and F1 threshold curves
    wandb_tables   — Per-sample tables and file artifacts
    aug_wandb_config — Augmentation hyperparameter config for wandb
"""
from utils.wandb_run import init_wandb_run, finish_wandb_run
from utils.wandb_metrics import log_training_loss
from utils.wandb_curves import log_pr_roc_to_wandb, log_f1_curve_to_wandb
from utils.wandb_tables import (
    log_sample_table_to_wandb,
    log_summary_table_to_wandb,
    log_generated_files_to_wandb,
)
from utils.aug_wandb_config import build_aug_wandb_config
from utils.wandb_config import build_wandb_config

__all__ = [
    # wandb_run
    "build_wandb_config",
    "init_wandb_run",
    "finish_wandb_run",
    # wandb_metrics
    "log_training_loss",
    # wandb_curves
    "log_pr_roc_to_wandb",
    "log_f1_curve_to_wandb",
    # wandb_tables
    "log_sample_table_to_wandb",
    "log_summary_table_to_wandb",
    "log_generated_files_to_wandb",
    # aug_wandb_config
    "build_aug_wandb_config",
]

