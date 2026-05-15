import argparse
from pathlib import Path

import wandb

from config.val_config import get_validation_config
from evaluator import evaluate_model
from train_3d_unet_model import train


def find_latest_best_model(model_results_dir: str = "./model_results") -> str:
    """Find the newest run_*/unet_3d_best.pth after training."""
    base = Path(model_results_dir)
    candidates = list(base.glob("run_*/unet_3d_best.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No best checkpoint found under ./model_results/run_*/unet_3d_best.pth"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 3D UNet, then automatically run evaluator on the best checkpoint."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit checkpoint path. If omitted, uses latest run_*/unet_3d_best.pth.",
    )
    parser.add_argument("--val-img-dir", default=None, help="Validation image directory override.")
    parser.add_argument("--val-label-dir", default=None, help="Validation label directory override.")
    parser.add_argument("--threshold", type=float, default=None, help="Evaluation threshold override.")
    parser.add_argument("--loss-type", default=None, help="Loss type for validation loss calculation.")
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save prediction/probability tif files during evaluation.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Disable saving prediction/probability tif files during evaluation.",
    )
    parser.add_argument(
        "--eval-wandb",
        action="store_true",
        help="Enable wandb logging for the automatic evaluation stage.",
    )
    parser.add_argument(
        "--no-eval-wandb",
        action="store_true",
        help="Disable wandb logging for the automatic evaluation stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Step 1/2: Training ===")
    train()

    print("\n=== Step 2/2: Auto Evaluation ===")
    config = get_validation_config()

    model_path = args.model_path or find_latest_best_model()
    val_img_dir = args.val_img_dir or config["val_img_dir"]
    val_label_dir = args.val_label_dir or config["val_label_dir"]
    patch_size = tuple(config["patch_size"])
    stride = tuple(config["stride"])
    threshold = args.threshold if args.threshold is not None else config["threshold"]
    loss_type = args.loss_type or config["loss_type"]

    if args.save_results and args.no_save_results:
        raise ValueError("Use either --save-results or --no-save-results, not both.")
    if args.save_results:
        save_results = True
    elif args.no_save_results:
        save_results = False
    else:
        save_results = config["save_results"]

    if args.eval_wandb and args.no_eval_wandb:
        raise ValueError("Use either --eval-wandb or --no-eval-wandb, not both.")
    if args.eval_wandb:
        use_eval_wandb = True
    elif args.no_eval_wandb:
        use_eval_wandb = False
    else:
        use_eval_wandb = bool(config.get("wandb", False))

    wandb_run = None
    if use_eval_wandb:
        model_tag = Path(model_path).parent.name
        wandb_run = wandb.init(
            project=config["wandb_project"],
            name=f"{model_tag}_auto_eval",
            config={
                "model_path": model_path,
                "val_img_dir": val_img_dir,
                "val_label_dir": val_label_dir,
                "patch_size": patch_size,
                "stride": stride,
                "threshold": threshold,
                "loss_type": loss_type,
                "save_results": save_results,
            },
            job_type="auto-validation",
            settings=wandb.Settings(silent=True, console="off"),
        )

    try:
        summary = evaluate_model(
            model_path=model_path,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            loss_type=loss_type,
            save_results=save_results,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb.finish()

    print("\n=== Auto Eval Summary ===")
    print(f"Model: {model_path}")
    print(f"Dice: {summary['dice']:.4f}")
    print(f"IoU: {summary['iou']:.4f}")
    print(f"F1: {summary['f1']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"Specificity: {summary['specificity']:.4f}")
    if "loss" in summary:
        print(f"Validation Loss: {summary['loss']:.4f}")


if __name__ == "__main__":
    main()