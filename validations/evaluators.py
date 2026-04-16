from validations.validation_utils import validate_with_full_metrics  # pyright: ignore[reportMissingImports]


def evaluate_with_optional_limit(model, dataset, device, controls, criterion):
    """Evaluate validation dataset with optional volume cap for speed."""
    original_volumes = dataset.volumes
    original_labels = dataset.labels
    max_val_volumes = controls["max_val_volumes"]

    if max_val_volumes is not None:
        dataset.volumes = dataset.volumes[:max_val_volumes]
        dataset.labels = dataset.labels[:max_val_volumes]

    metrics = validate_with_full_metrics(
        model,
        dataset,
        device,
        patch_size=controls["val_patch_size"],
        stride=controls["val_stride"],
        threshold=controls["val_threshold"],
        criterion=criterion,
    )

    dataset.volumes = original_volumes
    dataset.labels = original_labels
    return metrics


def maybe_evaluate_train_set(model, train_eval_dataset, device, controls, criterion):
    """Optionally evaluate training set to monitor overfitting."""
    if not controls["eval_train_set"]:
        return None

    return validate_with_full_metrics(
        model,
        train_eval_dataset,
        device,
        patch_size=controls["val_patch_size"],
        stride=controls["val_stride"],
        threshold=controls["val_threshold"],
        criterion=criterion,
    )
