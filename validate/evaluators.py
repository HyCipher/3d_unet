from validate.validation_utils import validate_with_full_metrics  # pyright: ignore[reportMissingImports]


def evaluate_with_optional_limit(model, dataset, device, controls, criterion):
    """Evaluate validation dataset with optional volume cap for speed."""
    max_val_volumes = controls["max_val_volumes"]
    dust_remove_min_size = int(controls.get("dust_remove_min_size", 0))

    if max_val_volumes is None:
        return validate_with_full_metrics(
            model,
            dataset,
            device,
            patch_size=controls["val_patch_size"],
            stride=controls["val_stride"],
            threshold=controls["val_threshold"],
            dust_remove_min_size=dust_remove_min_size,
            criterion=criterion,
        )

    original_volumes = dataset.volumes
    original_labels = dataset.labels
    try:
        dataset.volumes = dataset.volumes[:max_val_volumes]
        dataset.labels = dataset.labels[:max_val_volumes]
        return validate_with_full_metrics(
            model,
            dataset,
            device,
            patch_size=controls["val_patch_size"],
            stride=controls["val_stride"],
            threshold=controls["val_threshold"],
            dust_remove_min_size=dust_remove_min_size,
            criterion=criterion,
        )
    finally:
        dataset.volumes = original_volumes
        dataset.labels = original_labels


def maybe_evaluate_train_set(model, train_eval_dataset, device, controls, criterion):
    """Optionally evaluate training set to monitor overfitting."""
    if not controls["eval_train_set"]:
        return None

    dust_remove_min_size = int(controls.get("dust_remove_min_size", 0))

    return validate_with_full_metrics(
        model,
        train_eval_dataset,
        device,
        patch_size=controls["val_patch_size"],
        stride=controls["val_stride"],
        threshold=controls["val_threshold"],
        dust_remove_min_size=dust_remove_min_size,
        criterion=criterion,
    )
