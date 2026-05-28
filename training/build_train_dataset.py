from dataset import Tif3DPatchDataset


def build_train_dataset(controls, augment):
    return Tif3DPatchDataset(
        img_dir=controls["train_img_dir"],
        label_dir=controls["train_label_dir"],
        patch_size=controls["patch_size"],
        patches_per_volume=controls["patches_per_volume"],
        augment=augment,
        pos_sample_ratio=float(controls.get("pos_sample_ratio", 0.4)),
        edge_sample_ratio=float(controls.get("edge_sample_ratio", 0.1)),
        hard_negative_dir=controls.get("hard_negative_dir") if controls.get("hard_negative_enable", False) else None,
        hard_negative_sample_ratio=float(controls.get("hard_negative_sample_ratio", 0.0))
        if controls.get("hard_negative_enable", False)
        else 0.0,
        hardest_negative=bool(controls.get("hardest_negative", False)),
        hardest_negative_erosion_iters=int(controls.get("hardest_negative_erosion_iters", 1)),
    )