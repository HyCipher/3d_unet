
import os
import glob
import logging
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from training.axis_utils import normalize_to_zyx
from augmentations.data_augmentation import apply_augmentation
from .hard_sampling import (
    validate_hard_negative_setup,
    build_hard_negative_coordinates,
    build_hard_positive_coordinates,
    choose_hard_negative_coords,
    active_hard_negative_coords,
)


logger = logging.getLogger(__name__)
def compute_patches_per_volume(total_volume_size, patch_size):
    """Heuristic to compute how many patches to sample from each volume per epoch."""
    coverage=2.5
    patch_volume = int(np.prod(patch_size))
    total_volume_size = int(total_volume_size)

    return int(np.ceil((coverage * total_volume_size) / patch_volume))


class Tif3DPatchDataset(Dataset):
    """
    Dataset for 3D TIFF images and corresponding labels.
    Expects:
        - img_dir: Directory containing 3D TIFF image volumes (uint16).
        - label_dir: Directory containing 3D TIFF label volumes (uint8), same shape as images.
    Each __getitem__ returns a randomly cropped 3D patch (image and label) with optional augmentation.
    """
    def __init__(
        self,
        img_dir,
        label_dir,
        patch_size=(4, 128, 128),
        augment=True,
        pos_sample_ratio=0.2,
        edge_sample_ratio=0.1,
        hard_negative_dir=None,
        hard_negative_sample_ratio=0.0,
        hard_positive_sample_ratio=0.0,
        hardest_negative=False,
        hardest_negative_erosion_iters=1,
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
        assert len(self.img_paths) == len(self.label_paths)

        self.patch_size = patch_size
        self.augment = augment
        self.volumes = []
        self.labels = []

        for ip, lp in zip(self.img_paths, self.label_paths):
            vol = tiff.imread(ip).astype(np.float32)
            lab = tiff.imread(lp).astype(np.float32)

            vol, vol_order = normalize_to_zyx(vol, ip, self.patch_size)
            lab, lab_order = normalize_to_zyx(lab, lp, self.patch_size)

            assert vol.shape == lab.shape
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Loaded %s (source %s) and %s (source %s) as (Z,H,W)=%s",
                    os.path.basename(ip),
                    vol_order,
                    os.path.basename(lp),
                    lab_order,
                    vol.shape,
                )

            self.volumes.append(vol)
            self.labels.append(lab)
        
        self.volume_patch_counts = [
            compute_patches_per_volume(lab.size, self.patch_size)
            for lab in self.labels
        ]

        self._cumulative_patch_counts = np.cumsum(self.volume_patch_counts)
        self.total_patches = int(self._cumulative_patch_counts[-1]) if self.volume_patch_counts else 0
        self.pos_sample_ratio = float(pos_sample_ratio)
        self.edge_sample_ratio = float(edge_sample_ratio)
        self.hard_negative_dir = hard_negative_dir
        self.hard_negative_sample_ratio = float(hard_negative_sample_ratio)
        self.hard_positive_sample_ratio = float(hard_positive_sample_ratio)
        self.hardest_negative = bool(hardest_negative)
        self.hardest_negative_erosion_iters = int(hardest_negative_erosion_iters)

        if (
            self.pos_sample_ratio < 0.0
            or self.edge_sample_ratio < 0.0
            or self.hard_negative_sample_ratio < 0.0
            or self.hard_positive_sample_ratio < 0.0
        ):
            raise ValueError("Sampling ratios must be non-negative.")
        if (
            self.pos_sample_ratio
            + self.edge_sample_ratio
            + self.hard_negative_sample_ratio
            + self.hard_positive_sample_ratio
            > 1.0
        ):
            raise ValueError(
                "pos_sample_ratio + edge_sample_ratio + hard_negative_sample_ratio + hard_positive_sample_ratio must be <= 1.0."
            )

        # Cache positive coordinates to avoid scanning the full volume every __getitem__ call.
        self.pos_coords = []
        self.edge_coords = []
        self.hard_negative_coords = []
        self.hardest_negative_coords = []
        self.hard_positive_coords = []

        validate_hard_negative_setup(
            self.img_paths,
            self.hard_negative_dir,
            self.hard_negative_sample_ratio,
            self.hard_positive_sample_ratio,
        )

        for lab in self.labels:
            pos_mask = lab > 0
            self.pos_coords.append(np.argwhere(pos_mask))

            # Edge voxels are positive voxels that touch background in 6-neighborhood.
            p = np.pad(pos_mask, 1, mode="constant", constant_values=False)
            c = p[1:-1, 1:-1, 1:-1]
            interior = (
                c
                & p[:-2, 1:-1, 1:-1]
                & p[2:, 1:-1, 1:-1]
                & p[1:-1, :-2, 1:-1]
                & p[1:-1, 2:, 1:-1]
                & p[1:-1, 1:-1, :-2]
                & p[1:-1, 1:-1, 2:]
            )
            edge_mask = c & (~interior)
            self.edge_coords.append(np.argwhere(edge_mask))

        self.hard_negative_coords, self.hardest_negative_coords = build_hard_negative_coordinates(
            self.img_paths,
            self.labels,
            self.hard_negative_dir,
            self.patch_size,
            normalize_to_zyx,
            hardest_negative=self.hardest_negative,
            hardest_negative_erosion_iters=self.hardest_negative_erosion_iters,
        )
        self.hard_positive_coords = build_hard_positive_coordinates(
            self.img_paths,
            self.labels,
            self.hard_negative_dir,
            self.patch_size,
            normalize_to_zyx,
        )

        active_coords = active_hard_negative_coords(
            self.hard_negative_coords,
            self.hardest_negative_coords,
            hardest_negative=self.hardest_negative,
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "volume_patch_counts: %s, pos coords cached: %s, edge coords cached: %s, hard negatives cached: %s",
                self.volume_patch_counts,
                [len(c) for c in self.pos_coords],
                [len(c) for c in self.edge_coords],
                [len(c) for c in active_coords],
            )
            logger.info("hard positives cached: %s", [len(c) for c in self.hard_positive_coords])

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        vid = int(np.searchsorted(self._cumulative_patch_counts, idx, side="right"))
        vol = self.volumes[vid]
        lab = self.labels[vid]

        d, h, w = vol.shape
        pd, ph, pw = self.patch_size

        # Random crop policy: positive-centered, edge-centered, hard-negative-centered, or fully random.
        r = np.random.rand()
        if r < self.pos_sample_ratio:
            coords = self.pos_coords[vid]
        elif r < (self.pos_sample_ratio + self.edge_sample_ratio):
            coords = self.edge_coords[vid]
        elif r < (
            self.pos_sample_ratio
            + self.edge_sample_ratio
            + self.hard_negative_sample_ratio
        ):
            coords = choose_hard_negative_coords(
                vid,
                self.hard_negative_coords,
                self.hardest_negative_coords,
                hardest_negative=self.hardest_negative,
            )
        elif r < (
            self.pos_sample_ratio
            + self.edge_sample_ratio
            + self.hard_negative_sample_ratio
            + self.hard_positive_sample_ratio
        ):
            coords = self.hard_positive_coords[vid]
        else:
            coords = None

        if coords is not None and len(coords) > 0:
            zc, yc, xc = coords[np.random.randint(len(coords))]
            z = np.clip(zc - pd // 2, 0, d - pd)
            y = np.clip(yc - ph // 2, 0, h - ph)
            x = np.clip(xc - pw // 2, 0, w - pw)
        else:
            z = np.random.randint(0, d - pd + 1)
            y = np.random.randint(0, h - ph + 1)
            x = np.random.randint(0, w - pw + 1)

        x_patch = vol[z : z + pd, y : y + ph, x : x + pw]
        y_patch = lab[z : z + pd, y : y + ph, x : x + pw]

        x_patch = (x_patch - x_patch.mean()) / (x_patch.std() + 1e-8)
        y_patch = (y_patch > 0).astype(np.float32)

        x_patch, y_patch = apply_augmentation(x_patch, y_patch, augment=self.augment)

        x_patch = torch.from_numpy(x_patch).unsqueeze(0)
        y_patch = torch.from_numpy(y_patch).unsqueeze(0)

        return x_patch.float(), y_patch.float()
