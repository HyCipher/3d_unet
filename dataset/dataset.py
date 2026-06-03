
import os
import glob
import logging
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion
from augmentations.data_augmentation import apply_augmentation


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
        pos_sample_ratio=0.4,
        edge_sample_ratio=0.1,
        hard_negative_dir=None,
        hard_negative_sample_ratio=0.0,
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

            # (H,W,Z) -> (Z,H,W)
            vol = np.transpose(vol, (2, 0, 1))
            lab = np.transpose(lab, (2, 0, 1))

            assert vol.shape == lab.shape

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
        self.hardest_negative = bool(hardest_negative)
        self.hardest_negative_erosion_iters = int(hardest_negative_erosion_iters)

        if (
            self.pos_sample_ratio < 0.0
            or self.edge_sample_ratio < 0.0
            or self.hard_negative_sample_ratio < 0.0
        ):
            raise ValueError("Sampling ratios must be non-negative.")
        if self.pos_sample_ratio + self.edge_sample_ratio + self.hard_negative_sample_ratio > 1.0:
            raise ValueError(
                "pos_sample_ratio + edge_sample_ratio + hard_negative_sample_ratio must be <= 1.0."
            )

        # Cache positive coordinates to avoid scanning the full volume every __getitem__ call.
        self.pos_coords = []
        self.edge_coords = []
        self.hard_negative_coords = []
        self.hardest_negative_coords = []

        if self.hard_negative_dir is not None:
            hard_negative_paths = {
                os.path.basename(path): path
                for path in glob.glob(os.path.join(self.hard_negative_dir, "*.tif"))
            }
            if len(hard_negative_paths) != len(self.img_paths):
                raise ValueError(
                    f"Hard negative masks must match the training set size: "
                    f"{len(hard_negative_paths)} masks for {len(self.img_paths)} volumes."
                )
        elif self.hard_negative_sample_ratio > 0.0:
            raise ValueError("hard_negative_sample_ratio > 0 requires hard_negative_dir.")

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

        if self.hard_negative_dir is not None:
            for img_path in self.img_paths:
                mask_path = os.path.join(self.hard_negative_dir, os.path.basename(img_path))
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Missing hard negative mask: {mask_path}")

                hard_negative_mask = tiff.imread(mask_path).astype(np.float32)
                hard_negative_mask = np.transpose(hard_negative_mask, (2, 0, 1)) > 0
                self.hard_negative_coords.append(np.argwhere(hard_negative_mask))

                if self.hardest_negative:
                    hardest_mask = binary_erosion(
                        hard_negative_mask,
                        iterations=max(self.hardest_negative_erosion_iters, 1),
                        border_value=0,
                    )
                    if hardest_mask.any():
                        self.hardest_negative_coords.append(np.argwhere(hardest_mask))
                    else:
                        self.hardest_negative_coords.append(np.argwhere(hard_negative_mask))

        active_hard_negative_coords = (
            self.hardest_negative_coords
            if self.hardest_negative and self.hardest_negative_coords
            else self.hard_negative_coords
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "volume_patch_counts: %s, pos coords cached: %s, edge coords cached: %s, hard negatives cached: %s",
                self.volume_patch_counts,
                [len(c) for c in self.pos_coords],
                [len(c) for c in self.edge_coords],
                [len(c) for c in active_hard_negative_coords],
            )

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
        elif r < (self.pos_sample_ratio + self.edge_sample_ratio + self.hard_negative_sample_ratio):
            if self.hardest_negative and self.hardest_negative_coords:
                coords = self.hardest_negative_coords[vid]
            else:
                coords = self.hard_negative_coords[vid]
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
