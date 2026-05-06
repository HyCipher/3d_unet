
import os
import glob
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
from augmentations.data_augmentation import apply_augmentation


class Tif3DPatchDataset(Dataset):
    """
    Dataset for 3D TIFF images and corresponding labels.
    Expects:
        - img_dir: Directory containing 3D TIFF image volumes (uint16).
        - label_dir: Directory containing 3D TIFF label volumes (uint8), same shape as images.
    Each __getitem__ returns a randomly cropped 3D patch (image and label) with optional augmentation.
    """
    def __init__(self, img_dir, label_dir, patch_size=(4, 128, 128), patches_per_volume=200, augment=True):
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

        self.num_volumes = len(self.volumes)
        self.patches_per_volume = patches_per_volume

        # Cache positive coordinates to avoid scanning the full volume every __getitem__ call.
        self.pos_coords = []
        for lab in self.labels:
            coords = np.argwhere(lab > 0)
            self.pos_coords.append(coords)
        print(
            f"Dataset: {self.num_volumes} volumes, "
            f"pos coords cached: {[len(c) for c in self.pos_coords]}"
        )

    def __len__(self):
        return self.num_volumes * self.patches_per_volume

    def __getitem__(self, idx):
        vid = idx // self.patches_per_volume
        vol = self.volumes[vid]
        lab = self.labels[vid]

        d, h, w = vol.shape
        pd, ph, pw = self.patch_size

        # Random crop 3D patch
        if np.random.rand() < 0.8:
            pos = self.pos_coords[vid]
            if len(pos) > 0:
                zc, yc, xc = pos[np.random.randint(len(pos))]
                z = np.clip(zc - pd // 2, 0, d - pd)
                y = np.clip(yc - ph // 2, 0, h - ph)
                x = np.clip(xc - pw // 2, 0, w - pw)
            else:
                z = np.random.randint(0, d - pd + 1)
                y = np.random.randint(0, h - ph + 1)
                x = np.random.randint(0, w - pw + 1)
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
