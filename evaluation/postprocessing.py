import numpy as np
from scipy import ndimage


def remove_small_connected_components(mask, min_size=0, connectivity=1):
    """Remove connected components smaller than min_size from a 3D mask."""
    if min_size is None or min_size <= 0:
        return mask

    mask_bool = mask.astype(bool, copy=False)
    if not mask_bool.any():
        return mask.astype(np.uint8, copy=False)

    structure = ndimage.generate_binary_structure(mask_bool.ndim, connectivity)
    labeled_mask, num_features = ndimage.label(mask_bool, structure=structure)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    component_sizes = np.bincount(labeled_mask.ravel())
    small_labels = np.where((component_sizes < min_size) & (np.arange(component_sizes.size) != 0))[0]

    if small_labels.size == 0:
        return mask.astype(np.uint8, copy=False)

    cleaned = mask_bool.copy()
    cleaned[np.isin(labeled_mask, small_labels)] = False
    return cleaned.astype(np.uint8)