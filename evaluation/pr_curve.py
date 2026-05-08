import numpy as np


def sample_for_curves(gt_seg, prob_map, max_points=300000):
    y_true = gt_seg.reshape(-1).astype(np.uint8)
    y_score = prob_map.reshape(-1).astype(np.float32)

    if y_true.size > max_points:
        step = int(np.ceil(y_true.size / max_points))
        y_true = y_true[::step]
        y_score = y_score[::step]

    return y_true, y_score