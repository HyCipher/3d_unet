import numpy as np


def dice_coefficient(pred, gt, smooth=1e-6):
    """计算Dice系数"""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def iou_score(pred, gt, smooth=1e-6):
    """计算IoU"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1_specificity(pred_seg, gt_seg, smooth=1e-6):
    """根据二值分割结果计算 Precision/Recall/F1/Specificity。"""
    tp = np.sum(pred_seg * gt_seg)
    fp = np.sum(pred_seg * (1 - gt_seg))
    fn = np.sum((1 - pred_seg) * gt_seg)
    tn = np.sum((1 - pred_seg) * (1 - gt_seg))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    specificity = tn / (tn + fp + smooth)

    return precision, recall, f1, specificity
