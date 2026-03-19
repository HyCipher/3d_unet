import tifffile as tiff
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from detect import UNet   # 你的 3D UNet

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import tifffile as tiff
import numpy as np

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
# =========================
# Sliding window inference
# =========================
def sliding_window_inference(
    volume,
    model,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64),
    device="cuda"
):
    """
    volume: numpy array (Z, H, W)
    return: numpy array (Z, H, W) probability map
    """
    model.eval()

    Z, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = np.zeros((Z, H, W), dtype=np.float32)
    count_map = np.zeros((Z, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, Z - pd + 1, sd):
            for y in range(0, H - ph + 1, sh):
                for x in range(0, W - pw + 1, sw):

                    patch = volume[z:z+pd, y:y+ph, x:x+pw]

                    # normalize（和训练一致）
                    # patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                    patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
                    patch = patch.to(device)

                    pred = model(patch)
                    pred = torch.sigmoid(pred)   # [1,1,D,H,W]
                    pred = pred.cpu().numpy()[0, 0]

                    output[z:z+pd, y:y+ph, x:x+pw] += pred
                    count_map[z:z+pd, y:y+ph, x:x+pw] += 1

    output /= np.maximum(count_map, 1e-8)
    return output


# =========================
# Main inference function
# =========================
def infer_one_volume(
    img_path,
    model_path,
    save_path,
    patch_size=(8, 512, 512),
    stride=(2, 64, 64)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # load tif (H,W,Z)
    vol = tiff.imread(img_path).astype(np.float32)

    # (H,W,Z) → (Z,H,W)
    vol = np.transpose(vol, (2, 0, 1))

    print("Volume shape:", vol.shape)

    prob_map = sliding_window_inference(
        vol,
        model,
        patch_size=patch_size,
        stride=stride,
        device=device
    )

    # 二值化（可调阈值）
    # seg = (prob_map > 0.1).astype(np.uint8) * 255

    # 保存为 tif（转回 H,W,Z）
    # seg = np.transpose(seg, (1, 2, 0))
    # tiff.imwrite(save_path, seg)
    prob_map_save = np.transpose(prob_map, (1, 2, 0))
    tiff.imwrite("prob_map.tif", prob_map_save.astype(np.float32))

    print("Saved segmentation to:", save_path)
    return prob_map


# =========================
# plotobject by object evaluation
# =========================
def object_pr_curve(prob_map, gt, dist_thr=5):
        thrs = np.linspace(0, 1, 50)
        P, R = [], []

        for thr in thrs:
            p, r, _ = object_level_eval(prob_map, gt, thr, dist_thr)
            P.append(p)
            R.append(r)

        plt.figure()
        plt.plot(R, P)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Object-level PR curve")
        plt.grid()
        plt.savefig("object_pr.png")
        plt.close()

# =========================
# object by object evaluation
# =========================
def object_level_eval(prob_map, gt, thr=0.05, dist_thr=5):
    seg = (prob_map > thr).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # label connected components in Ground True and prediction
    gt_cc, gt_num = label(gt)
    pred_cc, pred_num = label(seg)
    
    # compute centers of mass
    gt_centers = np.array(center_of_mass(gt, gt_cc, range(1, gt_num+1)))
    pred_centers = np.array(center_of_mass(seg, pred_cc, range(1, pred_num+1)))

    if len(gt_centers)==0 or len(pred_centers)==0:
        return 0,0,0

    dist = cdist(gt_centers, pred_centers)
    row_ind, col_ind = linear_sum_assignment(dist)
    matched = dist[row_ind, col_ind] < dist_thr

    TP = np.sum(matched)
    FP = pred_num - TP
    FN = gt_num - TP

    # compute precision, recall, F1
    precision = TP/(TP+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)

    return precision, recall, f1


# =========================
# pixel-level evaluation
# =========================
def pixel_level_evaluation(prob_map, gt_path):
    """
    prob_map: (Z,H,W), float in [0,1]
    gt_path: path to GT tif (H,W,Z)
    """

    # load GT
    gt = tiff.imread(gt_path)

    # (H,W,Z) → (Z,H,W)
    gt = np.transpose(gt, (2, 0, 1))

    # 二值化 GT（非常重要）
    gt = (gt > 0).astype(np.uint8)

    print("GT mean:", gt.mean(), "GT max:", gt.max())

    # flatten
    y_true = gt.flatten()
    y_score = prob_map.flatten()

    # PR curve
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score
    )

    ap = average_precision_score(y_true, y_score)

    print(f"Average Precision (AP): {ap:.4f}")

    # plot
    plt.figure(figsize=(5,5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Pixel-level PR (AP={ap:.3f})")
    plt.grid(True)
    plt.savefig("pixel_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    return precision, recall, thresholds


# =========================
# Run
# =========================
if __name__ == "__main__":
    prob_map = infer_one_volume(
        img_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/data/validation/images/dauer_img_160_180.tif",
        model_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/models/unet_3d_epoch_50.pth",
        save_path="result_seg.tif",
        patch_size=(16, 512, 512),
        stride=(4, 128, 128)
    )
    
    # pixel-level evaluation
    precision, recall, thresholds = pixel_level_evaluation(
    prob_map,
    gt_path="/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/data/validation/labels/dauer_syn_160_180.tif"
)
    
    # load GT for object evaluation
    gt = tiff.imread("/BiO/Live/rooter/Downloads/C_elegans_UNet/3d_unet/data/validation/labels/dauer_syn_160_180.tif")
    gt = np.transpose(gt, (2, 0, 1))  # (Z,H,W)

    # object-level evaluation
    obj_p, obj_r, obj_f1 = object_level_eval(prob_map, gt, thr=0.05, dist_thr=5)
    print("Object-level Precision:", obj_p)
    print("Object-level Recall:", obj_r)
    print("Object-level F1:", obj_f1)
    object_pr_curve(prob_map, gt, dist_thr=5)