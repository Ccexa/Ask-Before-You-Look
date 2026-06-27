import numpy as np
from PIL import Image
import os
import glob
import argparse
import csv
from tqdm import tqdm
from calculate_seg_metric import MicroMetric

CONIC_CLASSES = {
    "neutrophil": 1,
    "epithelial": 2,
    "lymphocyte": 3,
    "plasma": 4,
    "eosinophil": 5,
    "connective": 6,
}

# Maps class name to the folder suffix used by our method
CLASS_FOLDER = {name: f"{name}_masks" for name in CONIC_CLASSES}
# CLASS_FOLDER = {name: f"conic_{name}" for name in CONIC_CLASSES} 

def dice_iou(gt: np.ndarray, pred: np.ndarray, eps=1e-6):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    union = np.logical_or(gt, pred).sum()
    if denom == 0:
        return 1.0, 1.0
    return float(2 * intersection / (denom + eps)), float(intersection / (union + eps))


def load_mask(path: str) -> np.ndarray:
    return (np.array(Image.open(path)) > 0).astype(np.uint8)


def get_image_names(input_dir: str, mode: str) -> list:
    if mode == "ours":
        # Use files from the first class folder
        first_cls = next(iter(CLASS_FOLDER.values()))
        folder = os.path.join(input_dir, first_cls)
        return sorted([os.path.basename(f) for f in glob.glob(os.path.join(folder, "*.png"))])
    else:
        return sorted([os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "*.png"))])


def load_pred_binary(input_dir: str, img_name: str, cls_name: str, mode: str) -> np.ndarray:
    if mode == "ours":
        path = os.path.join(input_dir, CLASS_FOLDER[cls_name], img_name)
        return load_mask(path)
    else:
        path = os.path.join(input_dir, img_name)
        arr = np.array(Image.open(path))
        return (arr == CONIC_CLASSES[cls_name]).astype(np.uint8)


def load_gt_binary(gt_dir: str, img_name: str, cls_name: str) -> np.ndarray:
    # GT is always multi-class
    path = os.path.join(gt_dir, img_name)
    arr = np.array(Image.open(path))
    return (arr == CONIC_CLASSES[cls_name]).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Ground truth multi-class mask directory")
    parser.add_argument("--input_dir", required=True, help="Prediction directory")
    parser.add_argument("--mode", choices=["ours", "multiclass"], default="ours",
                        help="'ours': one folder per class; 'multiclass': single multi-class mask file")
    parser.add_argument("--csv_file", default="conic_metrics.csv")
    args = parser.parse_args()

    img_names = get_image_names(args.input_dir, args.mode)

    # per-class accumulators: list of (dice, iou) per image
    class_scores = {cls: {"dice": [], "iou": []} for cls in CONIC_CLASSES}
    class_micro = {cls: MicroMetric() for cls in CONIC_CLASSES}

    for img_name in tqdm(img_names):
        for cls_name in CONIC_CLASSES:
            gt = load_gt_binary(args.gt_dir, img_name, cls_name)
            pred = load_pred_binary(args.input_dir, img_name, cls_name, args.mode)
            d, i = dice_iou(gt, pred)
            class_scores[cls_name]["dice"].append(d)
            class_scores[cls_name]["iou"].append(i)
            class_micro[cls_name](gt, pred)

    # Build result row
    row = {"input_dir": args.input_dir}
    all_dice, all_iou = [], []
    for cls_name in CONIC_CLASSES:
        d = float(np.mean(class_scores[cls_name]["dice"]))
        i = float(np.mean(class_scores[cls_name]["iou"]))
        row[f"{cls_name}_dice"] = f"{d:.4f}"
        row[f"{cls_name}_iou"] = f"{i:.4f}"
        all_dice.append(d)
        all_iou.append(i)
        m = class_micro[cls_name].aggregate()
        row[f"{cls_name}_micro_dice"] = f"{m['micro_dice']:.4f}"
        row[f"{cls_name}_micro_iou"] = f"{m['micro_iou']:.4f}"

    row["mean_dice"] = f"{np.mean(all_dice):.4f}"
    row["mean_iou"] = f"{np.mean(all_iou):.4f}"

    file_exists = os.path.exists(args.csv_file)
    with open(args.csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results appended to {args.csv_file}")
    for k, v in row.items():
        print(f"  {k}: {v}")
