import numpy as np
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm
import csv


class MicroMetric:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def __call__(self, gt: np.ndarray, pred: np.ndarray):
        assert gt.shape == pred.shape, "Shape mismatch"
        gt = gt.astype(bool)
        pred = pred.astype(bool)

        self.TP += np.logical_and(gt, pred).sum()
        self.FP += np.logical_and(pred, np.logical_not(gt)).sum()
        self.FN += np.logical_and(np.logical_not(pred), gt).sum()
        self.TN += np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    def aggregate(self, eps=1e-6):
        tp, fp, fn, tn = self.TP, self.FP, self.FN, self.TN

        return {
            "micro_dice": (2.0 * tp) / (2 * tp + fp + fn + eps),
            "micro_iou": tp / (tp + fp + fn + eps),
            "micro_accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
            "micro_precision": tp / (tp + fp + eps),
            "micro_recall": tp / (tp + fn + eps),
            "micro_fpr": fp / (fp + tn + eps),
            "micro_fnr": fn / (tp + fn + eps),
        }


class MacroMetric:
    def __init__(self):
        self.dice = []
        self.iou = []

    def __call__(self, gt: np.ndarray, pred: np.ndarray, eps=1e-6):
        assert gt.shape == pred.shape, "Shape mismatch"
        gt = gt.astype(bool)
        pred = pred.astype(bool)

        intersection = np.logical_and(gt, pred).sum()
        denominator = gt.sum() + pred.sum()
        union = np.logical_or(gt, pred).sum()

        if denominator == 0:
            self.dice.append(1.0)
            self.iou.append(1.0)
        else:
            self.dice.append(2 * intersection / (denominator + eps))
            self.iou.append(intersection / (union + eps))

    def aggregate(self):
        return {
            "macro_dice": float(np.mean(self.dice)) if self.dice else np.nan,
            "macro_iou": float(np.mean(self.iou)) if self.iou else np.nan,
        }


class PositiveMetric:
    """
    Only evaluate GT-positive patches.
    """
    def __init__(self):
        self.dice = []
        self.iou = []
        self.num_positive = 0

    def __call__(self, gt: np.ndarray, pred: np.ndarray, eps=1e-6):
        assert gt.shape == pred.shape, "Shape mismatch"
        gt = gt.astype(bool)
        pred = pred.astype(bool)

        if gt.sum() == 0:
            return

        self.num_positive += 1

        intersection = np.logical_and(gt, pred).sum()
        denominator = gt.sum() + pred.sum()
        union = np.logical_or(gt, pred).sum()

        self.dice.append(2 * intersection / (denominator + eps))
        self.iou.append(intersection / (union + eps))

    def aggregate(self):
        return {
            "num_positive": self.num_positive,
            "positive_dice": float(np.mean(self.dice)) if self.dice else np.nan,
            "positive_iou": float(np.mean(self.iou)) if self.iou else np.nan,
        }


class NegativeSafetyMetric:
    """
    Only evaluate GT-negative patches.

    Negative FP Mask Rate:
        # negative patches with predicted area ratio > threshold / # negative patches

    Mean FP Area:
        average predicted mask area ratio over all GT-negative patches
    """
    def __init__(self, fp_area_threshold: float = 0.005):
        self.fp_area_threshold = fp_area_threshold
        self.num_negative = 0
        self.num_fp_mask = 0
        self.pred_area_ratios = []

    def __call__(self, gt: np.ndarray, pred: np.ndarray):
        assert gt.shape == pred.shape, "Shape mismatch"
        gt = gt.astype(bool)
        pred = pred.astype(bool)

        if gt.sum() > 0:
            return

        self.num_negative += 1

        area_ratio = pred.sum() / float(pred.size)
        self.pred_area_ratios.append(area_ratio)

        if area_ratio > self.fp_area_threshold:
            self.num_fp_mask += 1

    def aggregate(self):
        return {
            "num_negative": self.num_negative,
            "negative_fp_mask_rate": (
                self.num_fp_mask / self.num_negative if self.num_negative > 0 else np.nan
            ),
            "mean_fp_area": (
                float(np.mean(self.pred_area_ratios)) if self.pred_area_ratios else np.nan
            ),
            "fp_area_threshold": self.fp_area_threshold,
        }


def load_binary_mask(path: str) -> np.ndarray:
    mask = np.array(Image.open(path))
    return (mask > 0).astype(np.uint8)


def append_to_csv(csv_file: str, row: dict):
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate binary tissue/tumor segmentation with positive-patch and negative-patch metrics."
    )
    parser.add_argument("--labels", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--preds", type=str, required=True, help="Path to predicted masks")
    parser.add_argument("--log_file", type=str, default="comparison.log", help="Path to log file")
    parser.add_argument("--csv_file", type=str, default="comparison.csv", help="Path to output CSV file")
    parser.add_argument(
        "--fp_area_threshold",
        type=float,
        default=0.000,
        help="Area ratio threshold for counting a prediction on GT-negative patches as false positive. Default: 0.005 = 0.5%%",
    )
    parser.add_argument(
        "--missing_pred_policy",
        type=str,
        default="skip",
        choices=["skip", "empty", "error"],
        help="How to handle missing prediction files: skip, empty mask, or error.",
    )

    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.labels, "*.png")))

    micro_metric = MicroMetric()
    macro_metric = MacroMetric()
    positive_metric = PositiveMetric()
    negative_metric = NegativeSafetyMetric(fp_area_threshold=args.fp_area_threshold)

    num_total = 0
    num_missing_pred = 0

    for gt_file in tqdm(gt_files):
        pred_file = os.path.join(args.preds, os.path.basename(gt_file))

        gt = load_binary_mask(gt_file)

        if not os.path.exists(pred_file):
            num_missing_pred += 1

            if args.missing_pred_policy == "error":
                raise FileNotFoundError(f"Missing prediction file: {pred_file}")
            elif args.missing_pred_policy == "skip":
                continue
            elif args.missing_pred_policy == "empty":
                pred = np.zeros_like(gt, dtype=np.uint8)
        else:
            pred = load_binary_mask(pred_file)

        assert gt.shape == pred.shape, f"Shape mismatch: {gt_file}, {pred_file}"

        num_total += 1

        micro_metric(gt, pred)
        macro_metric(gt, pred)
        positive_metric(gt, pred)
        negative_metric(gt, pred)

    results = {
        "inputs": args.preds,
        "labels": args.labels,
        "num_total_evaluated": num_total,
        "num_missing_pred": num_missing_pred,
    }

    results.update(micro_metric.aggregate())
    results.update(macro_metric.aggregate())
    results.update(positive_metric.aggregate())
    results.update(negative_metric.aggregate())

    with open(args.log_file, "a") as f:
        f.write(f"Labels: {args.labels}\n")
        f.write(f"Inputs: {args.preds}\n")
        f.write(f"Num evaluated: {num_total}\n")
        f.write(f"Num missing pred: {num_missing_pred}\n")

        for key, value in results.items():
            if key in ["inputs", "labels"]:
                continue

            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n")

    append_to_csv(args.csv_file, results)

    print("Done.")
    print(f"Results written to log: {args.log_file}")
    print(f"Results appended to CSV: {args.csv_file}")
    print(results)