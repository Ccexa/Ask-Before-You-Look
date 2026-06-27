import json
import csv
import os
import argparse
from typing import Dict, List, Any


def normalize_label(x: Any) -> str:
    """
    Normalize model answer / GT label to Yes or No.

    Supports:
    - "Yes"
    - "No"
    - "Yes. description..."
    - "No. description..."
    - lowercase variants
    """
    if x is None:
        return "Unknown"

    s = str(x).strip().lower()

    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"

    return "Unknown"


def compute_vqa_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    unknown_pred = 0
    unknown_gt = 0

    for item in records:
        pred = normalize_label(item.get("answer", ""))
        gt = normalize_label(item.get("gt", ""))

        if gt == "Unknown":
            unknown_gt += 1
            continue

        if pred == "Unknown":
            unknown_pred += 1
            # Unknown prediction is treated as incorrect.
            # For medical safety, you may alternatively treat Unknown as Yes.
            if gt == "Yes":
                fn += 1
            elif gt == "No":
                fp += 1
            continue

        if gt == "Yes" and pred == "Yes":
            tp += 1
        elif gt == "Yes" and pred == "No":
            fn += 1
        elif gt == "No" and pred == "No":
            tn += 1
        elif gt == "No" and pred == "Yes":
            fp += 1

    total = tp + tn + fp + fn
    gt_yes = tp + fn
    gt_no = tn + fp
    pred_yes = tp + fp
    pred_no = tn + fn

    eps = 1e-8

    accuracy = (tp + tn) / (total + eps)
    t_acc = tp / (gt_yes + eps)
    n_acc = tn / (gt_no + eps)

    precision = tp / (pred_yes + eps)
    recall = t_acc
    specificity = n_acc
    f1 = 2 * precision * recall / (precision + recall + eps)

    retained_ratio = pred_yes / (total + eps)

    return {
        "total": total,
        "gt_yes": gt_yes,
        "gt_no": gt_no,
        "pred_yes": pred_yes,
        "pred_no": pred_no,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "t_acc": t_acc,
        "n_acc": n_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "retained_ratio": retained_ratio,
        "unknown_pred": unknown_pred,
        "unknown_gt": unknown_gt,
    }


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Support {"results": [...]} or similar structures.
        for key in ["results", "data", "records"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"JSON dict in {path} does not contain a supported list key.")

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON format in {path}")


def main():
    parser = argparse.ArgumentParser(description="Compute VQA Yes/No gate metrics from JSON files.")
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more VQA result JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="vqa_metrics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = []

    for json_path in args.json_files:
        records = read_json(json_path)
        metrics = compute_vqa_metrics(records)

        row = {
            "file": json_path,
            "name": os.path.splitext(os.path.basename(json_path))[0],
        }
        row.update(metrics)

        rows.append(row)

    fieldnames = [
        "file",
        "name",
        "total",
        "gt_yes",
        "gt_no",
        "pred_yes",
        "pred_no",
        "TP",
        "TN",
        "FP",
        "FN",
        "accuracy",
        "t_acc",
        "n_acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "retained_ratio",
        "unknown_pred",
        "unknown_gt",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")

                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
                else:
                    formatted[key] = value

            writer.writerow(formatted)

    print(f"Saved VQA metrics to: {args.output_csv}")

    for row in rows:
        print("=" * 80)
        print(row["name"])
        print(f"Total: {row['total']}")
        print(f"GT Yes / No: {row['gt_yes']} / {row['gt_no']}")
        print(f"Pred Yes / No: {row['pred_yes']} / {row['pred_no']}")
        print(f"TP/TN/FP/FN: {row['TP']} / {row['TN']} / {row['FP']} / {row['FN']}")
        print(f"Accuracy: {row['accuracy']:.4f}")
        print(f"T-acc: {row['t_acc']:.4f}")
        print(f"N-acc: {row['n_acc']:.4f}")
        print(f"Precision: {row['precision']:.4f}")
        print(f"F1: {row['f1']:.4f}")
        print(f"Retained ratio: {row['retained_ratio']:.4f}")


if __name__ == "__main__":
    main()