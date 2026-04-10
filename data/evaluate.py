import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

LENGTH_KEY = "catheter_length_mm"

# Expected layout:
#   gt_dir/
#     subj001/
#       subj001_artifact.nii.gz
#       subj001_catheter.nii.gz
#       subj001.json
#     subj002/
#       ...
#   pred_dir/
#     subj001_taskA.nii.gz
#     subj001_taskB.nii.gz
#     ...
#     taskC.csv
#
#   taskC.csv:
#     subject,length
#     subj001,123.4


def load_nifti(path):
    return nib.load(str(path)).get_fdata()


def dice_score(pred, gt):
    pred, gt = pred > 0, gt > 0
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def relative_absolute_error(pred_length, gt_length):
    if gt_length == 0:
        return 0.0 if pred_length == 0 else np.inf
    return abs(pred_length - gt_length) / gt_length


def read_length_csv(csv_path):
    with open(csv_path, newline="") as f:
        return {row["subject"]: float(row["length"]) for row in csv.DictReader(f)}


def read_gt_length(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return float(data[LENGTH_KEY])


SUBJECTS = [f"subj{i:03d}" for i in range(1, 31)]


def evaluate(gt_dir, pred_dir, output_csv=None):
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    pred_lengths = read_length_csv(pred_dir / "taskC.csv")
    subjects = SUBJECTS

    results = []
    skipped = []

    for subj in subjects:
        gt = gt_dir / subj

        missing = []
        if not (pred_dir / f"{subj}_taskA.nii.gz").exists():
            missing.append("Task A")
        if not (pred_dir / f"{subj}_taskB.nii.gz").exists():
            missing.append("Task B")
        if subj not in pred_lengths:
            missing.append("Task C")
        if missing:
            skipped.append((subj, missing))
            continue

        dice_A = dice_score(load_nifti(pred_dir / f"{subj}_taskA.nii.gz"),
                            load_nifti(gt / f"{subj}_artifact.nii.gz"))
        dice_B = dice_score(load_nifti(pred_dir / f"{subj}_taskB.nii.gz"),
                            load_nifti(gt / f"{subj}_catheter.nii.gz"))
        err_C = relative_absolute_error(
            pred_lengths[subj],
            read_gt_length(gt / f"{subj}.json"),
        )

        results.append({
            "subject": subj,
            "taskA_dice": dice_A,
            "taskB_dice": dice_B,
            "taskC_rel_abs_error": err_C,
        })

    for subj, missing in skipped:
        print(f"[SKIP] {subj}: missing {', '.join(missing)}")

    if results:
        print("=" * 40)
        print(f"Evaluated: {len(results)} / {len(subjects)} subjects")
        print(f"Task A  mean Dice:        {np.mean([r['taskA_dice'] for r in results]):.6f}")
        print(f"Task B  mean Dice:        {np.mean([r['taskB_dice'] for r in results]):.6f}")
        print(f"Task C  mean RelAbsError: {np.mean([r['taskC_rel_abs_error'] for r in results]):.6f}")
        print("=" * 40)
    else:
        print("No subjects were evaluated.")

    if output_csv and results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Per-subject results saved to {output_csv}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MIA challenge predictions (Tasks A, B, C)."
    )
    parser.add_argument("--gt-dir", required=True,
                        help="Ground-truth directory (contains subj001/, subj002/, ...)")
    parser.add_argument("--pred-dir", required=True,
                        help="Prediction directory (flat: subj001_taskA.nii.gz, ..., taskC.csv)")
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to save per-subject results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.gt_dir, args.pred_dir, args.output_csv)