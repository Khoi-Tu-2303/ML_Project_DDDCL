"""Evaluation metrics for classification tasks in DDDCL.

Follows protocols from docs/04_PROTOCOLS.md and docs/07_VERIFICATION.md:
- Accuracy, Macro-F1, Balanced Accuracy, Per-class recall, Confusion Matrix.
- Explicit handling of zero divisions and edge cases.
- Support for saving prediction CSVs with row_id, y_true, y_pred, probabilities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray | Sequence[int]
        Ground truth integer labels.
    y_pred : np.ndarray | Sequence[int]
        Predicted integer labels.
    num_classes : int | None, default=None
        Total number of classes. If None, inferred from unique labels.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'accuracy': float
        - 'macro_f1': float
        - 'balanced_accuracy': float
        - 'per_class_recall': dict[int, float]
        - 'confusion_matrix': list[list[int]]
        - 'sample_count': int
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}"
        )

    if len(y_true_arr) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "per_class_recall": {},
            "confusion_matrix": [],
            "sample_count": 0,
        }

    # Determine unique labels
    if num_classes is not None:
        labels = list(range(num_classes))
    else:
        labels = sorted(list(set(y_true_arr) | set(y_pred_arr)))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0.0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    # Per-class recall
    recalls = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0.0)
    per_class_recall = {int(c): float(recalls[idx]) for idx, c in enumerate(labels)}

    # Confusion matrix: rows = true, cols = pred
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_list = [[int(val) for val in row] for row in cm]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm_list,
        "sample_count": int(len(y_true_arr)),
    }


def save_predictions_csv(
    filepath: str | Path,
    row_ids: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
) -> None:
    """Save prediction details to CSV format.
    
    Columns: row_id, y_true, y_pred, prob_0, prob_1, ...
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    prob_arr = np.asarray(y_prob, dtype=np.float32)
    num_classes = prob_arr.shape[1] if prob_arr.ndim == 2 else 2

    header = ["row_id", "y_true", "y_pred"] + [f"prob_class_{c}" for c in range(num_classes)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row_id in enumerate(row_ids):
            probs = prob_arr[i].tolist() if prob_arr.ndim == 2 else [1.0 - float(prob_arr[i]), float(prob_arr[i])]
            writer.writerow([row_id, int(y_true[i]), int(y_pred[i])] + [f"{p:.6f}" for p in probs])


def save_metrics_json(filepath: str | Path, metrics: dict[str, Any]) -> None:
    """Save metrics dictionary to formatted JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
