from typing import Dict, List

import numpy as np

from .pseudo_label import CLASS_NAMES


def confusion_matrix_1d(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, pred_value in zip(y_true, y_pred):
        if 0 <= true_value < num_classes and 0 <= pred_value < num_classes:
            confusion[int(true_value), int(pred_value)] += 1
    return confusion


def segmentation_metrics_from_confusion(confusion: np.ndarray) -> Dict[str, float | dict]:
    confusion = np.asarray(confusion, dtype=np.float64)
    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total > 0 else 0.0
    per_class = {}
    ious = []
    f1s = []
    valid_ious = []
    valid_f1s = []
    valid_class_names = []
    for cls in range(confusion.shape[0]):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        support = confusion[cls, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        name = CLASS_NAMES.get(cls, str(cls))
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
            "f1": float(f1),
            "support": float(support),
        }
        ious.append(iou)
        f1s.append(f1)
        if support > 0:
            valid_ious.append(iou)
            valid_f1s.append(f1)
            valid_class_names.append(name)
    return {
        "point_accuracy": accuracy,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "macro_f1": float(np.mean(valid_f1s)) if valid_f1s else 0.0,
        "mean_iou_all_classes": float(np.mean(ious)) if ious else 0.0,
        "macro_f1_all_classes": float(np.mean(f1s)) if f1s else 0.0,
        "valid_classes": valid_class_names,
        "per_class": per_class,
    }


def average_sample_metrics(confusions: List[np.ndarray]) -> Dict[str, float]:
    sample_ious = []
    sample_f1s = []
    for confusion in confusions:
        metrics = segmentation_metrics_from_confusion(confusion)
        sample_ious.append(metrics["mean_iou"])
        sample_f1s.append(metrics["macro_f1"])
    return {
        "sample_mean_iou": float(np.mean(sample_ious)) if sample_ious else 0.0,
        "sample_macro_f1": float(np.mean(sample_f1s)) if sample_f1s else 0.0,
    }
