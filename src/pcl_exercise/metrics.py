from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def tune_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray, step: float = 0.01) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    t = 0.0
    while t <= 1.0 + 1e-9:
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
        t += step
    return best_t, best_f1


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
