from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


F1_THRESHOLD = 0.5


def heart_failure_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.unique(labels).size < 2:
        raise ValueError("AUROC and AUPRC require both classes in an evaluation split.")
    prediction = (probabilities > F1_THRESHOLD).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
