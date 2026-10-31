from typing import Dict, List
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .constants import REPRESENTATION_NAMES


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> List[float]:
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    auc = roc_auc_score(y_true, y_probs)
    aupr = average_precision_score(y_true, y_probs)
    return [auc, aupr]


def compute_fusion_imbalance_from_contributions(contributions: np.ndarray) -> Dict[str, float]:
    contributions = np.asarray(contributions, dtype=np.float64)
    if (
        contributions.ndim != 2
        or contributions.shape[0] == 0
        or contributions.shape[1] != len(REPRESENTATION_NAMES)
    ):
        return {
            "fci_sample": np.nan,
            "fci_global": np.nan,
            "contribution_gap": np.nan,
            **{f"contribution_{name}": np.nan for name in REPRESENTATION_NAMES},
        }
    k = contributions.shape[1]
    normalizer = 1.0 - 1.0 / k
    sample_concentration = np.sum(np.square(contributions), axis=1)
    fci_sample = np.mean((sample_concentration - 1.0 / k) / normalizer)
    mean_contribution = np.mean(contributions, axis=0)
    fci_global = (np.sum(np.square(mean_contribution)) - 1.0 / k) / normalizer
    contribution_gap = float(np.max(mean_contribution) - np.min(mean_contribution))
    return {
        "fci_sample": float(fci_sample),
        "fci_global": float(fci_global),
        "contribution_gap": contribution_gap,
        **{
            f"contribution_{name}": float(mean_contribution[idx])
            for (idx, name) in enumerate(REPRESENTATION_NAMES)
        },
    }


def compute_icg_from_branch_confidences(
    branch_confidences: Dict[str, np.ndarray],
) -> Dict[str, float]:
    if not branch_confidences:
        return {
            "icg": np.nan,
            "icg_ratio": np.nan,
            **{f"confidence_{name}": np.nan for name in REPRESENTATION_NAMES},
        }
    mean_confidences = []
    stats: Dict[str, float] = {}
    for name in REPRESENTATION_NAMES:
        values = np.asarray(branch_confidences.get(name, []), dtype=np.float64)
        confidence = float(np.nanmean(values)) if values.size else np.nan
        stats[f"confidence_{name}"] = confidence
        mean_confidences.append(confidence)
    confidence_array = np.asarray(mean_confidences, dtype=np.float64)
    valid = confidence_array[~np.isnan(confidence_array)]
    if valid.size == 0:
        stats["icg"] = np.nan
        stats["icg_ratio"] = np.nan
    else:
        min_confidence = float(np.min(valid))
        max_confidence = float(np.max(valid))
        stats["icg"] = max_confidence - min_confidence
        stats["icg_ratio"] = max_confidence / max(min_confidence, 1e-12)
    return stats
