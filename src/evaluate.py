"""
Evaluation utilities — F₀.₅ scoring and blocking recall ceiling.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def per_entity_f_half(pred_set: set, true_set: set) -> float:
    """F₀.₅ for a single S1 entity."""
    if not pred_set and not true_set:
        return 1.0  # correct singleton
    if not pred_set or not true_set:
        # predicted empty on a non-singleton, or predicted something on a true singleton
        return 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f_half(predictions: dict, ground_truth: dict) -> float:
    """
    Macro-averaged F₀.₅ over all S1 entities.

    Parameters
    ----------
    predictions : dict
        {source1_entity_id: set of predicted matched entity ids}
    ground_truth : dict
        {source1_entity_id: set of true matched entity ids}

    Returns
    -------
    float
        Macro F₀.₅ score.
    """
    all_s1_ids = set(predictions.keys()) | set(ground_truth.keys())
    if not all_s1_ids:
        return 0.0

    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(per_entity_f_half(pred, true))

    return float(np.mean(scores))


def blocking_recall_ceiling(candidates: dict, ground_truth: dict) -> float:
    """
    Fraction of true matches present in the candidate set (recall ceiling).

    If blocking misses a true match, no downstream model can recover it.

    Parameters
    ----------
    candidates : dict
        {source1_entity_id: set of candidate entity ids}
    ground_truth : dict
        {source1_entity_id: set of true matched entity ids}

    Returns
    -------
    float
        Recall ceiling (0-1).
    """
    total_true = 0
    found = 0
    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue  # singletons don't affect blocking recall
        cand_set = candidates.get(s1_id, set())
        total_true += len(true_set)
        found += len(true_set & cand_set)

    if total_true == 0:
        return 1.0
    recall = found / total_true
    logger.info("  blocking recall ceiling: %.4f (%d / %d true matches found)",
                recall, found, total_true)
    return recall


def reduction_ratio(n_candidates: int, n_s1: int, n_s2s3: int) -> float:
    """Fraction of all-pairs comparisons eliminated by blocking."""
    all_pairs = n_s1 * n_s2s3
    if all_pairs == 0:
        return 1.0
    return 1.0 - (n_candidates / all_pairs)
