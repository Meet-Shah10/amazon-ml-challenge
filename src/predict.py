"""
Stage 4 — Inference and output generation

Loads trained models, runs inference on test candidate pairs, applies
calibration + threshold, and writes both output TSVs.
"""

import os
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config
from .train_model import load_models

logger = logging.getLogger(__name__)


def predict_and_write(test_features: pd.DataFrame,
                      models: list = None,
                      calibrator=None,
                      threshold: float = None,
                      s1_entity_ids: set = None):
    """
    Run inference on test features and write both output files.

    Parameters
    ----------
    test_features : DataFrame
        Feature matrix for test candidate pairs (from features.compute_features).
    models : list
        Trained LightGBM models (Booster or LGBMClassifier). Loaded from disk
        if None.
    calibrator : IsotonicRegression
        Probability calibrator. Loaded from disk if None.
    threshold : float
        Decision threshold. Loaded from disk if None.
    s1_entity_ids : set
        All test S1 entity IDs (to ensure every one appears in output).

    Returns
    -------
    dict
        {source1_entity_id: set of predicted matched entity ids}
    """
    logger.info("=== PREDICTION ===")

    if models is None or calibrator is None or threshold is None:
        models, calibrator, threshold = load_models()

    features = config.FEATURE_COLS

    # Filter to active pairs
    active = test_features[test_features["candidate_entity_id"] != ""].copy()

    if active.empty:
        logger.warning("  no active test pairs — writing empty outputs")
        predictions = {}
    else:
        # Ensure feature columns exist
        for col in features:
            if col not in active.columns:
                active[col] = 0.0
        active[features] = active[features].fillna(0.0).astype(np.float32)

        # Ensemble average probabilities across folds
        X = active[features].values
        probs = np.zeros(len(active), dtype=np.float64)

        for model in models:
            if isinstance(model, lgb.Booster):
                probs += model.predict(X)
            else:
                probs += model.predict_proba(X)[:, 1]
        probs /= len(models)

        # Calibrate
        calibrated = calibrator.predict(probs)
        active["prob"] = calibrated

        # Apply threshold
        active["predicted"] = (active["prob"] >= threshold).astype(int)

        n_predicted = active["predicted"].sum()
        logger.info("  threshold=%.3f → %d predicted matches out of %d candidates",
                    threshold, n_predicted, len(active))

        # Build predictions dict
        predictions = {}
        for s1_id, group in active.groupby("source1_entity_id"):
            matched = set(
                group.loc[group["predicted"] == 1, "candidate_entity_id"]
            )
            predictions[s1_id] = matched

    # Ensure ALL S1 entities appear (even singletons)
    if s1_entity_ids:
        for s1_id in s1_entity_ids:
            if s1_id not in predictions:
                predictions[s1_id] = set()

    # Write outputs
    _write_matching_results(predictions)
    _write_candidate_pairs(test_features, s1_entity_ids)

    logger.info("=== PREDICTION DONE ===")
    return predictions


def _write_matching_results(predictions: dict):
    """Write matching_results.tsv in submission format."""
    path = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")

    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = predictions[s1_id]
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)

    n_non_empty = sum(1 for r in rows if r["matched_entity_ids"])
    logger.info("  wrote %s: %d rows (%d with matches, %d singletons)",
                path, len(df), n_non_empty, len(df) - n_non_empty)


def _write_candidate_pairs(test_features: pd.DataFrame,
                           s1_entity_ids: set = None):
    """Write candidate_pairs.tsv in submission format."""
    path = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")

    active = test_features[test_features["candidate_entity_id"] != ""]
    grouped = (
        active
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]

    # Add missing S1 entities
    if s1_entity_ids:
        present = set(grouped["source1_entity_id"].values)
        missing = s1_entity_ids - present
        if missing:
            empty_rows = pd.DataFrame({
                "source1_entity_id": sorted(missing),
                "candidate_entity_ids": "",
            })
            grouped = pd.concat([grouped, empty_rows], ignore_index=True)

    grouped = grouped.sort_values("source1_entity_id")
    grouped.to_csv(path, sep="\t", index=False)
    logger.info("  wrote %s: %d rows", path, len(grouped))
