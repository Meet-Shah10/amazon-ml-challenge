"""
Stage 3 — Matching model (LightGBM)

Binary pairwise classifier trained on blocking-stage candidates with
hard-negative-aware weighting for chain/franchise disambiguation.

Training uses GroupKFold (by source1_entity_id) so that entity-level
features don't leak across folds.  OOF predictions are used for
probability calibration and threshold tuning (Stage 4).
"""

import os
import gc
import pickle
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

from . import config
from .evaluate import macro_f_half

logger = logging.getLogger(__name__)


def _get_lgb_device_params() -> dict:
    """
    Detect whether LightGBM can use GPU (CUDA or OpenCL) and return appropriate parameters.
    Falls back safely to CPU if GPU is unavailable or unsupported in the current LightGBM build.
    """
    if not getattr(config, "USE_GPU", False):
        return {"device": "cpu"}

    X_dummy = np.random.randn(20, 4).astype(np.float32)
    y_dummy = np.array([0, 1] * 10, dtype=np.int32)

    # 1. Try CUDA device (LightGBM >= 4.0 CUDA backend)
    try:
        clf = lgb.LGBMClassifier(device="cuda", n_estimators=2, verbose=-1, min_child_samples=2)
        clf.fit(X_dummy, y_dummy)
        logger.info("  [GPU] LightGBM CUDA acceleration verified and active (device='cuda')")
        return {"device": "cuda"}
    except Exception as e:
        logger.debug("LightGBM CUDA probe failed: %s", e)

    # 2. Try OpenCL GPU device
    try:
        clf = lgb.LGBMClassifier(device="gpu", n_estimators=2, verbose=-1, min_child_samples=2)
        clf.fit(X_dummy, y_dummy)
        logger.info("  [GPU] LightGBM OpenCL acceleration verified and active (device='gpu')")
        return {"device": "gpu"}
    except Exception as e:
        logger.debug("LightGBM OpenCL probe failed: %s", e)

    logger.info("  [CPU] LightGBM GPU build not detected; training on multi-core CPU (n_jobs=-1)")
    return {"device": "cpu"}


def train_matcher(train_df: pd.DataFrame,
                  ground_truth: dict) -> tuple:
    """
    Train LightGBM with GroupKFold and hard-negative weighting.

    Parameters
    ----------
    train_df : DataFrame
        Feature matrix with columns in config.FEATURE_COLS plus
        source1_entity_id, candidate_entity_id, and blocking scores.
    ground_truth : dict
        {source1_entity_id: set of true matched entity ids}

    Returns
    -------
    tuple of (models: list, calibrator: IsotonicRegression,
              best_threshold: float, best_f05: float, oof_df: DataFrame)
    """
    logger.info("=== MODEL TRAINING ===")

    # Probe and set GPU/CPU device
    device_params = _get_lgb_device_params()

    # Filter to active pairs only (non-empty candidates)
    active = train_df[train_df["candidate_entity_id"] != ""].copy()

    # Vectorized label assignment — avoids apply() on millions of rows
    # Build a flat set of all positive (s1_id, cand_id) pairs from the GT dict
    gt_pairs = set()
    for s1_id, cand_set in ground_truth.items():
        for cand_id in cand_set:
            gt_pairs.add((s1_id, cand_id))

    active["is_match"] = [
        int((s1_id, cand_id) in gt_pairs)
        for s1_id, cand_id in zip(
            active["source1_entity_id"].values,
            active["candidate_entity_id"].values,
        )
    ]

    n_pos = active["is_match"].sum()
    n_neg = len(active) - n_pos
    logger.info("  labeled pairs: %d positive, %d negative (ratio 1:%.1f)",
                n_pos, n_neg, n_neg / max(n_pos, 1))

    # Ensure feature columns exist (fill missing with 0)
    features = config.FEATURE_COLS
    for col in features:
        if col not in active.columns:
            active[col] = 0.0
    active[features] = active[features].fillna(0.0).astype(np.float32)

    # GroupKFold: split by source1_entity_id
    groups = active["source1_entity_id"].values
    gkf    = GroupKFold(n_splits=config.N_FOLDS)

    models    = []
    oof_probs = np.zeros(len(active), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(active, groups=groups)):
        logger.info("  fold %d/%d: train=%d, val=%d",
                    fold + 1, config.N_FOLDS, len(tr_idx), len(val_idx))

        tr  = active.iloc[tr_idx]
        val = active.iloc[val_idx]

        pos_count = int((tr["is_match"] == 1).sum())
        neg_count = int((tr["is_match"] == 0).sum())

        # Per-row weights: upweight hard negatives (same name, different address)
        weights = np.ones(len(tr), dtype=np.float32)
        hard_neg_mask = (tr["is_match"] == 0) & (tr["name_match_addr_mismatch"] == 1)
        weights[hard_neg_mask.values] = config.HARD_NEG_WEIGHT

        model = lgb.LGBMClassifier(
            n_estimators=config.LGBM_N_ESTIMATORS,
            learning_rate=config.LGBM_LEARNING_RATE,
            num_leaves=config.LGBM_NUM_LEAVES,
            min_child_samples=config.LGBM_MIN_CHILD_SAMPLES,
            scale_pos_weight=neg_count / max(pos_count, 1),
            objective="binary",
            verbose=-1,
            n_jobs=-1,
            **device_params,
        )

        try:
            model.fit(
                tr[features], tr["is_match"],
                sample_weight=weights,
            )
        except Exception as e:
            if device_params.get("device") != "cpu":
                logger.warning("    [GPU Fallback] Fold %d fit failed on GPU (%s); retrying on CPU", fold + 1, e)
                model.set_params(device="cpu")
                model.fit(
                    tr[features], tr["is_match"],
                    sample_weight=weights,
                )
            else:
                raise

        fold_probs      = model.predict_proba(val[features])[:, 1]
        oof_probs[val_idx] = fold_probs
        models.append(model)

        # Log fold performance at default 0.5 threshold
        fold_preds = (fold_probs >= 0.5).astype(int)
        fold_acc   = np.mean(fold_preds == val["is_match"].values)
        logger.info("    fold %d accuracy@0.5: %.4f", fold + 1, fold_acc)

    active["oof_prob"] = oof_probs

    # Calibrate probabilities with isotonic regression
    logger.info("  calibrating probabilities (isotonic regression) ...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, active["is_match"].values)
    active["calibrated_prob"] = calibrator.predict(oof_probs)

    # Tune threshold against F₀.₅, using ALL trained S1 IDs as singleton pool
    # (not just GT-positive ones) so the F₀.₅ denominator is correct.
    all_trained_s1_ids = set(active["source1_entity_id"].unique())
    best_t, best_f05 = _tune_threshold(active, ground_truth,
                                       all_s1_ids=all_trained_s1_ids)
    logger.info("  best threshold: %.3f  →  validation F₀.₅: %.4f", best_t, best_f05)

    # Feature importance
    _log_feature_importance(models, features)

    # Save models and calibrator
    _save_models(models, calibrator, best_t)

    logger.info("=== TRAINING DONE ===")
    return models, calibrator, best_t, best_f05, active


def _tune_threshold(oof_df: pd.DataFrame, ground_truth: dict,
                    all_s1_ids: set = None) -> tuple:
    """
    Sweep thresholds against macro F₀.₅ on calibrated OOF predictions.

    Vectorized: builds prediction dicts using groupby + boolean mask
    on the full DataFrame rather than iterating per-threshold in Python.

    Parameters
    ----------
    all_s1_ids : set
        All S1 entity IDs in the training sample (including true singletons
        that have no GT match). Required for a correct F₀.₅ denominator.
    """
    logger.info("  tuning threshold against F₀.₅ ...")

    grid = np.arange(config.THRESHOLD_LOW, config.THRESHOLD_HIGH, config.THRESHOLD_STEP)
    best_t, best_score = 0.5, -1.0

    # Pre-group once; threshold only changes the boolean mask
    # Sort descending by calibrated_prob so we can use a single groupby
    sorted_df = oof_df[["source1_entity_id", "candidate_entity_id",
                         "calibrated_prob"]].sort_values(
        ["source1_entity_id", "calibrated_prob"], ascending=[True, False]
    )

    # Use all_s1_ids if provided (includes true singletons);
    # fall back to GT-positive S1 IDs only (old behaviour) if not.
    if all_s1_ids:
        fill_s1s = all_s1_ids
    else:
        logger.warning("  _tune_threshold: all_s1_ids not provided — "
                       "singleton entities excluded from F₀.₅ denominator")
        fill_s1s = set(ground_truth.keys())

    for t in grid:
        matched_df = sorted_df[sorted_df["calibrated_prob"] >= t]

        # Build preds dict via groupby aggregation (faster than per-group loop)
        if matched_df.empty:
            preds = {s1_id: set() for s1_id in fill_s1s}
        else:
            preds = (
                matched_df
                .groupby("source1_entity_id")["candidate_entity_id"]
                .apply(set)
                .to_dict()
            )
            for s1_id in fill_s1s:
                if s1_id not in preds:
                    preds[s1_id] = set()

        score = macro_f_half(preds, ground_truth)
        if score > best_score:
            best_t, best_score = float(t), score

    return best_t, best_score


def _log_feature_importance(models: list, features: list):
    """Log top-10 features by average gain importance across folds."""
    importances = np.zeros(len(features))
    for m in models:
        importances += m.feature_importances_
    importances /= len(models)

    sorted_idx = np.argsort(importances)[::-1]
    logger.info("  top-10 features by importance:")
    for rank, idx in enumerate(sorted_idx[:10]):
        logger.info("    %2d. %-30s  %.1f", rank + 1, features[idx], importances[idx])


def _save_models(models: list, calibrator: IsotonicRegression,
                 threshold: float):
    """Persist models, calibrator, and threshold to disk."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    for i, model in enumerate(models):
        path = os.path.join(config.MODEL_DIR, f"lgbm_fold{i}.txt")
        model.booster_.save_model(path)

    with open(os.path.join(config.MODEL_DIR, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    with open(os.path.join(config.MODEL_DIR, "threshold.txt"), "w") as f:
        f.write(f"{threshold:.6f}\n")

    logger.info("  models saved to %s", config.MODEL_DIR)


def load_models() -> tuple:
    """Load saved models, calibrator, and threshold."""
    models = []
    for i in range(config.N_FOLDS):
        path = os.path.join(config.MODEL_DIR, f"lgbm_fold{i}.txt")
        if os.path.exists(path):
            booster = lgb.Booster(model_file=path)
            models.append(booster)

    with open(os.path.join(config.MODEL_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)

    with open(os.path.join(config.MODEL_DIR, "threshold.txt")) as f:
        threshold = float(f.read().strip())

    logger.info("  loaded %d models, threshold=%.3f", len(models), threshold)
    return models, calibrator, threshold
