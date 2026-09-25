#!/usr/bin/env python3
"""
Business Entity Resolution — End-to-End Pipeline
Amazon ML Challenge 2026

Usage:
    # Full pipeline: train + predict
    python run_pipeline.py --mode full

    # Training only (produces models + validation score)
    python run_pipeline.py --mode train

    # Prediction only (requires trained models in models/)
    python run_pipeline.py --mode predict

    # Quick validation run on a small sample
    python run_pipeline.py --mode train --sample 10000

Run from the code/business_entity_resolution/ directory.
The pipeline reads data from ../../dataset/ and writes outputs to ../../output/.
"""

import argparse
import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.normalize import load_and_normalize, load_ground_truth
from src.blocking import run_blocking, candidates_to_tsv
from src.features import compute_features
from src.train_model import train_matcher
from src.predict import predict_and_write
from src.evaluate import macro_f_half, blocking_recall_ceiling, reduction_ratio


def setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config.PROJECT_DIR, "pipeline.log"),
                mode="w",
            ),
        ],
    )


def load_data(split: str, sample: int = None, force_norm: bool = False):
    """Load and normalize source files for a given split (train/test)."""
    logger = logging.getLogger(__name__)
    logger.info("── Loading %s data ──", split)

    if split == "train":
        s1_path, s2_path, s3_path = config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3
    else:
        s1_path, s2_path, s3_path = config.TEST_S1, config.TEST_S2, config.TEST_S3

    s1 = load_and_normalize(s1_path, force=force_norm)
    s2 = load_and_normalize(s2_path, force=force_norm)
    s3 = load_and_normalize(s3_path, force=force_norm)

    if sample and sample < len(s1):
        logger.info("  sampling %d S1 entities for quick validation", sample)
        s1_sampled = s1.sample(n=sample, random_state=42)
        # Filter S2/S3 to matching countries
        countries = set(s1_sampled["country_norm"].unique())
        s2 = s2[s2["country_norm"].isin(countries)]
        s3 = s3[s3["country_norm"].isin(countries)]
        # Also subsample S2/S3 to keep memory manageable in sample mode
        # Use proportional sampling per country to preserve distribution
        s2s3_cap = max(sample * 400, 100000)  # ~200K for 500-sample
        if len(s2) > s2s3_cap:
            per_country = max(1, s2s3_cap // len(countries))
            rng = np.random.default_rng(42)
            idx = np.concatenate([
                rng.choice(grp_idx, size=min(len(grp_idx), per_country), replace=False)
                for grp_idx in (
                    s2.index[s2["country_norm"] == c].to_numpy() for c in countries
                )
            ])
            s2 = s2.loc[idx].reset_index(drop=True)
            logger.info("  subsampled S2 to %d for quick validation", len(s2))
        if len(s3) > s2s3_cap:
            per_country = max(1, s2s3_cap // len(countries))
            rng = np.random.default_rng(42)
            idx = np.concatenate([
                rng.choice(grp_idx, size=min(len(grp_idx), per_country), replace=False)
                for grp_idx in (
                    s3.index[s3["country_norm"] == c].to_numpy() for c in countries
                )
            ])
            s3 = s3.loc[idx].reset_index(drop=True)
            logger.info("  subsampled S3 to %d for quick validation", len(s3))
        s1 = s1_sampled

    logger.info("  S1: %d  S2: %d  S3: %d", len(s1), len(s2), len(s3))
    return s1, s2, s3


def run_training(s1, s2, s3, ground_truth, sample=None):
    """
    Full training pipeline: blocking → features → model → evaluation.
    """
    logger = logging.getLogger(__name__)
    t0 = time.time()

    # ── Stage 1: Blocking ──
    logger.info("── Stage 1: Blocking ──")
    candidates = run_blocking(s1, s2, s3)

    # Measure blocking quality — vectorized groupby aggregation
    active = candidates[candidates["candidate_entity_id"] != ""]
    cand_dict = (
        active
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )

    # Only measure on S1 entities present in our (possibly sampled) data
    s1_id_set = set(s1["entity_id"].values)
    gt_subset = {k: v for k, v in ground_truth.items() if k in s1_id_set}

    recall_ceiling = blocking_recall_ceiling(cand_dict, gt_subset)
    n_s2s3 = len(s2) + len(s3)
    rr = reduction_ratio(len(active), len(s1), n_s2s3)
    logger.info("  blocking recall ceiling: %.4f", recall_ceiling)
    logger.info("  reduction ratio:         %.6f (%.2f%% of pairs eliminated)",
                rr, rr * 100)

    # Save candidates to cache
    cand_cache = os.path.join(config.CACHE_DIR, "train_candidates.parquet")
    candidates.to_parquet(cand_cache, index=False)
    logger.info("  candidates cached to %s", cand_cache)

    gc.collect()

    # ── Stage 2: Features ──
    logger.info("── Stage 2: Feature Engineering ──")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    train_features = compute_features(candidates, s1, s2s3)

    # Save features to cache
    feat_cache = os.path.join(config.CACHE_DIR, "train_features.parquet")
    # Drop name_tokens column (list type) before saving
    save_cols = [c for c in train_features.columns if c != "name_tokens"]
    train_features[save_cols].to_parquet(feat_cache, index=False)
    logger.info("  features cached to %s", feat_cache)

    del candidates, s2s3
    gc.collect()

    # ── Stage 3: Model Training ──
    logger.info("── Stage 3: Model Training ──")
    models, calibrator, best_t, best_f05, oof_df = train_matcher(
        train_features, gt_subset
    )

    elapsed = time.time() - t0
    logger.info("══════════════════════════════════════════════")
    logger.info("  TRAINING COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("  Validation F₀.₅:  %.4f", best_f05)
    logger.info("  Best threshold:   %.3f", best_t)
    logger.info("  Blocking recall:  %.4f", recall_ceiling)
    logger.info("  Reduction ratio:  %.6f", rr)
    logger.info("══════════════════════════════════════════════")

    return models, calibrator, best_t


def run_prediction(s1, s2, s3, models=None, calibrator=None, threshold=None):
    """
    Full prediction pipeline: blocking → features → predict → write output.
    """
    logger = logging.getLogger(__name__)
    t0 = time.time()

    # ── Stage 1: Blocking ──
    logger.info("── Stage 1: Blocking (test) ──")
    candidates = run_blocking(s1, s2, s3)

    # Write candidate_pairs.tsv from the blocking output
    # (predict.py also writes it, but this ensures consistency)
    gc.collect()

    # ── Stage 2: Features ──
    logger.info("── Stage 2: Feature Engineering (test) ──")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    test_features = compute_features(candidates, s1, s2s3)

    del candidates, s2s3
    gc.collect()

    # ── Stage 4: Predict + write ──
    logger.info("── Stage 4: Prediction ──")
    s1_ids = set(s1["entity_id"].values)
    predictions = predict_and_write(
        test_features, models=models, calibrator=calibrator,
        threshold=threshold, s1_entity_ids=s1_ids,
    )

    elapsed = time.time() - t0
    logger.info("══════════════════════════════════════════════")
    logger.info("  PREDICTION COMPLETE in %.1f minutes", elapsed / 60)
    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    logger.info("  S1 entities:  %d total (%d matched, %d singletons)",
                len(predictions), n_matched, n_singleton)
    logger.info("══════════════════════════════════════════════")

    return predictions


def validate_output():
    """Run the organiser's validation script on the output files."""
    logger = logging.getLogger(__name__)
    validator = os.path.join(config.RESOURCE_ROOT, "utils", "validate_submission.py")
    matching = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")
    candidate = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")

    if not os.path.exists(validator):
        logger.warning("  validation script not found at %s", validator)
        return

    import subprocess
    cmd = [
        sys.executable, validator,
        "--matching", matching,
        "--candidate", candidate,
        "--test-dir", config.TEST_DIR,
    ]
    logger.info("  running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info("  validator stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("  validator stderr:\n%s", result.stderr)
    if result.returncode == 0:
        logger.info("  ✓ validation PASSED")
    else:
        logger.error("  ✗ validation FAILED (exit code %d)", result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "full"], default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N S1 entities for quick validation runs",
    )
    parser.add_argument(
        "--force-normalize", action="store_true",
        help="Force re-normalization even if cache exists",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started (mode=%s, sample=%s)", args.mode, args.sample)
    logger.info("  resource root: %s", config.RESOURCE_ROOT)
    logger.info("  output dir:    %s", config.OUTPUT_DIR)
    logger.info("  cache dir:     %s", config.CACHE_DIR)

    overall_start = time.time()

    if args.mode in ("train", "full"):
        # Load training data
        s1_train, s2_train, s3_train = load_data(
            "train", sample=args.sample, force_norm=args.force_normalize
        )
        ground_truth = load_ground_truth()

        # If sampled, filter ground truth to sampled S1 entities
        if args.sample:
            s1_ids = set(s1_train["entity_id"].values)
            ground_truth = {k: v for k, v in ground_truth.items() if k in s1_ids}

        models, calibrator, threshold = run_training(
            s1_train, s2_train, s3_train, ground_truth, sample=args.sample
        )

        del s1_train, s2_train, s3_train, ground_truth
        gc.collect()
    else:
        models, calibrator, threshold = None, None, None

    if args.mode in ("predict", "full"):
        # Load test data
        s1_test, s2_test, s3_test = load_data(
            "test", force_norm=args.force_normalize
        )

        run_prediction(
            s1_test, s2_test, s3_test,
            models=models, calibrator=calibrator, threshold=threshold,
        )

        del s1_test, s2_test, s3_test
        gc.collect()

        # Validate output
        validate_output()

    total_time = time.time() - overall_start
    logger.info("Pipeline finished in %.1f minutes total.", total_time / 60)


if __name__ == "__main__":
    main()
