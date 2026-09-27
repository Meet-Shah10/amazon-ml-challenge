# Business Entity Resolution — Amazon ML Challenge 2026

## Quick Start

**Python version: 3.9+** (tested on 3.9 and 3.11)

```bash
cd student_resource/code/business_entity_resolution

# Install dependencies (CPU-only)
pip install -r requirements.txt

# Install GPU acceleration for Kaggle (optional but recommended)
pip install cupy-cuda12x   # or cupy-cuda11x depending on CUDA driver

# Quick validation run (small sample — ~10 min)
python run_pipeline.py --mode train --sample 10000

# Full training + prediction (produces both output TSVs)
# Expected runtime: ~1 hr 15 min on Kaggle GPU T4 x2
python run_pipeline.py --mode full

# Training only (creates models in models/)
python run_pipeline.py --mode train

# Prediction only (loads trained models, writes output/)
python run_pipeline.py --mode predict
```

### Running on Kaggle
1. Set **Accelerator = GPU T4 x2** in the right-hand sidebar.
2. Set **Internet = ON** in the right-hand sidebar.
3. Run the full pipeline with `!python run_pipeline.py --mode full`.


## Architecture

Five-stage pipeline: normalization → multi-strategy blocking → pairwise
feature engineering → LightGBM pairwise classifier → F₀.₅ threshold tuning.

```
raw TSVs
  → normalize (name, address, country)           # src/normalize.py
  → partition by canonical country
  → blocking (TF-IDF NN + token index            # src/blocking.py
               + address-anchor, unioned)
  → prune to top-K per S1 entity
  → candidate_pairs.tsv
  → pairwise feature engineering                  # src/features.py
  → LightGBM pairwise classifier                 # src/train_model.py
  → calibrated probability → threshold
  → matching_results.tsv                          # src/predict.py
```

## Directory Layout

```
business_entity_resolution/
├── src/
│   ├── config.py          # all paths and hyperparameters
│   ├── normalize.py       # Stage 0: text normalization
│   ├── blocking.py        # Stage 1: candidate generation
│   ├── features.py        # Stage 2: pairwise features
│   ├── train_model.py     # Stage 3: LightGBM training
│   ├── predict.py         # Stage 4: inference + output
│   └── evaluate.py        # F₀.₅ scoring utilities
├── run_pipeline.py        # main entry point
├── requirements.txt       # pinned dependencies
├── cache/                 # normalized parquet + feature caches (auto-created)
└── models/                # trained LightGBM + calibrator (auto-created)
```

## Output

```
student_resource/output/
├── matching_results.tsv   # final matches (scored on leaderboard)
└── candidate_pairs.tsv    # blocking candidates (audited)
```

## Reproducing from scratch

```bash
# 1. Clean all caches
rm -rf cache/ models/

# 2. Run full pipeline
python run_pipeline.py --mode full

# 3. Validate output
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

## Key Design Decisions

- **LightGBM over neural models**: labeled pairs are limited; trees
  generalise better, train faster, and trivially satisfy ≤8B params.
- **Hard-negative weighting**: chain/franchise pairs (same brand, different
  address) are upweighted 2.5× during training to combat the #1 false-positive
  source under F₀.₅.
- **Country fallback**: entities with missing/unrecognised country get a
  second blocking pass against the full corpus, closing a silent recall gap.
- **No external data**: all features derived from competition data only.
  Dependencies (scikit-learn, rapidfuzz, lightgbm) are offline computation
  libraries, not data sources.

## Dependencies

All MIT/Apache 2.0 licensed:

| Package | Purpose |
|---|---|
| pandas + pyarrow | data loading, parquet caching |
| numpy | numerical operations |
| scikit-learn | TF-IDF, isotonic calibration, GroupKFold |
| lightgbm | gradient-boosted tree classifier |
| rapidfuzz | Jaro-Winkler, Levenshtein, token-set/sort |
| tqdm | progress bars |
