"""
Configuration — all paths, hyperparameters, and constants in one place.

Edit the TUNABLE PARAMETERS section to experiment; leave DERIVED PATHS alone
unless the directory layout changes.
"""

import os

# ── Derived paths ──────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_DIR = os.path.dirname(SRC_DIR)                        # business_entity_resolution/
CODE_DIR = os.path.dirname(PROJECT_DIR)                       # code/
RESOURCE_ROOT = os.environ.get("RESOURCE_ROOT", os.path.dirname(CODE_DIR))                     # student_resource/

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(RESOURCE_ROOT, "dataset"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Auto-detect dataset directory on Kaggle if default DATA_DIR does not exist
if not os.path.exists(DATA_DIR) and os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_source1.tsv" in files:
            TRAIN_DIR = root
            DATA_DIR = os.path.dirname(root)
            TEST_DIR = os.path.join(DATA_DIR, "test")
            break
        elif "train" in dirs and os.path.exists(os.path.join(root, "train", "train_source1.tsv")):
            DATA_DIR = root
            TRAIN_DIR = os.path.join(DATA_DIR, "train")
            TEST_DIR = os.path.join(DATA_DIR, "test")
            break

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(RESOURCE_ROOT, "output") if os.path.exists(RESOURCE_ROOT) else os.path.join(PROJECT_DIR, "output"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(PROJECT_DIR, "cache"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_DIR, "models"))

# Source file paths
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

# ── Tunable parameters ────────────────────────────────────────────────────

# Blocking
TFIDF_TOP_K = 25             # neighbours per S1 entity from TF-IDF NN
TOKEN_BLOCK_MAX_DF = 5000    # inverted-index: skip tokens in >N docs
TOKEN_BLOCK_MIN_SHARED = 2   # min shared tokens to keep a candidate
ADDR_ANCHOR_ENABLED = True   # address-anchor blocking on/off

# Pruning
PRUNE_TOP_K = 20             # hard top-K per S1 entity after union
PRUNE_MIN_SCORE = 0.35       # score-floor exception for high-confidence extras
PRUNE_HARD_CAP = 60          # absolute cap even with score-floor exception

# TF-IDF vectoriser
# TFIDF_MAX_FEATURES:  35_000 is the Kaggle-safe default (30 GB RAM limit).
#   Raise to 50_000-100_000 if you have ≥ 16 GB free RAM.
# TFIDF_BATCH_SIZE:   100 keeps the per-batch intermediate dense matrix
#   at ~800 MB even for the largest partition (India: 4.1 M candidates).
#   The previous default of 1000 caused an ~8 GB spike and OOM kill on Kaggle.
#   Raise back to 1000 only on machines with ≥ 64 GB RAM.
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MIN_DF = 3             # prune rare n-grams (typo singletons)
TFIDF_MAX_DF = 0.5
TFIDF_MAX_FEATURES = 35_000  # caps vocabulary → bounds sparse matrix size
TFIDF_BATCH_SIZE = 100       # S1-entity batch size for sparse matmul (Kaggle-safe)
TFIDF_SIM_THRESHOLD = 0.1    # drop cosine similarities below this

# LightGBM
LGBM_N_ESTIMATORS = 600
LGBM_LEARNING_RATE = 0.03
LGBM_NUM_LEAVES = 31
LGBM_MIN_CHILD_SAMPLES = 20

# Training
N_FOLDS = 5
HARD_NEG_WEIGHT = 2.5        # upweight for name_match_addr_mismatch negatives
VALIDATION_FRAC = 0.15       # held-out fraction of S1 entities

# Threshold tuning
THRESHOLD_LOW = 0.30
THRESHOLD_HIGH = 0.95
THRESHOLD_STEP = 0.01

# Feature columns (order must match features.py output)
FEATURE_COLS = [
    "name_exact", "name_jw", "name_lev", "name_token_set", "name_token_sort",
    "suffix_match", "first_token_match", "acronym_match",
    "name_common_tokens", "name_common_ratio", "name_token_count_diff",
    "addr_jw", "addr_lev", "addr_tfidf_cos",
    "pin_match", "street_num_match", "country_match",
    "name_match_addr_mismatch", "pin_conflict", "street_num_conflict",
    "blend_score", "name_rank", "addr_rank",
    "candidate_count", "is_reciprocal_nn",
]

# Ensure output directories exist
for d in (OUTPUT_DIR, CACHE_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)
