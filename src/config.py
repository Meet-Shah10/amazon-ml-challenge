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
# TFIDF_NGRAM_RANGE: (3,5) uses discriminative 3-grams instead of noisy 2-grams.
#   2-grams like "in","st","co" appear in almost every business name, creating
#   a dense similarity matrix that makes every S1 query match millions of docs.
#   3-grams "sta","tar","buc" are truly sparse → allows a much larger batch size.
# TFIDF_MAX_FEATURES: 30_000 is the Kaggle-safe default (30 GB RAM).
# TFIDF_BATCH_SIZE:   1500 is safe with 3-grams (sparse matrix, ~800 MB peak).
#   The previous batch size of 100 with 2-grams caused 8,832 batches + 8,832
#   gc.collect() calls for India alone (~45 min wasted on GC).
#   With 3-grams at batch=1500, India runs in ~590 batches instead.
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MIN_DF = 2             # prune rare n-grams (typo singletons)
TFIDF_MAX_DF = 0.5
TFIDF_MAX_FEATURES = 30_000  # caps vocabulary → bounds sparse matrix size
TFIDF_BATCH_SIZE = 1500      # safe with 3-grams; 15x fewer batches vs batch=100
TFIDF_SIM_THRESHOLD = 0.1    # drop cosine similarities below this

# Training sample cap
# LightGBM converges with 50k-100k S1 training entities (~1.5M pairwise rows).
# Training on all 2.2M S1 entities wastes 20+ hours for zero accuracy gain.
# Set to None to train on all S1 entities (not recommended on Kaggle).
TRAIN_SAMPLE_SIZE = 75_000   # stratified S1 sample for LightGBM training

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
