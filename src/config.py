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
TFIDF_TOP_K = 10             # neighbours per S1 entity from TF-IDF NN
TOKEN_BLOCK_MAX_DF = 3000    # inverted-index: skip tokens in >N docs
TOKEN_BLOCK_MIN_SHARED = 2   # min shared tokens to keep a candidate
ADDR_ANCHOR_ENABLED = True   # address-anchor blocking on/off

# Address TF-IDF blocking toggle.
# DISABLED: address char n-grams create extremely dense matrices.
# India addr_norm: 4.1M docs → nnz=423M, ~3.4 GB (vs name nnz=140M, ~1.1 GB).
# Root cause: tokens like "road","nagar","street","mumbai","delhi" overlap
# across almost every record → near-dense similarity matrix → OOM.
# Address similarity is still captured as a FEATURE (addr_tfidf_cos) in
# features.py via element-wise cosine on unique address pairs.
# Recall impact: minimal — true matches are caught by name TF-IDF + token
# overlap; renamed businesses are caught by address anchor (PIN+street_num).
# Enable only on machines with ≥ 40 GB free RAM.
ADDR_TFIDF_ENABLED = False

# Pruning
PRUNE_TOP_K = 12             # hard top-K per S1 entity after union
PRUNE_MIN_SCORE = 0.40       # score-floor exception for high-confidence extras
PRUNE_HARD_CAP = 25          # absolute cap even with score-floor exception

# TF-IDF vectoriser
# TFIDF_SIM_THRESHOLD: The most important runtime lever.
#   At 0.15: nearly every pair sharing ≥2 common 3-grams survives (dense output).
#   At 0.30: only genuinely name-similar pairs survive (sparse output).
#   Effect: ~5-10x fewer nnz in the output matrix → ~5-10x faster extraction.
#   Recall impact: minimal — pairs with cosine(name) < 0.30 are near-guaranteed
#   non-matches (different business names) and would be rejected by LightGBM.
# TFIDF_BATCH_SIZE: 2000 with threshold=0.30 is safe.
#   At 0.30 threshold, output nnz per batch ≈ 10x less than at 0.15.
#   2000 × 4.1M × ~0.1% nnz rate × 8 bytes ≈ 65 MB peak (vs 1 GB at 0.15).
#   4x fewer batches (441 vs 1766 for India) → 4x fewer Python loop iterations.
# TFIDF_MAX_FEATURES: 15k instead of 20k → tighter vocab, faster transform.
TFIDF_NGRAM_RANGE = (3, 5)
TFIDF_MIN_DF = 5             # prune rare n-grams
TFIDF_MAX_DF = 0.5
TFIDF_MAX_FEATURES = 15_000  # tighter vocab → faster transform + smaller matrix
TFIDF_BATCH_SIZE = 2000      # safe with threshold=0.30 (sparse output); 441 batches for India
TFIDF_SIM_THRESHOLD = 0.30   # primary runtime lever: ~5-10x fewer nnz than 0.15

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
