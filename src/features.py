"""
Stage 2 — Pairwise feature engineering

Computes ~25 features per (S1, candidate) pair.  Groups:

  • Name similarity  (exact, Jaro-Winkler, Levenshtein, token-set/sort,
                      suffix, first-token, acronym, common-token count/ratio)
  • Address similarity (Jaro-Winkler, Levenshtein, TF-IDF cosine, PIN,
                        street-num, landmark overlap)
  • Country match
  • Chain/franchise disambiguation (name_match_addr_mismatch, pin_conflict,
                                    street_num_conflict)
  • Structural (blend_score, name_rank, addr_rank, candidate_count,
                reciprocal_nn)

Feature computation strategy:
  • Per-pair string similarity (rapidfuzz) runs in batched chunks.
    Each chunk extracts raw NumPy arrays from the lookup tables,
    vectorizing across the chunk wherever possible.
  • Address TF-IDF cosine computed in one bulk pass over unique addresses.
  • Reciprocal-NN computed with a vectorized merge (no Python loop).
"""

import gc
import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_jw(a: str, b: str) -> float:
    """Jaro-Winkler similarity, safe for empty strings."""
    if not a or not b:
        return 0.0
    return JaroWinkler.similarity(a, b)


def _safe_ratio(a: str, b: str) -> float:
    """Levenshtein ratio (0-1), safe for empty strings."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _safe_token_set(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _safe_token_sort(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _common_tokens(a_tokens, b_tokens) -> tuple:
    """Return (count of common tokens, ratio)."""
    if not a_tokens or not b_tokens:
        return 0, 0.0
    a_set  = set(a_tokens) if isinstance(a_tokens, list) else set(str(a_tokens).split())
    b_set  = set(b_tokens) if isinstance(b_tokens, list) else set(str(b_tokens).split())
    common = len(a_set & b_set)
    union  = len(a_set | b_set)
    return common, (common / union) if union > 0 else 0.0


# ── Batch feature computation ─────────────────────────────────────────────

def compute_features(pairs_df: pd.DataFrame,
                     s1_df: pd.DataFrame,
                     s2s3_df: pd.DataFrame,
                     batch_size: int = 50000) -> pd.DataFrame:
    """
    Compute pairwise features for all candidate pairs.

    Parameters
    ----------
    pairs_df : DataFrame
        Must have columns: source1_entity_id, candidate_entity_id,
        plus blocking-stage columns (blend_score, name_rank, addr_rank,
        candidate_count).
    s1_df : DataFrame
        Normalized S1 source data.
    s2s3_df : DataFrame
        Normalized S2+S3 source data (concatenated).

    Returns
    -------
    DataFrame
        pairs_df enriched with feature columns from config.FEATURE_COLS.
    """
    logger.info("=== FEATURE ENGINEERING ===")

    # Filter out empty candidate rows
    mask   = pairs_df["candidate_entity_id"] != ""
    active = pairs_df[mask].copy()
    empty  = pairs_df[~mask].copy()

    if active.empty:
        logger.warning("  no active candidate pairs — returning empty features")
        return pairs_df

    logger.info("  computing features for %d active pairs", len(active))

    # Build lookup dicts from DataFrames for fast access
    s1_lookup    = s1_df.set_index("entity_id")
    s2s3_lookup  = s2s3_df.set_index("entity_id")

    # Pre-compute address TF-IDF cosine similarity in bulk
    addr_tfidf_cos = _batch_addr_tfidf(active, s1_lookup, s2s3_lookup)

    # Compute features in chunks to show progress
    n_chunks    = max(1, (len(active) + batch_size - 1) // batch_size)
    all_features = []

    for chunk_idx in tqdm(range(n_chunks), desc="  features", leave=False):
        start = chunk_idx * batch_size
        end   = min(start + batch_size, len(active))
        chunk = active.iloc[start:end]

        features = _compute_chunk_features(chunk, s1_lookup, s2s3_lookup)
        all_features.append(features)

    features_df = pd.concat(all_features, ignore_index=True)

    # Add address TF-IDF cosine
    features_df["addr_tfidf_cos"] = addr_tfidf_cos

    # Compute reciprocal-NN indicator (vectorized, no Python loop)
    features_df["is_reciprocal_nn"] = _reciprocal_nn(features_df)

    # Merge features back into active pairs
    for col in features_df.columns:
        if col not in active.columns:
            active[col] = features_df[col].values

    # Recombine with empty rows (empty rows get NaN features)
    result = pd.concat([active, empty], ignore_index=True)

    logger.info("=== FEATURES DONE: %d columns ===", len(config.FEATURE_COLS))
    return result


def _compute_chunk_features(chunk: pd.DataFrame,
                            s1_lookup: pd.DataFrame,
                            s2s3_lookup: pd.DataFrame) -> pd.DataFrame:
    """Compute string-similarity features for a chunk of pairs.

    Avoids iterrows() by using .loc[] bulk access for each column,
    then zipping the resulting arrays. This keeps the per-pair
    rapidfuzz calls but removes all the pandas row-object overhead.
    """
    s1_ids   = chunk["source1_entity_id"].values
    cand_ids = chunk["candidate_entity_id"].values
    n        = len(chunk)

    # Bulk-fetch all columns we need from both lookup tables
    # Missing keys become NaN rows which we fill to "" / 0
    s1_rows   = s1_lookup.reindex(s1_ids)
    cand_rows = s2s3_lookup.reindex(cand_ids)

    def _col(df, col, default=""):
        vals = df[col].values if col in df.columns else np.full(n, default)
        return [str(v) if v == v else default for v in vals]  # NaN-safe str cast

    def _col_int(df, col):
        vals = df[col].values if col in df.columns else np.zeros(n)
        return [int(v) if v == v else 0 for v in vals]

    def _col_list(df, col):
        vals = df[col].values if col in df.columns else [[] for _ in range(n)]
        return [v if isinstance(v, list) else (str(v).split() if v == v else []) for v in vals]

    s1_names   = _col(s1_rows,   "name_norm")
    cand_names = _col(cand_rows, "name_norm")
    s1_addrs   = _col(s1_rows,   "addr_norm")
    cand_addrs = _col(cand_rows, "addr_norm")
    s1_pins    = _col(s1_rows,   "addr_pin")
    cand_pins  = _col(cand_rows, "addr_pin")
    s1_sns     = _col(s1_rows,   "addr_street_num")
    cand_sns   = _col(cand_rows, "addr_street_num")
    s1_sfxs    = _col(s1_rows,   "name_suffix")
    cand_sfxs  = _col(cand_rows, "name_suffix")
    s1_firsts  = _col(s1_rows,   "name_first_token")
    cand_firsts= _col(cand_rows, "name_first_token")
    s1_acros   = _col(s1_rows,   "name_acronym")
    cand_acros = _col(cand_rows, "name_acronym")
    s1_cnorms  = _col(s1_rows,   "country_norm")
    cand_cnorms= _col(cand_rows, "country_norm")
    s1_toks    = _col_list(s1_rows,   "name_tokens")
    cand_toks  = _col_list(cand_rows, "name_tokens")
    s1_tcnts   = _col_int(s1_rows,   "name_token_count")
    cand_tcnts = _col_int(cand_rows, "name_token_count")

    features = []
    for i in range(n):
        s1_n  = s1_names[i]
        c_n   = cand_names[i]
        s1_a  = s1_addrs[i]
        c_a   = cand_addrs[i]
        s1_p  = s1_pins[i]
        c_p   = cand_pins[i]
        s1_sn = s1_sns[i]
        c_sn  = cand_sns[i]

        name_jw = _safe_jw(s1_n, c_n)
        addr_jw = _safe_jw(s1_a, c_a)

        common, common_ratio = _common_tokens(s1_toks[i], cand_toks[i])

        feat = {
            # Name features
            "name_exact":           int(s1_n == c_n and s1_n != ""),
            "name_jw":              name_jw,
            "name_lev":             _safe_ratio(s1_n, c_n),
            "name_token_set":       _safe_token_set(s1_n, c_n),
            "name_token_sort":      _safe_token_sort(s1_n, c_n),
            "suffix_match":         int(s1_sfxs[i] == cand_sfxs[i] and s1_sfxs[i] != ""),
            "first_token_match":    int(s1_firsts[i] == cand_firsts[i] and s1_firsts[i] != ""),
            "acronym_match":        int(s1_acros[i] != "" and s1_acros[i] == cand_acros[i]),
            "name_common_tokens":   common,
            "name_common_ratio":    common_ratio,
            "name_token_count_diff": abs(s1_tcnts[i] - cand_tcnts[i]),

            # Address features
            "addr_jw":              addr_jw,
            "addr_lev":             _safe_ratio(s1_a, c_a),
            # addr_tfidf_cos added externally from bulk computation
            "pin_match":            int(s1_p != "" and c_p != "" and s1_p == c_p),
            "street_num_match":     int(s1_sn != "" and c_sn != "" and s1_sn == c_sn),

            # Country
            "country_match":        int(s1_cnorms[i] == cand_cnorms[i] and s1_cnorms[i] != ""),

            # Chain/franchise disambiguation
            "name_match_addr_mismatch": int(name_jw >= 0.92 and addr_jw < 0.30),
            "pin_conflict":         int(s1_p != "" and c_p != "" and s1_p != c_p),
            "street_num_conflict":  int(s1_sn != "" and c_sn != "" and s1_sn != c_sn),
        }
        features.append(feat)

    return pd.DataFrame(features)


def _zero_features() -> dict:
    """Return a feature dict with all zeros (for missing entities)."""
    return {
        "name_exact": 0, "name_jw": 0.0, "name_lev": 0.0,
        "name_token_set": 0.0, "name_token_sort": 0.0,
        "suffix_match": 0, "first_token_match": 0, "acronym_match": 0,
        "name_common_tokens": 0, "name_common_ratio": 0.0,
        "name_token_count_diff": 0,
        "addr_jw": 0.0, "addr_lev": 0.0,
        "pin_match": 0, "street_num_match": 0, "country_match": 0,
        "name_match_addr_mismatch": 0, "pin_conflict": 0, "street_num_conflict": 0,
    }


def _batch_addr_tfidf(pairs: pd.DataFrame,
                      s1_lookup: pd.DataFrame,
                      s2s3_lookup: pd.DataFrame) -> np.ndarray:
    """
    Compute address TF-IDF cosine similarity for all pairs in bulk.

    Uses vectorized index lookup instead of a Python loop over every row.
    """
    logger.info("  computing address TF-IDF cosine in bulk ...")
    s1_ids   = pairs["source1_entity_id"].values
    cand_ids = pairs["candidate_entity_id"].values

    # Vectorized bulk lookup — reindex returns NaN rows for missing keys
    s1_addrs   = s1_lookup.reindex(s1_ids)["addr_norm"].fillna("").values.tolist()
    cand_addrs = s2s3_lookup.reindex(cand_ids)["addr_norm"].fillna("").values.tolist()

    # Fit TF-IDF on all unique addresses in one pass
    all_addrs = list(set(s1_addrs) | set(cand_addrs))
    if not all_addrs or all(a == "" for a in all_addrs):
        return np.zeros(len(pairs))

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        min_df=1, max_df=1.0, dtype=np.float32,
    )
    vectorizer.fit(all_addrs)

    s1_vecs   = sk_normalize(vectorizer.transform(s1_addrs),   norm="l2")
    cand_vecs = sk_normalize(vectorizer.transform(cand_addrs), norm="l2")

    # Element-wise dot product of L2-normalised vectors = cosine similarity
    cosines = np.asarray(s1_vecs.multiply(cand_vecs).sum(axis=1)).flatten()

    del vectorizer, s1_vecs, cand_vecs
    gc.collect()

    return cosines


def _reciprocal_nn(features_df: pd.DataFrame) -> np.ndarray:
    """
    Reciprocal nearest-neighbour indicator: is this candidate's top-ranked
    S1 entity the same S1 entity?  A strong 1:1 disambiguation signal.

    Fully vectorized — no Python-level loop over rows.
    """
    if "blend_score" not in features_df.columns:
        return np.zeros(len(features_df), dtype=np.int8)

    temp = features_df[["source1_entity_id", "candidate_entity_id", "blend_score"]].copy()

    # For each candidate, find the S1 with the highest blend_score
    best_s1 = (
        temp
        .sort_values("blend_score", ascending=False)
        .drop_duplicates("candidate_entity_id", keep="first")
        .rename(columns={"source1_entity_id": "best_s1_id"})
        [["candidate_entity_id", "best_s1_id"]]
    )

    # Merge back to get the best S1 for each row's candidate
    merged = temp.merge(best_s1, on="candidate_entity_id", how="left")

    # Reciprocal if the current row's S1 is the candidate's best S1
    reciprocal = (merged["source1_entity_id"] == merged["best_s1_id"]).astype(np.int8)
    return reciprocal.values
