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
from rapidfuzz.distance import JaroWinkler, Levenshtein
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

    # Pre-compute address TF-IDF cosine similarity in bulk (104 MB for 26M pairs)
    addr_tfidf_cos = _batch_addr_tfidf(active, s1_lookup, s2s3_lookup)

    # Stream feature chunks directly to parquet instead of accumulating in RAM.
    # Accumulating 26M rows × 30 cols as DataFrames peaks at ~8 GB before
    # pd.concat. Streaming keeps peak RAM at ~100 MB (one chunk at a time).
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    feat_stream_path = _os.path.join(config.CACHE_DIR, "_feat_stream_tmp.parquet")
    pq_writer = None

    n_chunks = max(1, (len(active) + batch_size - 1) // batch_size)
    n_workers = getattr(config, "N_CPU_WORKERS", 1)

    def _process_chunk_to_table(chunk_idx: int) -> pa.Table:
        start = chunk_idx * batch_size
        end   = min(start + batch_size, len(active))
        chunk = active.iloc[start:end]
        feat_chunk = _compute_chunk_features(chunk, s1_lookup, s2s3_lookup)
        feat_chunk["addr_tfidf_cos"] = addr_tfidf_cos[start:end]
        return pa.Table.from_pandas(feat_chunk, preserve_index=False)

    if n_workers > 1 and n_chunks > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for table in tqdm(executor.map(_process_chunk_to_table, range(n_chunks)),
                              total=n_chunks, desc=f"  features ({n_workers} threads)", leave=False):
                if pq_writer is None:
                    pq_writer = pq.ParquetWriter(feat_stream_path, table.schema,
                                                 compression="snappy")
                pq_writer.write_table(table)
                del table
    else:
        for chunk_idx in tqdm(range(n_chunks), desc="  features", leave=False):
            table = _process_chunk_to_table(chunk_idx)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(feat_stream_path, table.schema,
                                             compression="snappy")
            pq_writer.write_table(table)
            del table

    if pq_writer:
        pq_writer.close()

    del addr_tfidf_cos
    gc.collect()

    # Load back from parquet — at this point blocking memory is fully freed
    # so we have headroom. is_reciprocal_nn needs the full DataFrame to compute
    # the cross-candidate merge, so it runs after the full load.
    features_df = pd.read_parquet(feat_stream_path)
    try:
        _os.unlink(feat_stream_path)
    except OSError:
        pass

    # Compute reciprocal-NN indicator (vectorized, no Python loop)
    features_df["is_reciprocal_nn"] = _reciprocal_nn(features_df)

    # Merge features back into active pairs
    for col in features_df.columns:
        if col not in active.columns:
            active[col] = features_df[col].values

    del features_df
    gc.collect()

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

    # ── Pre-compute string similarities as numpy arrays ───────────────────────
    # rapidfuzz 3.x has no element-wise batch API (process.cdist is all-pairs,
    # not paired-row). List comprehensions still beat calling inside the for-loop
    # because numpy array creation batches the C extension calls and the inner
    # loop body can use fast numpy index reads instead of per-call Python lookups.
    name_jw_arr   = np.array(
        [JaroWinkler.normalized_similarity(a, b) for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )
    addr_jw_arr   = np.array(
        [JaroWinkler.normalized_similarity(a, b) for a, b in zip(s1_addrs, cand_addrs)],
        dtype=np.float32,
    )
    name_lev_arr  = np.array(
        [Levenshtein.normalized_similarity(a, b) for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )
    addr_lev_arr  = np.array(
        [Levenshtein.normalized_similarity(a, b) for a, b in zip(s1_addrs, cand_addrs)],
        dtype=np.float32,
    )
    name_tset_arr = np.array(
        [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )
    name_tsort_arr = np.array(
        [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )


    # ── Vectorized feature construction (no Python per-row loop) ─────────────
    # All arrays are already computed above; we just assemble them into columns.
    s1_names_arr   = np.array(s1_names,   dtype=object)
    cand_names_arr = np.array(cand_names, dtype=object)
    s1_pins_arr    = np.array(s1_pins,    dtype=object)
    cand_pins_arr  = np.array(cand_pins,  dtype=object)
    s1_sns_arr     = np.array(s1_sns,     dtype=object)
    cand_sns_arr   = np.array(cand_sns,   dtype=object)
    s1_sfxs_arr    = np.array(s1_sfxs,   dtype=object)
    cand_sfxs_arr  = np.array(cand_sfxs, dtype=object)
    s1_firsts_arr  = np.array(s1_firsts, dtype=object)
    cand_firsts_arr= np.array(cand_firsts, dtype=object)
    s1_acros_arr   = np.array(s1_acros,  dtype=object)
    cand_acros_arr = np.array(cand_acros, dtype=object)
    s1_cnorms_arr  = np.array(s1_cnorms, dtype=object)
    cand_cnorms_arr= np.array(cand_cnorms, dtype=object)
    s1_tcnts_arr   = np.array(s1_tcnts,  dtype=np.int32)
    cand_tcnts_arr = np.array(cand_tcnts, dtype=np.int32)

    # Boolean masks for non-empty pair comparisons
    both_names  = (s1_names_arr != "") & (cand_names_arr != "")
    both_pins   = (s1_pins_arr  != "") & (cand_pins_arr  != "")
    both_sns    = (s1_sns_arr   != "") & (cand_sns_arr   != "")
    both_sfx    = (s1_sfxs_arr  != "") & (cand_sfxs_arr  != "")
    both_first  = (s1_firsts_arr != "") & (cand_firsts_arr != "")
    both_acro   = (s1_acros_arr != "") & (cand_acros_arr != "")
    both_cnorm  = (s1_cnorms_arr != "") & (cand_cnorms_arr != "")

    # Common token features (still needs a small loop — rapidfuzz has no batch API)
    common_counts = np.zeros(n, dtype=np.int32)
    common_ratios = np.zeros(n, dtype=np.float32)
    for i in range(n):
        c, r = _common_tokens(s1_toks[i], cand_toks[i])
        common_counts[i] = c
        common_ratios[i] = r

    return pd.DataFrame({
        # Name features
        "name_exact":            (both_names & (s1_names_arr == cand_names_arr)).astype(np.int8),
        "name_jw":               name_jw_arr,
        "name_lev":              name_lev_arr,
        "name_token_set":        name_tset_arr,
        "name_token_sort":       name_tsort_arr,
        "suffix_match":          (both_sfx   & (s1_sfxs_arr   == cand_sfxs_arr  )).astype(np.int8),
        "first_token_match":     (both_first & (s1_firsts_arr == cand_firsts_arr)).astype(np.int8),
        "acronym_match":         (both_acro  & (s1_acros_arr  == cand_acros_arr )).astype(np.int8),
        "name_common_tokens":    common_counts,
        "name_common_ratio":     common_ratios,
        "name_token_count_diff": np.abs(s1_tcnts_arr - cand_tcnts_arr).astype(np.int32),
        # Address features
        "addr_jw":               addr_jw_arr,
        "addr_lev":              addr_lev_arr,
        # addr_tfidf_cos added externally from bulk computation
        "pin_match":             (both_pins  & (s1_pins_arr  == cand_pins_arr )).astype(np.int8),
        "street_num_match":      (both_sns   & (s1_sns_arr   == cand_sns_arr  )).astype(np.int8),
        # Country
        "country_match":         (both_cnorm & (s1_cnorms_arr == cand_cnorms_arr)).astype(np.int8),
        # Chain/franchise disambiguation
        "name_match_addr_mismatch": ((name_jw_arr >= 0.92) & (addr_jw_arr < 0.30)).astype(np.int8),
        "pin_conflict":          (both_pins  & (s1_pins_arr  != cand_pins_arr )).astype(np.int8),
        "street_num_conflict":   (both_sns   & (s1_sns_arr   != cand_sns_arr  )).astype(np.int8),
    })


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
        min_df=3,           # prune typo/rare n-grams
        max_df=0.10,        # prune stop-word n-grams ("road","nagar","pvt")
        max_features=20_000, # hard cap — keeps transform fast and matrix small
        dtype=np.float32,
    )
    vectorizer.fit(all_addrs)

    s1_vecs   = sk_normalize(vectorizer.transform(s1_addrs),   norm="l2")
    cand_vecs = sk_normalize(vectorizer.transform(cand_addrs), norm="l2")

    # Element-wise dot product of L2-normalised vectors = cosine similarity
    cosines = None
    if getattr(config, "USE_GPU", False):
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
            s1_gpu = cp_csr_matrix(s1_vecs)
            cand_gpu = cp_csr_matrix(cand_vecs)
            cos_gpu = s1_gpu.multiply(cand_gpu).sum(axis=1)
            cosines = cp.asnumpy(cos_gpu).flatten()
            del s1_gpu, cand_gpu, cos_gpu
            cp.get_default_memory_pool().free_all_blocks()
            logger.info("  [GPU] Computed address TF-IDF cosine on GPU via CuPy")
        except Exception as e:
            logger.debug("GPU cosine computation fallback to CPU: %s", e)
            cosines = None

    if cosines is None:
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
