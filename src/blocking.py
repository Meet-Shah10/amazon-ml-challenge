"""
Stage 1 — Blocking / candidate generation  (hybrid, memory-safe)

Four complementary strategies, unioned, then pruned to a capped top-K
per S1 entity by a blended score.  Country partitioning is the first
major cardinality reduction.

Strategies (all run WITHIN each country partition):
  1. TF-IDF char n-gram + batched sparse matmul  (name — high recall)
  2. TF-IDF char n-gram + batched sparse matmul  (address — catches DBA/renames)
  3. Token inverted index on name tokens          (fast, exact-match safety net)
  4. Address-anchor inverted index (PIN + street_num)

Memory constraints for the TF-IDF path:
  • max_features caps vocabulary → bounds sparse matrix size
  • min_df=3 prunes rare typo singletons from the feature space
  • float32 dtype halves data-array memory
  • Queries batched at TFIDF_BATCH_SIZE (default 1000)
  • Low-similarity entries thresholded out immediately per batch

Entities with missing/unknown country get a second pass against the full
corpus.
"""

import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sparse_top_k_rows(sim_matrix: csr_matrix, row_offset: int,
                       s1_ids: np.ndarray, other_ids: np.ndarray,
                       top_k: int,
                       out_s1: list, out_cand: list, out_scores: list) -> None:
    """Extract top-K entries per row from a sparse similarity matrix.

    Appends directly into three flat parallel lists (out_s1, out_cand,
    out_scores) instead of building Python dicts.  This is 5-10x cheaper
    on memory: 13M pairs as dicts ≈ 2.6 GB; as three flat lists ≈ 600 MB.

    Uses CSR indptr / indices / data arrays directly — avoids the overhead
    of calling getrow() for every row.
    """
    indptr  = sim_matrix.indptr
    indices = sim_matrix.indices
    data    = sim_matrix.data

    for i in range(sim_matrix.shape[0]):
        start_ptr = indptr[i]
        end_ptr   = indptr[i + 1]
        if start_ptr == end_ptr:
            continue  # empty row

        row_data    = data[start_ptr:end_ptr]
        row_indices = indices[start_ptr:end_ptr]

        global_i = row_offset + i
        s1_id    = s1_ids[global_i]
        n        = len(row_data)

        if n <= top_k:
            for j, score in zip(row_indices, row_data):
                out_s1.append(s1_id)
                out_cand.append(other_ids[j])
                out_scores.append(float(score))
        else:
            # O(n) partial sort instead of O(n log n) full sort
            top_local = np.argpartition(row_data, -top_k)[-top_k:]
            for idx in top_local:
                out_s1.append(s1_id)
                out_cand.append(other_ids[row_indices[idx]])
                out_scores.append(float(row_data[idx]))


# ── Strategy 1 & 2: TF-IDF blocking (memory-safe) ────────────────────────

def _tfidf_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                    text_col: str, top_k: int, score_col: str) -> pd.DataFrame:
    """
    TF-IDF char n-gram blocking with batched sparse matrix multiplication.

    Uses three flat parallel lists instead of per-row Python dicts to
    accumulate results — saves 5-10x RAM vs dict-per-row (2.6 GB → 600 MB
    for India's ~13M candidate pairs).
    """
    if other_df.empty or s1_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", score_col])

    n_s1, n_other = len(s1_df), len(other_df)
    logger.info("    TF-IDF blocking on '%s': %d S1 × %d cand  "
                "(max_features=%d, batch=%d)",
                text_col, n_s1, n_other,
                config.TFIDF_MAX_FEATURES, config.TFIDF_BATCH_SIZE)

    s1_texts    = s1_df[text_col].fillna("").values
    other_texts = other_df[text_col].fillna("").values

    # Fit on union so IDF weights are consistent across both sides
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        dtype=np.float32,
        sublinear_tf=True,
    )
    all_texts = np.concatenate([s1_texts, other_texts])
    vectorizer.fit(all_texts)
    vocab_size = len(vectorizer.vocabulary_)
    logger.info("      vocabulary size: %d", vocab_size)
    del all_texts
    gc.collect()

    if vocab_size == 0:
        logger.warning("      empty vocabulary for '%s' — skipping TF-IDF blocking "
                       "(partition too small for min_df/max_df constraints)", text_col)
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", score_col])

    # Transform & L2-normalise (so dot product = cosine similarity)
    s1_vecs    = normalize(vectorizer.transform(s1_texts),    norm="l2")
    other_vecs = normalize(vectorizer.transform(other_texts), norm="l2")

    # Log matrix stats for diagnosing memory
    other_mem_mb = (other_vecs.data.nbytes + other_vecs.indices.nbytes +
                    other_vecs.indptr.nbytes) / 1e6
    logger.info("      S2/S3 sparse matrix: %d×%d, nnz=%d, ~%.0f MB",
                other_vecs.shape[0], other_vecs.shape[1],
                other_vecs.nnz, other_mem_mb)

    del vectorizer
    gc.collect()

    s1_ids    = s1_df["entity_id"].values
    other_ids = other_df["entity_id"].values

    batch_size = config.TFIDF_BATCH_SIZE
    threshold  = config.TFIDF_SIM_THRESHOLD
    n_batches  = (n_s1 + batch_size - 1) // batch_size

    # Flat parallel lists — 5-10x lighter than list-of-dicts
    out_s1: list     = []
    out_cand: list   = []
    out_scores: list = []

    # GPU acceleration via CuPy (if CUDA and CuPy are available)
    use_gpu_matmul = False
    other_vecs_gpu = None
    if getattr(config, "USE_GPU", False):
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
            other_vecs_gpu = cp_csr_matrix(other_vecs.T)
            use_gpu_matmul = True
            logger.info("      [GPU] Accelerated sparse matmul active via CuPy (device=0)")
        except Exception as e:
            logger.debug("CuPy GPU acceleration not active: %s", e)
            use_gpu_matmul = False

    for b in tqdm(range(n_batches), desc=f"      tfidf-{text_col}", leave=False):
        start = b * batch_size
        end   = min(start + batch_size, n_s1)
        batch = s1_vecs[start:end]

        sim = None
        if use_gpu_matmul:
            try:
                batch_gpu = cp_csr_matrix(batch)
                sim_gpu   = batch_gpu.dot(other_vecs_gpu)
                if threshold > 0 and sim_gpu.nnz > 0:
                    sim_gpu.data[sim_gpu.data < threshold] = 0
                    sim_gpu.eliminate_zeros()
                sim = sim_gpu.get()
                del batch_gpu, sim_gpu
            except Exception as e:
                logger.warning("      [GPU Fallback] Batch %d failed on GPU (%s); using CPU", b, e)
                sim = None

        if sim is None:
            # Sparse × sparse^T → sparse cosine-similarity matrix
            sim = batch.dot(other_vecs.T).tocsr()
            if threshold > 0 and sim.nnz > 0:
                sim.data[sim.data < threshold] = 0
                sim.eliminate_zeros()

        _sparse_top_k_rows(
            sim, start, s1_ids, other_ids, top_k,
            out_s1, out_cand, out_scores,
        )

        del sim, batch

    if other_vecs_gpu is not None:
        del other_vecs_gpu
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    del s1_vecs, other_vecs
    gc.collect()

    n_pairs = len(out_s1)
    logger.info("      found %d candidate pairs", n_pairs)
    if not out_s1:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", score_col])

    return pd.DataFrame({
        "source1_entity_id":  out_s1,
        "candidate_entity_id": out_cand,
        score_col:            out_scores,
    })


# ── Strategy 3: Token overlap blocking (CountVectorizer + sparse dot product) ─

def _token_inverted_index_blocking(s1_df: pd.DataFrame,
                                   other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Name-token overlap blocking via CountVectorizer + batched sparse dot product.

    Replaces the Python nested-loop inverted index (millions of dict lookups)
    with a vectorized approach:
      1. CountVectorizer(binary=True) builds sparse binary token matrices in C++.
      2. overlap = s1_batch.dot(other^T) gives shared-token counts via BLAS.
      3. Filter by min_shared, extract top-K with Jaccard scoring.

    Memory: sparse binary float32 for 4.1M India candidates ≈ 160 MB.
    Speed: no Python dict loops; all token intersection work in C++ backend.
    """
    if other_df.empty or s1_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "token_jaccard"])

    from sklearn.feature_extraction.text import CountVectorizer

    min_shared = config.TOKEN_BLOCK_MIN_SHARED
    max_df_abs = config.TOKEN_BLOCK_MAX_DF
    top_k      = config.TFIDF_TOP_K
    batch_size = config.TFIDF_BATCH_SIZE

    logger.info("    token overlap (vectorized): %d S1 x %d cand "
                "(min_shared=%d, max_df=%d)",
                len(s1_df), len(other_df), min_shared, max_df_abs)

    def _safe_tokens(val):
        """Normalize to a list of tokens with min length 3."""
        if isinstance(val, list):
            return [t for t in val if len(t) >= 3]
        if val and val == val:   # not NaN
            return [t for t in str(val).split() if len(t) >= 3]
        return []

    s1_toks  = [_safe_tokens(v) for v in s1_df["name_tokens"].values]
    oth_toks = [_safe_tokens(v) for v in other_df["name_tokens"].values]

    # CountVectorizer: binary=True -> values in {0,1}, float32 saves RAM.
    # analyzer=lambda x: x passes pre-tokenised lists directly (no re-splitting).
    # max_df=int -> absolute document frequency cap (same as TOKEN_BLOCK_MAX_DF).
    cv = CountVectorizer(
        analyzer=lambda x: x,
        binary=True,
        min_df=2,
        max_df=max_df_abs,
        dtype=np.float32,
    )
    cv.fit(s1_toks + oth_toks)
    logger.info("      vocabulary size: %d tokens", len(cv.vocabulary_))

    s1_mat  = cv.transform(s1_toks)   # (n_s1,   vocab) sparse float32
    oth_mat = cv.transform(oth_toks)  # (n_other, vocab) sparse float32

    # Token counts per entity: row sums of binary matrix.
    # Needed for Jaccard denominator: union = s1_n + other_n - shared.
    s1_counts  = np.asarray(s1_mat.sum(axis=1)).ravel().astype(np.float32)
    oth_counts = np.asarray(oth_mat.sum(axis=1)).ravel().astype(np.float32)

    del cv, s1_toks, oth_toks
    gc.collect()

    s1_ids  = s1_df["entity_id"].values
    oth_ids = other_df["entity_id"].values
    n_s1    = len(s1_df)
    n_batches = (n_s1 + batch_size - 1) // batch_size

    out_s1:     list = []
    out_cand:   list = []
    out_scores: list = []

    # GPU acceleration via CuPy (if CUDA and CuPy are available)
    use_gpu_tok = False
    oth_mat_gpu = None
    if getattr(config, "USE_GPU", False):
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
            oth_mat_gpu = cp_csr_matrix(oth_mat.T)
            use_gpu_tok = True
            logger.info("      [GPU] Accelerated token overlap dot product active via CuPy (device=0)")
        except Exception as e:
            logger.debug("CuPy GPU acceleration not active for token index: %s", e)
            use_gpu_tok = False

    for b in tqdm(range(n_batches), desc="      token-cv", leave=False):
        start = b * batch_size
        end   = min(start + batch_size, n_s1)
        batch = s1_mat[start:end]

        overlap = None
        if use_gpu_tok:
            try:
                batch_gpu   = cp_csr_matrix(batch)
                overlap_gpu = batch_gpu.dot(oth_mat_gpu)
                if overlap_gpu.nnz > 0 and min_shared > 0:
                    overlap_gpu.data[overlap_gpu.data < min_shared] = 0
                    overlap_gpu.eliminate_zeros()
                overlap = overlap_gpu.get()
                del batch_gpu, overlap_gpu
            except Exception as e:
                logger.warning("      [GPU Fallback] Token batch %d failed on GPU (%s); using CPU", b, e)
                overlap = None

        if overlap is None:
            # Sparse dot: overlap[i,j] = shared token count (all in C++ BLAS)
            overlap = batch.dot(oth_mat.T).tocsr()
            if overlap.nnz > 0 and min_shared > 0:
                overlap.data[overlap.data < min_shared] = 0
                overlap.eliminate_zeros()

        # Extract top-K pairs with Jaccard scoring via flat parallel lists
        indptr  = overlap.indptr
        indices = overlap.indices
        data    = overlap.data

        for i in range(overlap.shape[0]):
            sp = indptr[i]; ep = indptr[i + 1]
            if sp == ep:
                continue
            row_data = data[sp:ep]
            row_idx  = indices[sp:ep]
            s1_id    = s1_ids[start + i]
            s1_n     = s1_counts[start + i]
            n        = len(row_data)

            if n <= top_k:
                for ki in range(n):
                    j      = row_idx[ki]
                    shared = row_data[ki]
                    union_ = s1_n + oth_counts[j] - shared
                    out_s1.append(s1_id)
                    out_cand.append(oth_ids[j])
                    out_scores.append(float(shared / union_) if union_ > 0 else 0.0)
            else:
                top_local = np.argpartition(row_data, -top_k)[-top_k:]
                for ki in top_local:
                    j      = row_idx[ki]
                    shared = row_data[ki]
                    union_ = s1_n + oth_counts[j] - shared
                    out_s1.append(s1_id)
                    out_cand.append(oth_ids[j])
                    out_scores.append(float(shared / union_) if union_ > 0 else 0.0)

        del overlap, batch

    if oth_mat_gpu is not None:
        del oth_mat_gpu
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    del s1_mat, oth_mat
    gc.collect()

    logger.info("      found %d candidate pairs", len(out_s1))
    if not out_s1:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "token_jaccard"])
    return pd.DataFrame({
        "source1_entity_id":  out_s1,
        "candidate_entity_id": out_cand,
        "token_jaccard":      out_scores,
    })


# ── Strategy 4: Address-anchor blocking ───────────────────────────────────

def _address_anchor_blocking(s1_df: pd.DataFrame,
                             other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Address-anchor inverted index on (PIN, street_num).
    Catches renamed businesses at the same physical address.
    """
    if other_df.empty or s1_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "addr_anchor"])

    logger.info("    address-anchor: %d S1 × %d cand", len(s1_df), len(other_df))

    idx_pin_sn = defaultdict(list)

    for eid, pin, sn in zip(other_df["entity_id"].values,
                            other_df["addr_pin"].values,
                            other_df["addr_street_num"].values):
        pin = str(pin).strip()
        sn  = str(sn).strip()
        if pin and sn:
            idx_pin_sn[(pin, sn)].append(eid)

    results = []
    for s1_id, pin, sn in zip(s1_df["entity_id"].values,
                              s1_df["addr_pin"].values,
                              s1_df["addr_street_num"].values):
        pin  = str(pin).strip()
        sn   = str(sn).strip()

        # High-precision anchor: same PIN + same street number = same building.
        # PIN-only lookup is deliberately removed: in India a single 6-digit PIN
        # covers an entire postal zone with thousands of businesses, so PIN-only
        # matching generates ~350M noisy candidate pairs with near-zero precision.
        if pin and sn:
            for eid in idx_pin_sn.get((pin, sn), []):
                results.append({
                    "source1_entity_id":  s1_id,
                    "candidate_entity_id": eid,
                    "addr_anchor":         1.0,
                })

    logger.info("      found %d candidate pairs", len(results))
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "addr_anchor"]
    )


# ── Union + Prune ─────────────────────────────────────────────────────────

def _union_and_prune(frames: list, s1_df: pd.DataFrame) -> pd.DataFrame:
    """Union all blocking frames, collapse per pair, prune to capped top-K.

    Replaced groupby(...).apply(_select) with a fully vectorized top-K
    selection using cumcount on a pre-sorted frame, which is an order
    of magnitude faster on millions of rows.
    """
    score_cols = ["name_sim", "addr_sim", "token_jaccard", "addr_anchor"]

    if not frames:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            *score_cols, "blend_score", "name_rank", "addr_rank",
        ])

    combined = pd.concat(frames, ignore_index=True)
    logger.info("    union: %d raw candidate rows", len(combined))

    for col in score_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    agg_dict = {col: "max" for col in score_cols}
    combined = (
        combined
        .groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(agg_dict)
    )
    combined[score_cols] = combined[score_cols].fillna(0.0)
    logger.info("    after dedup: %d unique pairs", len(combined))

    # Blend score: max across signals (single strong signal counts)
    combined["blend_score"] = combined[score_cols].max(axis=1)

    # Vectorized per-entity top-K via sort + cumcount ──────────────────────
    combined = combined.sort_values(
        ["source1_entity_id", "blend_score"], ascending=[True, False]
    )

    top_k     = config.PRUNE_TOP_K
    min_score = config.PRUNE_MIN_SCORE
    hard_cap  = config.PRUNE_HARD_CAP

    # cumcount gives 0-based rank within each source1_entity_id group
    combined["_rank"] = combined.groupby("source1_entity_id").cumcount()

    # Keep rows that are within top-K OR exceed min_score floor
    keep_mask = (combined["_rank"] < top_k) | (combined["blend_score"] > min_score)
    combined  = combined[keep_mask].copy()

    # Hard cap: re-rank after score-floor extension and apply hard_cap
    combined["_rank2"] = combined.groupby("source1_entity_id").cumcount()
    combined = combined[combined["_rank2"] < hard_cap].drop(
        columns=["_rank", "_rank2"]
    )

    # Per-entity rank features (vectorized — no apply)
    combined["name_rank"] = (
        combined.groupby("source1_entity_id")["name_sim"]
        .rank(ascending=False, method="min")
    )
    combined["addr_rank"] = (
        combined.groupby("source1_entity_id")["addr_sim"]
        .rank(ascending=False, method="min")
    )

    combined = combined.reset_index(drop=True)
    logger.info("    after prune: %d candidate pairs", len(combined))

    # Ensure every S1 entity has at least one row (even if no candidates)
    all_s1     = set(s1_df["entity_id"].values)
    present_s1 = set(combined["source1_entity_id"].values)
    missing_s1 = all_s1 - present_s1
    if missing_s1:
        empty_rows = pd.DataFrame({
            "source1_entity_id":  list(missing_s1),
            "candidate_entity_id": "",
            **{c: 0.0 for c in score_cols},
            "blend_score": 0.0, "name_rank": 0.0, "addr_rank": 0.0,
        })
        combined = pd.concat([combined, empty_rows], ignore_index=True)
        logger.info("    added %d S1 entities with no candidates", len(missing_s1))

    return combined


# ── Per-country blocking ──────────────────────────────────────────────────

def _block_partition(s1_part: pd.DataFrame, s2_part: pd.DataFrame,
                     s3_part: pd.DataFrame, label: str) -> list:
    """Run all blocking strategies on one country partition."""
    other  = pd.concat([s2_part, s3_part], ignore_index=True)
    frames = []
    top_k  = config.TFIDF_TOP_K

    logger.info("  [%s] S1=%d, S2=%d, S3=%d, other=%d",
                label, len(s1_part), len(s2_part), len(s3_part), len(other))

    if other.empty:
        return frames

    # 1. TF-IDF on name (high-recall driver)
    name_df = _tfidf_blocking(s1_part, other, "name_norm", top_k, "name_sim")
    if not name_df.empty:
        frames.append(name_df)
    gc.collect()

    # 2. TF-IDF on address — DISABLED by default (ADDR_TFIDF_ENABLED=False).
    # Address char n-grams produce 3× denser matrices than name (423M vs 140M
    # nnz for India), causing OOM on 30 GB machines and adding minimal recall:
    # true matches are name-similar; renamed businesses caught by addr anchor.
    # Enable on machines with ≥ 40 GB free RAM.
    if getattr(config, "ADDR_TFIDF_ENABLED", False):
        addr_df = _tfidf_blocking(s1_part, other, "addr_norm", top_k, "addr_sim")
        if not addr_df.empty:
            frames.append(addr_df)
        gc.collect()

    # 3. Token overlap blocking (CountVectorizer + sparse dot product)
    tok_df = _token_inverted_index_blocking(s1_part, other)
    if not tok_df.empty:
        frames.append(tok_df)
    gc.collect()

    # 4. Address-anchor blocking (PIN + street_num exact match)
    if config.ADDR_ANCHOR_ENABLED:
        anchor_df = _address_anchor_blocking(s1_part, other)
        if not anchor_df.empty:
            frames.append(anchor_df)

    del other
    gc.collect()
    return frames


# ── Public API ─────────────────────────────────────────────────────────────

def run_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                 s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full blocking pipeline: partition by country, run all strategies,
    union, prune.  Returns the candidate_pairs.tsv content.
    """
    logger.info("=== BLOCKING ===")

    all_frames = []
    countries  = sorted(c for c in s1_df["country_norm"].unique() if c)

    for country in countries:
        s1_part = s1_df[s1_df["country_norm"] == country]
        s2_part = s2_df[s2_df["country_norm"] == country]
        s3_part = s3_df[s3_df["country_norm"] == country]

        if s1_part.empty:
            continue

        frames = _block_partition(s1_part, s2_part, s3_part, label=country)
        all_frames.extend(frames)

        del s1_part, s2_part, s3_part
        gc.collect()

    # Fallback for missing/unknown country
    fallback_mask = (s1_df["country_norm"] == "") | (~s1_df["country_norm"].isin(countries))
    if fallback_mask.any():
        s1_fallback = s1_df[fallback_mask]
        logger.info("  fallback pass for %d S1 entities with missing/unknown country",
                    len(s1_fallback))
        frames = _block_partition(s1_fallback, s2_df, s3_df, label="FALLBACK")
        all_frames.extend(frames)

    candidates = _union_and_prune(all_frames, s1_df)

    # Candidate count per S1 entity (structural feature)
    non_empty = candidates[candidates["candidate_entity_id"] != ""]
    counts = non_empty.groupby("source1_entity_id").size().rename("candidate_count")
    candidates = candidates.join(counts, on="source1_entity_id")
    candidates["candidate_count"] = candidates["candidate_count"].fillna(0).astype(int)

    logger.info("=== BLOCKING DONE: %d total candidate pairs ===", len(non_empty))
    return candidates


def candidates_to_tsv(candidates: pd.DataFrame, path: str):
    """Write candidate_pairs.tsv in the submission format."""
    non_empty = candidates[candidates["candidate_entity_id"] != ""]
    grouped = (
        non_empty
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .reset_index()
    )
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]

    all_s1  = set(candidates["source1_entity_id"].unique())
    present = set(grouped["source1_entity_id"].values)
    missing = all_s1 - present
    if missing:
        empty_rows = pd.DataFrame({
            "source1_entity_id":   sorted(missing),
            "candidate_entity_ids": "",
        })
        grouped = pd.concat([grouped, empty_rows], ignore_index=True)

    grouped = grouped.sort_values("source1_entity_id")
    grouped.to_csv(path, sep="\t", index=False)
    logger.info("  wrote %s (%d rows)", path, len(grouped))
