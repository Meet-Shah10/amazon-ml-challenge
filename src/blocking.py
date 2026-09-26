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

    for b in tqdm(range(n_batches), desc=f"      tfidf-{text_col}", leave=False):
        start = b * batch_size
        end   = min(start + batch_size, n_s1)
        batch = s1_vecs[start:end]

        # Sparse × sparse^T → sparse cosine-similarity matrix
        sim = batch.dot(other_vecs.T).tocsr()

        # Threshold in-place to free non-zero memory quickly
        if threshold > 0 and sim.nnz > 0:
            sim.data[sim.data < threshold] = 0
            sim.eliminate_zeros()

        _sparse_top_k_rows(
            sim, start, s1_ids, other_ids, top_k,
            out_s1, out_cand, out_scores,
        )

        del sim, batch
        # gc.collect() omitted: Python ref-counting frees sim/batch immediately.
        # Calling gc.collect() inside a tight loop wastes 0.2-0.5s per call.

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


# ── Strategy 3: Token inverted-index blocking ─────────────────────────────

def _token_inverted_index_blocking(s1_df: pd.DataFrame,
                                   other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Name-token inverted index — fast, low-memory safety net.

    Builds an inverted index from S2/S3 name tokens (capped by max document
    frequency to skip ultra-common tokens).  For each S1 entity, retrieves
    candidates sharing ≥ min_shared tokens and scores by Jaccard.
    """
    if other_df.empty or s1_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "token_jaccard"])

    max_df     = config.TOKEN_BLOCK_MAX_DF
    min_shared = config.TOKEN_BLOCK_MIN_SHARED

    logger.info("    token inverted index: %d S1 × %d cand (max_df=%d, min_shared=%d)",
                len(s1_df), len(other_df), max_df, min_shared)

    # Build inverted index
    other_eids          = other_df["entity_id"].values
    other_tokens_list   = other_df["name_tokens"].values
    other_token_counts  = np.zeros(len(other_df), dtype=np.int32)

    inv_idx: defaultdict = defaultdict(list)
    for i, tokens in enumerate(other_tokens_list):
        if not isinstance(tokens, list):
            tokens = str(tokens).split() if tokens else []
        # Filter: min length 3 removes single digits, "of", "at", "st",
        # "rd", "nr" etc. that appear in almost every record and create
        # near-Cartesian joins without any discriminative value.
        unique = {t for t in tokens if len(t) >= 3}
        other_token_counts[i] = len(unique)
        for t in unique:
            inv_idx[t].append(i)

    # Prune high-frequency tokens
    pruned_count = 0
    filtered_idx = {}
    for t, posting in inv_idx.items():
        if len(posting) <= max_df:
            filtered_idx[t] = posting
        else:
            pruned_count += 1
    del inv_idx
    gc.collect()
    logger.info("      index: %d tokens kept, %d pruned (freq > %d)",
                len(filtered_idx), pruned_count, max_df)

    results  = []
    s1_eids  = s1_df["entity_id"].values
    s1_tokens_list = s1_df["name_tokens"].values

    for s1_i in range(len(s1_df)):
        s1_id     = s1_eids[s1_i]
        s1_tokens = s1_tokens_list[s1_i]
        if not isinstance(s1_tokens, list):
            s1_tokens = str(s1_tokens).split() if s1_tokens else []
        s1_set = set(s1_tokens)
        if not s1_set:
            continue
        s1_n = len(s1_set)

        cand_shared: defaultdict = defaultdict(int)
        for t in s1_set:
            for cand_i in filtered_idx.get(t, []):
                cand_shared[cand_i] += 1

        for cand_i, shared in cand_shared.items():
            if shared < min_shared:
                continue
            cand_n = int(other_token_counts[cand_i])
            union  = s1_n + cand_n - shared
            jaccard = shared / union if union > 0 else 0.0
            results.append({
                "source1_entity_id":  s1_id,
                "candidate_entity_id": other_eids[cand_i],
                "token_jaccard":       jaccard,
            })

    del filtered_idx
    gc.collect()
    logger.info("      found %d candidate pairs", len(results))
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "token_jaccard"]
    )


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

    # 2. TF-IDF on address (catches DBA / renamed businesses)
    addr_df = _tfidf_blocking(s1_part, other, "addr_norm", top_k, "addr_sim")
    if not addr_df.empty:
        frames.append(addr_df)
    gc.collect()

    # 3. Token inverted index on name (fast exact-token safety net)
    tok_df = _token_inverted_index_blocking(s1_part, other)
    if not tok_df.empty:
        frames.append(tok_df)
    gc.collect()

    # 4. Address-anchor blocking
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
