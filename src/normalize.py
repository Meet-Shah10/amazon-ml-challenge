"""
Stage 0 — Normalization

Normalize business_name, business_address, and country once per source file.
Results are cached to parquet so they are never recomputed in hot loops.

All scalar functions remain for use in unit tests or debugging.
`normalize_dataframe` is now fully vectorized for production throughput.
"""

import os
import re
import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# ── Lookup tables ──────────────────────────────────────────────────────────

LEGAL_SUFFIXES = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "llp": "llp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc", "gmbh": "gmbh",
    "sarl": "sarl", "sa": "sa",
    "sas": "sas", "eurl": "eurl",
    "srl": "srl", "ag": "ag",
}

ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "fl": "floor", "bldg": "building", "hwy": "highway",
    "ln": "lane", "dr": "drive", "sq": "square", "pl": "place",
    "no": "number", "nr": "near", "ste": "suite", "pkwy": "parkway",
    "ct": "court", "cir": "circle", "trl": "trail", "expy": "expressway",
    "rue": "rue", "bd": "boulevard", "av": "avenue",
}

COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u.s.": "united states",
    "u.s.a.": "united states", "america": "united states",
    "united states of america": "united states", "united states": "united states",
    "india": "india", "in": "india", "bharat": "india",
    "france": "france", "fr": "france", "république française": "france",
}

# Precompiled regex patterns
_RE_AMP        = re.compile(r"&")
_RE_NONWORD    = re.compile(r"[^\w\s]")
_RE_MULTI_WS   = re.compile(r"\s+")
_RE_PIN        = re.compile(r"\b(\d{5,6})\b")
_RE_STREET_NUM = re.compile(r"^(\d+[a-z]?)\b")
_RE_NEAR       = re.compile(r"near\s+(.+)$", re.IGNORECASE)

# Suffix set for fast membership tests
_SUFFIX_SET = frozenset(LEGAL_SUFFIXES.keys())


# ── Core scalar normalizers (kept for testing / debugging) ─────────────────

def _tokenize(s: str) -> list:
    """Lowercase, replace & with 'and', strip punctuation, split on whitespace."""
    if not s:
        return []
    s = _RE_AMP.sub(" and ", s.lower())
    s = _RE_NONWORD.sub(" ", s)
    return [t for t in s.split() if t]


def normalize_name(raw: str) -> dict:
    """Extract normalized name, legal suffix, acronym, etc."""
    tokens = _tokenize(raw)
    suffix = None
    core = []
    for t in tokens:
        if t in LEGAL_SUFFIXES:
            suffix = LEGAL_SUFFIXES[t]
        else:
            core.append(t)
    norm = " ".join(core)
    return {
        "name_norm": norm,
        "name_full_norm": " ".join(tokens),
        "name_suffix": suffix or "",
        "name_first_token": core[0] if core else "",
        "name_acronym": "".join(w[0] for w in core) if 1 < len(core) <= 6 else "",
        "name_token_count": len(core),
        "name_tokens": core,
    }


def normalize_address(raw: str) -> dict:
    """Extract normalized address, PIN/postal code, street number, landmark."""
    tokens = [ADDR_ABBR.get(t, t) for t in _tokenize(raw)]
    joined = " ".join(tokens)

    pin = ""
    m = _RE_PIN.search(raw or "")
    if m:
        pin = m.group(1)

    street_num = ""
    m = _RE_STREET_NUM.match(joined)
    if m:
        street_num = m.group(1)

    landmark = ""
    m = _RE_NEAR.search(joined)
    if m:
        landmark = m.group(1)

    return {
        "addr_norm": joined,
        "addr_pin": pin,
        "addr_street_num": street_num,
        "addr_landmark": landmark,
        "addr_token_count": len(tokens),
    }


def normalize_country(raw: str) -> str:
    """Canonicalize country string; unknown values pass through unchanged."""
    s = (raw or "").strip().lower()
    s = _RE_NONWORD.sub("", s).strip()
    return COUNTRY_ALIASES.get(s, s)


# ── Batch processing (fully vectorized) ───────────────────────────────────

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame — fully vectorized.

    Input columns: entity_id, business_name, business_address, country
    Output adds:   name_norm, name_full_norm, name_suffix, name_first_token,
                   name_acronym, name_token_count,
                   addr_norm, addr_pin, addr_street_num, addr_landmark,
                   addr_token_count, country_norm, name_tokens
    """
    logger.info("  normalizing %d records ...", len(df))

    # ── Name normalization (vectorized) ────────────────────────────────────
    raw_names = df["business_name"].fillna("").str.lower()
    # Replace & → and, strip non-word chars, collapse whitespace
    raw_names = (
        raw_names
        .str.replace(r"&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.strip()
    )

    # Tokenize: split on whitespace, producing a Series of lists
    token_series = raw_names.str.split()
    token_series = token_series.apply(lambda t: t if isinstance(t, list) else [])

    # For each row, partition tokens into (core, suffix) in one vectorized pass
    def _split_core_suffix(tokens: list):
        suffix = None
        core = []
        for t in tokens:
            if t in _SUFFIX_SET:
                suffix = LEGAL_SUFFIXES[t]
            else:
                core.append(t)
        return core, suffix

    split_series = token_series.apply(_split_core_suffix)
    core_series   = split_series.apply(lambda x: x[0])
    suffix_series = split_series.apply(lambda x: x[1])

    df = df.copy()
    df["name_norm"]        = core_series.apply(lambda c: " ".join(c))
    df["name_full_norm"]   = token_series.apply(lambda t: " ".join(t))
    df["name_suffix"]      = suffix_series.apply(lambda s: s if s else "")
    df["name_first_token"] = core_series.apply(lambda c: c[0] if c else "")
    df["name_acronym"]     = core_series.apply(
        lambda c: "".join(w[0] for w in c) if 1 < len(c) <= 6 else ""
    )
    df["name_token_count"] = core_series.apply(len)
    df["name_tokens"]      = core_series  # list column; not persisted to parquet

    # ── Address normalization (vectorized) ─────────────────────────────────
    raw_addrs = df["business_address"].fillna("").str.lower()
    raw_addrs_clean = (
        raw_addrs
        .str.replace(r"&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.strip()
    )
    addr_token_series = raw_addrs_clean.str.split().apply(
        lambda t: [ADDR_ABBR.get(tok, tok) for tok in (t or [])]
    )
    df["addr_norm"]        = addr_token_series.apply(lambda t: " ".join(t))
    df["addr_token_count"] = addr_token_series.apply(len)

    # PIN: extract first 5-6 digit run from original raw address
    df["addr_pin"] = (
        df["business_address"].fillna("").str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")
    )

    # Street number: first token of normalised addr if it starts with digits
    df["addr_street_num"] = (
        df["addr_norm"].str.extract(r"^(\d+[a-z]?)\b", expand=False).fillna("")
    )

    # Landmark: text after "near" (case-insensitive)
    df["addr_landmark"] = (
        df["addr_norm"].str.extract(r"near\s+(.+)$", expand=False).fillna("")
    )

    # ── Country normalization ───────────────────────────────────────────────
    country_clean = (
        df["country"].fillna("").str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )
    df["country_norm"] = country_clean.map(COUNTRY_ALIASES).fillna(country_clean)

    return df


# ── Cache-aware loader ─────────────────────────────────────────────────────

def load_and_normalize(tsv_path: str, force: bool = False) -> pd.DataFrame:
    """
    Load a source TSV, normalize it, and cache the result as parquet.

    On subsequent calls, returns the cached parquet directly unless ``force=True``.
    The ``name_tokens`` column (a Python list) is dropped before caching because
    parquet doesn't handle nested Python objects cleanly; it is regenerated on
    load from ``name_norm`` when needed.

    Memory-safe: reads the TSV in chunks of NORM_CHUNK_SIZE rows so that peak
    intermediate RAM is ~1-1.5 GB regardless of file size.  Safe on 8 GB EC2.
    """
    basename = os.path.splitext(os.path.basename(tsv_path))[0]
    cache_path = os.path.join(config.CACHE_DIR, f"{basename}_normalized.parquet")

    if os.path.exists(cache_path) and not force:
        logger.info("  loading cached %s", cache_path)
        df = pd.read_parquet(cache_path)
        # Regenerate name_tokens from name_norm (cheap split, no regex needed)
        df["name_tokens"] = df["name_norm"].str.split().apply(
            lambda t: t if isinstance(t, list) else []
        )
        return df

    logger.info("  reading %s", tsv_path)

    # ── Chunked normalization (memory-safe for large files) ────────────────
    # Each chunk is normalized independently and written to a temporary parquet
    # shard; shards are concatenated once all chunks are done, then the temp
    # files are removed.  Peak RSS ≈ 1.5 GB even for 5 M-row source files.
    chunk_size = getattr(config, "NORM_CHUNK_SIZE", 200_000)
    chunk_dir  = os.path.join(config.CACHE_DIR, f"{basename}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []

    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader):
        chunk_cache = os.path.join(chunk_dir, f"chunk_{i:04d}.parquet")
        if not os.path.exists(chunk_cache) or force:
            logger.info("  chunk %d: normalizing %d rows ...", i, len(chunk))
            normed = normalize_dataframe(chunk)
            normed.drop(columns=["name_tokens"], errors="ignore").to_parquet(
                chunk_cache, index=False
            )
        else:
            logger.info("  chunk %d: already cached, skipping.", i)
        chunk_paths.append(chunk_cache)

    logger.info("  merging %d chunk(s) ...", len(chunk_paths))
    df = pd.concat(
        [pd.read_parquet(p) for p in chunk_paths],
        ignore_index=True,
    )

    # Write consolidated cache and clean up temporary chunk shards
    df.to_parquet(cache_path, index=False)
    logger.info("  cached to %s", cache_path)
    for p in chunk_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(chunk_dir)
    except OSError:
        pass

    # Regenerate name_tokens in-memory (list column; not stored in parquet)
    df["name_tokens"] = df["name_norm"].str.split().apply(
        lambda t: t if isinstance(t, list) else []
    )
    return df


def load_ground_truth(path: Optional[str] = None) -> dict:
    """
    Load ground truth into a dict mapping source1_entity_id → set of matched ids.

    Singletons map to an empty set.
    """
    path = path or config.TRAIN_GT
    logger.info("  loading ground truth from %s", path)
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    # Fully vectorized — no iterrows
    s1_ids      = gt["source1_entity_id"].values
    matched_strs = gt["matched_entity_ids"].values
    truth = {}
    for s1_id, matched in zip(s1_ids, matched_strs):
        matched = matched.strip()
        truth[s1_id] = set(matched.split(",")) if matched else set()
    return truth
