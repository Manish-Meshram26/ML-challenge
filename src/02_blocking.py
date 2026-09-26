"""
02_blocking.py
Multi-strategy candidate generation (blocking) for Entity Resolution.

Goal: Maximize RECALL — every true match must appear in the candidate set.
The candidate set is then fed to the ML classifier for precision filtering.

Strategies (union of all):
  1. TF-IDF Cosine Similarity on business_name (char n-grams + word n-grams)
  2. Token Blocking — inverted index on name word tokens
  3. Address Token Blocking — inverted index on first address tokens
  4. Prefix Blocking — first 4 chars of normalized name

All strategies are country-aware: we only match records with the same country_clean.
This is a hard filter since country is always provided and matching across countries
is extremely unlikely for the same real-world business entity.

Output: candidate_pairs.tsv
  - source1_entity_id  →  comma-separated list of S2/S3 candidate IDs
"""

import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFIDF_COSINE_THRESHOLD = 0.20      # Min cosine similarity to keep candidate
TFIDF_TOP_K            = 20        # Max TF-IDF neighbors per S1 entity
TOKEN_MIN_TOKEN_LEN    = 3         # Ignore tokens shorter than this in token blocking
PREFIX_LEN             = 5         # Length of name prefix for prefix blocking
ADDR_PREFIX_LEN        = 3         # Number of address tokens to use for addr blocking
CHUNK_SIZE             = 10_000    # Process S1 in chunks for TF-IDF (memory)


def load_sources(split: str, data_dir: str):
    """Load all three source files for a given split ('train' or 'test')."""
    prefix = f"{split}_source"
    paths = {
        'S1': os.path.join(data_dir, split, f"{prefix}1.tsv"),
        'S2': os.path.join(data_dir, split, f"{prefix}2.tsv"),
        'S3': os.path.join(data_dir, split, f"{prefix}3.tsv"),
    }
    dfs = {}
    for src, path in paths.items():
        log.info(f"Loading {src} from {path} ...")
        df = pd.read_csv(path, sep='\t', dtype=str)
        df = df.fillna('')
        dfs[src] = df
    return dfs['S1'], dfs['S2'], dfs['S3']


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Import and apply preprocessing."""
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
    from preprocess_utils import preprocess_dataframe
    return preprocess_dataframe(df)


# ─── Strategy 1: TF-IDF Cosine Blocking ─────────────────────────────────────
def build_tfidf_candidates(s1: pd.DataFrame, s23: pd.DataFrame, country: str) -> dict:
    """
    Within a single country group, build TF-IDF index on S1 names and
    query each S2/S3 record to find top-K similar S1 entities.
    Returns: dict { s1_entity_id → set of s2/s3 candidate entity_ids }
    """
    candidates = defaultdict(set)

    s1_c  = s1[s1['country_clean'] == country].reset_index(drop=True)
    s23_c = s23[s23['country_clean'] == country].reset_index(drop=True)

    if s1_c.empty or s23_c.empty:
        return candidates

    log.info(f"  TF-IDF [{country}]: S1={len(s1_c):,}, S23={len(s23_c):,}")

    # Fit TF-IDF on S1 names using char n-grams (2–4) + word n-grams (1–2)
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
    )
    s1_matrix = vectorizer.fit_transform(s1_c['name_clean'].tolist())

    # Process S23 in chunks to avoid memory overload
    s1_ids = s1_c['entity_id'].tolist()
    for start in tqdm(range(0, len(s23_c), CHUNK_SIZE), desc=f"TF-IDF {country}", leave=False):
        chunk = s23_c.iloc[start:start + CHUNK_SIZE]
        chunk_matrix = vectorizer.transform(chunk['name_clean'].tolist())
        # Cosine similarity: (chunk_size × s1_size)
        sims = cosine_similarity(chunk_matrix, s1_matrix)
        # For each S23 row, find top-K S1 neighbors above threshold
        for i, row_sims in enumerate(sims):
            top_k_idx = np.argsort(row_sims)[::-1][:TFIDF_TOP_K]
            s23_id = chunk.iloc[i]['entity_id']
            for j in top_k_idx:
                if row_sims[j] >= TFIDF_COSINE_THRESHOLD:
                    candidates[s1_ids[j]].add(s23_id)
                else:
                    break  # Sorted, so no point continuing

    return candidates


# ─── Strategy 2: Token Blocking on Name Tokens ──────────────────────────────
def build_token_candidates(s1: pd.DataFrame, s23: pd.DataFrame, country: str) -> dict:
    """
    Build inverted index: name_token → [s1_entity_ids]
    For each S2/S3 record, look up S1 entities sharing ≥ 1 significant token.
    Very high recall, some noise from common tokens — filtered by min token length.
    """
    candidates = defaultdict(set)

    s1_c  = s1[s1['country_clean'] == country]
    s23_c = s23[s23['country_clean'] == country]

    if s1_c.empty or s23_c.empty:
        return candidates

    log.info(f"  Token blocking [{country}]: S1={len(s1_c):,}, S23={len(s23_c):,}")

    # Build inverted index for S1
    inverted = defaultdict(set)
    for _, row in s1_c.iterrows():
        for tok in row['name_tokens']:
            if len(tok) >= TOKEN_MIN_TOKEN_LEN:
                inverted[tok].add(row['entity_id'])

    # Query with S23 tokens
    for _, row in tqdm(s23_c.iterrows(), total=len(s23_c), desc=f"Token {country}", leave=False):
        for tok in row['name_tokens']:
            if len(tok) >= TOKEN_MIN_TOKEN_LEN and tok in inverted:
                for s1_id in inverted[tok]:
                    candidates[s1_id].add(row['entity_id'])

    return candidates


# ─── Strategy 3: Address Token Blocking ─────────────────────────────────────
def build_address_candidates(s1: pd.DataFrame, s23: pd.DataFrame, country: str) -> dict:
    """
    Block on the first ADDR_PREFIX_LEN tokens of the normalized address.
    Catches cases where business names differ but addresses are the same.
    """
    candidates = defaultdict(set)

    s1_c  = s1[s1['country_clean'] == country]
    s23_c = s23[s23['country_clean'] == country]

    if s1_c.empty or s23_c.empty:
        return candidates

    log.info(f"  Address blocking [{country}]: S1={len(s1_c):,}, S23={len(s23_c):,}")

    def addr_key(tokens):
        """Use first N non-trivial address tokens as block key."""
        meaningful = [t for t in tokens if len(t) >= 3][:ADDR_PREFIX_LEN]
        return ' '.join(meaningful)

    inverted = defaultdict(set)
    for _, row in s1_c.iterrows():
        key = addr_key(row['addr_tokens'])
        if key:
            inverted[key].add(row['entity_id'])

    for _, row in tqdm(s23_c.iterrows(), total=len(s23_c), desc=f"Addr {country}", leave=False):
        key = addr_key(row['addr_tokens'])
        if key and key in inverted:
            for s1_id in inverted[key]:
                candidates[s1_id].add(row['entity_id'])

    return candidates


# ─── Strategy 4: Prefix Blocking on Name ────────────────────────────────────
def build_prefix_candidates(s1: pd.DataFrame, s23: pd.DataFrame, country: str) -> dict:
    """
    Block on the first PREFIX_LEN characters of the normalized business name.
    Catches truncation and abbreviation variants.
    """
    candidates = defaultdict(set)

    s1_c  = s1[s1['country_clean'] == country]
    s23_c = s23[s23['country_clean'] == country]

    if s1_c.empty or s23_c.empty:
        return candidates

    log.info(f"  Prefix blocking [{country}]: S1={len(s1_c):,}, S23={len(s23_c):,}")

    inverted = defaultdict(set)
    for _, row in s1_c.iterrows():
        prefix = row['name_clean'][:PREFIX_LEN]
        if len(prefix) >= 3:
            inverted[prefix].add(row['entity_id'])

    for _, row in tqdm(s23_c.iterrows(), total=len(s23_c), desc=f"Prefix {country}", leave=False):
        prefix = row['name_clean'][:PREFIX_LEN]
        if prefix in inverted:
            for s1_id in inverted[prefix]:
                candidates[s1_id].add(row['entity_id'])

    return candidates


# ─── Main Blocking Pipeline ──────────────────────────────────────────────────
def run_blocking(split: str, data_dir: str, output_dir: str, models_dir: str):
    """
    Run all blocking strategies and produce candidate_pairs.tsv.
    
    Args:
        split: 'train' or 'test'
        data_dir: path to dataset/
        output_dir: path to output/
        models_dir: path to models/ (for caching)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Load data ──
    log.info("Loading data sources...")
    s1_raw, s2_raw, s3_raw = load_sources(split, data_dir)

    # ── Preprocess ──
    log.info("Preprocessing data sources...")
    sys_path_patch()
    from preprocess_utils import preprocess_dataframe
    s1  = preprocess_dataframe(s1_raw)
    s2  = preprocess_dataframe(s2_raw)
    s3  = preprocess_dataframe(s3_raw)

    # Combine S2 and S3 as candidates
    s23 = pd.concat([s2, s3], ignore_index=True)

    # Cache preprocessed data
    cache_path = os.path.join(models_dir, f'{split}_preprocessed.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump({'s1': s1, 's2': s2, 's3': s3, 's23': s23}, f)
    log.info(f"Cached preprocessed data to {cache_path}")

    # ── Get unique countries ──
    all_countries = set(s1['country_clean'].unique()) | set(s23['country_clean'].unique())
    all_countries.discard('')
    log.info(f"Countries found: {all_countries}")

    # ── Run all strategies ──
    all_candidates = defaultdict(set)

    for country in sorted(all_countries):
        log.info(f"\n=== Processing country: '{country}' ===")

        # Strategy 1: TF-IDF
        tfidf_cands = build_tfidf_candidates(s1, s23, country)
        for s1_id, cands in tfidf_cands.items():
            all_candidates[s1_id].update(cands)

        # Strategy 2: Token Blocking
        token_cands = build_token_candidates(s1, s23, country)
        for s1_id, cands in token_cands.items():
            all_candidates[s1_id].update(cands)

        # Strategy 3: Address Blocking
        addr_cands = build_address_candidates(s1, s23, country)
        for s1_id, cands in addr_cands.items():
            all_candidates[s1_id].update(cands)

        # Strategy 4: Prefix Blocking
        prefix_cands = build_prefix_candidates(s1, s23, country)
        for s1_id, cands in prefix_cands.items():
            all_candidates[s1_id].update(cands)

    # ── Ensure every S1 entity has a row (even if empty candidates) ──
    log.info("Writing candidate_pairs.tsv ...")
    rows = []
    for _, row in s1.iterrows():
        s1_id = row['entity_id']
        cands = all_candidates.get(s1_id, set())
        # Deduplicate and sort
        cands_str = ','.join(sorted(cands))
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cands_str})

    cand_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    cand_df.to_csv(out_path, sep='\t', index=False)
    log.info(f"Written {len(cand_df):,} rows to {out_path}")

    # ── Stats ──
    non_empty = cand_df[cand_df['candidate_entity_ids'] != '']
    total_cands = non_empty['candidate_entity_ids'].apply(lambda x: len(x.split(','))).sum()
    log.info(f"Stats: {len(non_empty):,} S1 entities with candidates, "
             f"{total_cands:,} total candidate pairs")

    return all_candidates, s1, s23


def sys_path_patch():
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run blocking pipeline')
    parser.add_argument('--split', default='test', choices=['train', 'test'])
    parser.add_argument('--data-dir', default=os.path.join(BASE_DIR, 'dataset'))
    parser.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'output'))
    parser.add_argument('--models-dir', default=os.path.join(BASE_DIR, 'models'))
    args = parser.parse_args()

    run_blocking(args.split, args.data_dir, args.output_dir, args.models_dir)
