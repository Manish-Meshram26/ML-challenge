"""
03_features.py
Pairwise similarity feature engineering for candidate pairs.

For each (S1 entity, S2/S3 candidate) pair, compute ~20 features:
  - Name similarity: Jaro-Winkler, Levenshtein ratio, Jaccard tokens,
                     TF-IDF cosine, char-3gram Jaccard, sorted-token Levenshtein
  - Address similarity: Jaccard tokens, Levenshtein ratio, numeric token match
  - Country: exact match flag
  - Combined: max name sim, name×addr average

Uses rapidfuzz for fast string similarity (much faster than pure Python).
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# rapidfuzz provides fast Levenshtein, Jaro-Winkler
from rapidfuzz import fuzz, distance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from preprocess_utils import char_ngrams, tokenize

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── Individual Feature Functions ────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def token_set_ratio(a: str, b: str) -> float:
    """
    Jaro-Winkler on sorted-token version of strings.
    Handles word order transpositions (e.g. 'ABC Corp' vs 'Corp ABC').
    """
    a_sorted = ' '.join(sorted(a.split()))
    b_sorted = ' '.join(sorted(b.split()))
    return fuzz.ratio(a_sorted, b_sorted) / 100.0


def extract_numeric_tokens(text: str) -> set:
    """Extract all numeric tokens (street numbers, PIN codes, etc.)"""
    import re
    return set(re.findall(r'\b\d+\b', text))


def compute_pair_features(
    s1_name_clean: str,
    s1_addr_clean: str,
    s1_country: str,
    s1_name_tokens: list,
    s1_addr_tokens: list,
    s23_name_clean: str,
    s23_addr_clean: str,
    s23_country: str,
    s23_name_tokens: list,
    s23_addr_tokens: list,
) -> dict:
    """
    Compute all pairwise features for a single (S1, S23) candidate pair.
    Returns a dict of feature_name → float value.
    """
    feats = {}

    # ── Name features ──────────────────────────────────────────────────────
    n1, n2 = s1_name_clean, s23_name_clean

    feats['name_jaro_winkler'] = (
        distance.JaroWinkler.normalized_similarity(n1, n2) if n1 and n2 else 0.0
    )
    feats['name_levenshtein_ratio'] = fuzz.ratio(n1, n2) / 100.0
    feats['name_partial_ratio'] = fuzz.partial_ratio(n1, n2) / 100.0
    feats['name_token_set_ratio'] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats['name_token_sort_ratio'] = fuzz.token_sort_ratio(n1, n2) / 100.0

    # Jaccard on word tokens
    t1_set = set(s1_name_tokens)
    t2_set = set(s23_name_tokens)
    feats['name_jaccard_tokens'] = jaccard(t1_set, t2_set)

    # Jaccard on char trigrams
    c1 = char_ngrams(n1, 3)
    c2 = char_ngrams(n2, 3)
    feats['name_char3gram_jaccard'] = jaccard(c1, c2)

    # Length ratio
    if max(len(n1), len(n2)) > 0:
        feats['name_len_ratio'] = min(len(n1), len(n2)) / max(len(n1), len(n2))
    else:
        feats['name_len_ratio'] = 1.0

    # Common prefix length (normalized)
    common_prefix = 0
    for a, b in zip(n1, n2):
        if a == b:
            common_prefix += 1
        else:
            break
    max_len = max(len(n1), len(n2), 1)
    feats['name_common_prefix_ratio'] = common_prefix / max_len

    # ── Address features ───────────────────────────────────────────────────
    a1, a2 = s1_addr_clean, s23_addr_clean

    feats['addr_levenshtein_ratio'] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats['addr_token_set_ratio'] = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats['addr_partial_ratio'] = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    # Jaccard on address word tokens
    at1 = set(s1_addr_tokens)
    at2 = set(s23_addr_tokens)
    feats['addr_jaccard_tokens'] = jaccard(at1, at2)

    # Numeric token match (street numbers, pin codes)
    n1_nums = extract_numeric_tokens(a1)
    n2_nums = extract_numeric_tokens(a2)
    feats['addr_numeric_jaccard'] = jaccard(n1_nums, n2_nums)
    feats['addr_has_common_number'] = float(bool(n1_nums & n2_nums))

    # First address token match
    at1_list = s1_addr_tokens
    at2_list = s23_addr_tokens
    feats['addr_first_token_match'] = float(
        bool(at1_list) and bool(at2_list) and at1_list[0] == at2_list[0]
    )

    # ── Country feature ────────────────────────────────────────────────────
    feats['country_exact_match'] = float(s1_country == s23_country)

    # ── Combined features ──────────────────────────────────────────────────
    name_sims = [
        feats['name_jaro_winkler'],
        feats['name_levenshtein_ratio'],
        feats['name_jaccard_tokens'],
        feats['name_char3gram_jaccard'],
        feats['name_token_set_ratio'],
        feats['name_token_sort_ratio'],
    ]
    feats['max_name_sim']  = max(name_sims)
    feats['mean_name_sim'] = sum(name_sims) / len(name_sims)

    addr_sims = [
        feats['addr_levenshtein_ratio'],
        feats['addr_jaccard_tokens'],
        feats['addr_numeric_jaccard'],
    ]
    feats['max_addr_sim']  = max(addr_sims)

    feats['name_addr_product'] = feats['max_name_sim'] * feats['max_addr_sim']

    return feats


FEATURE_COLS = [
    'name_jaro_winkler', 'name_levenshtein_ratio', 'name_partial_ratio',
    'name_token_set_ratio', 'name_token_sort_ratio', 'name_jaccard_tokens',
    'name_char3gram_jaccard', 'name_len_ratio', 'name_common_prefix_ratio',
    'addr_levenshtein_ratio', 'addr_token_set_ratio', 'addr_partial_ratio',
    'addr_jaccard_tokens', 'addr_numeric_jaccard', 'addr_has_common_number',
    'addr_first_token_match', 'country_exact_match',
    'max_name_sim', 'mean_name_sim', 'max_addr_sim',
    'name_addr_product',
]


def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    s1_lookup: dict,
    s23_lookup: dict,
    batch_size: int = 50_000
) -> pd.DataFrame:
    """
    Build feature matrix for all candidate pairs.

    Args:
        candidate_pairs: DataFrame with columns [source1_entity_id, candidate_entity_id]
        s1_lookup:  dict {entity_id → dict of preprocessed fields}
        s23_lookup: dict {entity_id → dict of preprocessed fields}
        batch_size: process in batches for progress reporting

    Returns:
        DataFrame with all feature columns + source1_entity_id + candidate_entity_id
    """
    log.info(f"Building features for {len(candidate_pairs):,} pairs ...")
    records = []

    for i, (_, row) in enumerate(tqdm(candidate_pairs.iterrows(), total=len(candidate_pairs))):
        s1_id  = row['source1_entity_id']
        s23_id = row['candidate_entity_id']

        s1_data  = s1_lookup.get(s1_id, {})
        s23_data = s23_lookup.get(s23_id, {})

        feats = compute_pair_features(
            s1_name_clean=s1_data.get('name_clean', ''),
            s1_addr_clean=s1_data.get('addr_clean', ''),
            s1_country=s1_data.get('country_clean', ''),
            s1_name_tokens=s1_data.get('name_tokens', []),
            s1_addr_tokens=s1_data.get('addr_tokens', []),
            s23_name_clean=s23_data.get('name_clean', ''),
            s23_addr_clean=s23_data.get('addr_clean', ''),
            s23_country=s23_data.get('country_clean', ''),
            s23_name_tokens=s23_data.get('name_tokens', []),
            s23_addr_tokens=s23_data.get('addr_tokens', []),
        )
        feats['source1_entity_id']  = s1_id
        feats['candidate_entity_id'] = s23_id
        records.append(feats)

    df = pd.DataFrame(records)
    return df


def build_lookups(s1: pd.DataFrame, s23: pd.DataFrame) -> tuple:
    """
    Build fast lookup dicts from entity_id → field values for feature computation.
    """
    def df_to_lookup(df):
        return {
            row['entity_id']: {
                'name_clean': row['name_clean'],
                'addr_clean': row['addr_clean'],
                'country_clean': row['country_clean'],
                'name_tokens': row['name_tokens'],
                'addr_tokens': row['addr_tokens'],
            }
            for _, row in df.iterrows()
        }

    log.info("Building S1 lookup...")
    s1_lookup = df_to_lookup(s1)
    log.info("Building S23 lookup...")
    s23_lookup = df_to_lookup(s23)
    return s1_lookup, s23_lookup


def expand_candidates(cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand candidate_pairs.tsv from wide format (comma-separated IDs)
    to long format (one row per pair) for feature computation.
    """
    rows = []
    for _, row in tqdm(cand_df.iterrows(), total=len(cand_df), desc="Expanding candidates"):
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids']
        if pd.isna(cands) or cands == '':
            continue
        for cid in str(cands).split(','):
            cid = cid.strip()
            if cid:
                rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': cid})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Smoke test
    feats = compute_pair_features(
        'prime money incorporated', '17560 ellis road tahlequah ok', 'us',
        ['prime', 'money', 'incorporated'], ['17560', 'ellis', 'road', 'tahlequah'],
        'prime money inc', '17560 ellis rd tahlequah oklahoma', 'us',
        ['prime', 'money', 'incorporated'], ['17560', 'ellis', 'road', 'tahlequah'],
    )
    for k, v in feats.items():
        print(f"  {k:35s} = {v:.4f}")
    print("\nFeature computation OK")
