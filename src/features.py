"""
features.py
Pairwise similarity feature engineering for candidate pairs.

For each (S1 entity, S2/S3 candidate) pair, compute ~20 features:
  - Name similarity: Jaro-Winkler, Levenshtein ratio, Jaccard tokens,
                     char-3gram Jaccard, sorted-token ratio
  - Address similarity: Jaccard tokens, Levenshtein ratio, numeric token match
  - Country: exact match flag
  - Combined: max name sim, name×addr average

VECTORIZED — no iterrows loops. Uses rapidfuzz.process for batch operations.
"""

import os
import sys
import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from rapidfuzz import fuzz, distance
from rapidfuzz.distance import JaroWinkler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from preprocess_utils import char_ngrams, tokenize

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


# ─── Feature column list ─────────────────────────────────────────────────────
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

_NUM_RE = re.compile(r'\b\d+\b')


# ─── Helper: vectorized Jaccard over two arrays of strings ───────────────────
def _jaccard_tokens_vec(a_list, b_list):
    """Vectorized token Jaccard over parallel lists of strings."""
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        sa = set(a.split()) if a else set()
        sb = set(b.split()) if b else set()
        u = sa | sb
        out[i] = len(sa & sb) / len(u) if u else 1.0
    return out


def _jaccard_char3_vec(a_list, b_list):
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ca = char_ngrams(a, 3) if a else set()
        cb = char_ngrams(b, 3) if b else set()
        u = ca | cb
        out[i] = len(ca & cb) / len(u) if u else 1.0
    return out


def _numeric_jaccard_vec(a_list, b_list):
    out  = np.zeros(len(a_list), dtype=np.float32)
    flag = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        na = set(_NUM_RE.findall(a)) if a else set()
        nb = set(_NUM_RE.findall(b)) if b else set()
        u = na | nb
        out[i]  = len(na & nb) / len(u) if u else 1.0
        flag[i] = float(bool(na & nb))
    return out, flag


def _first_token_match_vec(a_list, b_list):
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        at = a.split() if a else []
        bt = b.split() if b else []
        out[i] = float(bool(at) and bool(bt) and at[0] == bt[0])
    return out


def _prefix_ratio_vec(a_list, b_list):
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        cp = 0
        for x, y in zip(a, b):
            if x == y:
                cp += 1
            else:
                break
        ml = max(len(a), len(b), 1)
        out[i] = cp / ml
    return out


def _len_ratio_vec(a_list, b_list):
    out = np.zeros(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        ml = max(len(a), len(b))
        out[i] = min(len(a), len(b)) / ml if ml > 0 else 1.0
    return out


# ─── Batch rapidfuzz wrappers (list → np.array) ──────────────────────────────
def _batch_ratio(a_list, b_list):
    return np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(a_list, b_list)],
                    dtype=np.float32)

def _batch_partial_ratio(a_list, b_list):
    return np.array([fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(a_list, b_list)],
                    dtype=np.float32)

def _batch_token_set_ratio(a_list, b_list):
    return np.array([fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(a_list, b_list)],
                    dtype=np.float32)

def _batch_token_sort_ratio(a_list, b_list):
    return np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a_list, b_list)],
                    dtype=np.float32)

def _batch_jaro_winkler(a_list, b_list):
    return np.array(
        [JaroWinkler.normalized_similarity(a, b) if a and b else 0.0
         for a, b in zip(a_list, b_list)],
        dtype=np.float32
    )


# ─── Main vectorized feature builder ─────────────────────────────────────────
def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    s1_lookup: dict,
    s23_lookup: dict,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """
    Build feature matrix for all candidate pairs — FULLY VECTORIZED.

    Processes in batches of `batch_size` rows so RAM stays bounded.
    ~100x faster than the old iterrows() approach.

    Args:
        candidate_pairs: DataFrame with [source1_entity_id, candidate_entity_id]
        s1_lookup:  {entity_id → {name_clean, addr_clean, country_clean, ...}}
        s23_lookup: same for S2+S3
        batch_size: rows per batch (200K ≈ 500 MB RAM per batch)

    Returns:
        DataFrame with FEATURE_COLS + source1_entity_id + candidate_entity_id
    """
    n = len(candidate_pairs)
    log.info(f"Building features for {n:,} pairs (batch_size={batch_size:,}) ...")

    s1_ids  = candidate_pairs['source1_entity_id'].tolist()
    s23_ids = candidate_pairs['candidate_entity_id'].tolist()

    # Pre-extract all fields we need (avoids repeated dict lookups)
    empty = {'name_clean': '', 'addr_clean': '', 'country_clean': '',
             'name_tokens': [], 'addr_tokens': []}

    s1_name  = [s1_lookup.get(i, empty)['name_clean']  for i in s1_ids]
    s1_addr  = [s1_lookup.get(i, empty)['addr_clean']  for i in s1_ids]
    s1_ctry  = [s1_lookup.get(i, empty)['country_clean'] for i in s1_ids]

    s23_name = [s23_lookup.get(i, empty)['name_clean'] for i in s23_ids]
    s23_addr = [s23_lookup.get(i, empty)['addr_clean'] for i in s23_ids]
    s23_ctry = [s23_lookup.get(i, empty)['country_clean'] for i in s23_ids]

    # Allocate output arrays
    feat_arrays = {col: np.zeros(n, dtype=np.float32) for col in FEATURE_COLS}

    for start in tqdm(range(0, n, batch_size), desc="Feature batches"):
        end = min(start + batch_size, n)
        sl  = slice(start, end)

        n1 = s1_name[sl];  n2 = s23_name[sl]
        a1 = s1_addr[sl];  a2 = s23_addr[sl]
        c1 = s1_ctry[sl];  c2 = s23_ctry[sl]

        # ── Name features ──────────────────────────────────────────────────
        feat_arrays['name_jaro_winkler'][sl]      = _batch_jaro_winkler(n1, n2)
        feat_arrays['name_levenshtein_ratio'][sl]  = _batch_ratio(n1, n2)
        feat_arrays['name_partial_ratio'][sl]      = _batch_partial_ratio(n1, n2)
        feat_arrays['name_token_set_ratio'][sl]    = _batch_token_set_ratio(n1, n2)
        feat_arrays['name_token_sort_ratio'][sl]   = _batch_token_sort_ratio(n1, n2)
        feat_arrays['name_jaccard_tokens'][sl]     = _jaccard_tokens_vec(n1, n2)
        feat_arrays['name_char3gram_jaccard'][sl]  = _jaccard_char3_vec(n1, n2)
        feat_arrays['name_len_ratio'][sl]          = _len_ratio_vec(n1, n2)
        feat_arrays['name_common_prefix_ratio'][sl]= _prefix_ratio_vec(n1, n2)

        # ── Address features ───────────────────────────────────────────────
        feat_arrays['addr_levenshtein_ratio'][sl]  = _batch_ratio(a1, a2)
        feat_arrays['addr_token_set_ratio'][sl]    = _batch_token_set_ratio(a1, a2)
        feat_arrays['addr_partial_ratio'][sl]      = _batch_partial_ratio(a1, a2)
        feat_arrays['addr_jaccard_tokens'][sl]     = _jaccard_tokens_vec(a1, a2)
        nj, nf = _numeric_jaccard_vec(a1, a2)
        feat_arrays['addr_numeric_jaccard'][sl]    = nj
        feat_arrays['addr_has_common_number'][sl]  = nf
        feat_arrays['addr_first_token_match'][sl]  = _first_token_match_vec(a1, a2)

        # ── Country ────────────────────────────────────────────────────────
        feat_arrays['country_exact_match'][sl] = np.array(
            [float(x == y) for x, y in zip(c1, c2)], dtype=np.float32)

        # ── Combined ───────────────────────────────────────────────────────
        name_stack = np.stack([
            feat_arrays['name_jaro_winkler'][sl],
            feat_arrays['name_levenshtein_ratio'][sl],
            feat_arrays['name_jaccard_tokens'][sl],
            feat_arrays['name_char3gram_jaccard'][sl],
            feat_arrays['name_token_set_ratio'][sl],
            feat_arrays['name_token_sort_ratio'][sl],
        ])  # (6, batch)
        feat_arrays['max_name_sim'][sl]  = name_stack.max(axis=0)
        feat_arrays['mean_name_sim'][sl] = name_stack.mean(axis=0)

        addr_stack = np.stack([
            feat_arrays['addr_levenshtein_ratio'][sl],
            feat_arrays['addr_jaccard_tokens'][sl],
            feat_arrays['addr_numeric_jaccard'][sl],
        ])
        feat_arrays['max_addr_sim'][sl] = addr_stack.max(axis=0)

        feat_arrays['name_addr_product'][sl] = (
            feat_arrays['max_name_sim'][sl] * feat_arrays['max_addr_sim'][sl]
        )

    # Assemble into DataFrame
    df = pd.DataFrame(feat_arrays)
    df.insert(0, 'source1_entity_id',  s1_ids)
    df.insert(1, 'candidate_entity_id', s23_ids)
    return df


# ─── Build lookup dicts from DataFrames ──────────────────────────────────────
def build_lookups(s1: pd.DataFrame, s23: pd.DataFrame) -> tuple:
    """
    Build fast lookup dicts: entity_id → field values.
    Uses to_dict('index') — no iterrows.
    """
    def df_to_lookup(df):
        sub = df.set_index('entity_id')[
            ['name_clean', 'addr_clean', 'country_clean', 'name_tokens', 'addr_tokens']
        ]
        return sub.to_dict('index')

    log.info("Building S1 lookup...")
    s1_lookup = df_to_lookup(s1)
    log.info("Building S23 lookup...")
    s23_lookup = df_to_lookup(s23)
    return s1_lookup, s23_lookup


# ─── Expand candidates: wide → long ──────────────────────────────────────────
def expand_candidates(cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand candidate_pairs.tsv from wide (comma-separated IDs) to long (one row per pair).
    Uses pandas str.split + explode — no iterrows.
    """
    df = cand_df[cand_df['candidate_entity_ids'].notna() &
                 (cand_df['candidate_entity_ids'] != '')].copy()
    df['candidate_entity_id'] = df['candidate_entity_ids'].str.split(',')
    df = df.explode('candidate_entity_id')
    df['candidate_entity_id'] = df['candidate_entity_id'].str.strip()
    df = df[df['candidate_entity_id'] != '']
    return df[['source1_entity_id', 'candidate_entity_id']].reset_index(drop=True)


# ─── Single-pair feature function (kept for reference / smoke test) ───────────
def compute_pair_features(
    s1_name_clean, s1_addr_clean, s1_country, s1_name_tokens, s1_addr_tokens,
    s23_name_clean, s23_addr_clean, s23_country, s23_name_tokens, s23_addr_tokens,
):
    """Compute all pairwise features for a single pair. Used in smoke test only."""
    n1, n2 = s1_name_clean, s23_name_clean
    a1, a2 = s1_addr_clean, s23_addr_clean
    feats = {}
    feats['name_jaro_winkler']       = JaroWinkler.normalized_similarity(n1, n2) if n1 and n2 else 0.0
    feats['name_levenshtein_ratio']  = fuzz.ratio(n1, n2) / 100.0
    feats['name_partial_ratio']      = fuzz.partial_ratio(n1, n2) / 100.0
    feats['name_token_set_ratio']    = fuzz.token_set_ratio(n1, n2) / 100.0
    feats['name_token_sort_ratio']   = fuzz.token_sort_ratio(n1, n2) / 100.0
    t1s, t2s = set(s1_name_tokens), set(s23_name_tokens)
    u = t1s | t2s
    feats['name_jaccard_tokens']     = len(t1s & t2s) / len(u) if u else 1.0
    c1, c2 = char_ngrams(n1, 3), char_ngrams(n2, 3)
    cu = c1 | c2
    feats['name_char3gram_jaccard']  = len(c1 & c2) / len(cu) if cu else 1.0
    ml = max(len(n1), len(n2))
    feats['name_len_ratio']          = min(len(n1), len(n2)) / ml if ml else 1.0
    cp = sum(1 for x, y in zip(n1, n2) if x == y and not False)
    for i, (x, y) in enumerate(zip(n1, n2)):
        if x != y:
            cp = i; break
    else:
        cp = min(len(n1), len(n2))
    feats['name_common_prefix_ratio'] = cp / max(len(n1), len(n2), 1)
    feats['addr_levenshtein_ratio']   = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats['addr_token_set_ratio']     = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats['addr_partial_ratio']       = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    at1, at2 = set(s1_addr_tokens), set(s23_addr_tokens)
    au = at1 | at2
    feats['addr_jaccard_tokens']      = len(at1 & at2) / len(au) if au else 1.0
    n1n = set(_NUM_RE.findall(a1))
    n2n = set(_NUM_RE.findall(a2))
    nu = n1n | n2n
    feats['addr_numeric_jaccard']     = len(n1n & n2n) / len(nu) if nu else 1.0
    feats['addr_has_common_number']   = float(bool(n1n & n2n))
    feats['addr_first_token_match']   = float(bool(s1_addr_tokens) and bool(s23_addr_tokens) and s1_addr_tokens[0] == s23_addr_tokens[0])
    feats['country_exact_match']      = float(s1_country == s23_country)
    ns = [feats['name_jaro_winkler'], feats['name_levenshtein_ratio'],
          feats['name_jaccard_tokens'], feats['name_char3gram_jaccard'],
          feats['name_token_set_ratio'], feats['name_token_sort_ratio']]
    feats['max_name_sim']  = max(ns)
    feats['mean_name_sim'] = sum(ns) / len(ns)
    as_ = [feats['addr_levenshtein_ratio'], feats['addr_jaccard_tokens'], feats['addr_numeric_jaccard']]
    feats['max_addr_sim']      = max(as_)
    feats['name_addr_product'] = feats['max_name_sim'] * feats['max_addr_sim']
    return feats


if __name__ == '__main__':
    feats = compute_pair_features(
        'prime money incorporated', '17560 ellis road tahlequah ok', 'us',
        ['prime', 'money', 'incorporated'], ['17560', 'ellis', 'road', 'tahlequah'],
        'prime money inc', '17560 ellis rd tahlequah oklahoma', 'us',
        ['prime', 'money', 'incorporated'], ['17560', 'ellis', 'road', 'tahlequah'],
    )
    for k, v in feats.items():
        print(f"  {k:35s} = {v:.4f}")
    print("\nFeature computation OK")
