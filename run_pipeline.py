"""
run_pipeline.py
================
ONE-COMMAND entry point for the Amazon ML Challenge 2026 Entity Resolution pipeline.

Run from the student_resource/ directory:

  # Step 1: Run blocking on TRAIN data (generates training candidates)
  python run_pipeline.py --step block_train

  # Step 2: Run EDA + check blocking recall on train
  python run_pipeline.py --step eda

  # Step 3: Train the LightGBM classifier
  python run_pipeline.py --step train

  # Step 4: Run blocking on TEST data
  python run_pipeline.py --step block_test

  # Step 5: Predict on test data → matching_results.tsv + candidate_pairs.tsv
  python run_pipeline.py --step predict

  # Step 6: Validate output format (official checker)
  python run_pipeline.py --step check

  # Step 7: Score on your validation split (before submitting)
  python run_pipeline.py --step score

  # Run everything end-to-end
  python run_pipeline.py --step all
"""

import os
import sys
import time
import shutil
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SRC_DIR    = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Import pipeline modules ─────────────────────────────────────────────────
from preprocess_utils import preprocess_dataframe
from features import (
    FEATURE_COLS, build_feature_matrix, build_lookups, expand_candidates
)


# ════════════════════════════════════════════════════════════════════════════
# BLOCKING — FULLY VECTORIZED (pandas merge, no iterrows loops)
# ════════════════════════════════════════════════════════════════════════════
#
# WHY THE REWRITE:
#   Old approach:  `for _, row in df.iterrows()` → Python object per row → ~1-2 it/s on 4M rows
#   New approach:  pandas explode → merge (C-level hash join) → groupby → ~100-1000x faster
#
# Key design:
#   1. Token blocking: explode name_tokens, filter high-IDF tokens, merge, groupby
#   2. Prefix blocking: vectorized str[:N] column, merge, groupby
#   3. Address blocking: vectorized addr_key column, merge, groupby
#   4. TF-IDF blocking: pre-transform all S23 at once, batch slice sparse matrix
#
# IDF filter: tokens appearing in >MAX_TOKEN_DF S1 records are skipped.
# "limited", "private", "services" etc. appear in 100K+ records →
# joining them would create billions of intermediate rows → OOM.

TOKEN_MIN_LEN   = 3
PREFIX_LEN      = 5
MAX_TOKEN_DF    = 30     # Tokens in >30 S1 records are too common to discriminate
TFIDF_THRESHOLD = 0.20
TFIDF_TOP_K     = 20
TFIDF_BATCH     = 200    # S23 rows per dense multiply batch (~706 MB peak RAM)


def load_split(split: str):
    """Load and preprocess all 3 sources for a split, with caching."""
    cache_path = os.path.join(MODELS_DIR, f'{split}_preprocessed.pkl')
    if os.path.exists(cache_path):
        log.info(f"Loading preprocessed {split} data from cache...")
        with open(cache_path, 'rb') as f:
            c = pickle.load(f)
        return c['s1'], c['s2'], c['s3'], c['s23']

    log.info(f"Preprocessing {split} data from raw files...")
    d   = os.path.join(DATA_DIR, split)
    pfx = f"{split}_source"
    s1_raw = pd.read_csv(os.path.join(d, f'{pfx}1.tsv'), sep='\t', dtype=str).fillna('')
    s2_raw = pd.read_csv(os.path.join(d, f'{pfx}2.tsv'), sep='\t', dtype=str).fillna('')
    s3_raw = pd.read_csv(os.path.join(d, f'{pfx}3.tsv'), sep='\t', dtype=str).fillna('')
    s1  = preprocess_dataframe(s1_raw)
    s2  = preprocess_dataframe(s2_raw)
    s3  = preprocess_dataframe(s3_raw)
    s23 = pd.concat([s2, s3], ignore_index=True)

    with open(cache_path, 'wb') as f:
        pickle.dump({'s1': s1, 's2': s2, 's3': s3, 's23': s23}, f)
    log.info(f"Cached to {cache_path}")
    return s1, s2, s3, s23


# ─── Helper: explode tokens into long format ─────────────────────────────────
def _explode_tokens(df, id_col, token_col, min_len, country):
    """
    Given a dataframe with a list column (token_col), produce a
    long-format DataFrame: one row per (entity_id, token) pair.
    Applies min token length filter and country filter.
    """
    sub = df[df['country_clean'] == country][[id_col, token_col]].copy()
    if sub.empty:
        return pd.DataFrame(columns=[id_col, 'token'])
    exp = sub.explode(token_col).rename(columns={token_col: 'token'})
    exp = exp[exp['token'].str.len() >= min_len]
    return exp.reset_index(drop=True)


# ─── Strategy 1: Chunked Inverted-Index Token Blocking ──────────────────────
def token_block_fast(s1, s23, country):
    """
    Token blocking via inverted index + chunked S23 processing.

    WHY NOT PANDAS MERGE:
      Merge of 626K S1 tokens × 3M S23 tokens produced 254M rows → OOM.
      Even with IDF filter (MAX=500), too many pairs for groupby.apply(set).

    THIS APPROACH:
      1. Build S1 inverted index as a Python dict: token → [s1_entity_ids]
         (fits in ~50MB RAM; O(n_s1 × avg_tokens) to build)
      2. Filter dict to MAX_TOKEN_DF=30: removes "limited","private","services" etc.
      3. Process S23 in chunks of 200K records via Python list iteration.
         Dict lookup is O(1). At 10M lookups/sec → each chunk takes ~1 sec.
      4. Aggregate chunk pairs into running result dict.

    Expected runtime: ~5-15 min for India (4.1M S23) vs 1000+ hrs with iterrows.
    """
    import gc

    s1c  = s1[s1['country_clean'] == country]
    s23c = s23[s23['country_clean'] == country]
    if s1c.empty or s23c.empty:
        return {}

    log.info(f"  Token [{country}]: S1={len(s1c):,}, S23={len(s23c):,}")

    # ── Step 1: Build S1 inverted index ──────────────────────────────────
    inv = {}  # token (str) → list of s1_entity_id (str)
    for s1_id, tokens in zip(s1c['entity_id'], s1c['name_tokens']):
        for tok in tokens:
            if len(tok) >= TOKEN_MIN_LEN:
                if tok in inv:
                    inv[tok].append(s1_id)
                else:
                    inv[tok] = [s1_id]

    # IDF filter: drop tokens appearing in >MAX_TOKEN_DF S1 entities
    inv = {tok: ids for tok, ids in inv.items() if len(ids) <= MAX_TOKEN_DF}
    log.info(f"    S1 index: {len(inv):,} discriminative tokens (MAX_DF={MAX_TOKEN_DF})")

    if not inv:
        return {}

    # ── Step 2: Process S23 in chunks ────────────────────────────────────
    result  = {}   # s1_entity_id → set of s23_entity_ids
    s23_ids   = s23c['entity_id'].tolist()
    s23_toks  = s23c['name_tokens'].tolist()
    n_s23     = len(s23_ids)
    CHUNK     = 100_000  # S23 records per iteration; 100K = ~300MB peak per chunk (safe for 16GB)

    for chunk_start in tqdm(range(0, n_s23, CHUNK),
                            desc=f"Token {country}", leave=False):
        chunk_end = min(chunk_start + CHUNK, n_s23)

        # Collect (s1_id, s23_id) pairs from this S23 chunk
        pairs_s1  = []
        pairs_s23 = []
        for s23_id, tokens in zip(s23_ids[chunk_start:chunk_end],
                                   s23_toks[chunk_start:chunk_end]):
            for tok in tokens:
                if tok in inv:
                    matched = inv[tok]
                    pairs_s23.extend([s23_id] * len(matched))
                    pairs_s1.extend(matched)

        if not pairs_s1:
            continue

        # Build DataFrame, dedup, aggregate — cleaner than numpy object arrays
        chunk_df = pd.DataFrame({'s1_id': pairs_s1, 's23_id': pairs_s23})
        del pairs_s1, pairs_s23
        chunk_df.drop_duplicates(inplace=True)

        for s1_id, grp in chunk_df.groupby('s1_id')['s23_id']:
            vals = set(grp.values)
            if s1_id in result:
                result[s1_id].update(vals)
            else:
                result[s1_id] = vals
        del chunk_df
        gc.collect()


    log.info(f"    Token [{country}]: {sum(len(v) for v in result.values()):,} candidate pairs")
    return result





# ─── Strategy 2: Prefix Blocking (inverted index) ────────────────────────────
def prefix_block_fast(s1, s23, country):
    """
    Block on first PREFIX_LEN chars of normalized name.
    Uses inverted index — no pandas merge, no OOM risk from common prefixes.
    Common prefixes ('shri', 'ravi') filtered by MAX_PREFIX_DF.
    Runtime: ~5-10 sec for 4M S23 records (pure dict lookup loop).
    """
    s1c  = s1[s1['country_clean'] == country]
    s23c = s23[s23['country_clean'] == country]
    if s1c.empty or s23c.empty:
        return {}

    MAX_PREFIX_DF = 50   # skip prefixes shared by >50 S1 entities

    # Build inverted index: prefix → [s1_entity_ids]
    inv = {}
    for s1_id, name in zip(s1c['entity_id'], s1c['name_clean']):
        pfx = name[:PREFIX_LEN]
        if len(pfx) >= 3:
            if pfx in inv:
                inv[pfx].append(s1_id)
            else:
                inv[pfx] = [s1_id]

    # IDF filter: drop very common prefixes
    inv = {pfx: ids for pfx, ids in inv.items() if len(ids) <= MAX_PREFIX_DF}

    # Query S23
    result = {}
    for s23_id, name in zip(s23c['entity_id'], s23c['name_clean']):
        pfx = name[:PREFIX_LEN]
        if pfx in inv:
            for s1_id in inv[pfx]:
                if s1_id in result:
                    result[s1_id].add(s23_id)
                else:
                    result[s1_id] = {s23_id}

    log.info(f"    Prefix [{country}]: {sum(len(v) for v in result.values()):,} candidate pairs")
    return result


# ─── Strategy 3: Address Blocking (inverted index) ───────────────────────────
def address_block_fast(s1, s23, country):
    """
    Block on first 3 meaningful address tokens (joined as a key).
    Inverted index — no OOM from common address keys.
    Runtime: ~5-15 sec for 4M S23 records.
    """
    s1c  = s1[s1['country_clean'] == country]
    s23c = s23[s23['country_clean'] == country]
    if s1c.empty or s23c.empty:
        return {}

    MAX_ADDR_DF = 100   # skip address keys shared by >100 S1 entities

    def addr_key(tokens):
        return ' '.join([t for t in tokens if len(t) >= 3][:3])

    # Build inverted index
    inv = {}
    for s1_id, tokens in zip(s1c['entity_id'], s1c['addr_tokens']):
        key = addr_key(tokens)
        if len(key) >= 3:
            if key in inv:
                inv[key].append(s1_id)
            else:
                inv[key] = [s1_id]

    # IDF filter
    inv = {k: ids for k, ids in inv.items() if len(ids) <= MAX_ADDR_DF}

    # Query S23
    result = {}
    for s23_id, tokens in zip(s23c['entity_id'], s23c['addr_tokens']):
        key = addr_key(tokens)
        if key in inv:
            for s1_id in inv[key]:
                if s1_id in result:
                    result[s1_id].add(s23_id)
                else:
                    result[s1_id] = {s23_id}

    log.info(f"    Address [{country}]: {sum(len(v) for v in result.values()):,} candidate pairs")
    return result


# ─── Strategy 4: Char-Bigram Blocking (inverted index) ───────────────────────
def bigram_block_fast(s1, s23, country):
    """
    Block on character bigrams of first 12 chars. Pairs with >=2 shared
    bigrams are candidates. Inverted index — no explode+merge OOM.
    Runtime: ~10-20 sec for 4M S23 records.
    """
    s1c  = s1[s1['country_clean'] == country]
    s23c = s23[s23['country_clean'] == country]
    if s1c.empty or s23c.empty:
        return {}

    MAX_BG_DF = 150   # skip bigrams shared by >150 S1 entities

    def get_bigrams(text):
        t = text[:12]
        return list({t[i:i+2] for i in range(len(t)-1) if len(t[i:i+2]) == 2})

    # Build inverted index: bigram → [s1_entity_ids]
    inv = {}
    for s1_id, name in zip(s1c['entity_id'], s1c['name_clean']):
        for bg in get_bigrams(name):
            if bg in inv:
                inv[bg].append(s1_id)
            else:
                inv[bg] = [s1_id]

    # IDF filter
    inv = {bg: ids for bg, ids in inv.items() if len(ids) <= MAX_BG_DF}

    # Query S23: count bigram overlaps per (s1_id, s23_id) pair
    result = {}
    for s23_id, name in zip(s23c['entity_id'], s23c['name_clean']):
        hits = {}   # s1_id → overlap count
        for bg in get_bigrams(name):
            if bg in inv:
                for s1_id in inv[bg]:
                    hits[s1_id] = hits.get(s1_id, 0) + 1
        # Only keep pairs with >=2 shared bigrams
        for s1_id, cnt in hits.items():
            if cnt >= 2:
                if s1_id in result:
                    result[s1_id].add(s23_id)
                else:
                    result[s1_id] = {s23_id}

    log.info(f"    Bigram [{country}]: {sum(len(v) for v in result.values()):,} candidate pairs")
    return result




# ─── Strategy 5: TF-IDF Blocking (optional, vectorized) ─────────────────────
def tfidf_block_fast(s1, s23, country):
    """
    TF-IDF cosine blocking — pre-transforms all S23 in ONE call, then
    batch-slices the sparse matrix. No Python loop over rows.
    ~30-60 min per country (vs 59 hrs with iterrows).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize

    s1c  = s1[s1['country_clean'] == country].reset_index(drop=True)
    s23c = s23[s23['country_clean'] == country].reset_index(drop=True)
    if s1c.empty or s23c.empty:
        return {}

    n_s1, n_s23 = len(s1c), len(s23c)
    log.info(f"  TF-IDF [{country}]: S1={n_s1:,}, S23={n_s23:,}")

    vec = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        min_df=2, max_features=150_000, sublinear_tf=True, dtype=np.float32,
    )
    log.info(f"    Fitting on S1...")
    s1_mat  = vec.fit_transform(s1c['name_clean'].tolist())
    log.info(f"    Transforming S23 in one shot...")
    s23_mat = vec.transform(s23c['name_clean'].tolist())

    s1_norm  = sk_normalize(s1_mat,  norm='l2', copy=False)
    s23_norm = sk_normalize(s23_mat, norm='l2', copy=False)
    s1_ids   = s1c['entity_id'].tolist()
    s23_ids  = s23c['entity_id'].tolist()

    result = {}
    for start in tqdm(range(0, n_s23, TFIDF_BATCH),
                      desc=f"TF-IDF {country}", leave=False):
        batch = s23_norm[start:start + TFIDF_BATCH]
        sims  = (batch @ s1_norm.T).toarray()          # (BATCH, n_s1) float32
        for i, row_sims in enumerate(sims):
            k       = min(TFIDF_TOP_K, n_s1)
            top_idx = np.argpartition(row_sims, -k)[-k:]
            top_idx = top_idx[np.argsort(row_sims[top_idx])[::-1]]
            s23_id  = s23_ids[start + i]
            for j in top_idx:
                if row_sims[j] >= TFIDF_THRESHOLD:
                    result.setdefault(s1_ids[j], set()).add(s23_id)
                else:
                    break
    log.info(f"    TF-IDF [{country}]: {sum(len(v) for v in result.values()):,} candidate pairs")
    return result


# ─── Master Blocking Orchestrator ────────────────────────────────────────────
def run_blocking(split: str, out_filename: str = 'candidate_pairs.tsv',
                 skip_tfidf: bool = False):
    """
    Run all blocking strategies (vectorized) and write candidate_pairs.tsv.

    Expected runtime:
      - Token + Prefix + Address + Bigram: ~20-40 min total
      - + TF-IDF: additional ~1-2 hrs per country
    """
    import gc
    t0 = time.time()
    s1, s2, s3, s23 = load_split(split)

    # ── FREE s2 and s3 immediately — they're already merged into s23 ──────
    # This saves ~7 GB RAM (s2 ≈ 3.5 GB, s3 ≈ 3.7 GB with token lists).
    # s23 = pd.concat([s2, s3]) was built at cache time; we never need s2/s3 again here.
    del s2, s3
    gc.collect()
    log.info("Freed s2 + s3 from RAM (using s23 only)")

    countries = sorted(set(s1['country_clean'].unique()) | set(s23['country_clean'].unique()))
    countries = [c for c in countries if c]
    log.info(f"Countries in {split}: {countries}")

    all_cands = {}   # { s1_entity_id → set of s23_ids }

    for country in countries:
        log.info(f"\n{'='*50}\n  Country: {country}\n{'='*50}")

        for fn, name in [
            (token_block_fast,   'Token  '),
            (prefix_block_fast,  'Prefix '),
            (address_block_fast, 'Address'),
            (bigram_block_fast,  'Bigram '),
        ]:
            t1 = time.time()
            cands = fn(s1, s23, country)
            for s1_id, cset in cands.items():
                if s1_id in all_cands:
                    all_cands[s1_id].update(cset)
                else:
                    all_cands[s1_id] = set(cset)
            log.info(f"    {name}: done in {time.time()-t1:.1f}s")

        if not skip_tfidf:
            t1    = time.time()
            cands = tfidf_block_fast(s1, s23, country)
            for s1_id, cset in cands.items():
                if s1_id in all_cands:
                    all_cands[s1_id].update(cset)
                else:
                    all_cands[s1_id] = set(cset)
            log.info(f"    TF-IDF: done in {(time.time()-t1)/60:.1f}m")
        else:
            log.info("    TF-IDF: SKIPPED (--no-tfidf)")

    # Write candidate_pairs.tsv — every S1 entity must appear
    log.info("\nWriting candidate_pairs.tsv ...")
    rows = []
    for _, row in s1.iterrows():
        s1_id = row['entity_id']
        cands = all_cands.get(s1_id, set())
        rows.append({
            'source1_entity_id'   : s1_id,
            'candidate_entity_ids': ','.join(sorted(cands)),
        })

    cand_df  = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cand_df.to_csv(out_path, sep='\t', index=False)

    non_empty   = cand_df[cand_df['candidate_entity_ids'] != '']
    total_cands = non_empty['candidate_entity_ids'].apply(lambda x: len(x.split(','))).sum()
    log.info(f"\n✅ Blocking done in {(time.time()-t0)/60:.1f}m")
    log.info(f"   {len(non_empty):,} S1 entities with candidates | {total_cands:,} total pairs")
    log.info(f"   Written to: {out_path}")
    return all_cands, s1, s23

# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════
import lightgbm as lgb

LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'num_leaves':        127,
    'learning_rate':     0.05,
    'n_estimators':      1000,
    'min_child_samples': 20,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
}


def f05_score(prec, rec):
    """F0.5: weights precision 2x over recall."""
    if prec + rec == 0:
        return 0.0
    return 1.25 * prec * rec / (0.25 * prec + rec)


def load_gt(gt_path):
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    return {
        r['source1_entity_id']: (
            set(r['matched_entity_ids'].split(','))
            if r['matched_entity_ids'].strip() else set()
        )
        for r in gt_df.to_dict('records')
    }


def sweep_threshold(pairs_df, probs, gt_dict, s1_ids_all):
    """Find optimal F0.5 decision threshold."""
    best_t, best_f = 0.5, 0.0
    for thresh in np.linspace(0.30, 0.80, 51):
        matched = (
            pairs_df[probs >= thresh]
            .groupby('source1_entity_id')['candidate_entity_id']
            .apply(lambda x: set(x))
        ).to_dict()

        scores = []
        for s1_id in s1_ids_all:
            true_set = gt_dict.get(s1_id, set())
            pred_set = matched.get(s1_id, set())
            if not true_set and not pred_set:
                scores.append(1.0)
            elif not true_set and pred_set:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                prec = tp / len(pred_set) if pred_set else 0.0
                rec  = tp / len(true_set) if true_set else 0.0
                scores.append(f05_score(prec, rec))

        score = float(np.mean(scores))
        if score > best_f:
            best_f, best_t = score, thresh

    return best_t, best_f


def run_training():
    """Train LightGBM on training candidates."""
    t0 = time.time()

    # Load train split
    s1, s2, s3, s23 = load_split('train')

    # Load train candidates
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs_train.tsv')
    if not os.path.exists(cand_path):
        log.error(f"Run 'block_train' step first. Missing: {cand_path}")
        sys.exit(1)

    log.info("Loading & expanding train candidates...")
    cand_df  = pd.read_csv(cand_path, sep='\t', dtype=str).fillna('')
    pairs_df = expand_candidates(cand_df)
    log.info(f"  {len(pairs_df):,} candidate pairs")

    # Ground truth labels — vectorized (NOT apply(axis=1) which OOMs on 82M rows)
    gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')
    gt_dict  = load_gt(gt_path)
    # Build a set of "s1_id|s23_id" keys for O(1) lookup
    gt_key_set = {f"{s1_id}|{mid}"
                  for s1_id, mids in gt_dict.items()
                  for mid in mids}
    log.info(f"  Ground truth: {len(gt_key_set):,} true match pairs")

    # Vectorized label assignment — isin() is C-level, no Python loop
    pair_keys       = pairs_df['source1_entity_id'] + '|' + pairs_df['candidate_entity_id']
    pairs_df['label'] = pair_keys.isin(gt_key_set).astype(np.int8)
    del pair_keys, gt_key_set

    pos_df = pairs_df[pairs_df['label'] == 1]
    neg_df = pairs_df[pairs_df['label'] == 0]
    pos    = len(pos_df)
    neg    = len(neg_df)
    log.info(f"  Labels: {pos:,} positive, {neg:,} negative (ratio 1:{neg//max(pos,1)})")

    # Sample negatives — keep all positives + 5× negatives (enough for LightGBM)
    NEG_RATIO  = 5
    neg_sample = neg_df.sample(n=min(pos * NEG_RATIO, neg), random_state=42)
    pairs_df   = pd.concat([pos_df, neg_sample], ignore_index=True)
    log.info(f"  Sampled training set: {len(pairs_df):,} pairs ({pos:,} pos + {len(neg_sample):,} neg)")
    del pos_df, neg_df, neg_sample

    import gc; gc.collect()

    # Build lookups + features
    log.info("Building lookups...")
    s1_lookup, s23_lookup = build_lookups(s1, s23)
    log.info("Computing features...")
    feat_df = build_feature_matrix(
        pairs_df[['source1_entity_id', 'candidate_entity_id']], s1_lookup, s23_lookup
    )
    feat_df['label'] = pairs_df['label'].values
    feat_df.to_pickle(os.path.join(MODELS_DIR, 'train_features.pkl'))


    # Train/Val split (by S1 entity, not by pair)
    np.random.seed(42)
    s1_ids = s1['entity_id'].tolist()
    np.random.shuffle(s1_ids)
    split_n = int(len(s1_ids) * 0.80)
    train_ids = set(s1_ids[:split_n])
    val_ids   = set(s1_ids[split_n:])

    train_mask = feat_df['source1_entity_id'].isin(train_ids)
    val_mask   = feat_df['source1_entity_id'].isin(val_ids)

    X_tr = feat_df[train_mask][FEATURE_COLS].values.astype(np.float32)
    y_tr = feat_df[train_mask]['label'].values
    X_v  = feat_df[val_mask][FEATURE_COLS].values.astype(np.float32)
    y_v  = feat_df[val_mask]['label'].values

    scale_pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    log.info(f"Training: {len(X_tr):,} | Val: {len(X_v):,} | scale_pos_weight: {scale_pw:.1f}")

    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, 'scale_pos_weight': scale_pw})
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    log.info("Top features:")
    for feat, v in imp.head(10).items():
        log.info(f"  {feat:40s}: {v:.0f}")

    # Threshold sweep
    log.info("Sweeping threshold on validation set...")
    val_pairs = feat_df[val_mask][['source1_entity_id', 'candidate_entity_id']].reset_index(drop=True)
    val_probs = model.predict_proba(X_v)[:, 1]
    val_gt    = {k: v for k, v in gt_dict.items() if k in val_ids}
    val_s1_all = [i for i in s1['entity_id'] if i in val_ids]
    best_thresh, best_f05 = sweep_threshold(val_pairs, val_probs, val_gt, val_s1_all)

    log.info(f"Best threshold: {best_thresh:.4f} → Val F0.5 = {best_f05:.4f}")

    # Save
    model_path  = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
    thresh_path = os.path.join(MODELS_DIR, 'best_threshold.txt')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(thresh_path, 'w') as f:
        f.write(str(best_thresh))

    log.info(f"\n✅ Training complete in {(time.time()-t0)/60:.1f}m")
    log.info(f"   Model: {model_path}")
    log.info(f"   Threshold: {best_thresh:.4f}  |  Val F0.5: {best_f05:.4f}")
    return model, best_thresh, best_f05


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════════════════

def run_prediction(split: str = 'test', threshold_override: float = None):
    """Run inference and write matching_results.tsv."""
    import gc
    t0 = time.time()

    # Load model + threshold
    model_path  = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
    thresh_path = os.path.join(MODELS_DIR, 'best_threshold.txt')
    if not os.path.exists(model_path):
        log.error("No model found. Run 'train' step first.")
        sys.exit(1)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    threshold = threshold_override
    if threshold is None:
        with open(thresh_path) as f:
            threshold = float(f.read().strip())
    log.info(f"Loaded model. Threshold: {threshold:.4f}")

    # Load data
    s1, s2, s3, s23 = load_split(split)

    # Load candidates
    cand_fname = 'candidate_pairs.tsv'
    cand_path  = os.path.join(OUTPUT_DIR, cand_fname)
    if not os.path.exists(cand_path):
        log.error(f"Run 'block_test' step first. Missing: {cand_path}")
        sys.exit(1)

    cand_df  = pd.read_csv(cand_path, sep='\t', dtype=str).fillna('')
    pairs_df = expand_candidates(cand_df)
    n_pairs  = len(pairs_df)
    log.info(f"  {n_pairs:,} candidate pairs to score")

    # Build lookups ONCE and pre-extract ALL field arrays upfront
    # This avoids per-chunk dict construction (the bottleneck in the old approach)
    log.info("Building lookups + extracting fields...")
    s1_lookup, s23_lookup = build_lookups(s1, s23)

    empty = {'name_clean': '', 'addr_clean': '', 'country_clean': ''}
    s1_ids_arr  = pairs_df['source1_entity_id'].values
    s23_ids_arr = pairs_df['candidate_entity_id'].values

    log.info("  Extracting S1 fields...")
    s1_name = [s1_lookup.get(i, empty)['name_clean']    for i in s1_ids_arr]
    s1_addr = [s1_lookup.get(i, empty)['addr_clean']    for i in s1_ids_arr]
    s1_ctry = [s1_lookup.get(i, empty)['country_clean'] for i in s1_ids_arr]
    log.info("  Extracting S23 fields...")
    s23_name = [s23_lookup.get(i, empty)['name_clean']    for i in s23_ids_arr]
    s23_addr = [s23_lookup.get(i, empty)['addr_clean']    for i in s23_ids_arr]
    s23_ctry = [s23_lookup.get(i, empty)['country_clean'] for i in s23_ids_arr]
    del s1_lookup, s23_lookup
    gc.collect()
    log.info("  Fields extracted. Starting chunked scoring...")

    # Score in chunks — call vectorized feature helpers DIRECTLY on string slices
    # No dict construction per chunk → each 500K chunk takes ~20 sec not ~8 min
    from src.features import (
        _batch_jaro_winkler, _batch_ratio, _batch_partial_ratio,
        _batch_token_set_ratio, _batch_token_sort_ratio,
        _jaccard_tokens_vec, _jaccard_char3_vec, _len_ratio_vec,
        _prefix_ratio_vec, _numeric_jaccard_vec, _first_token_match_vec,
    )

    PRED_CHUNK = 13_000_000   # 13M×136B = ~1.77 GB peak RAM per chunk (within 1.8 GB budget)
    matched_s1  = []
    matched_s23 = []
    n_matches   = 0

    log.info(f"Scoring {n_pairs:,} pairs in chunks of {PRED_CHUNK:,}...")
    n_chunks = (n_pairs + PRED_CHUNK - 1) // PRED_CHUNK

    for chunk_idx in tqdm(range(n_chunks), desc="Scoring chunks"):
        start = chunk_idx * PRED_CHUNK
        end   = min(start + PRED_CHUNK, n_pairs)
        sz    = end - start

        n1 = s1_name[start:end];  n2 = s23_name[start:end]
        a1 = s1_addr[start:end];  a2 = s23_addr[start:end]
        c1 = s1_ctry[start:end];  c2 = s23_ctry[start:end]

        # Build feature matrix directly from string lists — no dicts, no DataFrame
        feat = np.empty((sz, len(FEATURE_COLS)), dtype=np.float32)
        col = 0
        feat[:, col] = _batch_jaro_winkler(n1, n2);          col += 1
        feat[:, col] = _batch_ratio(n1, n2);                  col += 1
        feat[:, col] = _batch_partial_ratio(n1, n2);          col += 1
        feat[:, col] = _batch_token_set_ratio(n1, n2);        col += 1
        feat[:, col] = _batch_token_sort_ratio(n1, n2);       col += 1
        feat[:, col] = _jaccard_tokens_vec(n1, n2);           col += 1
        feat[:, col] = _jaccard_char3_vec(n1, n2);            col += 1
        feat[:, col] = _len_ratio_vec(n1, n2);                col += 1
        feat[:, col] = _prefix_ratio_vec(n1, n2);             col += 1
        feat[:, col] = _batch_ratio(a1, a2);                  col += 1
        feat[:, col] = _batch_token_set_ratio(a1, a2);        col += 1
        feat[:, col] = _batch_partial_ratio(a1, a2);          col += 1
        feat[:, col] = _jaccard_tokens_vec(a1, a2);           col += 1
        nj, nf = _numeric_jaccard_vec(a1, a2)
        feat[:, col] = nj;                                     col += 1
        feat[:, col] = nf;                                     col += 1
        feat[:, col] = _first_token_match_vec(a1, a2);        col += 1
        feat[:, col] = np.array([float(x==y) for x,y in zip(c1,c2)], dtype=np.float32); col += 1
        # Combined
        name_stack = feat[:, 0:6]   # jaro, lev, tsr, tsor, jac, char3 — wait, reorder
        # Correct indices: jaro=0, lev=1, partial=2, tsr=3, tsort=4, jac_tok=5, char3=6
        name_stack = np.stack([feat[:,0], feat[:,1], feat[:,5], feat[:,6], feat[:,3], feat[:,4]])
        feat[:, col] = name_stack.max(axis=0);                 col += 1   # max_name_sim
        feat[:, col] = name_stack.mean(axis=0);                col += 1   # mean_name_sim
        addr_stack = np.stack([feat[:,9], feat[:,12], feat[:,13]])
        feat[:, col] = addr_stack.max(axis=0);                 col += 1   # max_addr_sim
        feat[:, col] = feat[:, col-3] * feat[:, col-1]        # name_addr_product (max_name * max_addr)

        probs = model.predict_proba(feat)[:, 1]
        mask  = probs >= threshold
        n_matches += int(mask.sum())
        matched_s1.extend(s1_ids_arr[start:end][mask].tolist())
        matched_s23.extend(s23_ids_arr[start:end][mask].tolist())

        del feat, probs, mask
        gc.collect()

    log.info(f"  {n_matches:,} matches predicted (threshold={threshold:.4f})")

    # Build matching_results.tsv
    matched_df = pd.DataFrame({'source1_entity_id': matched_s1,
                               'candidate_entity_id': matched_s23})
    matched = (
        matched_df
        .groupby('source1_entity_id')['candidate_entity_id']
        .apply(lambda x: ','.join(sorted(set(x))))
        .reset_index()
    )
    matched.columns = ['source1_entity_id', 'matched_entity_ids']

    # Ensure ALL S1 entities appear
    all_s1 = pd.DataFrame({'source1_entity_id': s1['entity_id'].tolist()})
    results = all_s1.merge(matched, on='source1_entity_id', how='left')
    results['matched_entity_ids'] = results['matched_entity_ids'].fillna('')

    # Write outputs
    suffix = '' if split == 'test' else f'_{split}'
    out_path = os.path.join(OUTPUT_DIR, f'matching_results{suffix}.tsv')
    results.to_csv(out_path, sep='\t', index=False)

    singletons = (results['matched_entity_ids'] == '').sum()
    total_links = results['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x else 0
    ).sum()
    log.info(f"\n✅ Prediction complete in {(time.time()-t0)/60:.1f}m")
    log.info(f"   {len(results):,} S1 entities | {singletons:,} singletons | {total_links:,} links")
    log.info(f"   matching_results: {out_path}")
    return results



# ════════════════════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════════════════════

def run_scoring():
    """Score train-split predictions against ground truth."""
    pred_path = os.path.join(OUTPUT_DIR, 'matching_results_train.tsv')
    gt_path   = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

    if not os.path.exists(pred_path):
        log.error(f"No validation predictions at {pred_path}. "
                  "Run: python run_pipeline.py --step predict --split train")
        sys.exit(1)

    gt_dict  = load_gt(gt_path)
    pred_df  = pd.read_csv(pred_path, sep='\t', dtype=str).fillna('')
    pred_map = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        mids  = str(row['matched_entity_ids']).strip()
        pred_map[s1_id] = set(mids.split(',')) if mids else set()

    scores = []
    tp_t, fp_t, fn_t = 0, 0, 0
    sing_ok, sing_bad = 0, 0

    for s1_id, true_set in gt_dict.items():
        pred_set = pred_map.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0); sing_ok += 1
        elif not true_set and pred_set:
            scores.append(0.0); sing_bad += 1; fp_t += len(pred_set)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec  = tp / len(true_set) if true_set else 0.0
            scores.append(f05_score(prec, rec))
            tp_t += tp; fp_t += fp; fn_t += fn

    macro = float(np.mean(scores))
    gp = tp_t / (tp_t + fp_t) if tp_t + fp_t else 0.0
    gr = tp_t / (tp_t + fn_t) if tp_t + fn_t else 0.0

    print("\n" + "="*60)
    print(f"  📊 MACRO F0.5:    {macro:.6f}")
    print(f"  Global Precision: {gp:.4f}")
    print(f"  Global Recall:    {gr:.4f}")
    print(f"  TP={tp_t:,}  FP={fp_t:,}  FN={fn_t:,}")
    print(f"  Singletons correct: {sing_ok:,}  |  False merges: {sing_bad:,}")
    print("="*60 + "\n")
    return macro


# ════════════════════════════════════════════════════════════════════════════
# FORMAT CHECK
# ════════════════════════════════════════════════════════════════════════════

def run_format_check():
    """Run official validate_submission.py."""
    import subprocess
    validator = os.path.join(BASE_DIR, 'utils', 'validate_submission.py')
    matching  = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    test_dir  = os.path.join(DATA_DIR, 'test')

    result = subprocess.run(
        [sys.executable, validator,
         '--matching', matching, '--candidate', candidate, '--test-dir', test_dir],
        capture_output=True, text=True, cwd=BASE_DIR
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode == 0:
        log.info("✅ PASS — safe to upload!")
    else:
        log.error("❌ FAIL — fix errors before submitting.")
    return result.returncode == 0


# ════════════════════════════════════════════════════════════════════════════
# EDA
# ════════════════════════════════════════════════════════════════════════════

def run_eda():
    """Quick EDA + blocking recall check."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(SRC_DIR, '00_eda.py')], cwd=BASE_DIR)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Amazon ML Challenge 2026 Pipeline')
    parser.add_argument(
        '--step',
        choices=['all', 'block_train', 'block_test', 'eda', 'train', 'predict', 'score', 'check'],
        default='all',
    )
    parser.add_argument('--split', default='test', choices=['train', 'test'],
                        help='Data split for predict step')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override decision threshold for predict step')
    parser.add_argument('--no-tfidf', action='store_true',
                        help='Skip TF-IDF blocking (fast first submission). '
                             'Token+Address+Prefix+Bigram still give good recall.')
    args = parser.parse_args()

    skip_tfidf = args.no_tfidf
    step = args.step
    log.info(f"\n🚀 Amazon ML Challenge 2026 — Step: {step.upper()}")
    if skip_tfidf:
        log.info("⚡ Mode: FAST (no TF-IDF)")
    t0 = time.time()

    if step == 'all':
        run_blocking('train', 'candidate_pairs_train.tsv', skip_tfidf=skip_tfidf)
        run_blocking('test',  'candidate_pairs.tsv',       skip_tfidf=skip_tfidf)
        run_training()
        run_prediction('test', args.threshold)
        run_format_check()

    elif step == 'block_train':
        run_blocking('train', 'candidate_pairs_train.tsv', skip_tfidf=skip_tfidf)

    elif step == 'block_test':
        run_blocking('test', 'candidate_pairs.tsv', skip_tfidf=skip_tfidf)

    elif step == 'eda':
        run_eda()

    elif step == 'train':
        run_training()

    elif step == 'predict':
        run_prediction(args.split, args.threshold)

    elif step == 'score':
        run_scoring()

    elif step == 'check':
        run_format_check()

    log.info(f"\n⏱  Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
