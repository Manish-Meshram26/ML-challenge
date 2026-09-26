"""
00_eda.py
Exploratory Data Analysis for the Amazon ML Challenge 2026 dataset.
Run this FIRST to understand the data before building the pipeline.

Outputs summary statistics to console and saves key findings.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, 'dataset')


def load_all(split='train'):
    """Load all source files for a given split."""
    prefix = f"{split}_source"
    d = os.path.join(DATA_DIR, split)
    s1 = pd.read_csv(os.path.join(d, f'{prefix}1.tsv'), sep='\t', dtype=str).fillna('')
    s2 = pd.read_csv(os.path.join(d, f'{prefix}2.tsv'), sep='\t', dtype=str).fillna('')
    s3 = pd.read_csv(os.path.join(d, f'{prefix}3.tsv'), sep='\t', dtype=str).fillna('')
    return s1, s2, s3


def analyze_ground_truth():
    """Analyze the ground truth matching distribution."""
    gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')
    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')

    print("\n" + "="*60)
    print("GROUND TRUTH ANALYSIS")
    print("="*60)

    match_counts = gt['matched_entity_ids'].apply(
        lambda x: 0 if not x.strip() else len(x.split(','))
    )

    print(f"Total S1 entities in GT:     {len(gt):,}")
    print(f"Singletons (0 matches):      {(match_counts == 0).sum():,} ({(match_counts==0).mean()*100:.1f}%)")
    print(f"Entities with ≥1 match:      {(match_counts > 0).sum():,} ({(match_counts>0).mean()*100:.1f}%)")
    print(f"\nMatch count distribution:")
    print(f"  Mean:    {match_counts.mean():.2f}")
    print(f"  Median:  {match_counts.median():.1f}")
    print(f"  Max:     {match_counts.max()}")
    print(f"  95th %:  {np.percentile(match_counts, 95):.1f}")

    print(f"\nTop match count frequencies:")
    for count, freq in Counter(match_counts).most_common(10):
        print(f"  {count} match(es): {freq:,} entities")

    # Source breakdown
    all_match_ids = []
    for mids in gt['matched_entity_ids']:
        if mids.strip():
            all_match_ids.extend(mids.split(','))

    s2_matches = sum(1 for m in all_match_ids if m.startswith('S2-'))
    s3_matches = sum(1 for m in all_match_ids if m.startswith('S3-'))
    print(f"\nTotal match links:  {len(all_match_ids):,}")
    print(f"  from Source 2:    {s2_matches:,} ({s2_matches/max(len(all_match_ids),1)*100:.1f}%)")
    print(f"  from Source 3:    {s3_matches:,} ({s3_matches/max(len(all_match_ids),1)*100:.1f}%)")

    return gt, match_counts


def analyze_source(df: pd.DataFrame, name: str):
    """Analyze a source dataframe."""
    print(f"\n{'='*60}")
    print(f"SOURCE {name} ANALYSIS")
    print(f"{'='*60}")
    print(f"Total records:        {len(df):,}")
    print(f"Missing business_name: {(df['business_name']=='').sum():,} ({(df['business_name']=='').mean()*100:.1f}%)")
    print(f"Missing address:      {(df['business_address']=='').sum():,} ({(df['business_address']=='').mean()*100:.1f}%)")
    print(f"Missing country:      {(df['country']=='').sum():,}")

    print(f"\nCountry distribution:")
    for country, count in df['country'].value_counts().items():
        print(f"  {country:15s}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\nBusiness name length (chars):")
    name_lens = df['business_name'].str.len()
    print(f"  Mean: {name_lens.mean():.1f} | Median: {name_lens.median():.1f} | Max: {name_lens.max()}")

    print(f"\nAddress length (chars):")
    addr_lens = df['business_address'].str.len()
    print(f"  Mean: {addr_lens.mean():.1f} | Median: {addr_lens.median():.1f} | Max: {addr_lens.max()}")

    print(f"\nSample records:")
    for _, row in df.sample(min(5, len(df)), random_state=42).iterrows():
        print(f"  [{row['entity_id']}] {row['business_name'][:50]:50s} | {row['business_address'][:40]:40s} | {row['country']}")


def check_blocking_recall(gt_path: str, cand_path: str):
    """
    Measure blocking recall: what fraction of true matches appear in candidates?
    Run AFTER blocking to verify your recall ceiling is high enough.
    """
    print("\n" + "="*60)
    print("BLOCKING RECALL ANALYSIS")
    print("="*60)

    gt = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    cand = pd.read_csv(cand_path, sep='\t', dtype=str).fillna('')

    # Build candidate set as (s1_id, match_id) pairs
    cand_pairs = set()
    for _, row in cand.iterrows():
        s1_id = row['source1_entity_id']
        cids = str(row.get('candidate_entity_ids', '')).strip()
        if cids:
            for cid in cids.split(','):
                cand_pairs.add((s1_id, cid.strip()))

    # Check how many GT pairs are in candidates
    found, total = 0, 0
    missed_examples = []
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        mids = str(row['matched_entity_ids']).strip()
        if not mids:
            continue
        for mid in mids.split(','):
            total += 1
            if (s1_id, mid) in cand_pairs:
                found += 1
            elif len(missed_examples) < 5:
                missed_examples.append((s1_id, mid))

    recall = found / total if total > 0 else 0.0
    print(f"Ground truth pairs:    {total:,}")
    print(f"Found in candidates:   {found:,}")
    print(f"Blocking recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"Missed pairs:          {total - found:,}")

    if missed_examples:
        print(f"\nExample missed pairs:")
        for s1_id, mid in missed_examples:
            print(f"  S1={s1_id}  →  {mid}")

    return recall


if __name__ == '__main__':
    print("\n🔍 Amazon ML Challenge 2026 — EDA Report")
    print("="*60)

    # Load train split
    log.info("Loading training data...")
    s1, s2, s3 = load_all('train')

    # Analyze ground truth
    gt, match_counts = analyze_ground_truth()

    # Analyze each source
    analyze_source(s1, '1')
    analyze_source(s2, '2')
    analyze_source(s3, '3')

    # Check blocking recall if candidate file exists
    cand_path_train = os.path.join(BASE_DIR, 'output', 'candidate_pairs_train.tsv')
    cand_path_main  = os.path.join(BASE_DIR, 'output', 'candidate_pairs.tsv')
    gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

    for cand_path in [cand_path_train, cand_path_main]:
        if os.path.exists(cand_path):
            check_blocking_recall(gt_path, cand_path)
            break
    else:
        print("\n[INFO] No candidate file found yet — run blocking first to check recall.")

    print("\n✅ EDA complete.")
