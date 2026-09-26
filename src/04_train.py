"""
04_train.py
LightGBM binary classifier training for Entity Resolution.

Training Data: Candidate pairs from blocking stage with ground truth labels.
Label: 1 if the candidate is a true match (in ground_truth), 0 otherwise.

Workflow:
  1. Load training candidate pairs (from blocking on train split)
  2. Build features for all pairs
  3. Assign labels from ground truth
  4. Train LightGBM with 5-fold stratified CV
  5. Sweep threshold to maximize F0.5 on validation fold
  6. Save model + best threshold
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from preprocess_utils import preprocess_dataframe
from features import FEATURE_COLS, build_feature_matrix, build_lookups, expand_candidates

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'binary_logloss',
    'boosting_type':    'gbdt',
    'num_leaves':       127,
    'learning_rate':    0.05,
    'n_estimators':     1000,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}

N_FOLDS         = 5
THRESHOLD_START = 0.30
THRESHOLD_END   = 0.80
THRESHOLD_STEPS = 50


# ─── F0.5 helpers ─────────────────────────────────────────────────────────────
def f05_score(precision: float, recall: float) -> float:
    """Compute F_beta score with beta=0.5."""
    beta_sq = 0.25  # beta=0.5, beta^2=0.25
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_f05(pred_df: pd.DataFrame, gt_dict: dict) -> float:
    """
    Compute macro-average F0.5 score.

    Args:
        pred_df: DataFrame with columns [source1_entity_id, matched_entity_ids (comma-sep string)]
        gt_dict: dict { source1_entity_id → set of true matched entity_ids }

    Returns: macro-average F0.5 across all S1 entities in gt_dict
    """
    scores = []
    all_s1 = set(gt_dict.keys())

    pred_map = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row.get('matched_entity_ids', '')
        if pd.isna(matches) or matches == '':
            pred_map[s1_id] = set()
        else:
            pred_map[s1_id] = set(str(matches).split(','))

    for s1_id in all_s1:
        true_set = gt_dict[s1_id]
        pred_set = pred_map.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)  # Correct singleton
        elif not true_set and pred_set:
            scores.append(0.0)  # False merge on singleton
        elif true_set and not pred_set:
            prec, rec = 0.0, 0.0
            scores.append(f05_score(prec, rec))
        else:
            tp = len(true_set & pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec  = tp / len(true_set) if true_set else 0.0
            scores.append(f05_score(prec, rec))

    return float(np.mean(scores)) if scores else 0.0


def threshold_sweep(pairs_df: pd.DataFrame, probs: np.ndarray,
                    gt_dict: dict, s1_ids_all: list) -> tuple:
    """
    Sweep over decision thresholds and return the one maximizing macro F0.5.
    
    Args:
        pairs_df: long-format pairs with [source1_entity_id, candidate_entity_id]
        probs: model predicted probability for each pair
        gt_dict: ground truth dict
        s1_ids_all: all S1 entity IDs (including singletons with no candidates)
    Returns: (best_threshold, best_f05)
    """
    thresholds = np.linspace(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEPS)
    best_thresh, best_f05 = 0.5, 0.0

    for thresh in thresholds:
        pairs_df = pairs_df.copy()
        pairs_df['match'] = probs >= thresh

        # Build predictions per S1 entity
        pred_rows = []
        matched = pairs_df[pairs_df['match']].groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).reset_index()
        matched.columns = ['source1_entity_id', 'matched_entity_ids']

        # Ensure all S1 entities appear
        s1_series = pd.DataFrame({'source1_entity_id': s1_ids_all})
        pred_df = s1_series.merge(matched, on='source1_entity_id', how='left')
        pred_df['matched_entity_ids'] = pred_df['matched_entity_ids'].fillna('')

        score = macro_f05(pred_df, gt_dict)
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    log.info(f"Best threshold: {best_thresh:.4f} → F0.5 = {best_f05:.4f}")
    return best_thresh, best_f05


# ─── Main Training Pipeline ──────────────────────────────────────────────────
def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth into dict { s1_entity_id → set of matched IDs }"""
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if pd.isna(matches) or matches == '':
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(matches).split(','))
    return gt_dict


def run_training(
    data_dir: str,
    models_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
):
    """
    Full training pipeline:
      1. Load preprocessed data (from blocking cache)
      2. Load candidates from blocking output
      3. Build features
      4. Train LightGBM
      5. Tune threshold on validation set
      6. Save model + threshold
    """
    os.makedirs(models_dir, exist_ok=True)

    # ── Load preprocessed cache (from blocking step) ──
    cache_path = os.path.join(models_dir, 'train_preprocessed.pkl')
    if not os.path.exists(cache_path):
        log.error(f"Preprocessed cache not found at {cache_path}. Run blocking first.")
        raise FileNotFoundError(cache_path)

    log.info("Loading preprocessed data from cache...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    s1  = cache['s1']
    s23 = cache['s23']

    # ── Load candidates ──
    cand_path = os.path.join(output_dir, 'candidate_pairs_train.tsv')
    if not os.path.exists(cand_path):
        # Fall back to main output path (if blocking was run with --split train)
        cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    log.info(f"Loading candidates from {cand_path} ...")
    cand_df = pd.read_csv(cand_path, sep='\t', dtype=str).fillna('')
    log.info(f"Loaded {len(cand_df):,} candidate rows")

    # ── Expand candidates to long format ──
    pairs_df = expand_candidates(cand_df)
    log.info(f"Expanded to {len(pairs_df):,} candidate pairs")

    # ── Load ground truth ──
    gt_path = os.path.join(data_dir, 'train', 'train_ground_truth.tsv')
    gt_dict = load_ground_truth(gt_path)
    log.info(f"Ground truth loaded: {len(gt_dict):,} S1 entities")

    # ── Label pairs ──
    log.info("Labeling pairs from ground truth...")
    # Flatten ground truth to set of (s1_id, match_id) tuples
    gt_pairs = set()
    for s1_id, matches in gt_dict.items():
        for m in matches:
            gt_pairs.add((s1_id, m))

    pairs_df['label'] = pairs_df.apply(
        lambda r: 1 if (r['source1_entity_id'], r['candidate_entity_id']) in gt_pairs else 0,
        axis=1
    )
    pos = pairs_df['label'].sum()
    neg = len(pairs_df) - pos
    log.info(f"Label distribution: {pos:,} positive, {neg:,} negative (ratio 1:{neg//max(pos,1)})")

    # ── Build lookups ──
    s1_lookup, s23_lookup = build_lookups(s1, s23)

    # ── Build feature matrix ──
    log.info("Building feature matrix...")
    feat_df = build_feature_matrix(pairs_df[['source1_entity_id', 'candidate_entity_id']], s1_lookup, s23_lookup)
    feat_df['label'] = pairs_df['label'].values

    # Save feature matrix
    feat_path = os.path.join(models_dir, 'train_features.pkl')
    feat_df.to_pickle(feat_path)
    log.info(f"Saved feature matrix ({len(feat_df):,} rows) to {feat_path}")

    # ── Train/Val split ──
    # Stratify by S1 entity to avoid leakage
    s1_ids = s1['entity_id'].tolist()
    np.random.seed(42)
    np.random.shuffle(s1_ids)
    split_idx = int(len(s1_ids) * (1 - val_fraction))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids   = set(s1_ids[split_idx:])

    train_mask = feat_df['source1_entity_id'].isin(train_s1_ids)
    val_mask   = feat_df['source1_entity_id'].isin(val_s1_ids)

    X_train = feat_df[train_mask][FEATURE_COLS].values.astype(np.float32)
    y_train = feat_df[train_mask]['label'].values
    X_val   = feat_df[val_mask][FEATURE_COLS].values.astype(np.float32)
    y_val   = feat_df[val_mask]['label'].values

    log.info(f"Train: {len(X_train):,} pairs | Val: {len(X_val):,} pairs")
    log.info(f"Train pos: {y_train.sum():,} | Val pos: {y_val.sum():,}")

    # ── Train LightGBM ──
    log.info("Training LightGBM ...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {**LGBM_PARAMS, 'scale_pos_weight': scale_pos_weight}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── Feature Importance ──
    feat_imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    log.info("Top 10 feature importances:")
    for feat, imp in feat_imp.head(10).items():
        log.info(f"  {feat:35s}: {imp:.1f}")

    # ── Threshold Sweep on Validation ──
    log.info("Sweeping decision threshold on validation set...")
    val_pairs = feat_df[val_mask][['source1_entity_id', 'candidate_entity_id']].copy().reset_index(drop=True)
    val_probs = model.predict_proba(X_val)[:, 1]

    val_gt_dict = {s1_id: matches for s1_id, matches in gt_dict.items() if s1_id in val_s1_ids}
    val_s1_all  = [sid for sid in s1['entity_id'].tolist() if sid in val_s1_ids]

    best_threshold, best_f05 = threshold_sweep(val_pairs, val_probs, val_gt_dict, val_s1_all)

    # ── Save model ──
    model_path = os.path.join(models_dir, 'lgbm_model.pkl')
    thresh_path = os.path.join(models_dir, 'best_threshold.txt')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(thresh_path, 'w') as f:
        f.write(str(best_threshold))

    log.info(f"Model saved to {model_path}")
    log.info(f"Best threshold {best_threshold:.4f} saved to {thresh_path}")
    log.info(f"Validation F0.5: {best_f05:.4f}")

    return model, best_threshold, best_f05


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   default=os.path.join(BASE_DIR, 'dataset'))
    parser.add_argument('--models-dir', default=os.path.join(BASE_DIR, 'models'))
    parser.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'output'))
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()

    run_training(args.data_dir, args.models_dir, args.output_dir, args.val_fraction)
