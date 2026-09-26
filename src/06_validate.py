"""
06_validate.py
Local F0.5 scorer — run this on your validation split BEFORE every leaderboard submission.

Usage:
  python src/06_validate.py \
    --pred output/matching_results_train.tsv \
    --gt dataset/train/train_ground_truth.tsv \
    --s1 dataset/train/train_source1.tsv

This script implements the EXACT same macro-average F0.5 formula as the leaderboard.
Use it to iterate without burning precious submissions.
"""

import sys
import argparse
import numpy as np
import pandas as pd


def f05_score(precision: float, recall: float) -> float:
    """F_beta with beta=0.5: weights precision 2x over recall."""
    beta_sq = 0.25
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_macro_f05(pred_path: str, gt_path: str, s1_path: str = None) -> dict:
    """
    Compute macro-average F0.5 score.
    
    Args:
        pred_path: path to matching_results.tsv (your predictions)
        gt_path:   path to train_ground_truth.tsv
        s1_path:   path to source1.tsv (to ensure all S1 entities are included)
    
    Returns: dict with score breakdown
    """
    # Load ground truth
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str).fillna('')
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).strip()
        if not matches:
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matches.split(','))

    # Load predictions
    pred_df = pd.read_csv(pred_path, sep='\t', dtype=str).fillna('')
    pred_dict = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).strip()
        if not matches:
            pred_dict[s1_id] = set()
        else:
            pred_dict[s1_id] = set(matches.split(','))

    # Compute per-entity scores
    scores = []
    tp_total, fp_total, fn_total = 0, 0, 0
    singleton_correct = 0
    singleton_false_merge = 0
    per_entity_scores = {}

    # Use GT keys as the evaluation universe
    all_s1 = list(gt_dict.keys())

    for s1_id in all_s1:
        true_set = gt_dict[s1_id]
        pred_set = pred_dict.get(s1_id, set())

        if not true_set and not pred_set:
            # Correct singleton prediction
            entity_score = 1.0
            singleton_correct += 1
        elif not true_set and pred_set:
            # False merge on a singleton
            entity_score = 0.0
            singleton_false_merge += 1
            fp_total += len(pred_set)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec  = tp / len(true_set) if true_set else 0.0
            entity_score = f05_score(prec, rec)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        scores.append(entity_score)
        per_entity_scores[s1_id] = entity_score

    macro_f05 = float(np.mean(scores)) if scores else 0.0

    # Global precision/recall for reference
    global_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    global_rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    # Bottom-10 worst predictions
    worst = sorted(per_entity_scores.items(), key=lambda x: x[1])[:10]

    return {
        'macro_f05':            macro_f05,
        'global_precision':     global_prec,
        'global_recall':        global_rec,
        'total_entities':       len(all_s1),
        'singleton_correct':    singleton_correct,
        'singleton_false_merge': singleton_false_merge,
        'tp':                   tp_total,
        'fp':                   fp_total,
        'fn':                   fn_total,
        'worst_entities':       worst,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute local F0.5 score')
    parser.add_argument('--pred', required=True, help='Path to matching_results.tsv')
    parser.add_argument('--gt',   required=True, help='Path to train_ground_truth.tsv')
    parser.add_argument('--s1',   default=None,  help='Path to source1.tsv (optional)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Amazon ML Challenge 2026 — Local F0.5 Evaluator")
    print("="*60)

    results = compute_macro_f05(args.pred, args.gt, args.s1)

    print(f"\n  📊 MACRO F0.5 SCORE:  {results['macro_f05']:.6f}")
    print(f"\n  Global Precision:     {results['global_precision']:.4f}")
    print(f"  Global Recall:        {results['global_recall']:.4f}")
    print(f"\n  Total S1 entities:    {results['total_entities']:,}")
    print(f"  True Positives (TP):  {results['tp']:,}")
    print(f"  False Positives (FP): {results['fp']:,}")
    print(f"  False Negatives (FN): {results['fn']:,}")
    print(f"\n  Singletons correct:   {results['singleton_correct']:,}")
    print(f"  Singletons w/ false merge: {results['singleton_false_merge']:,}")

    print(f"\n  ⚠  Worst-10 entities (lowest F0.5):")
    for s1_id, score in results['worst_entities']:
        print(f"     {s1_id}  →  {score:.4f}")

    print("\n" + "="*60 + "\n")
    return results['macro_f05']


if __name__ == '__main__':
    score = main()
    # Exit with non-zero code if score is very low (useful for CI)
    sys.exit(0 if score > 0.0 else 1)
