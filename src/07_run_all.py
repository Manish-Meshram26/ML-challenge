"""
07_run_all.py
Master end-to-end pipeline runner for the Amazon ML Challenge 2026.

Usage:
  # Full pipeline — blocking + training + prediction on test set
  python src/07_run_all.py --mode full

  # Blocking only (on train split for training)
  python src/07_run_all.py --mode block --split train

  # Blocking on test split
  python src/07_run_all.py --mode block --split test

  # Train model (requires train blocking to be done)
  python src/07_run_all.py --mode train

  # Predict on test set (requires test blocking + trained model)
  python src/07_run_all.py --mode predict

  # Validate predictions against ground truth
  python src/07_run_all.py --mode validate
"""

import os
import sys
import logging
import argparse
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def run_blocking_train():
    """Run blocking on training data to generate candidates for model training."""
    log.info("\n" + "="*60)
    log.info("STEP 1: Blocking on TRAIN split")
    log.info("="*60)
    from blocking import run_blocking
    candidates, s1, s23 = run_blocking('train', DATA_DIR, OUTPUT_DIR, MODELS_DIR)

    # Rename output for train split
    import shutil
    src = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    dst = os.path.join(OUTPUT_DIR, 'candidate_pairs_train.tsv')
    if os.path.exists(src):
        shutil.copy(src, dst)
        log.info(f"Copied train candidates to {dst}")
    return candidates


def run_blocking_test():
    """Run blocking on test data to generate candidates for inference."""
    log.info("\n" + "="*60)
    log.info("STEP 2: Blocking on TEST split")
    log.info("="*60)
    from blocking import run_blocking
    candidates, s1, s23 = run_blocking('test', DATA_DIR, OUTPUT_DIR, MODELS_DIR)
    return candidates


def run_training():
    """Train the LightGBM model."""
    log.info("\n" + "="*60)
    log.info("STEP 3: Training LightGBM model")
    log.info("="*60)
    from train import run_training as _train
    model, threshold, f05 = _train(DATA_DIR, MODELS_DIR, OUTPUT_DIR)
    log.info(f"Training complete. Validation F0.5: {f05:.4f}, Threshold: {threshold:.4f}")
    return model, threshold, f05


def run_prediction():
    """Run inference on test set."""
    log.info("\n" + "="*60)
    log.info("STEP 4: Predicting on TEST split")
    log.info("="*60)
    from predict import run_prediction as _predict
    results, feat_df = _predict('test', DATA_DIR, MODELS_DIR, OUTPUT_DIR)
    return results


def run_validation():
    """Score predictions on train validation split."""
    log.info("\n" + "="*60)
    log.info("STEP 5: Validating predictions")
    log.info("="*60)
    from validate import compute_macro_f05

    pred_path = os.path.join(OUTPUT_DIR, 'matching_results_train.tsv')
    gt_path   = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

    if not os.path.exists(pred_path):
        log.error(f"No validation predictions found at {pred_path}. "
                  f"Run predict with --split train first.")
        return None

    results = compute_macro_f05(pred_path, gt_path)
    log.info(f"\n  MACRO F0.5: {results['macro_f05']:.6f}")
    log.info(f"  Precision: {results['global_precision']:.4f}")
    log.info(f"  Recall:    {results['global_recall']:.4f}")
    return results['macro_f05']


def run_format_check():
    """Run the official submission validator."""
    log.info("\n" + "="*60)
    log.info("STEP 6: Validating submission format")
    log.info("="*60)
    validator = os.path.join(BASE_DIR, 'utils', 'validate_submission.py')
    matching  = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    candidate = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    test_dir  = os.path.join(DATA_DIR, 'test')

    result = subprocess.run(
        [sys.executable, validator,
         '--matching', matching,
         '--candidate', candidate,
         '--test-dir', test_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode == 0:
        log.info("✅ Submission format PASSED!")
    else:
        log.error("❌ Submission format FAILED. Fix issues before submitting.")
        print(result.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Amazon ML Challenge 2026 — Pipeline Runner')
    parser.add_argument(
        '--mode',
        choices=['full', 'block', 'train', 'predict', 'validate', 'check'],
        default='full',
        help='Pipeline mode to run'
    )
    parser.add_argument('--split', default='test', choices=['train', 'test'],
                        help='Data split (for block mode)')
    args = parser.parse_args()

    start_time = time.time()
    log.info(f"\n🚀 Amazon ML Challenge 2026 Pipeline — Mode: {args.mode.upper()}")

    if args.mode == 'full':
        run_blocking_train()
        run_blocking_test()
        run_training()
        run_prediction()
        run_format_check()

    elif args.mode == 'block':
        if args.split == 'train':
            run_blocking_train()
        else:
            run_blocking_test()

    elif args.mode == 'train':
        run_training()

    elif args.mode == 'predict':
        run_prediction()

    elif args.mode == 'validate':
        run_validation()

    elif args.mode == 'check':
        run_format_check()

    elapsed = time.time() - start_time
    log.info(f"\n✅ Pipeline complete in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
