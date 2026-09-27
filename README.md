# Entity-resolution optimization

This is the consolidated optimized implementation. The final refitted model achieved **0.961507 macro F0.5 on 3,619 local audit references**. The first experiment achieved 0.944929. These are local measurements, not leaderboard scores; a hidden-test score of 0.99 has not been established. Exact measurements are in `output/.cache/metrics.json`.

## Run

The active project tree is:

```text
run_pipeline.py                 # optimized training, prediction, resume, checks
requirements.txt
README.md
Documentation_template.md       # methodology and measured validation results
utils/validate_submission.py    # official checker plus streaming verification
dataset/                        # supplied data, excluded from Git
models/model.pkl                # trained model, excluded from Git
output/matching_results.tsv     # leaderboard upload after inference completes
output/candidate_pairs.tsv      # candidate audit file
output/.cache/                  # checkpoints, logs, metrics, temporary data
```

Prediction automatically runs exhaustive output verification after assembly. A successful run writes `output/.cache/submission_verification.json` with `passed: true` and hashes of the two TSVs. Until that exists for the current files, the optimized submission is not ready. The earlier submission is retained only as an ignored backup in the cache during generation.

Use Python with numpy, pandas, scikit-learn, lightgbm, rapidfuzz, and text-unidecode installed. This workstation has LightGBM and RapidFuzz in `.deps`, which the script discovers automatically.

```powershell
python run_pipeline.py --step selftest
python run_pipeline.py --step prepare --sample-rate 0.008
python run_pipeline.py --step fit --threads 8
python run_pipeline.py --step predict --threads 4 --country-workers 2 --shard-size 1000000
python run_pipeline.py --step check
```

Use an absolute script path if the shell starts outside this folder. All data and output paths are resolved relative to the script, regardless of the working directory. `--resume` resumes completed inference shards only with the same model and configuration; a fingerprint rejects incompatible checkpoints. Final submission files are written directly to `output/`. Default production shard size is 1,000,000 reference entities. `--step diagnose` exports development-set errors after fitting.

The current workstation run uses a limit of 1,000,000 references, allowing one index per test country. The complete India index initially used approximately 3.5 GB process RAM on this workstation. To recover an interrupted run, repeat the same prediction command with `--resume`; completed shards are reused and the interrupted shard restarts. Do not change the model, script, threshold, or shard size while resuming. With the supplied trained `models/model.pkl`, skip preparation and fitting and run prediction directly. Keep the script beside `dataset/` as in this project.

The streaming verifier checks both output files, including target ID existence and matched-subset-of-candidates. It loads the valid target ID set but streams output rows to avoid holding every candidate set in memory. The original official validator remains available in `utils/validate_submission.py`. Passing a verifier proves format correctness, not prediction accuracy.

Resume an interrupted production run with `python run_pipeline.py --step predict --threads 4 --country-workers 2 --shard-size 1000000 --resume`. Two country workers share the machine; the four-thread and eight-thread benchmark produced byte-identical predictions and candidates. Use `--country-workers 1` if memory is constrained. Checkpoints commit every 50,000 source records, so each incomplete country resumes at its last committed position. Keep the machine running until completion. Large datasets, model files, dependencies, caches, and generated TSVs are excluded from Git; submit the TSV directly to the competition. Active parallel progress is logged in `output/.cache/parallel.log`.

## What is being optimized, and why

1. **Validation fidelity.** The old code removes most validation negatives before tuning the threshold. This makes the decision problem artificially easy. The replacement samples reference entities, then scans *all* Source 2/3 records and retains every candidate surviving the fixed blocking/filter stage. Reference groups split into 60% fit, 20% threshold tuning, and 20% audit. Exact duplicate reference name/address/country combinations stay together. IDs are used only for joins and deterministic selection, never as model features.
2. **Candidate coverage.** Country-specific inverted indexes combine complete names, names without legal suffixes, sorted name tokens, rare name tokens, name prefixes/suffixes, token fragments, complete addresses, address prefixes, rare address words, adjacent address words, and numeric-plus-word address keys. Address retrieval handles renamed businesses and translated names. Full-reference block frequencies are used even when training only on sampled reference entities, preserving the production retrieval policy. Very common blocks are discarded to control the comparison count.
3. **Measured retrieval ceiling.** Missing true candidates remain false negatives in the macro metric. The report includes true-pair recall and the score an ideal classifier could achieve using the retrieved candidates. Ground-truth injection is permitted only for fitting entities; it never supplies validation or audit candidates.
4. **Text normalization.** Strip punctuation, normalize whitespace/case, normalize legal abbreviations, transliterate scripts, retain useful domain stems, normalize numeric leading zeros, and compare compact names. This fixes the earlier accent handling that split decomposed accented letters. The provided source records are the only data source; no geocoding, business databases, or external record lookup is used.
5. **Features that separate evidence from absence.** Name, core-name, and address ratios, token sorting, token containment, partial matching, Jaro-Winkler similarity, Jaccard similarity, length ratios, exact agreement and missingness are combined with numeric overlap/conflict, leading number agreement, compact-name similarity, and name/address interaction. Two absent fields do not constitute agreement. Numeric disagreements are learned signals, not absolute rejection rules, because the training examples include address variation. Country-specific reference name/core/address frequencies tell the model how ambiguous an apparent exact match is. Label-free source name and name/address frequencies add corroboration evidence; unmatched-number deltas, near-miss numbers, character edit operations, and token-level similarity address observed decoys. The expanded model uses 57 features.
6. **Precision-sensitive learning.** Train a regularized LightGBM classifier without artificially amplifying positives. Early stopping chooses tree count; a broad threshold search optimizes the actual entity-macro F0.5 on the tuning entities. Report countries separately to expose distribution differences. France has no labeled training examples, so its hidden-test performance cannot be directly validated.
7. **Memory and speed.** Stream Source 2/3 rather than storing millions of nested Python records; write pair features in binary float32 batches; use memory-mapped base features (training builds an augmented matrix for the reference sample). Cache normalized records in bounded chunks for repeated shard scans. The reference shard size is configurable. Batch similarity kernels run in RapidFuzz's C++ implementation across CPU threads. A 14-feature screening model reduces the number of pairs needing the full model. Its initial cutoff targets 99.95% tuning-positive retention; the final cutoff is selected by end-to-end tuning F0.5, preferring the highest tested cutoff within 0.0002 of the best score. Actual audit retention and score are reported, rather than assuming the screening model never misses a true pair. SQLite groups the official output files without storing all candidates in RAM. Avoid dense all-pairs TF-IDF and the original 4 GB pickle cache. Source-frequency dictionaries and reference indexes still consume memory; this is not a constant-memory algorithm.
8. **Reproducibility.** Save the feature schema, sampled reference records, truth, deterministic splits, model, decision threshold, audit errors and metrics. Keep a model trained without audit entities. The reported validation model and final refit model are distinguished.
9. **Submission correctness.** Produce exactly one row for every test reference, including empty match sets. Deduplicate IDs and include all accepted pairs in the candidate file. Run the supplied validator before submission.

## Further experiments, contingent on measured errors and time

- Add retrieval routes only where missed-truth analysis demonstrates a useful gain: typo-tolerant character grams, compact names, address fragments, or phonetic keys.
- Mine difficult negative pairs rather than uniformly removing negatives.
- Test state/region abbreviation normalization learned or specified without external record lookups.
- Test a second model or ensemble only if it improves tuning F0.5 and remains computationally feasible.
- Test match-set confidence or competing-reference margins only after checking whether references have exclusive ownership of Source 2/3 records. Do not assume one-to-one matching.
- Test cautious graph propagation for aliases only with safeguards and independent validation; uncontrolled connected components can create severe false merges.
- Increase the training-reference sample if learning curves justify the runtime.
- Use separate development and final audit samples if repeated error-driven tuning is performed, to avoid claiming a repeatedly consulted set is untouched.

## Time budget

The intended order is correctness and profiling, sampled end-to-end validation, focused error analysis, production inference, then output verification. Runtime must be estimated from observed throughput, not promised from hardware alone. Large neural fine-tuning, exhaustive hyperparameter sweeps, and dense similarity matrices are low-priority within seven hours on this 16 GB / 4 GB GPU workstation.

## Metric

For each reference, with `T` true matches, `P` predicted matches and `TP` correct predictions:

`F0.5 = 1.25 * TP / (0.25 * T + P)`.

When `T=P=0`, score 1; when `T=0, P>0`, score 0. Average equally across all evaluated references. A confidence interval from the audit sample measures sampling uncertainty only; it does not account for unseen-country shift or repeated model selection.
