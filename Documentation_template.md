# Amazon ML Challenge 2026: optimized business entity resolution

Team name and members: **fill in before submission**.

## Approach

The solution matches Source 2/3 records to Source 1 references using country-aware retrieval followed by a two-stage LightGBM classifier. It uses only the supplied data. No business databases, geocoding, APIs, or external record lookup are used. Generic local text libraries provide normalization and transliteration.

The metric is entity-macro F0.5: `1.25 * TP / (0.25 * true_count + predicted_count)`. A reference with no true or predicted matches scores one; a false merge on a singleton scores zero. Therefore threshold selection uses complete per-reference candidate sets and includes missed retrievals as false negatives.

## Validation and data handling

Training contains 2,206,821 references and 10,320,219 Source 2/3 records. We deterministically sample 17,720 reference entities to fit the workstation/time budget, while scanning **all** Source 2/3 records. Groups defined by normalized reference name, address and country separate 10,647 fitting, 3,454 tuning, and 3,619 audit references. Exact duplicate reference descriptions stay in one group. IDs are join keys only, never predictive features.

The final candidate sample contains 5,611,868 pairs. No negative downsampling is applied to tuning or audit candidates. Ground-truth positives missed by retrieval are injected only for fitting references, never for tuning or audit retrieval. Final refitting uses fitting plus tuning references and excludes audit references. Several development iterations were compared, so these local results should not be interpreted as a guarantee of hidden-test performance.

## Retrieval and features

Normalization handles case, punctuation, accents, scripts, legal abbreviations, website stems, and numeric leading zeros. The retrieval union includes exact/core/sorted names, rare tokens, token pairs, token fragments, compact names, prefixes/suffixes, exact and partial address keys, rare address tokens, numeric pairs, and name-number combinations. Block frequency limits are measured against the complete reference population even when evaluating sampled references.

The full classifier has 57 features: name/core/address similarities; token overlap and containment; partial and Jaro-Winkler similarities; length and missingness indicators; number agreement and contradictions; compact-name similarity; reference ambiguity frequencies; label-free source name and name/address frequencies; numeric near-miss indicators; character edit operations; and token-level fuzzy agreement. Missing values are not treated as exact agreement.

LightGBM uses 47 leaves, learning rate 0.045, regularization, up to 900 trees, and deterministic seeds. A 14-feature, 250-tree screening model reduces expensive comparisons. Its cutoff is selected on end-to-end tuning F0.5, preferring the largest cutoff within 0.0002 of the best tuning score. The selected screening threshold is 0.003 and final match threshold is 0.75.

## Measured results

| Measurement | Result |
|---|---:|
| First experiment, validation-model audit macro F0.5 | 0.944929 |
| Expanded experiment, validation-model audit macro F0.5 | 0.961092 |
| Final refitted model, audit macro F0.5 | **0.961507** |
| Final audit macro F0.5, India | 0.948746 |
| Final audit macro F0.5, US | 0.969950 |
| Audit retrieval true-pair recall | 0.978301 |
| Ideal-classifier audit score using retrieved pairs | 0.992878 |
| Ideal-classifier audit score after screening | 0.991081 |
| Candidate fraction retained for expensive scoring | 4.045% |

The user's reported earlier leaderboard score (~0.65) was not reproduced on the hidden test set and is not directly comparable to these local measurements. **A 0.99 submission score has not been established.** France occurs only in test, so no labeled French score is available. Remaining errors include near-duplicate decoys, ambiguous names, incomplete addresses, and transliteration differences.

Regional thresholds produced negligible tuning gain and were rejected. Lowering the first-match threshold also produced no gain and was rejected. Full training labels show unique ownership of Source 2/3 matches, but most inspected false positives were unowned decoys, so a simple ownership constraint was not adopted.

## Execution and artifacts

`run_pipeline.py` is the entry point. `requirements.txt` lists dependencies and `README.md` gives commands and design details. Normalized source caches, bounded reference shards, batched multi-threaded RapidFuzz kernels, float32 features, and SQLite output grouping control runtime and memory. Checkpoint fingerprints prevent mixing models/configurations.

Outputs are written separately under `output/`; the original solution remains intact. `matching_results.tsv` contains one row per test reference, with empty lists where appropriate. `candidate_pairs.tsv` contains the retained candidate set. Submission verification status and final output counts are recorded separately after inference completes.
