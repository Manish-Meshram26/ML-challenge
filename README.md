# Amazon ML Challenge 2026 — Entity Resolution Pipeline

## Team Setup
- **Task**: Match business records from Source 2 & Source 3 to Source 1 reference entities
- **Metric**: Macro-average F₀.₅ (precision-weighted)
- **Countries**: US, India (train) + France (test)

---

## Environment Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution

All commands run from the `student_resource/` directory.

### Step 1: Run blocking on TRAIN data
*(generates candidates for model training — ~30–60 min)*
```bash
python run_pipeline.py --step block_train
```

### Step 2: Exploratory Data Analysis (optional, run anytime)
```bash
python run_pipeline.py --step eda
```

### Step 3: Train the model
*(requires Step 1 to be done)*
```bash
python run_pipeline.py --step train
```

### Step 4: Run blocking on TEST data
*(generates candidates for test inference — ~30–60 min)*
```bash
python run_pipeline.py --step block_test
```

### Step 5: Generate test predictions
```bash
python run_pipeline.py --step predict
```

### Step 6: Score on validation split (BEFORE uploading)
```bash
python run_pipeline.py --step score
```

### Step 7: Validate submission format (official checker)
```bash
python run_pipeline.py --step check
```

### All-in-one (Steps 1–5 + 7)
```bash
python run_pipeline.py --step all
```

---

## Output Files

After running, these files will be in `output/`:
- `matching_results.tsv` — upload this to the leaderboard
- `candidate_pairs.tsv` — blocking candidates (for submission package audit)
- `matching_results_train.tsv` — validation predictions (for local scoring)

---

## Tuning the Decision Threshold

To try a different threshold without retraining:
```bash
python run_pipeline.py --step predict --threshold 0.60
```

---

## Project Structure

```
student_resource/
├── run_pipeline.py          ← Main entry point (run this!)
├── requirements.txt
├── README.md
├── features.py              ← Feature engineering (imported by run_pipeline)
├── dataset/
│   ├── train/               ← Training TSV files + ground truth
│   └── test/                ← Test TSV files
├── output/                  ← Generated outputs (matching_results, candidates)
├── models/                  ← Saved model + threshold + data caches
├── src/
│   ├── 00_eda.py            ← EDA script
│   ├── 01_preprocess.py     ← Preprocessing documentation
│   ├── preprocess_utils.py  ← Preprocessing utilities (imported everywhere)
│   ├── features.py          ← Feature computation (imported by run_pipeline)
│   ├── 04_train.py          ← Training (standalone version)
│   ├── 05_predict.py        ← Prediction (standalone version)
│   ├── 06_validate.py       ← Local F0.5 scorer
│   └── 07_run_all.py        ← Alternative runner
└── utils/
    └── validate_submission.py  ← Official format checker (provided)
```

---

## Architecture

### Two-Stage Pipeline

```
Raw Data → Preprocessing → Blocking → Candidate Pairs → Features → LightGBM → Threshold → Output
```

**Stage 1 — Blocking** (Maximize Recall):
1. TF-IDF character n-gram cosine similarity on business names
2. Inverted-index token blocking on name word tokens
3. Address token blocking (first 3 address tokens)
4. Prefix blocking (first 5 chars of normalized name)

All strategies are country-aware and their results are union-merged.

**Stage 2 — LightGBM Classifier** (Maximize F₀.₅):
- 21 pairwise features: name similarity (Jaro-Winkler, Levenshtein, Jaccard, char n-grams), address similarity, country match
- Decision threshold tuned to maximize validation F₀.₅

---

## Key Design Decisions

- **F₀.₅ = precision-heavy**: Threshold is tuned conservatively to avoid false merges
- **Country-aware blocking**: Only compare records within the same country (string match, not hard-coded)
- **No external data**: Fully self-contained — no API calls, no geocoding, no external DBs
- **Singletons handled**: Every S1 entity appears in output; correct empty predictions score 1.0
