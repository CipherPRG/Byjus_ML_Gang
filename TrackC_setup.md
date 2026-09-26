# Track C — Setup & Step-by-Step Run Guide

Track C owns the **validation harness**, **error analysis**, and **documentation/packaging** tools.
These scripts do not train or predict — they consume output from Tracks A/B/D and report on it.

---

## Files Created by Track C

| File | Purpose |
|------|---------|
| `code/business_entity_resolution/src/val_harness.py` | Per-country F0.5, TP/FP/FN, worst entities, singleton check |
| `code/business_entity_resolution/src/error_analysis.py` | Categorised FP/FN report with actionable advice |
| `utils/check_and_validate.bat` | Pre-submission wrapper (Windows) |
| `utils/check_and_validate.sh` | Pre-submission wrapper (Linux/Mac) |
| `build_package.bat` | Assembles `submission_package/` dir (Windows) |
| `build_package.sh` | Assembles `submission_package/` dir (Linux/Mac) |
| `Documentation_template.md` | Fully filled solution documentation |

---

## Prerequisites

### 1. Python environment

Python 3.9+ is required. Install all dependencies from the repo root:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Required packages:

```
numpy>=1.24
pandas>=2.0
lightgbm>=4.0
rapidfuzz>=3.0
scikit-learn>=1.3
```

> `rapidfuzz` is used by `error_analysis.py` for similarity scoring. If it is missing the script
> falls back to token Jaccard automatically — but install it for accurate category labels.

### 2. Data you need

All commands below are run from the **`student_resource/` root** (the folder containing `dataset/`,
`utils/`, `code/`).

You need at least one of:

| Dataset | Path | Used for |
|---------|------|---------|
| Full training data | `dataset/train/` | Validation against real ground truth |
| Dense sample | `sample_dense/` | Fast development iteration |

Minimum files needed:

```
dataset/train/train_ground_truth.tsv   ← ground truth (S1 entity → matched IDs)
dataset/train/train_source1.tsv        ← S1 records (has entity_id, country columns)
dataset/train/train_source2.tsv        ← S2 records (for error_analysis.py)
dataset/train/train_source3.tsv        ← S3 records (for error_analysis.py)
output/matching_results.tsv            ← predictions to evaluate (from Track D)
output/candidate_pairs.tsv            ← candidates (from Track D, needed for validate_submission)
dataset/test/                          ← test dir (needed for validate_submission)
```

Replace `dataset/train/` with `sample_dense/` everywhere if running on the sample.

### 3. Git branch

Confirm you are on the correct branch before making any changes:

```bash
git checkout track-c-validation
```

---

## Step-by-Step Run Guide

### Step 1 — Validate prediction format (always first)

Checks that `matching_results.tsv` and `candidate_pairs.tsv` meet the submission format spec.
Must print `PASS` before any leaderboard upload.

```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

Expected output: `PASS`

---

### Step 2 — Run val_harness.py (F0.5 breakdown)

Prints overall and per-country macro F0.5, TP/FP/FN counts, top-20 worst entities, and singleton analysis.

**On the full training val split (30% held-out, recommended):**

```bash
python code/business_entity_resolution/src/val_harness.py \
    --gt   dataset/train/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --s1   dataset/train/train_source1.tsv \
    --val-only
```

**On all training entities (full ground truth):**

```bash
python code/business_entity_resolution/src/val_harness.py \
    --gt   dataset/train/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --s1   dataset/train/train_source1.tsv
```

**With sample_dense instead of full data:**

```bash
python code/business_entity_resolution/src/val_harness.py \
    --gt   sample_dense/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --s1   sample_dense/train_source1.tsv \
    --val-only
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | (required) | Path to `train_ground_truth.tsv` |
| `--pred` | (required) | Path to `matching_results.tsv` to evaluate |
| `--s1` | (required) | Path to `train_source1.tsv` (for country lookup) |
| `--top` | `20` | How many worst entities to print |
| `--val-only` | off | Restrict to 30% hash-based val split (same split as `train.py`) |

**What to look for in the output:**

- Overall macro F0.5 ≥ 0.95 → GO
- Per-country F0.5: large gaps between countries indicate a country-specific issue
- Singleton wrong-match count: should be low (high = threshold too low)
- Top-20 worst entities: use these IDs to investigate FP/FN patterns in Step 3

---

### Step 3 — Run error_analysis.py (FP/FN categorisation)

Categorises every false positive and false negative into one of: `name_collision`,
`address_collision`, `near_dup`, `script_mismatch`, `singleton_fp`, `other`.
Prints actionable advice for Tracks A and B at the end.

```bash
python code/business_entity_resolution/src/error_analysis.py \
    --gt   dataset/train/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --src1 dataset/train/train_source1.tsv \
    --src2 dataset/train/train_source2.tsv \
    --src3 dataset/train/train_source3.tsv \
    --n    50 \
    --show 20
```

**With sample_dense:**

```bash
python code/business_entity_resolution/src/error_analysis.py \
    --gt   sample_dense/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --src1 sample_dense/train_source1.tsv \
    --src2 sample_dense/train_source2.tsv \
    --src3 sample_dense/train_source3.tsv
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--gt` | (required) | Path to `train_ground_truth.tsv` |
| `--pred` | (required) | Path to `matching_results.tsv` |
| `--src1/2/3` | (required) | Source TSV files (for name/address lookup) |
| `--n` | `50` | Max FP/FN pairs to collect per type |
| `--show` | `20` | How many example pairs to print per category |
| `--val-only` | off | Restrict to 30% hash-based val split |

**Error categories explained:**

| Category | Meaning | Typical fix |
|----------|---------|------------|
| `name_collision` | Same name, clearly different address | Raise threshold or add address weight |
| `address_collision` | Same address, clearly different name | Raise threshold or add name weight |
| `near_dup` | Both name and address similar, borderline | Raise thr/margin or add more features |
| `script_mismatch` | Cross-script pair (Indic vs Latin) | Check `norm.py` transliteration keys |
| `singleton_fp` | True singleton incorrectly matched | Raise threshold |
| `other` | Doesn't fit above patterns | Inspect examples manually |

---

### Step 4 — Pre-submission wrapper (combines Steps 1 + 2)

Run this wrapper before **every** leaderboard upload. It runs `validate_submission.py` then
`val_harness.py --val-only` and prints a final GO / NO-GO verdict.

**Windows:**

```bat
utils\check_and_validate.bat ^
    output\matching_results.tsv ^
    output\candidate_pairs.tsv ^
    dataset\test ^
    dataset\train\train_ground_truth.tsv ^
    dataset\train\train_source1.tsv
```

**Linux / Mac:**

```bash
bash utils/check_and_validate.sh \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    dataset/test \
    dataset/train/train_ground_truth.tsv \
    dataset/train/train_source1.tsv
```

**Argument order (positional):**

| # | Argument | Example |
|---|----------|---------|
| 1 | matching_results.tsv | `output/matching_results.tsv` |
| 2 | candidate_pairs.tsv | `output/candidate_pairs.tsv` |
| 3 | test dataset dir | `dataset/test` |
| 4 | train_ground_truth.tsv | `dataset/train/train_ground_truth.tsv` |
| 5 | train_source1.tsv | `dataset/train/train_source1.tsv` |

**Interpret the output:**

- `[PASS] validate_submission.py` → format is correct, safe to upload
- `[FAIL] validate_submission.py exited with code N` → fix format issues, do NOT upload
- `FINAL VERDICT: GO` → format valid, F0.5 breakdown printed above
- `FINAL VERDICT: NO-GO` → do not upload until the FAIL is fixed

---

### Step 5 — Assemble submission package (Day 3)

Run once on Day 3 after the final prediction run, before zipping and uploading.

**Windows:**

```bat
build_package.bat output models_v4
```

**Linux / Mac:**

```bash
bash build_package.sh output models_v4
```

This creates `submission_package/` with the correct structure:

```
submission_package/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/          (all .py files)
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

If `output/matching_results.tsv` does not exist yet, placeholder files are inserted and a
warning is printed — replace them with the final output before zipping.

**Zip the package:**

```powershell
# PowerShell (Windows)
Compress-Archive -Path submission_package\* -DestinationPath Byjus_ML_Gang_submission.zip
```

```bash
# Linux / Mac
zip -r Byjus_ML_Gang_submission.zip submission_package/
```

---

## Recommended Run Order Per Integration Cycle

```
After Track D generates output/matching_results.tsv:

1. python utils/validate_submission.py ...          ← format check
2. python src/val_harness.py ... --val-only         ← F0.5 breakdown
3. python src/error_analysis.py ... --val-only      ← FP/FN categories
4. Share error_analysis output with Tracks A & B
5. (Day 3 only) bash build_package.sh → zip → upload
```

Or run steps 1+2 together via:

```bash
bash utils/check_and_validate.sh <args>
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'rapidfuzz'`**
→ `pip install rapidfuzz>=3.0` — or ignore, `error_analysis.py` falls back to Jaccard automatically.

**`ERROR: --gt path not found`**
→ Check you are running from `student_resource/` root, not from inside `code/` or `src/`.

**`validate_submission.py` prints FAIL**
→ Do not submit. Read the FAIL message — common causes: missing S1 entity rows, duplicate rows,
matched IDs that don't exist in the test source files.

**`val_harness.py` shows very low F0.5 on one country**
→ Run `error_analysis.py` filtered to that country and look at the dominant category.
If `near_dup` dominates → model threshold issue. If `script_mismatch` dominates → `norm.py` issue.

**`build_package.sh` warns about missing output files**
→ Track D hasn't run `predict.py` yet, or the output dir name doesn't match. Pass the correct
output dir as argument: `bash build_package.sh output_v4`.
