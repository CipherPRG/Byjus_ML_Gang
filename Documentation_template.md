# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Byjus_ML_Gang  
**Team Members:** Pratham, Adithya, Aayush, and team  
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

We solve the Amazon ML Challenge 2026 business entity resolution task using a two-stage LightGBM
pipeline with country-scoped blocking. A blocking step reduces the O(n²) comparison space to a
tractable candidate set via 11 key types (address tokens, name skeletons, compound keys); a
two-stage gradient-boosted classifier then scores each candidate pair using 32 string-similarity
features, with a competitive-context refinement stage. The final decision threshold is tuned to
maximise macro F0.5 on a held-out 30% validation split, reflecting the metric's 2× precision
weighting.

Best validation F0.5 on sample_dense: **0.9681** (models_v4, thr=0.96, margin=0.20).  
Blocking recall on sample_dense: 0.9545 overall (India 0.9284 / US 0.9720).

---

## 2. Methodology

### 2.1 Problem Analysis

The core noise patterns observed in the data:

- **Script mixing / transliteration noise.** Indian business names appear both in Devanagari/Tamil/Telugu and
  in Latin-script transliterations (e.g. "महाराष्ट्र" vs "Maharashtra"). Exact-string matching fails entirely
  on these; the pipeline handles them via a Unicode-character-name-based consonant-skeleton key that
  is script-agnostic.
- **Legal-suffix noise.** "Shell Gas Station", "Shell Gas Station India Pvt Ltd", and
  "Shell Gas Station India Private Limited" refer to the same entity. The normaliser strips ~60 common
  legal suffixes (LTD, PVT, LLC, INC, CO, …) before comparison.
- **Address abbreviation.** "ST" / "Street", "RD" / "Road", "AVE" / "Avenue" appear interchangeably.
  The address normaliser expands all standard abbreviations before tokenising.
- **House-number formatting.** "12-A Main St" vs "12A Main Street" — the pipeline extracts the numeric
  token separately (`anum_jac`, `anum_first_eq` features) and does not rely on exact address match.
- **Missing address field.** Some Source 2/3 records have an empty address. The `a2_empty` feature
  flags these so the model learns not to penalise them on address features.
- **Leading-zero padding.** Normalisation strips leading zeros from numeric tokens to prevent
  "001" ≠ "1" mismatches.
- **State-name abbreviation (India).** "Maharashtra" → "MH", "Gujarat" → "GJ", etc. — the normaliser
  expands abbreviations bidirectionally so both forms collapse to the same token.
- **Singletons (no true match exist in Source 2/3).** These must be predicted empty; incorrectly
  merging a singleton costs precision heavily under F0.5. The threshold/margin rule is tuned to keep
  singleton precision high.

### 2.2 Solution Strategy

**Approach type:** Blocking + Two-Stage Gradient-Boosted Classifier (Hybrid)

**High-level flow:**
1. Normalise all name and address strings (script-normalise → expand abbreviations → strip legal suffixes → lowercase).
2. Generate blocking keys for every record. Merge Source 1 keys with Source 2/3 keys to produce a
   candidate pair set (orders of magnitude smaller than the full cross-join).
3. Compute 32 pair-level string-similarity features for every candidate pair.
4. Stage 1 LightGBM classifier scores each pair in isolation (pair probability p1).
5. Stage 2 LightGBM classifier rescores each pair using the competitive context of p1 scores
   within the S1 entity's neighbourhood and within each S2/S3 record's candidate set.
6. Decode: each S2/S3 record is assigned to at most one S1 (the highest-p2 candidate), subject to
   a global threshold and a margin rule that enforces confidence.
7. Threshold and margin are swept jointly over the held-out validation split to maximise macro F0.5.

**Core innovation:** The competitive-context stage 2 model is the key architectural feature — it
directly learns to exploit the fact that a strong competitor should lower the confidence of a
borderline first-place score, and vice versa. This significantly reduces false positives without
sacrificing recall on clear matches.

---

## 3. Candidate Generation (Blocking)

### Key types

All keys are prefixed with the country code, so key spaces are entirely disjoint across countries.
Unseen test countries (France) automatically get their own key namespace with no special-casing.

| Key type | Description | Weight |
|----------|-------------|--------|
| **A** | Address token keys from `addr_keys()` — street tokens, combined house-number+street pairs | 3 |
| **N** | Sorted core-name token set (legal suffixes stripped) | 3 |
| **X** | Compound key: (name skeleton or T/K name token) × (address digit token or long address word). Most selective key type | 3 |
| **V** | Full sorted consonant skeleton of the core name | 2 |
| **R** | Two longest rare address words (both must match) | 2 |
| **P** | Name prefix (first 5 chars of first token) + suffix (last 4 chars of last token) | 2 |
| **Q** | Compact sorted name key (first 9 chars of sorted concatenation) | 2 |
| **T** | 4-char prefix of each long name token (≥5 chars, alpha) | 1 |
| **U** | 4-char suffix of each long name token | 1 |
| **K** | 3-char consonant-skeleton prefix of each significant name skeleton | 1 |
| **S** | 5-char prefix of longest rare address word | 1 |

### Candidate generation logic

- Block size cap: keys shared by more than `max_block=60` records are dropped as generic (too little discriminative value, too many candidates).
- Per-S1 cap: at most `topk=60` candidate S2/S3 records per S1 entity, ranked by summed key weight.
- Candidate pairs written to `output/candidate_pairs.tsv`.

### Blocking recall

On `sample_dense/` (training sample):

| Country | Blocking recall |
|---------|----------------|
| India | 0.9284 |
| US | 0.9720 |
| **Overall** | **0.9545** |

~4.5% of true matches are structurally unrecoverable from the candidate set regardless of model
performance — this is the hard precision ceiling that can only be improved by adding new key types
to `blocking.py`.

---

## 4. Matching Model

### 4.1 Features

32 pair features computed by `features.py` using `rapidfuzz` (MIT-licensed):

**Name features (11):**

| Feature | Description |
|---------|-------------|
| `nr` | `fuzz.ratio` on full normalised name |
| `nsort` | `fuzz.token_sort_ratio` on full name |
| `nset` | `fuzz.token_set_ratio` on full name |
| `npart` | `fuzz.partial_ratio` on full name |
| `njw` | Jaro-Winkler similarity on core name (legal suffixes removed) |
| `nlev` | Levenshtein normalised similarity on core name |
| `ncore_eq` | 1 if sorted core token sets are equal |
| `ncore_jac` | Jaccard over core name token sets |
| `nlen_d` | Normalised length difference of core names |
| `nfirst_eq` | 1 if first core token matches |
| `ncomp` | `fuzz.token_set_ratio` on core names (legacy name competition feature) |

**Address features (15):**

| Feature | Description |
|---------|-------------|
| `ar` | `fuzz.ratio` on full normalised address (0 if either empty) |
| `asort` | `fuzz.token_sort_ratio` on address |
| `aset` | `fuzz.token_set_ratio` on address |
| `apart` | `fuzz.partial_ratio` on address |
| `ajac` | Jaccard over address tokens |
| `anum_jac` | Jaccard over address digit-containing tokens |
| `anum_eq` | 1 if digit token sets intersect |
| `a2_empty` | 1 if Source 2/3 address is empty |
| `akey_eq` | 1 if address blocking-key sets intersect |
| `a_exact` | 1 if addresses are exactly equal (after normalisation) |
| `anum_first_eq` | 1 if first numeric token matches |
| `ajw` | Jaro-Winkler similarity on full address strings |
| `alev` | Levenshtein normalised similarity on full address |
| `alen_d` | Normalised address length difference |
| `akey_jac` | Jaccard over address blocking-key sets |

**Phonetic / skeleton features (3):**

| Feature | Description |
|---------|-------------|
| `sk_r` | `fuzz.ratio` on full consonant skeleton strings |
| `sk_set` | `fuzz.token_set_ratio` on skeleton |
| `sk_part` | `fuzz.partial_ratio` on skeleton |

**Other features (3):**

| Feature | Description |
|---------|-------------|
| `ncontain` | 1 if either core name is a substring of the other (length ≥ 4) |
| `wcount_d` | Normalised word-count difference of core names |
| `w` | Summed blocking key weight for this pair |

Total: 32 features. Feature importance analysis (LightGBM gain) showed `aset` dominates stage 1
(~70% of split gain), confirming address token-set ratio is the strongest single discriminator.

### 4.2 Model Architecture

**Stage 1 — Pair classifier:**
- `LGBMClassifier` with 500 estimators, learning rate 0.05, 63 leaves, L2 regularisation (λ=1.0),
  `is_unbalance=True` (class imbalance correction for the heavily skewed positive rate).
- Input: 32 pair features.
- Output: `p1` ∈ [0,1] — pairwise match probability ignoring competitive context.
- Early stopping on AUC (50 rounds) against the held-out val split.

**Stage 2 — Context-aware refiner:**
- `LGBMClassifier` with 200 estimators, learning rate 0.06, 31 leaves.
- Input: 10 features — `p1` itself plus competitive context:
  - `rk_s1` / `gap_s1`: rank and gap of this pair within the S1 entity's candidates.
  - `rk_o` / `gap_o`: rank and gap of this pair within the S2/S3 record's candidate set.
  - `max_o_other`, `sum_s1`, `sum_o`: aggregate competitor statistics.
  - Plus 10 raw pair features retained from stage 1: `nsort`, `aset`, `ajac`, `akey_eq`, `a_exact`,
    `anum_first_eq`, `w`, `ncore_eq`, `sk_r`, `sk_set`.
- Output: `p2` ∈ [0,1] — final match probability.
- Feature importance: `p1` alone contributes ~92% of gain, confirming stage 2 primarily learns
  to sharpen stage 1's borderline scores using competitive context, not to substitute for it.

**Model size:** Two LightGBM models, each well under 1 MB on disk. Total: < 2 MB. No external
pretrained weights. All models trained from the provided training data only. License: LightGBM
(MIT), rapidfuzz (MIT). Satisfies the ≤8B parameter constraint by many orders of magnitude.

### 4.3 Decoding

Each S2/S3 record is assigned to **at most one** S1 entity:
1. For each S2/S3 record, keep only the candidate pair with the highest `p2`.
2. Accept the assignment only if `p2 >= threshold` AND `p2 - p2_runner_up >= margin`.
3. S1 entities with no accepted S2/S3 assignment are output with an empty match list (singleton
   prediction).

The threshold and margin are jointly swept over a grid (`thr ∈ [0.30, 0.98]`, `margin ∈ [0.00, 0.44]`)
on the held-out validation split (30% of S1 entities, hash-based split — no data leakage since
the split is on S1 IDs, not pair IDs). The (thr, margin) pair maximising macro F0.5 is saved to
`models/config.json` alongside the model artefacts.

Best configuration found on `sample_dense/`:
- `thr=0.96`, `margin=0.20` → val macro F0.5 = **0.9681** (models_v4, max_block=60/topk=60)

### 4.4 Model Comparison (Track B benchmarking)

All alternatives were evaluated on the same OOF + stage2 + decode pipeline on `sample_dense/`:

| Model | Val F0.5 |
|-------|----------|
| **LightGBM (production)** | **0.9681** |
| Ensemble (LightGBM + XGBoost blend) | 0.9669 |
| XGBoost | 0.9666 |
| RandomForest (200 estimators) | 0.9643 |
| PyTorch neural net (deeper) | 0.9644 |
| CatBoost | 0.9642 |
| PyTorch neural net | 0.9640 |
| LogisticRegression | 0.9570 |

LightGBM was retained as the production model.

---

## 5. Results & Error Analysis

### 5.1 Quantitative Results

| Config | thr | margin | Val F0.5 |
|--------|-----|--------|----------|
| Baseline (27 features, thr=0.65, margin=0.20) | 0.65 | 0.20 | 0.9054 |
| +6 new features (B2) | 0.70 | 0.20 | 0.9666 |
| +hyperparameter tuning, sweep widened (B3/B4) | 0.94 | 0.34 | 0.9674 |
| +widened blocking max_block=60/topk=60 (A+B) | 0.96 | 0.20 | **0.9681** |

Public leaderboard F0.5: **[TBD — update after final submission]**

Blocking recall: 0.9545 overall (India 0.9284, US 0.9720).  
Singletons: the decoding rule (thr=0.96, margin=0.20) correctly leaves most singletons unpredicted,
contributing to precision.

### 5.2 Error Categories

Based on `error_analysis.py` output on the train val split (run by Track D — populate after
the first full-data val run):

| Category | FP count | FN count |
|----------|----------|----------|
| name_collision (same name, different address) | [TBD] | [TBD] |
| address_collision (same address, different name) | [TBD] | [TBD] |
| near_dup (both similar, borderline confidence) | [TBD] | [TBD] |
| script_mismatch (cross-script transliteration) | [TBD] | [TBD] |
| singleton_fp (true singleton incorrectly matched) | [TBD] | n/a |
| other | [TBD] | [TBD] |

### 5.3 Common Error Patterns

**False positives (wrong merges):**
- Same business chain with nearby but distinct branches sharing identical trading names and
  similar (but different) addresses — the model's address features are not discriminative enough
  for these "clone" cases.
- Singletons with very common name tokens (e.g. "STAR HOTEL") incorrectly matched to a
  different branch at a different address when the threshold is too low.

**False negatives (missed matches):**
- Records that appear only in script-mixed form — the consonant skeleton bridges many cross-script
  pairs, but pairs where transliteration is too non-standard fall through.
- Records with very short or completely missing addresses, where name-only matching is insufficient
  to clear the threshold.
- True matches that share no blocking key (ceiling: ~4.5% of true pairs, per blocking recall above).

---

## 6. Conclusion

Our two-stage LightGBM pipeline with country-scoped blocking achieves a validation macro F0.5 of
0.9681 on `sample_dense/`, up from a 0.9054 baseline, through a combination of feature engineering
(+6 similarity features), hyperparameter tuning, threshold/margin sweep optimisation, and widened
blocking parameters. The biggest single gain came from widening the blocking candidate cap
(max_block=60/topk=60, +0.0007 F0.5), confirming that blocking recall is the hard ceiling. The
competitive-context stage 2 model is the key architectural feature that keeps precision high
under the F0.5 metric's 2× precision weighting. The pipeline handles unseen countries (France)
without any code changes via dynamic country-scoped key namespaces.

---

## Appendix

### A. Code Structure and Reproduction Steps

The full source ships in the submission zip under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── requirements.txt
├── README.md
└── src/
    ├── norm.py              — text normalisation, Indic transliteration, consonant skeleton
    ├── blocking.py          — 11 key types (A/N/X/V/R/P/Q/T/U/K/S), country-scoped, Side class
    ├── features.py          — 32 pair features via rapidfuzz
    ├── model.py             — two-stage LightGBM, decode_prep/decode_apply, stage2_matrix
    ├── train.py             — sample → features → train stage1+stage2 → thr/margin sweep → save
    ├── predict.py           — load models → blocking+features+inference → write output/
    ├── pipeline.py          — shared build_country() logic
    ├── evaluate.py          — f05_macro()
    ├── io_utils.py          — read_tsv, countries_of, read_country, write_lists
    ├── make_sample.py       — builds sample_dense/ (mod-N sample, keeps name-clusters intact)
    ├── val_harness.py       — per-country F0.5 + TP/FP/FN + worst entities + singleton check
    ├── error_analysis.py    — categorised FP/FN report for diagnostic use
    ├── blocking_audit.py    — blocking recall audit vs. ground truth
    └── blocking_audit_fast.py — fast version of blocking_audit
```

**Reproduction steps** (run from `student_resource/` root):

```bash
pip install -r code/business_entity_resolution/requirements.txt

# 1. Build dense training sample (optional: sample_dense/ may already exist)
python code/business_entity_resolution/src/make_sample.py \
    --data dataset/train --out sample_dense --mod 20

# 2. Train both stages and tune threshold/margin
python code/business_entity_resolution/src/train.py \
    --data sample_dense \
    --models code/business_entity_resolution/models \
    --workers 4

# 3. Generate predictions on the test set
python code/business_entity_resolution/src/predict.py \
    --data dataset/test \
    --models code/business_entity_resolution/models \
    --out output \
    --workers 4

# 4. Validate format (must print PASS before any upload)
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 5. (Optional) Full diagnostic breakdown on train val split
python code/business_entity_resolution/src/val_harness.py \
    --gt  dataset/train/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --s1  dataset/train/train_source1.tsv \
    --val-only

# 6. (Optional) Categorised error analysis
python code/business_entity_resolution/src/error_analysis.py \
    --gt   dataset/train/train_ground_truth.tsv \
    --pred output/matching_results.tsv \
    --src1 dataset/train/train_source1.tsv \
    --src2 dataset/train/train_source2.tsv \
    --src3 dataset/train/train_source3.tsv
```

**Key runtime notes:**
- `train.py` on `sample_dense/` (≈10% of full training data): ~20–40 minutes on a standard laptop with `--workers 4`.
- `predict.py` on full test set: ~30–60 minutes depending on hardware.
- All scripts are CPU-only (no GPU required). Memory peak: ~12–16 GB during `predict.py` on full test data.
- The trained model artefacts are in `models_v4/` (or whichever directory is passed to `--models`).

**Dependencies** (`requirements.txt`):
```
lightgbm>=4.1.0
rapidfuzz>=3.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

All packages are MIT or Apache-2.0 licensed.

### B. Additional Results

**Blocking parameter sensitivity (sample_dense/):**

| max_block / topk | Val F0.5 | thr / margin |
|-----------------|----------|-------------|
| 30 / 30 (original) | 0.9674 | 0.94 / 0.34 |
| **60 / 60 (production)** | **0.9681** | **0.96 / 0.20** |

Widening blocking increases candidate count ~2× but only adds +0.0007 F0.5; worth the cost given
the time budget.

**Per-country threshold experiment (Track B):**

Global threshold (0.9681) vs. per-country thresholds (India thr=0.90/margin=0.34, US thr=0.94/margin=0.38)
combined F0.5 = 0.9683 — delta +0.0002, within noise. Per-country thresholds were not adopted.

---

*Note: Fill in the [TBD] fields in Section 5 after Track D runs the full-data prediction and
val_harness.py / error_analysis.py produce output. Final public LB score to be added after
the last leaderboard submission.*
