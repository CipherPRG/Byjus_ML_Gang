# WORKPLAN — Amazon ML Challenge 2026: Business Entity Resolution

> Created: 25 Sep 2026 19:45 IST  
> Deadline: ~27 Sep 2026 evening IST (~22:00 IST)  
> Remaining: ~50 hours  
> Max submissions: 5 per calendar day (IST midnight resets)

---

## 0. Ground Rules — READ THESE FIRST

1. **One evolving pipeline, not 4 competing ones** — all 4 people improve the SAME codebase in parallel on separate branches, integrated by Track D.
2. **Max 5 leaderboard submissions per calendar day** (per problem_statement.md). Resets at IST midnight. Never waste a slot on an unvalidated change.
3. **Hackathon duration: 3 days / 72 hours total** (started 25 Sep 2026 ~19:45 IST). Deadline is **[VERIFY ON PORTAL]** ~22:00 IST 27 Sep 2026 — EXACT TIME UNCONFIRMED.
4. **Final rankings uncertainty** — problem_statement.md says rankings are based on the private leaderboard but does NOT state whether that means your single latest submission or your best submission across the challenge. **[VERIFY ON PORTAL]** — until confirmed, treat every submission as if it might be the one that counts. Never submit a worse/experimental version late in the timeline.
5. **TWO distinct submission types** — do not conflate these:
   - **(a) Leaderboard uploads** — `matching_results.tsv` only, up to 5/day — drives public+private leaderboard scores.
   - **(b) ONE final package** — zip with `output/`, `code/`, `Documentation_template.md` — submitted once through the portal before the deadline.
6. **Team structure: 4 people, shared GitHub repo.** Branch-per-person workflow: `track-a-blocking`, `track-b-model`, `track-c-validation`, `main` (Track D integrates). Git already initialized locally; `.gitignore` excludes `dataset/`, `sample/`, `sample_dense/`, `output/`, `models/`.
7. **Real dataset is large** (~1.15GB train sources, ~1.13GB test sources). Any approach must actually run end-to-end multiple times in the remaining hours, not just work in theory.
8. **Hard constraints from problem_statement.md**: no external APIs/geocoding/internet lookups; model MIT/Apache-2.0 licensed, ≤8B parameters (current LightGBM satisfies both); every test S1 entity must appear in the submission; `validate_submission.py` must print PASS before every upload.
9. **France (unseen country) is in the test set** — pipeline handles it correctly via dynamic country reading (`countries_of()` in `io_utils.py`; blocking keys are country-prefixed strings, so France gets its own namespace automatically with zero hard-coding). Track A verifies this explicitly in `blocking_audit.py`.
10. **Person C owns `Documentation_template.md`** — filled progressively Days 1–3 (sections 1–4 + Appendix A by Day 2 EOD), final version ready Day 3 13:00 IST for package assembly.

**Independent technical assessment of the current pipeline:** Two-stage LightGBM (blocking → stage1 classifier → stage2 context-aware refiner) is the right architecture for this problem at this data scale — standard for entity resolution, MIT-licensed, <8B params, precision-focused decoding matches F0.5's β=0.5 weighting. Current baseline: **val F0.5 = 0.9054** on `sample_dense` (thr=0.65, margin=0.2), models trained, but **the full test dataset has not been run yet** — that is Track D's first task (D1). Do not attempt a from-scratch architecture change (transformer embeddings, graph-based matching, etc.) with the time remaining; focus effort on blocking recall, feature engineering and hyperparameter tuning as scoped in Tracks A/B below.

---

## 0b. Submission Plan

**A. Leaderboard submissions** (up to 5/day, `matching_results.tsv` only):

| # | Trigger | Local Go/No-Go Check | Expected Time |
|---|---|---|---|
| 1 | Baseline full-test run complete | Val F0.5 > 0 (sanity) | Day 1 22:30 or Day 2 early |
| 2 | Integration 1 (A+B merged) | Val F0.5 > Submission #1 | Day 2 15:00 |
| 3 | Integration 2 (error-driven fixes) | Val F0.5 > Submission #2 | Day 2 22:00 |
| 4 | Final freeze run | Val F0.5 ≥ Submission #3 (no regression) | Day 3 12:30 |
| 5 | RESERVE (only if #4 regresses or a format issue appears) | Emergency fix validated | Day 3 16:00 if needed |

**Go/No-Go rule — never upload without all three:**
1. `utils/validate_submission.py` prints `PASS`.
2. Val F0.5 (on the train val split) ≥ previous submission's val F0.5 (or it's the first submission).
3. Track C's `val_harness.py` shows no catastrophic per-country drop.

**B. Final package submission** (once, due before the deadline ~22:00 IST Day 3):
- Assembly starts Day 3 14:00 IST (Track C task C5 + Track D task D6).
- Structure (per `problem_statement.md`):
  ```
  <team_name>_submission.zip
  ├── output/                          (matching_results.tsv + candidate_pairs.tsv from the BEST run)
  ├── code/business_entity_resolution/ (src/, README.md, requirements.txt)
  └── Documentation_template.md        (filled in)
  ```
- Track D reviews the structure against the README spec before zipping.
- Target upload time: Day 3 18:00 IST — leaves buffer before the ~22:00 IST deadline.

---

## 1. Problem & Constraint Summary

**Task.** For every Source 1 entity (the clean reference), find all matching records in Source 2 and Source 3 (noisy, multi-script, partial-info duplicates). The two output files — `output/matching_results.tsv` and `output/candidate_pairs.tsv` — are both required; only `matching_results.tsv` is scored.

**Metric.** Macro-averaged F0.5 over all Source 1 entities in the evaluation set. F0.5 weights precision twice as heavily as recall. A singleton predicted empty scores 1.0; a false merge on a singleton scores 0.0.

**Data sizes (real dataset).** ~200 MB Source 1 train, ~490–504 MB each for Source 2/3 train, ~175 MB Source 1 test, ~506–509 MB each for Source 2/3 test. Countries in train: US, India. Test adds France (unseen; pipeline must handle it without special-casing).

**Hard rules.**
- No external APIs, geocoding, or internet lookups.
- Model must be MIT/Apache-2.0 licensed and ≤8B parameters.
- Every Source 1 test entity must appear in the submission; no duplicate rows; matched IDs must exist in the test set.
- Run `python utils/validate_submission.py` before every upload.

**Submission budget.**
- Day 1 (25 Sep): started late; aim to use at most 2 submissions (baseline smoke-test + first real run).
- Day 2 (26 Sep): 5 submissions available — use them for iterative improvement.
- Day 3 (27 Sep): 5 submissions available — reserve the last 2 for the final polished run + 1 safety re-upload.
- Final submission package (zip with `output/`, `code/`, `Documentation_template.md`) due by deadline.

---

## 2. Existing Pipeline — Snapshot

```
code/business_entity_resolution/
  src/
    norm.py       — text normalisation, Indic transliteration, consonant skeleton
    blocking.py   — country-scoped blocking keys (A/N/P/Q/T/U/V/K/R/S/X types), Side class, candidates()
    features.py   — 27 pair features via rapidfuzz (name + address fuzz ratios, Jaccard, house-number agreement)
    model.py      — two-stage LightGBM; stage2 adds rank/gap/competitor context; decode() threshold+margin rule
    train.py      — dense sample → features → train stage1+stage2 → tune thr/margin → save models/
    predict.py    — load models → per-country blocking+features+inference → write output/
    pipeline.py   — build_country() shared logic
    evaluate.py   — f05_macro()
    io_utils.py   — read_tsv, countries_of, read_country, write_lists
    make_sample.py — builds sample_dense/ (1/mod of the data, keeping name-clusters intact)
  models/         — trained model artefacts (stage1.txt, stage2.txt, config.json)
  requirements.txt
  README.md

utils/validate_submission.py  — run before every submission
sample_dense/   — pre-built dense training sample (~10% of train)
dataset/train/  — full training data (~200+490+504 MB)
dataset/test/   — full test data (~175+506+509 MB)
```

The sample-based validation showed ~0.96 F0.5 but the pipeline has **not been run on the full dataset yet**.

---

## 3. Team Tracks

Four people, four parallel tracks. Tracks A, B, C work independently on separate files and branches; Track D (integration lead) is the only one who touches the full-data pipeline runs and actual submissions.

---

### Track A — Blocking & Candidate Recall
**Owner:** Person A  
**Branch:** `track-a-blocking`  
**Owns:** `code/business_entity_resolution/src/blocking.py`, `code/business_entity_resolution/src/norm.py`  
**Working dataset:** `sample_dense/` (fast iteration; full-data tests via Track D)

#### Why this track matters
Blocking recall is the hard ceiling on model F0.5. Every true match that blocking misses is a guaranteed false negative. The current `decode()` also enforces a 1-to-1 assignment (each S2/S3 to its best S1), so high-weight shared keys are critical.

**Note (Pratham, 26 Sep ~05:49 IST):** as of the current `sample_dense` runs (with Track B's 6 new features on top), overall blocking recall is still **0.9545** (India 0.9284 / US 0.9720) — unchanged, since features.py doesn't touch blocking. That means ~4.5% of true matches are structurally unrecoverable no matter what the model/features do — this is a harder ceiling on F0.5 than feature engineering. **Track A: this is probably higher-leverage right now than anything Track B can still squeeze out of features** — if `blocking_audit.py` (A2) hasn't identified the missed-pair patterns yet, that's likely the best use of remaining time.

#### Concrete tasks

**A1 — Measure current blocking recall on sample_dense (Day 1, tonight)**
Run `train.py` on `sample_dense/` and record the per-country blocking recall printed in the output. Capture these numbers as baseline in the Day 1 status update.
```
python code/business_entity_resolution/src/train.py \
  --data sample_dense --models code/business_entity_resolution/models --workers 4
```
Read the printed lines: `[US] ... blocking recall=X.XXXX`, `[India] ...`.

**A2 — Identify missed true pairs (Day 1–2)**
Add a diagnostic script `src/blocking_audit.py` (new file — does not modify any existing file) that:
- Loads `sample_dense/` ground truth.
- Runs `build_country()` for each country.
- Prints the S1 entity IDs and S2/S3 IDs for every true pair that did **not** appear in the candidate set.
- Groups misses by key type: are they missed because no key matched, or because the matching key was too common and was dropped by `max_block=30`?

Focus on patterns: Do missed pairs share a numeric address token? Do they only differ in script (Hindi vs Latin)? Does the `decode()` 1-to-1 constraint cause any misses (true multi-match entities)?

**A3 — Extend key types for known miss patterns (Day 2)**
Based on A2 findings, add new key types to `keys_for()` in `blocking.py` (edit blocking.py directly). Candidate improvements — implement whichever address the actual miss patterns:
- **Postal code key**: India has 6-digit PIN codes (`^\d{6}$`), US has 5-digit ZIPs. A key `Z|country|pincode` combined with a name skeleton (`X`-style compound) would be very selective.
- **Trigram / 3-char prefix key for address tokens** longer than 6 chars — covers transliteration variants of the same street name.
- **Relaxed `max_block` for high-weight A-type (address) keys**: currently capped at 30; raising to 50 only for `A`-type keys may recover missed pairs without exploding candidate counts.
- **Name-only fallback key**: for records with empty or very short addresses, fall back to a longer name n-gram key (`F|country|first4_last4_core`) that isn't generated for records with good addresses.
- **France/Latin skeleton key**: `skel()` in `norm.py` handles Latin text; verify that French accented words (café → cafe via `strip_accents`) then produce correct skeleton keys. Add a test in `blocking_audit.py`.

**A4 — Re-run blocking audit after changes, check recall gain vs. candidate count (Day 2)**
After each change, re-run `train.py` on `sample_dense/` and compare:
- Blocking recall (higher is better).
- Average candidates per S1 entity (`pairs / S1 count` from the printed line) — should not grow >20% from the baseline.
- Wall-clock time — blocking must still complete in reasonable time for the full dataset.

**A5 — Hand off clean diff to Track D for full-data integration (Day 2 EOD)**
Commit `blocking.py`, `norm.py`, and `blocking_audit.py` to `track-a-blocking` branch with a summary comment: "blocking recall on sample_dense: before X.XXXX → after X.XXXX; avg candidates/S1: before Y → after Y".

#### Done criteria for Track A
- [ ] Blocking recall on `sample_dense/` is measured and documented.
- [ ] Miss-pattern analysis exists in `blocking_audit.py` with printed output.
- [ ] At least one concrete key-type addition is committed to `blocking.py`.
- [ ] Re-run confirms recall improved without >20% candidate-count growth.
- [ ] Branch is cleanly rebased/merged-ready for Track D by Day 2 ~20:00 IST.

---

### Track B — Features & Model Tuning
**Owner:** Person B  
**Branch:** `track-b-model`  
**Owns:** `code/business_entity_resolution/src/features.py`, `code/business_entity_resolution/src/model.py`  
**Working dataset:** `sample_dense/` (fast iteration)

#### Why this track matters
The 27 current features cover name/address fuzz ratios and house-number agreement well but are missing several potentially high-signal features (TF-IDF cosine, numeric span comparison, country-specific PIN/ZIP). The two-stage LightGBM structure is sound; hyperparameters in `P1`/`P2` dicts are currently defaults and have not been tuned on the full data.

#### Concrete tasks

**B1 — Measure baseline model performance on sample_dense (Day 1, tonight)**
Run `train.py` on `sample_dense/` with the existing code and record: validation macro F0.5, chosen thr, chosen margin, per-country blocking recall. This is the hard number everything else is compared against.

**B2 — Add new features to `features.py` (Day 2)**
All additions go in `_chunk()`. The `F1` list at the top must be updated to match. New candidate features — implement whichever improve validation F0.5:

- **Numeric span Jaccard** (`nspan_jac`): extract all digit runs from each address; Jaccard over those sets. More robust than exact `anum_eq` when numbers are formatted differently (e.g., "12-A" vs "12").
- **Longest common substring ratio** (`nlcs`): `len(lcs(n1,n2)) / max(len(n1),len(n2))` for the core name strings. Captures partial matches that token-level metrics miss.
- **Name length ratio** (`nlen_ratio`): `min(len,len)/max(len,len)` — distinguishes short "Shell" from long "Shell Gas Station India Private Ltd".
- **Country is same** (`country_eq`): always 1.0 within current country-scoped blocking, but useful if blocking ever runs cross-country.
- **Address numeric count difference** (`anum_cnt_diff`): `|len(nu1) - len(nu2)|`; penalises pairs where one address has many numbers and the other has none.
- **TF-IDF cosine on address tokens** (`atf_cos`): fit a `TfidfVectorizer` once per country in a preprocessing step (or per batch) and add the cosine as a feature. This can be done without external data.
- **Skeleton exact match** (`sk_eq`): `float(sk1 == sk2)` for the full sorted skeleton string — a tighter version of `sk_r`.

After adding features, update `F1` list, re-run `train.py` on `sample_dense/`, check that F0.5 improves. Use LightGBM feature importances (available via `m1.booster_.feature_importance()`) to prune features that score near zero.

**B2 — Result (Pratham, 26 Sep ~05:40 IST):**
Added 6 new features to `features.py` (pushed to `pathu` branch, commit `3d2e58e`):
- `ajw`, `alev` — address JaroWinkler/Levenshtein similarity (name already had these via `njw`/`nlev`; address had none)
- `alen_d` — address length difference, normalised (name had `nlen_d`; address had none)
- `ncontain` — substring containment flag between core name strings (catches branding noise, e.g. `"Crestline Crestline Clean LP"` vs `"Crestline Clean"`)
- `akey_jac` — jaccard overlap of address blocking-keys, replacing the old boolean `akey_eq`
- `wcount_d` — name core-token count difference

`F1` grew from 27 → 33 features, all appended at the end (safe — `model.py`'s `RAW2` indexes by name via `F1.index(c)`, not position).

Ran `train.py --data sample_dense --models ../models --workers 4`:
- Blocking recall: India 0.9284, US 0.9720, overall 0.9545 — **identical** to the pre-change baseline (expected: features.py doesn't touch blocking.py, so this just confirms no regression)
- Validation macro F0.5: **0.9653 → 0.9666** at thr=0.70, margin=0.20 (val S1: 32,931) — small but real improvement, no regression anywhere

Then ran full-dataset `predict.py` with these new models → `output_v2/`:
- `matching_results.tsv`: 1,732,544 rows, 93,332 empty / **1,639,212 non-empty** (vs original submission's 1,642,130 non-empty — ~2,900 fewer, consistent with the new margin being stricter, likely trading a few borderline matches for precision)
- `utils/validate_submission.py` → **PASS**, safe to submit

**Thinking about next:**
- Plan is to submit the original (`output/`) as one leaderboard upload and this new one (`output_v2/`) as a second, back-to-back, to get a real leaderboard-score comparison rather than trusting internal val F0.5 alone (leaderboard scores against real held-out ground truth, which could behave differently)
- Have **not** yet trained on the full `dataset/train` (only ever `sample_dense/`) — considering whether it's worth the RAM/time cost (full-dataset training would need ~27GB+ RAM based on the `predict.py` full-test-set experience; laptop can do it, Colab free tier can't). Kaggle Notebooks (~29-30GB RAM, free) is the fallback if the laptop is tied up
- Given the gain from B2's features was modest (+0.0013 F0.5), still open to more feature ideas (numeric-span jaccard, TF-IDF cosine per B2's original suggestion list) if there's time left after the leaderboard comparison lands

**B3 — Tune stage1 and stage2 hyperparameters (Day 2)**
The dicts `P1` and `P2` in `model.py` use fixed learning rates and tree counts. Try a small grid search directly in a scratch script `src/tune_hyperparams.py` (new file) using the held-out validation split that `train.py` already constructs:
- `n_estimators`: [200, 300, 500] for stage1
- `num_leaves`: [31, 63, 127] for stage1
- `learning_rate`: [0.05, 0.08, 0.10]
- Stage2 is already lightweight; focus effort on stage1.

Do not run a full grid — pick 4–6 most promising combos manually and pick the best by validation F0.5.

**B4 — Tune the threshold/margin sweep range in `train.py` (Day 2)**
The current sweep in `train.py` is `thr in np.arange(0.30, 0.96, 0.05)` and `margin in (0.0, 0.05, 0.1, 0.2)`. After model improvements, the optimal threshold may shift. Verify the sweep is wide enough to capture the true optimum; if the best thr falls near either edge of the range, widen it. (This requires editing `train.py` — discuss with Track D before changing, as `train.py` is also used by Track D for the full-data run.)

**B4.5 — Light-to-heavy stage1 model comparison with runtime benchmarking (Day 2, time-boxed to ~2 hours)**
This is internal experimentation inside Track B on `sample_dense/` — it does NOT create a second pipeline; the winner simply replaces `fit_stage1()`'s estimator in `model.py`, everything downstream (stage2, decode, blocking) stays the same. In a scratch script `src/model_bench.py` (new file), train stage1 on the same train/val split `train.py` already builds, swapping only the classifier, and record for each: validation F0.5 (via the same threshold/margin sweep as `train.py`), train wall-clock time, and predict wall-clock time on the val set:
- **Light:** `sklearn.linear_model.LogisticRegression` (baseline floor, near-instant)
- **Light-medium:** `sklearn.ensemble.RandomForestClassifier` (n_estimators=200)
- **Current:** `LightGBM` (existing stage1 config — the reference point)
- **Heavier (only if time allows):** `LightGBM` with higher `num_estimators`/`num_leaves` from B3's grid, or `xgboost.XGBClassifier` if installed — both MIT/Apache-2.0, well under 8B params, so both are contest-legal.
Report a simple table (model, val F0.5, train time, predict time) in the Day 2 status update. Keep whichever model wins on F0.5 first, runtime second — do not swap the production model without Track D's sign-off, since it affects the full-data run's timing budget.

**B3/B4/B4.5 — Result (Pratham, 26 Sep ~15:45 IST):**
Pushed to `pathu` branch (commits `c9b7817`, `bb36851`):
- Hyperparameters: added `is_unbalance=True`, `min_child_samples=20`, `reg_lambda=1.0` to P1/P2; added optional early stopping (50 rounds, AUC) via an `eval_set` param on `fit_stage1`/`fit_stage2`.
- Fixed a real perf bug in the threshold/margin sweep (`decode()` was re-sorting the full dataframe on every combo) by splitting it into `decode_prep()`/`decode_apply()` — identical results, much faster sweeps.
- Widened the sweep grid (thr 0.30–0.98, margin 0.00–0.44) after the previous optimum landed at its edge; confirmed new interior optimum at **thr=0.94, margin=0.34**.
- Validation macro F0.5: **0.9666 → 0.9674** (B2 baseline → after this round of tuning).
- B4.5 model comparison (`src/model_bench.py`, same OOF + stage2 + decode pipeline for every candidate):

  | model | val F0.5 |
  |---|---|
  | current (LightGBM, production config) | **0.9674** |
  | rf200 (RandomForestClassifier) | 0.9643 |
  | logreg (LogisticRegression) | 0.9570 |

  (`lgb_700_127` was already tried standalone earlier and reverted — no improvement, so not re-run.) Decision: keep current LightGBM stage1 config, no production model swap.
- Feature importance check (`src/check_feature_importance.py` for stage1, `src/check_stage2_importance.py` for stage2): stage1 dominated by `aset` (70% of gain); stage2 dominated by `p1` (92%, expected — stage2 refines stage1's own score using competitive context). Two near-zero features found: `a1_empty` (exactly 0.0, confirmed unused anywhere else) was removed from `features.py`, re-verified F0.5 unchanged (0.9674). `a_exact` (0.00% in stage1, 0.00% in stage2 but non-zero) was kept since it's still occasionally used and not provably dead.
- Also ran a fixed/fast version of the adithya-sundar branch's blocking audit against ground truth (commit `a79c68e`): overall blocking recall = **0.9545** (India 0.9284, US 0.9720) — this is Track B/Track A's shared ceiling info, useful for Track C/D's documentation.

Branch is merge-ready for Track D.

**B3/B4/B4.5 — Extended model comparison (Pratham, 26 Sep ~18:30 IST):**
Benchmarked several alternative stage1 model families through the exact same OOF + stage2 +
decode evaluation pipeline (`src/model_bench.py`), to check whether a different classifier could
beat the current LightGBM config:

| model | val F0.5 |
|---|---|
| **LightGBM (current)** | **0.9674** |
| Ensemble blend (LightGBM + XGBoost) | 0.9669 |
| XGBoost | 0.9666 |
| RandomForest(200) | 0.9643 |
| GPU-trained neural net (PyTorch, deeper config) | 0.9644 |
| CatBoost | 0.9642 |
| GPU-trained neural net (PyTorch) | 0.9640 |
| LogisticRegression | 0.9570 |

Result: LightGBM stays the production model — none of the alternatives beat it. This closes out
the "which classifier" question with real evidence across 8 model families rather than assumption.

Since model choice isn't the remaining lever, now investigating two other angles instead:
- Per-country threshold/margin (`src/per_country_threshold.py`) — blocking recall already differs a
  lot by country (India 0.9284 vs US 0.9720), so one global thr/margin may be a compromise
- Cross-source corroboration as a new stage2 feature (`src/cross_source_experiment.py`) — does an
  S1 entity having independent strong evidence from both Source2 and Source3 improve F0.5 over
  today's per-pair-only scoring

Results (per-country threshold): tested — global F0.5=0.9674 vs per-country combined F0.5=0.9676
(India thr=0.90/margin=0.34, US thr=0.94/margin=0.38). Delta +0.0002 — within noise, not adopted.

**B6 — Model/feature-count mismatch fix + widened blocking, credit Track A (Pratham/Adithya, 26 Sep ~20:05 IST):**
Found `models/` and `models_tuned/` were trained on a pre-`bb36851` 33-feature version of `features.py`;
that commit dropped a dead feature, nobody retrained, so `predict.py` crashed with a LightGBM
shape-mismatch error. Retrained clean on current code → `models_v3/` (32 features, thr=0.94/margin=0.34,
val F0.5=0.9674) — this generated the real `output_v3` submission candidate.

Separately, Adithya's `adithya-sundar` branch (Track A) widened blocking from
`max_block=30/topk=30` to `max_block=60/topk=60` (more candidates survive blocking per S1 entity).
That branch had diverged from `pathu` (missing this session's later Track B work), so rather than
merging it directly, the parameter change was tested on top of current `pathu` code:

| config | val F0.5 | thr / margin |
|---|---|---|
| max_block=30 / topk=30 (`models_v3`) | 0.9674 | 0.94 / 0.34 |
| max_block=60 / topk=60 (`models_v4`, Adithya's change) | **0.9681** | 0.96 / 0.20 |

+0.0007 — real, adopted as the new production config (`models_v4`). Cost: ~2x candidates per
S1 entity, so training and prediction both take roughly 2x longer. `train.py`'s `CFG` now defaults
to the widened values. Not yet re-run against the full test set (still on `output_v3`/`models_v3`
for the actual submission as of this note) — next step is generating `output_v4` from `models_v4`
before using it for a real submission.

Also flagged for later testing: Adithya's branch adds a new stage-1 feature `a1_empty` (mirrors the
existing `a2_empty`) — untested by Track B, could stack on top of the blocking win for more gain.

Cross-source corroboration (`src/cross_source_experiment.py`): still not run.

**B5 — Hand off clean diff to Track D (Day 2 EOD)**
Commit `features.py` and `model.py` (and any new scratch files like `tune_hyperparams.py`) to `track-b-model` with a summary: "validation F0.5 on sample_dense: before X.XXXX → after X.XXXX; new features: [...]; best hyperparams: [...]".

#### Done criteria for Track B
- [x] Baseline F0.5 on `sample_dense/` measured and recorded.
- [x] At least 2 new features added, validated to improve F0.5 on `sample_dense/`.
- [x] LightGBM feature importance checked; no-contribution features removed or flagged.
- [x] Hyperparameter tuning done; best `P1`/`P2` dicts updated in `model.py`.
- [x] Light-to-heavy stage1 model comparison run (B4.5), results table recorded, decision on which model to keep documented.
- [x] Branch is cleanly merge-ready for Track D by Day 2 ~20:00 IST.

---

### Track C — Validation Harness, Error Analysis & Documentation
**Owner:** Person C  
**Branch:** `track-c-validation`  
**Owns:** `code/business_entity_resolution/src/evaluate.py`, `Documentation_template.md`, new scripts `src/error_analysis.py` and `src/val_harness.py`  
**Working dataset:** `sample_dense/` for development; feeds full-data results from Track D back for analysis

#### Why this track matters
The current `evaluate.py` only has `f05_macro()`. There is no per-entity breakdown, no false-positive/negative inspection, and no automated check that submission files are valid before Track D uploads them. Track C builds the diagnostic infrastructure that makes every integration cycle faster and every submission safer.

#### Concrete tasks

**C1 — Build `src/val_harness.py` (Day 1, tonight)**
A single script that, given a ground truth file and a `matching_results.tsv`, prints:
- Overall macro F0.5.
- Per-country macro F0.5 (US, India, and any other country).
- Breakdown: count of true positives, false positives, false negatives across all entities.
- Top 20 worst-performing S1 entities (lowest individual F0.5), with their predicted and true match sets printed side-by-side.
- Count of singletons (no true match) correctly predicted empty vs. incorrectly given a match.

Usage (run from `student_resource/`):
```
python code/business_entity_resolution/src/val_harness.py \
  --gt dataset/train/train_ground_truth.tsv \
  --pred output/matching_results.tsv \
  --s1  dataset/train/train_source1.tsv
```
This script is used by Track D after every full-data run to get the breakdown before deciding whether to submit.

**C2 — Build `src/error_analysis.py` (Day 2)**
A script that, given a prediction TSV and the ground truth, produces:
- False positive pairs: S1 entity + incorrectly matched S2/S3 entity. For each, print: S1 name/address, S2/S3 name/address, country.
- False negative pairs: true matches that were missed. Same printout.
- Group false positives by likely cause: same name different address ("name collision"), same address different name ("address collision"), both name and address similar ("near-duplicate confusion"), other.

Usage:
```
python code/business_entity_resolution/src/error_analysis.py \
  --gt dataset/train/train_ground_truth.tsv \
  --pred output/matching_results.tsv \
  --src1 dataset/train/train_source1.tsv \
  --src2 dataset/train/train_source2.tsv \
  --src3 dataset/train/train_source3.tsv \
  --n 50
```
Track D runs this after each integration cycle; output is pasted into the Day 2/Day 3 status update so all four people can see what errors remain.

**C3 — Pre-submission validation wrapper (Day 2)**
Write a short shell/bat wrapper `utils/check_and_validate.bat` (Windows) and `utils/check_and_validate.sh` (Linux/Mac) that:
1. Runs `utils/validate_submission.py` (already exists — do not modify it).
2. Runs `val_harness.py` against the train ground truth to print the F0.5 breakdown.
3. Prints a final GO / NO-GO line.

Track D runs this wrapper before every leaderboard upload.

**C4 — Fill in `Documentation_template.md` progressively (Days 1–3)**
Fill in the template as the pipeline results become known. Do not wait for the final numbers — fill in the methodology sections (blocking strategy, features, model architecture, threshold tuning method) now, leaving placeholder `[TBD]` only for the final F0.5 score. Update the score fields each time Track D reports a new leaderboard score.

Sections to complete:
- 1 Executive Summary
- 2.1 Problem Analysis (noise patterns observed in the data: script mixing, legal suffix noise, address abbreviation)
- 2.2 Solution Strategy
- 3 Candidate Generation (describe all key types: A/N/P/Q/T/U/V/K/R/S/X from `blocking.py`)
- 4 Matching Model (27 base features + any new ones from Track B; two-stage LightGBM; decode logic)
- 5 Results & Error Analysis (populate from `error_analysis.py` output)
- 6 Conclusion
- Appendix A (code structure and reproduction steps)

**C5 — Prepare final submission package structure (Day 3)**
Create the zip-ready directory structure under a local folder `submission_package/`:
```
submission_package/
  output/
    matching_results.tsv    (copy from output/ after final run)
    candidate_pairs.tsv
  code/
    business_entity_resolution/
      src/                  (all .py files)
      README.md
      requirements.txt
  Documentation_template.md (filled-in)
```
Do not create the zip yet — Track D will copy the final output files in and zip it. Just set up the structure with a `Makefile` or `build_package.bat` script that copies everything into place.

#### Done criteria for Track C
- [ ] `val_harness.py` runs against `sample_dense/` predictions and prints per-country F0.5 + top-20 worst entities.
- [ ] `error_analysis.py` runs and produces a categorised FP/FN report.
- [ ] Pre-submission validation wrapper exists and runs end-to-end.
- [ ] `Documentation_template.md` sections 1–4 and Appendix A are filled in (no `[TBD]` left in methodology sections).
- [ ] Submission package directory structure and copy script are ready.

---

### Track D — Integration, Full-Data Runs & Submission Management
**Owner:** Person D (integration lead)  
**Branch:** `main` (integrates from all other branches)  
**Owns:** All files during integration; specifically `train.py`, `predict.py`, `pipeline.py`, `io_utils.py`, and the `output/` folder. Also owns the daily submission log (see Section 5).

#### Why this track matters
The other three tracks work on `sample_dense/` which is ~5% of the real data. The full dataset is ~1.2 GB of source files. Full pipeline runs take significant wall-clock time and must be scheduled carefully across the 50-hour window to leave buffer before the deadline. Track D is also the single point of contact with the leaderboard.

#### Concrete tasks

**D1 — First full-pipeline run with baseline code (Day 1 tonight, ~21:00–23:00 IST)**
Before any changes from A/B/C are merged, run the unmodified pipeline on the real full dataset to establish the true baseline:
```bash
# Step 1: build dense sample from full train (already done — sample_dense/ exists; skip if present)
# python code/business_entity_resolution/src/make_sample.py --data dataset/train --out sample_dense --mod 20

# Step 2: train on sample_dense
python code/business_entity_resolution/src/train.py \
  --data sample_dense --models code/business_entity_resolution/models --workers 4

# Step 3: predict on full test set
python code/business_entity_resolution/src/predict.py \
  --data dataset/test --models code/business_entity_resolution/models --out output --workers 4

# Step 4: validate format
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
Record: train time, predict time, countries processed, S1 entities with ≥1 match, total matched pairs.

**D2 — First leaderboard submission (Day 1 or early Day 2)**
If validation passes (`PASS`), upload `output/matching_results.tsv`. Record: submission time, public F0.5 score. This is Submission #1.

**D3 — Integration of Track A + B changes (Day 2, ~14:00–16:00 IST)**
After Tracks A and B commit their branches:
1. Merge `track-a-blocking` into `main`.
2. Merge `track-b-model` into `main`. Resolve any conflicts (most likely in `features.py` imports if Track B added new features that reference blocking-level data).
3. Re-run full pipeline (steps D1 steps 2–4).
4. Run `val_harness.py` and `error_analysis.py` from Track C against the train split (use held-out entities from `train_ground_truth.tsv` to avoid label leakage — use the same hash-based val split that `train.py` already uses).
5. If F0.5 improves on the val split, upload. This is Submission #2.

**D4 — Integration cycle 2 (Day 2 evening, ~20:00–22:00 IST)**
Incorporate any Track C diagnostic findings (from `error_analysis.py` output), and any follow-up tuning from Tracks A/B. Re-run full pipeline. Submit if improved. This is Submission #3.

**D5 — Day 3 final runs (Day 3, ~10:00–16:00 IST)**
At most 2 more submissions. By 16:00 IST, freeze the code, do the final full pipeline run with the best known configuration, validate, and upload. Reserve 1 submission slot as a safety re-upload in case of a last-minute format issue.

**D6 — Submission package assembly (Day 3, ~16:00–18:00 IST)**
Using Track C's package script:
- Copy final `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Copy all `src/*.py` files.
- Copy filled `Documentation_template.md`, `README.md`, `requirements.txt`.
- Zip as `<team_name>_submission.zip` and verify structure matches the README spec.

**Monitoring and communication.** After every full pipeline run, paste the 5-line status update into WORKPLAN.md (Section 5). The blocking recall and validation F0.5 from each run must be recorded so the team can see trend without a meeting.

#### Submission log (Track D maintains this table)

| # | Date/Time (IST) | Code State | Val F0.5 (train split) | LB F0.5 | Notes |
|---|---|---|---|---|---|
| 1 | | baseline | | | first real-data run |
| 2 | | A+B merged | | | |
| 3 | | A+B+C fixes | | | |
| 4 | | best config | | | |
| 5 | | final | | | |

#### Done criteria for Track D
- [ ] At least 1 leaderboard submission made by end of Day 1 or early Day 2.
- [ ] Integration of Tracks A+B completed by Day 2 ~16:00 IST with a full pipeline run.
- [ ] At least 3 leaderboard submissions made by end of Day 2.
- [ ] Final submission (best known config) uploaded by Day 3 ~16:00 IST.
- [ ] Submission package zip assembled and ready by Day 3 ~18:00 IST.

---

## 4. Integration Phase

Integration happens exactly twice (plus the Day 3 final):

### Integration 1 — Day 2, ~13:00 IST
**Trigger:** Tracks A and B both commit their branches by Day 2 13:00 IST.  
**Who:** Track D does the merge; Track B reviews feature list conflicts.

Steps:
1. `git checkout main && git merge track-a-blocking`
2. `git merge track-b-model`  
   - Likely conflict: `features.py` if B added new features and A added address-token logic that feeds features. Resolve: keep both additions; recheck `F1` list length matches `_chunk()` return tuple length.
   - `norm.py`: if A changed `keys_for()` or added key types, and B changed `skel()` — merge carefully; run `python -c "from norm import norm_name, norm_addr; print(norm_name('test'))"` to check imports.
3. Full pipeline run (D3 above). Record train/predict time.
4. Track C runs `val_harness.py` on train-val split output.
5. Go/no-go for Submission #2.

### Integration 2 — Day 2, ~20:00 IST
**Trigger:** Track C has error analysis output; Tracks A/B may have follow-up patches.  
**Who:** Track D merges; all four people review the `error_analysis.py` output together (15-min sync).

Steps:
1. Merge any follow-up commits from A and B.
2. Full pipeline run.
3. Validate, check val F0.5 improvement.
4. Go/no-go for Submission #3.

### Integration 3 (Final) — Day 3, ~12:00–15:00 IST
**Trigger:** No more code changes. Only config/threshold adjustments allowed after this point.  
**Who:** Track D runs; Track C validates submission package structure.

Steps:
1. Final full pipeline run with best configuration.
2. `utils/validate_submission.py` must print `PASS`.
3. `val_harness.py` must show no regression from Submission #3 val F0.5.
4. Upload `matching_results.tsv`. This is the final leaderboard submission.
5. Track C zips the package.

**Merge conflict prevention rules (follow from Day 1):**
- Track A only edits: `norm.py`, `blocking.py`, and new `blocking_audit.py`.
- Track B only edits: `features.py`, `model.py`, and new `tune_hyperparams.py`.
- Track C only creates: `val_harness.py`, `error_analysis.py`, wrapper scripts, `Documentation_template.md`.
- Track D only edits: `train.py`, `predict.py` (and only when integrating), `pipeline.py`, `io_utils.py`.
- `evaluate.py` belongs to Track C; if Track D needs it, read-only.
- Nobody edits `make_sample.py` (sample already built in `sample_dense/`).
- Nobody edits `utils/validate_submission.py` (it is provided by the organisers).

---

## 5. Daily Status Updates

Each person writes exactly 5 bullet lines: `DONE`, `DONE`, `DONE`, `BLOCKED` (or `DONE`), `NEXT`. Update in place under the dated section — do not append duplicate dated headers.

---

### Day 1 — 25 Sep 2026

**Person A (Blocking)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person B (Features/Model)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person C (Validation/Docs)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person D (Integration Lead)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

---

### Day 2 — 26 Sep 2026

**Person A (Blocking)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person B (Features/Model)** — Pratham
- [x] DONE: Added 6 new features to `features.py` (ajw, alev, alen_d, ncontain, akey_jac, wcount_d) — see B2 result above for full detail
- [x] DONE: Verified no regression — blocking recall identical (India 0.9284/US 0.9720), val F0.5 improved 0.9653 → 0.9666
- [x] DONE: Fixed a stale `blocking.py`/`norm.py` mismatch on the `pathu` branch's Colab clone (branch itself was already correct — false alarm, but worth noting since it cost time)
- [x] DONE: Ran full `predict.py` with new models → `output_v2/`, validated PASS (1,639,212 non-empty matches)
- [ ] NEXT: Submit original `output/` as one leaderboard upload + `output_v2/` as a second, compare real LB scores; decide whether full-`dataset/train` training is worth the RAM/time cost

**Person C (Validation/Docs)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person D (Integration Lead)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

---

### Day 3 — 27 Sep 2026

**Person A (Blocking)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person B (Features/Model)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person C (Validation/Docs)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

**Person D (Integration Lead)**
- [ ] DONE:
- [ ] DONE:
- [ ] DONE:
- [ ] BLOCKED/DONE:
- [ ] NEXT:

---

## 6. Timeline

> All times IST. Each full pipeline run is estimated at 30–90 minutes depending on hardware (make_sample is ~10–15 min, train ~20–40 min on sample_dense, predict ~30–60 min on full test data).

```
Day 1 — 25 Sep (tonight, ~4 hours remaining)
19:45  Hackathon kickoff / WORKPLAN.md created
20:00  All four people read WORKPLAN.md, claim their track, create their git branch
20:00  Person D: start baseline full-data pipeline run in background (train + predict)
20:00  Person A: run train.py on sample_dense/, record blocking recall baseline
20:00  Person B: run train.py on sample_dense/, record F0.5 baseline
20:00  Person C: start writing val_harness.py
22:00  Person D: validate output/ if run finished; prepare for Submission #1
22:30  Person D: Submit #1 to leaderboard (if PASS); record LB score in submission log
23:00  All: update Day 1 status lines in WORKPLAN.md
23:00  Person A: start blocking_audit.py (miss-pattern analysis)
23:00  Person B: start new feature additions
       — sleep / continue working overnight —

Day 2 — 26 Sep (main working day, ~18 hours)
09:00  All: update status if overnight work happened
09:00  Person A: blocking recall improvement from new key types (A3 task)
09:00  Person B: feature tuning + hyperparameter search (B2–B3 tasks)
09:00  Person C: finish val_harness.py, start error_analysis.py
09:00  Person D: if Submission #1 LB score received, record + share with team
11:00  Person A: re-run blocking audit after changes; check recall gain (A4 task)
11:00  Person B: LightGBM feature importance check; prune weak features
11:00  Person C: fill Documentation_template.md sections 1–3
12:00  Person D: check if Tracks A+B are merge-ready early; if yes, start Integration 1 early
13:00  INTEGRATION 1: Track D merges track-a-blocking + track-b-model → main
13:00  Person D: start full pipeline run #2 (Integration 1 run)
14:00  Person C: run val_harness.py on pipeline run #2 output (train val split)
14:30  Person D: go/no-go for Submission #2
15:00  Person D: Submit #2 (if improved); record LB score
15:00  Person C: run error_analysis.py; share output with full team
15:30  Persons A+B: read error analysis; identify 1–2 targeted fixes each
16:00  Person A: implement fix (e.g., raise max_block for A-type, add PIN key)
16:00  Person B: implement fix (e.g., adjust threshold range in train.py via Track D)
17:00  Person C: finish filling Documentation_template.md sections 4–5
18:00  Person A: commit follow-up fixes to track-a-blocking
18:00  Person B: commit follow-up fixes to track-b-model
19:00  All four: 15-min sync — review error analysis together
20:00  INTEGRATION 2: Track D merges follow-up changes → main
20:00  Person D: start full pipeline run #3
21:30  Person C: validate with val_harness.py + pre-submission wrapper
22:00  Person D: go/no-go + Submit #3; record LB score
22:30  All: update Day 2 status lines in WORKPLAN.md
23:00  Person C: submission package directory structure ready (C5 task)
       — sleep —

Day 3 — 27 Sep (final day, deadline ~22:00 IST)
09:00  All: read LB scores from Day 2 submissions; decide if any last targeted change is worth it
09:30  Person A: at most 1 more small blocking improvement (if Day 2 error analysis shows clear recall gap)
09:30  Person B: at most 1 more model/threshold tweak (e.g., widen thr sweep range if optimum was at edge)
10:00  HARD FREEZE: no more code changes after 10:00 IST
10:30  Person D: start final full pipeline run (INTEGRATION 3 run)
12:00  Person C: validate output with full pre-submission wrapper; check val_harness.py
12:30  Person D: Submit #4 (final intended submission); record LB score
13:00  Person C: finalise Documentation_template.md with final F0.5 score
14:00  Person C: run build_package script; assemble submission_package/ dir
14:30  Person D: review package structure against README spec; verify zip contents
15:00  SUBMISSION PACKAGE ready
15:30  Buffer: if LB score for #4 is lower than #3 (regression), investigate quickly
16:00  Person D: Submit #5 only if there is a clear regression fix or if #4 was a format issue
16:00  Otherwise: DONE — team can relax
16:00–22:00  Buffer before deadline; Submit #5 slot held in reserve
18:00  Final zip submitted to the portal (the zip package, separately from leaderboard upload)
```

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Full-data pipeline run takes >2 hours | Start D1 run tonight; if too slow, increase `--workers`; if memory-bound, process one country at a time in predict.py |
| France entities produce 0 candidates (new country, unseen in train) | `countries_of()` in `io_utils.py` reads country from the test file dynamically; blocking keys are already country-prefixed strings so France just gets its own key namespace. Track A verifies this in blocking_audit.py |
| Merge conflict in features.py (A and B both edit related code) | A only edits `blocking.py`/`norm.py`; B only edits `features.py`/`model.py`; split is clean |
| `decode()` 1-to-1 constraint misses multi-match S1 entities | Track A audit identifies this; if significant, Track B can relax to top-K per S1 in decode() instead of strict 1-to-1 per S2/S3 |
| `sample_dense/` F0.5 doesn't predict full-data F0.5 | Track D always runs on full test set; val_harness.py compares against train val split, not sample |
| Day 3 submission fails validation | Pre-submission wrapper (Track C) catches format issues before upload; 1 extra submission slot held in reserve |

---

*This document is living. Track D updates the submission log after every upload. Each person updates their 5 status lines each evening. Do not edit any section other than your own status lines and the submission log table unless you are Track D doing an integration.*