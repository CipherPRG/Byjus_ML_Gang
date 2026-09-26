# Business Entity Resolution — Architecture & Dataset Reference

Companion to `README.md` (quick-start commands). This file goes deeper: exact dataset
shape, what every source file does internally, and the relations/gotchas that aren't
obvious from reading one file at a time. Last verified against the code and data on
disk on 2026-09-26 (post commit `bb36851`, `models_v3` retrain).

## 1. Problem recap

3 noisy business-record sources per country. Source 1 = clean reference entities.
Source 2/3 = noisy duplicate listings (typos, missing fields, some non-English
scripts). For every Source 1 entity, find all matching Source 2/3 records. Scored by
macro-averaged F0.5 per S1 entity; an empty prediction for a true singleton scores a
perfect 1.0. No external APIs/lookups; model trained only on the provided data.

## 2. Dataset — exact shape

| file | rows | columns |
|---|---|---|
| `dataset/train/train_source1.tsv` | 2,206,821 | entity_id, business_name, business_address, country |
| `dataset/train/train_source2.tsv` | 5,034,616 | same |
| `dataset/train/train_source3.tsv` | 5,285,603 | same |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 (1 row per S1) | source1_entity_id, matched_entity_ids (comma-joined S2/S3 ids, empty = true singleton) |
| `dataset/test/test_source1.tsv` | 1,732,544 | entity_id, business_name, business_address, country |
| `dataset/test/test_source2.tsv` | 4,887,273 | same |
| `dataset/test/test_source3.tsv` | 5,082,316 | same |

Train countries: **India, US** only. Test countries: **India, US, France** — France is
*unseen at training time*, a deliberate zero-shot generalization check (see §6).
`entity_id` prefixes (`S1-`, `S2-`, `S3-`) only identify source; they carry no relation
to each other numerically — the only ground-truth relation is `train_ground_truth.tsv`.
`business_name`/`business_address` for India rows are frequently in Devanagari or other
Indic scripts, not just Latin with typos.

`sample_dense/` (local-only, not in git — pushed as `sample_dense.zip` to `main` this
session, regenerate via `make_sample.py` if missing) is a smaller dense subsample of
`train/` used for all fast iteration (model_bench, train.py sweeps, the two experiment
scripts below) instead of the full 2.2M-entity train set.

## 3. Pipeline, file by file

**`norm.py`** — normalization primitives, no external data/APIs.
- `translit()`: Indic-script → Latin transliteration built *purely* from
  `unicodedata.name()` lookups (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
  Telugu, Kannada, Malayalam) — no transliteration library or lookup table beyond
  Unicode's own character names.
- `skel()`: crude consonant-skeleton phonetic key (drops vowels, collapses doubled
  consonants) so transliterated Indic text and native Latin text become comparable.
- `norm_name()` → (clean string, core tokens with legal suffixes like "llc"/"pvt"
  stripped). `norm_addr()` → (clean string, tokens; abbreviation expansion, US state
  name → abbreviation, leading zeros stripped). `addr_keys()` → up to 3 (house-number,
  following street word) pairs, e.g. `("41", "groton")`.

**`blocking.py`** — candidate generation (this is what `output/candidate_pairs.tsv` is).
- `keys_for()` builds several typed keys per record — address number+word (`A`), sorted
  core name (`N`), name prefix/suffix (`P`), phonetic-skeleton keys (`V`/`K`), a
  **compound key** = name-key × address-token (`X`, sharpens an otherwise-generic name
  key), rare long address tokens (`R`/`S`). `WEIGHT` ranks reliability per key type
  (address/compound/exact-name keys weighted highest).
- `Side` holds one source's normalized records + flattened (key, row, weight) arrays;
  `Side.concat()` merges Source2+Source3 into one "other" side.
- `candidates()` merges S1 keys against "other" keys sharing a hash, drops
  overly-generic key groups (size cap), sums shared-key weight per (r1, ro) pair, keeps
  top-K by weight per S1 record. **Country is folded into every key string, never used
  as a model feature** — this is exactly why an unseen country (France) still works:
  blocking only needs France records to share keys with other France records.

**`features.py`** — 32 hand-engineered features per candidate pair (`F1` list), via
rapidfuzz (ratio/token_sort/token_set/partial_ratio, JaroWinkler, Levenshtein) plus
custom exact/Jaccard/length-diff comparisons across cleaned name, core tokens, phonetic
skeleton, cleaned address, address number-tokens, address keys, and the blocking weight
`w` itself as a feature. Parallelized via `ProcessPoolExecutor` when `workers>1`.
*(Was 33 features before commit `bb36851` dropped one dead-weight feature — see §7.)*

**`model.py`** — two-stage LightGBM matcher.
- Stage 1: LightGBM on the 32 raw pair features → `p1`.
- Stage 2: LightGBM on **context** of `p1` — rank/gap of this pair within its S1's
  candidate list, rank/gap within the "other" record's S1 options, sum of competing
  probabilities — plus 10 raw features (`RAW2`) carried through. This is what lets the
  model reason about competition ("best match for both sides, or is there a stronger
  rival candidate"), not just score each pair in isolation.
- `decode()`: each S2/S3 record is assigned to **at most one** S1 — its single
  best-scoring match — only if `p2 >= thr` AND `(p2 - runner_up_p2) >= margin`.
  `decode_prep()`/`decode_apply()` split the expensive sort from the cheap thr/margin
  filter so a full sweep only sorts once (perf fix, see comments in the file).

**`pipeline.py`** — glue: `build_country()` combines Source2+Source3 into one "other"
side and runs `candidates()` for one country; `load_side()` streams a source file in
chunks filtered to one country.

**`io_utils.py`** — TSV I/O (all-string dtype, no NaN coercion); `write_lists()` writes
the two-column format both `matching_results.tsv` and `candidate_pairs.tsv` use.

**`evaluate.py`** — `f05_macro()`, the actual competition metric: per-S1-entity F0.5,
empty-predicted == empty-truth scores 1.0 (rewards true singletons), macro-averaged
(every S1 entity weighted equally regardless of match count).

**`train.py`** — normal mode: `build_country` per country on `sample_dense`, fit stage1
(2-fold out-of-fold to avoid stage2 overfitting on stage1's own training predictions),
fit stage2, sweep thr×margin (35×23=805 combos) on held-out **entities** (S1 ids held
out via `crc32` hash so no pair from the same S1 leaks between train/val), save
`config.json` + `stage1.txt` + `stage2.txt`. Final mode: retrain on 100% of data with a
pre-locked thr/margin — for the last training pass before the deadline.

**`predict.py`** (and Track D's hardened `predict_ajusbyjus.py`, currently only on
`origin/ajus-byjus`) — loads `config.json`+`stage1.txt`+`stage2.txt`, runs
`build_country`→`pair_features`→`stage2_matrix`→`decode` per test country, batches
stage-1 inference (default 2M pairs/batch) to bound RAM, writes both output TSVs.
Track D's hardened version adds: `psutil` RAM warning, configurable `--batch`,
France/unseen-country sanity check (warns if an expected-unseen country is missing from
test S1, or produces zero matches despite having candidates), per-batch progress, and a
final run-summary block.

**`make_sample.py`** — builds `sample_dense/` from `dataset/train/`.

**Track B experiment scripts (this session, `pathu` branch):**
- `model_bench.py` — pluggable model registry; benchmarked 8 model families through the
  identical OOF+stage2+decode pipeline. LightGBM won outright (0.9674) — closes the
  "which classifier" question.
- `per_country_threshold.py` — per-country thr/margin sweep vs one global value.
  Result: +0.0002, noise — not adopted.
- `cross_source_experiment.py` — new stage2 feature (does an S1 entity have independent
  strong evidence, p1≥0.70, from *both* Source2 and Source3). Result: not yet run.
- `check_feature_importance.py` / `check_stage2_importance.py` — LightGBM gain-based
  importance audits; identified the dead feature dropped in `bb36851`.

## 4. Model artifacts on disk (as of this session)

| dir | features trained on | thr / margin | val F0.5 | status |
|---|---|---|---|---|
| `models/` | 33 (pre-`bb36851`) | 0.70 / 0.20 | 0.9667 | **stale** |
| `models_tuned/` | 33 (pre-`bb36851`) | 0.70 / 0.20 | 0.9668 | **stale** |
| `models_v3/` | 32 (current) | 0.94 / 0.34 | 0.9674 | **current** — use this |

`output_v3/matching_results.tsv` + `candidate_pairs.tsv` were generated from
`models_v3` against the real full test set (this session). Note: `models/` and
`models_tuned/` live inside `code/business_entity_resolution/`, but `models_v3/` was
trained with `--models ..\..\models_v3` from inside `code/business_entity_resolution`,
so it landed at the **repo root** (`student_resource/models_v3/`), one level up from
the other two — same content either way, just a different path this one time.

## 5. Known relations / gotchas for the team

- **Model artifacts are tightly coupled to `features.py`'s exact column count/order at
  training time.** Any future feature add/drop requires retraining every `models*/`
  folder before `predict.py` runs again — LightGBM throws a hard shape-mismatch error
  otherwise (hit this exact issue this session: `bb36851` dropped a feature, nobody
  retrained, `models`/`models_tuned` became stale).
- `country` is a blocking key only, never a model feature — this is the entire reason
  the model generalizes zero-shot to France.
- Macro-F0.5 rewards correctly predicting true singletons as a perfect 1.0 — the
  `decode()` thr/margin choice directly controls this precision/recall trade-off.
- `sample_dense/` is local-only/untracked; regenerate via `make_sample.py` or unzip
  `sample_dense.zip` from `main` if missing.

## 6. Reproduction (see also `README.md`)

```
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/make_sample.py --data dataset/train --out sample_dense --mod 20
python code/business_entity_resolution/src/train.py --data sample_dense --models models_v3 --workers 8
python code/business_entity_resolution/src/predict.py --data dataset/test --models models_v3 --out output --workers 8
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
