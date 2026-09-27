# Business Entity Resolution — reproduction guide

Blocking + two-stage LightGBM matcher + metric-aware decoder. Pure Python (numpy, pandas, rapidfuzz,
lightgbm, scikit-learn). No external data, APIs, lookups or pretrained models: everything is learned from the
provided training data. `country` is treated as an open set (it only scopes blocking keys and is never a model
feature), so a country unseen in training (France) is processed exactly like the others.
`ARCHITECTURE.md` explains every file and design decision; the methodology write-up is `Documentation_template.md`
at the root of the submission zip (`ML_chads_Documentation.md` in the repository).

## Layout

```
src/                     the pipeline (everything needed to reproduce the submission)
  norm.py                normalisation: accents, legal suffixes, abbreviations, Indic-script transliteration,
                         phonetic consonant skeleton, per-country generic-word learning
  blocking.py            blocking keys + candidate generation (incl. optional keys_v3 extra candidates)
  features.py            pair features (36 base, 42 with --feat-v3: + IDF-weighted similarities)
  model.py               stage 1 / stage 2 LightGBM, thr/margin decoder, expected-F0.5 decoder
  pipeline.py            per-country glue: load -> normalise -> block (or recorded pairs) -> density/IDF
  evaluate.py            the competition metric (macro F0.5 per Source-1 entity, empty==empty scores 1)
  io_utils.py            TSV reading/writing
  make_sample_v3.py      builds the training sample WITH the real competing businesses (step 1)
  shrink_sample_v3.py    shrinks it to fit 16 GB RAM, exactly as a smaller --frac would (step 2)
  train.py               trains both stages, chooses everything on the ES half of validation (step 3)
  predict.py             writes output/matching_results.tsv + output/candidate_pairs.tsv (step 4)
  eval_full.py           scores a model on labelled data with a loss breakdown (analysis)
tests/test_pipeline.py   20 fast unit tests (no dataset needed)
experiments/             one-off analysis scripts, kept for transparency; NOT needed to reproduce
reports/                 blocking-audit tables referenced in the documentation
```

## Reproduce the submission

Run from `student_resource/code/business_entity_resolution` (the folder containing `src/`), with the data in
`student_resource/dataset/`. `--workers` = number of CPU processes for feature building.

```
pip install -r requirements.txt
python -m unittest discover -s tests            # optional: 20 unit tests, < 1 s

# 1) training sample with the real competition: 5% of Source-1 entities (hash-selected), every rival S1 that is a
#    real blocking candidate of their S2/S3 records, and the real full-density candidate pairs (train_pairs.tsv)
python src/make_sample_v3.py --data ../../dataset/train --out ../../sample_v3 --frac 0.05 --keys-v2

# 2) shrink to 2.5% of S1 so training fits in 16 GB RAM (identical to what --frac 0.025 would build)
python src/shrink_sample_v3.py --src ../../sample_v3 --out ../../sample_v3_25 --frac 0.025

# 3) train (early stopping, threshold/margin and decoder chosen on the ES half; honest score printed as CLEAN)
python src/train.py --data ../../sample_v3_25 --models ../../models_v11 --workers 10 --addr-stop-frac 0.01 --keys-v2 --feat-v3 --density-src ../../dataset/train/train_source1.tsv

# 4) predict the test set with the extra word-pair candidates
python src/predict.py --data ../../dataset/test --models ../../models_v11 --out ../../output --workers 10 --keys-v3

# 5) validate (from student_resource/)
cd ../..
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Everything the model needs at predict time is saved by `train.py` in the model folder: `config.json` (blocking
configuration and feature flags, so predict rebuilds exactly what the model was trained with), `stage1.txt`,
`stage2.txt` and `decoder.json` (expected-F0.5 decoder; used because it beat the threshold rule on the ES half).

Run times on the team laptop (ASUS TUF F16, 16 GB RAM, 10 workers): step 1 ≈ 1 h, step 2 ≈ 15 min,
step 3 ≈ 55 min, step 4 ≈ 80 min. Steps 3–4 use most of 16 GB RAM (close other apps).
All sampling and validation splits are hash-based (deterministic); LightGBM is run with its default seed.

## Pipeline in one paragraph

Records are normalised (`norm.py`). Country-scoped blocking keys (house number + street word, sorted core name,
phonetic skeletons, name × address compound keys, address bigrams, compact website-style names, and — with
`--keys-v3` — pairs of rare name words / rare address words) generate candidates; generic keys whose block is
too large are dropped and the top 60 candidates per Source-1 record are kept. Stage 1 (LightGBM) scores each pair
from string-similarity features; stage 2 (LightGBM) re-scores it from the competition around it (its rank and
gap among the S1's candidates and among the S1 options of the S2/S3 record). Each S2/S3 record is assigned to at
most one S1 (its best), kept only if the decoder accepts it. All choices are made on the ES half of the held-out
Source-1 entities; the other (REP) half is never used for any choice and gives the honest score.

## Flags

| flag | where | meaning | submitted model |
|---|---|---|---|
| `--keys-v2` | train | sharper blocking keys (address bigrams, number × rare word, compact names) | on |
| `--addr-stop-frac 0.01` | train | per-country generic address words learned from the data | 0.01 |
| `--feat-v3` + `--density-src` | train | IDF-weighted similarities + sibling density (counted on the FULL S1 file) | on |
| `--keys-v3` | predict / eval | extra candidates from rare word pairs; normal candidates untouched | on |
| `--learn-suffix` | train | learned legal suffixes (experimental, learns real words) | off |

All flags default to off, so older models reproduce exactly.
