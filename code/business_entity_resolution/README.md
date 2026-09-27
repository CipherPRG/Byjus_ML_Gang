# Business Entity Resolution: reproduction guide

Blocking + two-stage LightGBM matcher. Pure Python (numpy, pandas, rapidfuzz, lightgbm, scikit-learn);
no external data, APIs or pretrained models (all models are trained from the provided training data only).
`ARCHITECTURE.md` explains every file and design decision in detail.

## Reproduce the submission

Run from `student_resource/code/business_entity_resolution` (the folder with `src/`). Paths below
assume the data is in `student_resource/dataset/`. `--workers` = CPU processes for feature building.

```
pip install -r requirements.txt

# 1) training sample WITH rival businesses (real test-time competition): 5% of S1 + every rival S1 +
#    the real full-density candidate pairs (train_pairs.tsv). Runs the real blocking on full train.
python src/make_sample_v3.py --data ../../dataset/train --out ../../sample_v3 --frac 0.05 --keys-v2

# 2) shrink it to 2.5% of S1 so training fits in 16 GB RAM (exactly what --frac 0.025 would build)
python src/shrink_sample_v3.py --src ../../sample_v3 --out ../../sample_v3_25 --frac 0.025

# 3) train both stages; thr/margin chosen on the ES half of validation, honest score printed as CLEAN
python src/train.py --data ../../sample_v3_25 --models ../../models_v10 --workers 10 --addr-stop-frac 0.01 --keys-v2

# 4) predict the test set (writes matching_results.tsv, candidate_pairs.tsv, scores_<country>.npz)
python src/predict_ajusbyjus.py --data ../../dataset/test --models ../../models_v10 --out ../../output --workers 10

# 5) validate (from student_resource/)
cd ../..
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

[FILL: if the final model is v11, step 3 becomes
`python src/train.py --data ../../sample_v3_25 --models ../../models_v11 --workers 10 --addr-stop-frac 0.01 --keys-v2 --feat-v3 --density-src ../../dataset/train/train_source1.tsv`
and step 4 uses `--models ../../models_v11`. If a v10+v11 blend was submitted, add the two
`blend_scores.py` commands from the repo root.]

The blocking configuration and feature flags are saved in `<models>/config.json`, so predict and
eval always rebuild exactly what the model was trained with. Run times on the team laptop
(16 GB RAM): sample build ~1 h, train ~1.5–2 h, predict ~80 min.

## Pipeline
1. `norm.py`: name/address normalisation (accent strip, abbreviation expansion, legal-suffix removal incl.
   French/international forms, state names, leading zeros, spelled-out ordinals), Indic-script to Latin
   transliteration built only from Unicode character names, and a consonant-skeleton phonetic key.
2. `blocking.py`: country-scoped keys (house number + street word, sorted core name, name prefix/suffix,
   phonetic skeletons, rare address tokens, name × address compound keys, address bigrams, compact
   website-style names); oversized key groups are dropped as generic; top-60 candidates per S1 by
   shared-key weight. These candidates are written to `candidate_pairs.tsv`.
3. `features.py`: 36 pair features (rapidfuzz name/address similarities, token and number Jaccard,
   house-number edit distance, exact/normalised equality, empty-address flags, blocking weight);
   `--feat-v3` adds digit-span Jaccard, skeleton equality and IDF-weighted name/address similarity.
4. `model.py`: stage 1 LightGBM on pair features; stage 2 LightGBM on the competition around each pair
   (rank / gap / strongest and summed rivals among the S1's candidates and among the S1 options of each
   S2/S3 record). Decoding: every S2/S3 record goes to at most one S1 (its best), only if
   p ≥ thr and p − runner-up ≥ margin (or the expected-F0.5 decoder if `decoder.json` exists).
5. `train.py`: rival S1 businesses are context rows (scored like test rows, never trained on);
   early stopping and thr/margin on the ES half of the held-out S1 entities, CLEAN score on the other half.

`country` is only used to scope blocking; it is never a model feature, so unseen countries (France) work.
