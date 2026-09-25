# Business Entity Resolution: reproduction guide

Blocking + two-stage LightGBM matcher. Pure Python (numpy, pandas, rapidfuzz, lightgbm); no external data,
APIs or pretrained models (all models are trained from the provided training data only).

Run everything from the `student_resource/` folder (the one that contains `dataset/`, `utils/`, `code/`).

```
pip install -r code/business_entity_resolution/requirements.txt

# 1) dense training sample (keeps all same-name look-alike businesses together; ~10-15 min, low memory)
python code/business_entity_resolution/src/make_sample.py --data dataset/train --out sample_dense --mod 20

# 2) train both stages + tune the decision threshold for macro F0.5 on a held-out validation split
python code/business_entity_resolution/src/train.py --data sample_dense --models code/business_entity_resolution/models --workers 4

# 3) generate both output files for the test set
python code/business_entity_resolution/src/predict.py --data dataset/test --models code/business_entity_resolution/models --out output --workers 4

# 4) validate
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline
1. `norm.py`: name/address normalisation (accent strip, abbreviation expansion, legal-suffix removal, state names, leading zeros),
   Indic-script (Devanagari, Telugu, Kannada, Tamil, ...) to Latin transliteration built only from Unicode character names,
   and a consonant-skeleton phonetic key comparable across scripts.
2. `blocking.py`: country-scoped keys (house-number+street word, sorted core name, name prefix/suffix tokens, rare address tokens);
   compound keys (name key x address token) keep otherwise-generic name keys sharp; key blocks larger than a cap are dropped as generic; top-K candidates per Source 1 record by shared-key weight.
   The candidates fed to the model are written to `output/candidate_pairs.tsv`.
3. `features.py`: rapidfuzz name/address similarities, token Jaccard, house-number agreement, exact/normalised address equality.
4. `model.py`: stage 1 LightGBM on pair features; stage 2 LightGBM on the context of stage-1 probabilities
   (rank / gap / competitors among the S1's candidates and among the S1 options of each S2/S3 record).
   Decoding: every S2/S3 record is assigned to at most one S1 (its best), only if probability >= threshold.
5. `train.py` tunes threshold/margin on held-out Source 1 entities (macro F0.5, singletons included).

`country` is only used to scope blocking; it is never one-hot encoded, so unseen countries (France) work.
