# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [FILL]
**Team Members:** [FILL: names]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We solve entity resolution as **blocking + a two-stage LightGBM matcher + a metric-aware decoder**.
Country-scoped blocking keys (address number + street word, sorted core name, phonetic skeletons,
name × address compound keys, address bigrams, website-style compact names) generate candidates;
stage 1 scores each pair from 36 string-similarity features, and stage 2 re-scores it from the
*competition* around it (how it ranks among its S1's candidates and among the S1 options of the
S2/S3 record). The key finding of the project was that the test set has ~5x more competing
businesses per record than a naive training sample, so we built a training sample that keeps
the real competition (rival businesses as context); this closed most of our local-vs-leaderboard gap.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Data:** train 2.21M S1 / 5.03M S2 / 5.29M S3 (India, US); test 1.73M S1 / 4.89M S2 / 5.08M S3
  (India, US and **France, unseen in training**). Metric: macro F0.5 per S1 entity, a correct empty
  prediction for a true singleton scores 1.0 (~5.5% of S1 are singletons, mean 3.5 matches per S1).
- **Noise patterns:** typos and dropped letters, abbreviations (`rd`/`road`, `pvt`/`private`),
  word-order changes, legal suffixes in many spellings, Indic scripts (Devanagari, Tamil, Telugu,
  Kannada, ...) for India, transliterated legal suffixes (`praivarr limirrad`), leading zeros and
  house-number typos, **empty addresses** (a large share of our model's misses), website-style
  names (`allshivsystemscom`).
- **Sibling businesses:** many different businesses share one address or a near-identical name,
  so similarity alone is not enough; the model has to compare a pair against its rivals.
- **Competition density (the most important finding):** on the real test, each S2/S3 record is a
  blocking candidate of **~9.5 S1 businesses** on average (71% have ≥ 5). A training sample that
  keeps only sampled S1 has ~2 (3–7% have ≥ 5). Models trained and validated like that were
  over-confident: local validation over-promised the leaderboard by ~1.5 points, consistently.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage gradient-boosted classifier + expected-F0.5-aware decoding (hybrid).
**Core Innovation:** training on a sample that reproduces the real test-time competition
(`make_sample_v3.py`: sampled S1 + every rival S1 + the real full-density candidate pairs), combined
with a stage-2 model that reasons about that competition.

Rules we held ourselves to: no external data or pretrained models; no country-specific hardcoding
(France is handled like any other country); every choice (early stopping, threshold, margin,
decoder, blend weight) is made on one half of the validation entities (ES) and reported on the
other, never-touched half (REP); every new behaviour sits behind a config flag, so older models
reproduce exactly.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all prefixed with the country, `src/blocking.py`):
  - `A` house number + next street word (skipping per-country generic words learned from the data);
  - `N` sorted core name tokens (legal suffixes removed); `P` name prefix/suffix; `Q` compact sorted name;
  - `V`/`K` phonetic consonant skeleton of the name (works across Latin and transliterated Indic text);
  - `T`/`U` long rare name tokens; `R`/`S` rare long address tokens;
  - `X` compound name-key × address-token keys (turn generic name keys into sharp ones);
  - `B` address bigram containing a number, `Y` (house number × rare address word), `C` compact /
    initials-based names matching website-style names (keys_v2).
- **Pruning:** key groups larger than a cap (60 other-side records / 200 S1) are dropped as generic;
  per S1, the top 60 candidates by summed key weight are kept.
- **Candidate pairs generated:** [FILL from the final predict log: total over India / US / France].
- **How we ensured true matches were not lost:** we diagnosed every missed true pair on the training
  data (`diag_blocking.py`: no shared key / only oversized keys / cut by top-K). Raising caps or
  top-K barely helped (India +0.6 pt recall for 2.3x candidates), so we added sharper keys instead
  (`B`, `Y`, `C`): India recall 0.906 → 0.928, US 0.963 → 0.970 for ~10% more candidates. Measured
  at real density: blocking recall **0.950** (India 0.923, US 0.968), i.e. a perfect matcher on our
  candidates would score 0.982.

---

## 4. Matching Model

**Features used** (stage 1, `src/features.py`):
- **Name:** rapidfuzz ratio / token_sort / token_set / partial ratio, Jaro-Winkler and Levenshtein on
  core names, core-token Jaccard and exact equality, first-token equality, containment, length and
  word-count differences, phonetic-skeleton similarities.
- **Address:** the same string similarities on the cleaned address, token Jaccard, number-token Jaccard
  and equality, first-number equality, blocking-key overlap, exact equality, empty-address flags,
  **house-number edit distance and numeric gap** (a typo'd number vs a different building).
- **Other:** the blocking key weight of the pair.
- **v11 only (flag `--feat-v3`):** digit-span Jaccard, exact skeleton equality, **IDF-weighted name
  and address similarity** (word rarity learned from the S1 file itself, so a shared rare word counts
  more than a shared common one); stage 2 also gets how many S1 businesses share the record's exact
  address / name skeleton (sibling density).

**Stage 2** (`src/model.py`): the stage-1 score of the pair plus its rank and gap inside the S1's
candidate list and inside the S2/S3 record's S1 options, the strongest rival score and the sum of
rival scores, plus 11 raw pair features.

**Model type:** LightGBM, both stages (stage 1: up to 2000 trees, 63 leaves; stage 2: up to 800
trees, 31 leaves), early stopping on the ES half. Stage-1 scores used to train stage 2 are
out-of-fold. 8 model families were benchmarked through the same pipeline; LightGBM won.
Rival businesses enter training as context rows: they are scored by stage 1 exactly like test rows
and used only as competition for stage 2 and the decoder, never as labelled examples.

**Threshold selection method:** each S2/S3 record is assigned to at most one S1 (its best) and kept
only if p2 ≥ thr and p2 − runner-up ≥ margin; thr and margin are chosen by sweeping macro F0.5 on
the ES half (final model: thr [FILL], margin [FILL]). We also built an expected-F0.5 decoder
(isotonic calibration + choosing the best number of matches per S1); it is used only when it beats
the threshold rule on the ES half.

---

## 5. Results & Error Analysis

| model | main change | local clean-half F0.5 | leaderboard |
|---|---|---|---|
| v6 | 36 features, generic address words | – | 0.901 |
| v7 | realistic-density sample | – | 0.93 |
| v8 | sharper blocking keys, early stopping | 0.9532 (sparse sample) | 0.94 |
| v9 | + expected-F0.5 decoder | 0.9554 (rival sample) | 0.945 |
| v10 | trained with rival businesses as context | **0.9621** (rival sample) | [FILL] |
| v11 | + IDF / density features | [FILL] | [FILL] |

Local numbers from the rival sample (`sample_v3_25`) track the leaderboard (v9: 0.955 local vs 0.945 LB);
numbers from the older sparse sample did not.

- **F_0.5 Score (macro):** [FILL: final model] on the untouched validation half.
- **Where the remaining points go** (v9 on the rival sample, 4.8 points lost in total): partial matches
  (some right, some missed) 3.3; true pairs blocked but rejected 0.9; no true pair blocked 0.5;
  singletons we matched 0.1. India is the harder country (0.93 vs 0.97 for US on v9).
- **Common false positives (wrong merges):** sibling businesses at the same address with different
  or garbled names; common names with empty addresses; house numbers off by a few digits.
- **Common false negatives (missed matches):** empty or one-word addresses; heavy transliteration
  noise and unstripped legal-suffix variants in Indic names; records whose only shared keys are
  generic; the model being cautious when many rivals look similar (precision 0.99 vs recall 0.88 on v9).

---

## 6. Conclusion

A classical, fully data-driven pipeline (normalisation, targeted blocking keys, two-stage LightGBM)
reached [FILL] macro F0.5. The largest single improvement did not come from a new feature or model
but from making training and validation look like the test: once rival businesses were kept in the
training sample, local validation started to predict the leaderboard. Next steps would be linking
the S2 and S3 records of the same business to each other (to recover partial matches) and blocking
keys for records with empty addresses.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/`: `norm.py` (normalisation, transliteration, phonetic keys),
`blocking.py` (keys, candidate generation), `features.py` (pair features), `model.py` (two stages,
decoders), `pipeline.py` (per-country glue), `train.py`, `predict_ajusbyjus.py` (inference, writes
both output files), `eval_full.py` (scoring with loss breakdown), `make_sample_v3.py` +
`shrink_sample_v3.py` (training sample with rivals). Exact commands to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` are in
`code/business_entity_resolution/README.md`; the internals are in `ARCHITECTURE.md`.

### B. Additional Results

- Rival density (mean S1 candidates per S2/S3 record): sparse training sample India 2.0 / US 1.8;
  real test India 9.7 / US 8.9 / France 9.4.
- Tried and rejected: per-country thresholds (+0.0002, noise), a separate threshold for empty
  addresses (+0.0001), larger block caps / top-K (tiny recall gain for 2x candidates), learned legal
  suffixes (picked up real words such as "technologies").
