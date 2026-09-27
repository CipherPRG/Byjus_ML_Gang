# Analysis report: where we lose points, known bugs, what to improve (27 Sep 2026, 11:45 IST)

Written for teammates **and their CLI agents** to analyse. All numbers are measured, not
estimated, unless marked *(est.)*. Code lives in `code/business_entity_resolution/src/`
(branch `pathu`); see `code/business_entity_resolution/ARCHITECTURE.md` for the full pipeline.

**Ground rules for any fix (from the team lead):**
- No hardcoding (no country-specific rules, no lists written for France, no hand-picked thresholds).
- No overfitting: anything chosen from data must be chosen on the **ES half** of validation
  (`es_half_of()` in `train.py`); the **REP half is never used for any choice** and is the honest score.
- New blocking/decoding behaviour goes behind a config flag that defaults to off, so older models
  reproduce exactly.
- Heavy runs (train / predict / eval on full data) are done on Pratham's laptop only.
  Teammates: code analysis, small-sample tests (`sample/dataset/train`), suggestions.

---

## 1. Status

| model | what changed | local val (sample_v2) | clean half | leaderboard |
|---|---|---|---|---|
| v6 | 36 features, generic-address stop words, trained on sample_dense | 0.892 | – | **0.901** |
| v7 | trained on sample_v2 (realistic density), thr 0.98 | 0.9452 | – | **0.93** |
| v8 | + blocking keys_v2, 2000/800 trees + early stopping, clean val split | 0.9537 | 0.9532 | **0.94** |
| v9 (predicting) | v8 + expected-F0.5 decoder (`models_v9/decoder.json`) | ~0.955 | 0.9543 (v8 thr: 0.9530) | pending |
| v10 (planned) | trained on sample_v3 with rival context (§3.0) | – | – | – |

The leaderboard sits **~1.4–1.5 pt below local** for both v7 and v8 (systematic, see §3.3).

## 2. Where the F0.5 points are lost (v7 on sample_v2 val, thr 0.98; v8 is similar)

| | India | US | all |
|---|---|---|---|
| macro F0.5 | **0.9198** | 0.9617 | 0.9449 |
| ceiling (perfect decode on our candidates) | 0.9631 | 0.9882 | 0.9781 |
| blocking recall (true pair became a candidate) | 0.906 (v8: 0.928) | 0.963 (v8: 0.970) | 0.940 |
| FN from blocking / FN from model / FP (pairs) | 4,111 / 3,131 / 581 | 2,448 / 3,714 / 489 | |
| pts lost: partial (some right, some missed/extra) | 5.05 | 2.84 | 3.73 |
| pts lost: has matches, predicted none, none blocked | 1.27 | 0.24 | 0.65 |
| pts lost: has matches, predicted none, blocked but rejected | 1.35 | 0.59 | 0.89 |
| pts lost: singleton but we matched something | 0.29 | 0.14 | 0.20 |

**India is the biggest lever**: it is ~47% of the test set (810k of 1.73M S1) but scores ~4 pt below US.

## 3. Bugs / weaknesses, ranked by expected impact

### 3.0 NEW (12:00): training/validation sample has ~5x too few rival businesses - fix built, being run
`rival_density.py eval_cache_v8 output_v8/candidate_pairs.tsv` - how many S1 businesses are
blocking candidates of each S2/S3 record:

| | mean S1 rivals | >=2 | >=5 | >=10 |
|---|---|---|---|---|
| sample_v2 India / US (what we train + validate on) | 2.0 / 1.8 | 52% / 46% | 7% / 3% | 0.1% / 0% |
| real test India / US / France | 9.7 / 8.9 / 9.4 | 93% / 94% / 94% | 71% / 71% / 74% | 39% / 37% / 42% |

sample_v2 keeps only the sampled 5% of S1; the rival S1 were dropped (they had no GT rows). So
stage 2's competition features (rank/gap/strongest rival inside each S2/S3 record) and the
"each record -> its best S1" decode were trained and validated with ~5x too little competition,
and local val is optimistic. This is the most likely cause of the systematic LB gap (§3.3).
**Fix (built + smoke-tested, not hardcoding - it restores the real data distribution):**
- `src/make_sample_v3.py`: same sampled S1 as sample_v2 (same hash) + every rival S1, and
  `train_pairs.tsv` = the REAL full-density candidate pairs (no re-blocking inside the sample).
- `pipeline.build_country`: uses `<data>/<prefix>_pairs.tsv` if present (never present for test).
- `train.py`: trains/scores only S1 with GT rows; rival pairs are context rows - features to disk
  in batches, scored by the final stage-1 model like test rows, used only in stage 2 + decode.
  Old datasets (no pairs file) give byte-identical results (regression-checked).
- `eval_full.py`: scores only S1 with a GT row.
Plan: build sample_v3 -> train v10 -> compare v9 vs v10 on sample_v3 val (clean half) -> ship
only if v10 wins. This also gives the first local number measured with realistic competition.

### 3.1 Decoder did not optimise the metric - FIXED locally (not yet on LB)
- Old decoder (`model.decode`): each S2/S3 record -> its best S1, keep if `p2 >= 0.98`. One global
  per-pair cut, but the metric is **per-S1 F0.5, macro-averaged**, and an empty prediction on a true
  singleton scores 1.0. The best decision depends on the S1's whole candidate list.
- `is_unbalance=True` makes `p2` over-confident, which is why the optimum threshold drifted to 0.98.
- Measured: of India's *blocked* true pairs, **7.8% are the best S1 for their record but cut by
  thr** (median p2 = 0.945). Only 0.2% are lost to another S1 winning. In p2 ∈ [0.93, 0.98) 61% of
  pairs are true; above 0.98, 98% are true.
- Fix: `model.fit_ef_decoder` (isotonic calibration of p2 + λ = expected unseen matches per S1, fitted
  on ES only, pooled over countries) and `model.decode_ef_assigned` (per S1, pick the top-k that
  maximises expected F0.5, k=0 allowed). Result on v7 cache: **REP 0.9446 -> 0.9461** (India +0.28);
  stable for λ in [0, 0.5]. `train.py` now fits it and writes `decoder.json` only if it beats thr on
  ES; `predict_ajusbyjus.py` uses `decoder.json` when present.
- **Open:** calibration is pooled; France (unseen) may be calibrated differently. Ideas welcome that
  do not need France labels.

### 3.2 Genuinely ambiguous pairs in the high-score zone (model / features)
Examples of pairs the model scores 0.93–0.98 (from `hard_zone.py eval_cache_v7 India 0.93 0.98`):
- **Same address, scrambled/garbage name** (`calozeta`, `suryava`, `viobrixwex`, `mehul`): some true,
  some false (sibling businesses at the same address).
- **Same or near-same name, empty address** (`sun advisors limited` false, `meenam holdings` true).
- **House-number off by a few** (`1002` vs `1007`, `22 160` vs `2 160`): `hnum_edit` exists but these
  still land in the hard zone.
- **Transliterated legal suffixes not stripped**: `praivarr limirrad`, `limiteda`, `praiveta` - the
  skeleton-based `LEGAL_SK` in `norm.py` misses some variants, so they count as name content.
- Earlier measurement: records with an **empty address** are 29% (India) / 44% (US) of model FNs but
  ~3% of all true pairs.
- **Ask:** feature ideas that separate these using data only, e.g. "how many S1 records share this
  exact address" (an address shared by many S1 means siblings, not the same business), "is the
  other name a dictionary-less token", a data-driven legal-suffix learner (tokens that very often
  appear at the end of S1 names). Any new feature = retrain; test on the small sample first.

### 3.3 Leaderboard is ~1.4 pt below local val (systematic)
- Not noise: v7 -1.5, v8 -1.4. **Correction:** an earlier version of this report said local val
  matched full-scale train (0.9452 vs 0.9449) - wrong, that eval also ran on sample_v2. We have
  never measured at full density; §3.0 shows sample_v2 is far less competitive than the test.
- **Country mix explains about half**: local val is 40% India / 60% US; test is 47% India / 38% US /
  15% France. Re-weighting local India/US to the test mix gives ~0.939 *(est.)*.
- The rest is France *(est. ~0.88 on v7, unmeasurable: no France labels)*.
- **Rejected hypothesis:** "France collapsed from v6 to v7". Comparing submission files
  (`compare_outputs.py`): v7/v8 dropped ~10% of v6's pairs in **every** country the same way
  (France kept 0.894, US 0.914, India 0.820); France pairs/S1 3.13 and empty rate 6.2% look like US.
- **Ask:** label-free ways to check France quality (e.g. agreement between S2 and S3 matches for the
  same S1, score distributions vs US).

### 3.4 Blocking misses (mostly India)
- v8 keys_v2 raised India recall 0.906 -> 0.928, US 0.963 -> 0.970 (+~10% candidates).
- Remaining misses by cause (`diag_blocking.py`, small sample, v8 keys): India none=414 /
  capped=548 / topk=350; US 130 / 618 / 86. "capped" = the pair shares only keys whose block is
  bigger than `max_block=60` (median smallest block ~450-500). Raising caps/topk barely helps
  (India +0.6 pt recall for 2.3x candidates) - **sharper keys are the way, not bigger caps**.
- Note: with more keys, India top-K cuts went up (188 -> 350): more candidates compete for 60 slots.
- **Ask:** run `python diag_blocking.py sample\dataset\train --kv2` (small, ~2 min) and propose
  key types for the remaining "none" examples it prints.

### 3.5 Training data is only 5% of S1 (sample_v2)
- sample_v2 = 110,784 S1 (5%), 6.2M candidate pairs. Early stopping used 1304/532 trees.
- Small-sample hint: half data -> full-val 0.9813 vs 0.9825 with all data. Real check pending:
  `train.py --train-frac 0.5` on sample_v2. If half data costs a lot, build a 10% sample.

### 3.6 Predict speed (77 min for v8 on the full test set)
Profile (`profile_predict.py`, small sample, per pair): blocking 46% (file reads + normalisation +
keys; **each big test file is read once per country = 3x**), features 33% (Python loop + rapidfuzz),
stage-1 scoring 15%, rest 6%. Safe ideas, must give **byte-identical outputs**: read each file once;
reuse one `ProcessPoolExecutor` instead of one per 2M-pair batch (~45 pool starts); `--workers 12`.
Expected saving ~10 of 77 min.

### 3.7 Smaller issues
- `predict_ajusbyjus.py` is **not in git** - the final zip must be reproducible from what's committed.
- `train.py` OOF fold split uses `crc32(id+"f") % 2`, which is correlated with the val hash
  (`crc32` is affine); harmless for OOF but worth replacing with md5 like `es_half_of`.
- `Documentation_template.md` is still unfilled (required in the final zip).

## 4. Tried and rejected (don't repeat)
- Per-country thr/margin: +0.0002 (sample_dense) - noise.
- Separate lower threshold for empty-address records: best +0.0001 - not shipped.
- Raising max_block / topk: tiny recall gain, 2x+ candidates and RAM.
- 8 other model families vs LightGBM (`model_bench.py`): LightGBM won.

## 5. Useful scripts (repo root, read-only, run from `student_resource/`)
| script | what it does | cost |
|---|---|---|
| `diag_blocking.py <data> [--kv2]` | why each true pair is missed + caps/topk sweep | small sample ~2 min |
| `decode_experiment.py <eval_cache>` | threshold vs expected-F0.5 decoder, ES vs REP | ~1 min |
| `fit_decoder.py <eval_cache> <models>` | fits decoder.json for an existing model | ~1 min |
| `hard_zone.py <cache> <country> <lo> <hi>` | examples of true vs false pairs in a score band | ~1 min |
| `compare_outputs.py out_a out_b ...` | per-country pairs/S1, empty rate, agreement | ~1.5 min |
| `profile_predict.py <data> <country> <models> <workers>` | time per predict phase | small ~1 min |
| `score_output.py <matching_results.tsv> <ground_truth.tsv>` | competition metric on labelled data | seconds |
| `rival_density.py <eval_cache> <candidate_pairs.tsv>` | S1 rivals per S2/S3 record: sample vs test | ~3 min |
| `src/make_sample_v3.py --data <train> --out <dir> --frac F --keys-v2` | sample with rival context (§3.0) | small sample ~1.5 min (`--frac 0.2`) |
| `fit_decoder.py <cache> <models> [--eval-only]` | `--eval-only`: score an existing decoder.json as shipped | ~1 min |

Caches (`eval_cache_v7`, soon `eval_cache_v8`) and sample_v2 are local-only (too big for git);
ask Pratham for outputs of any heavy run.
