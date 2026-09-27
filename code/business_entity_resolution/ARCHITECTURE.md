# Business Entity Resolution — Architecture & Dataset Reference

Companion to `README.md` (quick-start commands). This file goes deeper: exact dataset
shape, what every source file does internally, and the relations/gotchas that aren't
obvious from reading one file at a time. Sections 1–7 describe the pipeline up to v8 (commit `58946ae`); **§8 lists everything
added after that (v9–v11: rival-context training sample, EF decoder, feat_v3, speed-ups).**

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

**`sample_v2/` (local-only, current training/validation set since v7).** `sample_dense`
samples by name hash, which keeps an S1 and its true matches together but thins out the
*different-name* neighbours ~20x - so blocking and hard negatives are far easier than on
the real test set, and local val over-promised (v6: 0.973 on sample_dense vs 0.901 LB).
`sample_v2` (built by `experiments/make_sample_v2_fixed.py`) takes 5% of S1 entities and keeps
ALL their real full-train blocking candidates, so candidate density matches the real
data (110,784 S1; 1,954,070 S2; 1,870,057 S3). The fixed script drops competitor S1s
that have no ground-truth row (Aayush's original labelled them wrongly).

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
- `LEGAL` also strips French/international legal suffixes (sarl, sas, eurl, gmbh, srl,
  spa, bv, ...) and `ABBR` expands French street abbreviations (r→rue, bd→boulevard,
  che→chemin, ...), so France (test-only) is normalised like the train countries.
- `generic_addr_tokens()` learns per-country generic address words (present in >1% of
  that country's S1 addresses: street types, city/state names, 'rue', 'de', ...) from
  the data itself (cfg `addr_stop_frac`), so address keys skip them for ANY country.
- `norm_addr(ords=True)` (only when cfg `keys_v2`) maps spelled-out ordinals to digits
  ('thirteenth' → '13th').

**`blocking.py`** — candidate generation (this is what `output/candidate_pairs.tsv` is).
- `keys_for()` builds several typed keys per record — address number+word (`A`), sorted
  core name (`N`), name prefix/suffix (`P`), phonetic-skeleton keys (`V`/`K`), a
  **compound key** = name-key × address-token (`X`, sharpens an otherwise-generic name
  key), rare long address tokens (`R`/`S`). `WEIGHT` ranks reliability per key type
  (address/compound/exact-name keys weighted highest).
- **`keys_v2()`** (cfg `keys_v2=True`, v8+; off = byte-identical to v7 blocking). Built
  from a diagnosis of *why* true pairs were missed (`diag_blocking.py`: no shared key /
  every shared key over a size cap / cut by top-K). Raising caps or top-K barely helped
  (India +0.6 pt recall for 2.3x candidates), so v8 adds sharper keys instead:
  `B` adjacent address bigram with a number ('4600 24th', 'c 25', '14 109');
  `Y` unordered (house number × rare address word), survives word-order changes;
  `C` compact name / initials+last word, matching website-style names
  ('allshivsystemscom' ↔ 'all shiv systems', 'wmbrokeragecom' ↔ 'wentworth midwest
  brokerage'). Blocking recall on sample_v2: India 0.906 → 0.928, US 0.963 → 0.970, for
  ~10% more candidates per S1.
- `Side` holds one source's normalized records + flattened (key, row, weight) arrays;
  `Side.concat()` merges Source2+Source3 into one "other" side.
- `candidates()` merges S1 keys against "other" keys sharing a hash, drops
  overly-generic key groups (size cap), sums shared-key weight per (r1, ro) pair, keeps
  top-K by weight per S1 record. **Country is folded into every key string, never used
  as a model feature** — this is exactly why an unseen country (France) still works:
  blocking only needs France records to share keys with other France records.

**`features.py`** — 36 hand-engineered features per candidate pair (`F1` list; 32 until
commit `2c8ee99` added `a1_empty`, `both_empty`, and the house-number features
`hnum_edit` = edit distance between house numbers (typo vs different building) and
`hnum_logdiff`), via
rapidfuzz (ratio/token_sort/token_set/partial_ratio, JaroWinkler, Levenshtein) plus
custom exact/Jaccard/length-diff comparisons across cleaned name, core tokens, phonetic
skeleton, cleaned address, address number-tokens, address keys, and the blocking weight
`w` itself as a feature. Parallelized via `ProcessPoolExecutor` when `workers>1`.
*(Was 33 features before commit `bb36851` dropped one dead-weight feature — see §7.)*

**`model.py`** — two-stage LightGBM matcher.
- Stage 1: LightGBM on the 36 raw pair features → `p1`.
- Stage 2: LightGBM on **context** of `p1` — rank/gap of this pair within its S1's
  candidate list, rank/gap within the "other" record's S1 options, sum of competing
  probabilities — plus 11 raw features (`RAW2`) carried through. This is what lets the
  model reason about competition ("best match for both sides, or is there a stronger
  rival candidate"), not just score each pair in isolation.
- Tree caps 2000 (stage 1) / 800 (stage 2) since v8, with early stopping (50 rounds,
  AUC) picking the real count. v7 used all of its old 500/200 cap (capacity-limited);
  v8 stopped at 1304 / 532.
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
Since v8 the val entities are split again (md5 hash) into an **ES half** (early-stopping
tree count + thr/margin choice) and a **REP half that is never used for any choice**, so
the printed `CLEAN report-half` F0.5 is an honest, unbiased score; `full-val` is
comparable to older runs. The stage-1 main model is fitted first and the 2 OOF fold
models reuse its early-stopped tree count. All three scores + tree counts go into
`config.json`. Flags: `--addr-stop-frac 0.01`, `--keys-v2` (both saved in config.json,
so predict/eval reproduce them automatically).

**`eval_full.py`** — full-scale evaluator: scores a saved model on any data folder
with a loss breakdown (blocking vs model FN, FP, singletons), thr/margin sweep,
`--val-split` (score only train.py's val entities) and `--cache`.

**`diag_blocking.py`** (repo root) — for every missed true pair says why (no shared key
/ capped / top-K cut), prints examples and a caps/top-K recall-vs-cost sweep;
`--kv2` to test keys_v2.

**`predict.py`** — loads `config.json` + `stage1.txt` + `stage2.txt` (+ `decoder.json` if present), runs
`build_country` → `pair_features` → `stage2_matrix` → decoder per test country, batches stage-1 inference
(default 2M pairs/batch) to bound RAM, and writes both output TSVs plus `scores_<country>.npz`. Prints a RAM
warning, per-batch progress, a warning for ANY country that gets 0 matches despite candidates, and a run
summary. `--keys-v3` adds the extra name-/address-word-pair candidates (§8). (This was `predict_ajusbyjus.py`
during development; renamed in the final clean-up, code unchanged except log messages - outputs byte-identical.)

**`experiments/`** — one-off analysis scripts kept for transparency, NOT needed to reproduce the submission: `make_sample.py` (sample_dense), `make_sample_v2_fixed.py` (sample_v2), `model_bench.py`, `per_country_threshold.py`, `cross_source_experiment.py`, `check_*_importance.py`, `blocking_audit*.py`, `blocking_recall_fast.py`. They import the pipeline from `../src`.

**Experiment scripts (now in `experiments/`):**
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

| dir | features trained on | blocking cfg | thr / margin | val F0.5 | status |
|---|---|---|---|---|---|
| `models/` | 33 (pre-`bb36851`) | max_block=30/topk=30 | 0.70 / 0.20 | 0.9667 | **stale** |
| `models_tuned/` | 33 (pre-`bb36851`) | max_block=30/topk=30 | 0.70 / 0.20 | 0.9668 | **stale** |
| `models_v3/` | 32 (current) | max_block=30/topk=30 | 0.94 / 0.34 | 0.9674 | superseded by v4 |
| `models_v4/` | 32 | max_block=60/topk=60 | 0.96 / 0.20 | 0.9681 (sample_dense) | superseded |
| `models_v5/` | 36 | 60/200/60 | — | 0.9731 (sample_dense) | superseded |
| `models_v6/` | 36 | 60/200/60 + stop 0.01 | — | 0.9734 (sample_dense) / 0.892 (sample_v2) | superseded — **LB 0.901** |
| `models_v7/` | 36 | 60/200/60 + stop 0.01 | 0.98 / 0.00 | 0.9452 (sample_v2) | superseded — **LB 0.93** |
| `models_v8/` | 36 | 60/200/60 + stop 0.01 + **keys_v2** | 0.98 / 0.00 | **0.9537** (sample_v2; clean half 0.9532) | **current** — submission #6 |

**Only sample_v2 numbers are comparable to the leaderboard.** sample_dense scores
(0.97+) over-promised badly (see §2). v6 was +0.9 pt above its sample_v2 score on the LB,
v7 was -1.5 pt below it; France (test-only, unmeasurable locally) is the most likely
source of the gap.

`output_v8/matching_results.tsv` + `candidate_pairs.tsv` (from `models_v8`, validator
PASS) are the current submission: 1,732,544 S1, 5,312,393 matched pairs (~3.1 per S1,
6.9% predicted singletons), France behaving like India/US (3.1 pairs/S1, 6.1% empty).
v8 cost: predict is slower than v7 (more trees, ~10% more candidates).

History of v4: `models_v4`'s widened blocking config (`max_block=60, max_s1_block=200, topk=60`, up
from 30/200/30) came from Track A (Adithya)'s `adithya-sundar` branch — his branch had
diverged from `pathu` (missing this session's later Track B work), so rather than
merging it wholesale, the parameter change alone was cherry-picked into `train.py`'s
`CFG` and retrained on top of current code. Real improvement (+0.0007), not noise, at
the cost of ~2x candidates/entity (both train and predict take ~2x longer). Adithya's
branch also added an `a1_empty` feature (mirrors existing `a2_empty`) — folded in with
commit `2c8ee99` (36 features).

Note on paths: `models/` and `models_tuned/` live inside
`code/business_entity_resolution/`, but `models_v3/`/`models_v4/` were trained with
`--models ..\..\models_v3` (etc.) from inside `code/business_entity_resolution`, so they
landed at the **repo root** (`student_resource/models_v3/`, `models_v4/`) instead —
same content either way, just a different path.

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
  `sample_dense.zip` from `main` if missing. **Don't judge models on it any more** —
  use `sample_v2/` (realistic density, see §2).
- **Blocking cfg lives in `config.json`** (`max_block`, `max_s1_block`, `topk`,
  `addr_stop_frac`, `keys_v2`) and predict/eval read it from there. A model must be
  predicted with the same blocking it was trained with; new blocking options are always
  added behind a cfg flag that defaults to off, so older models reproduce exactly.
- **Don't choose anything on the clean REP half.** If a new sweep or setting is tuned,
  tune it on the ES half (train.py does this) so `CLEAN report-half` stays honest.
- Predict RAM: v8 peaks high on the full test set (India ≈ 43.6M candidate pairs). Run
  it alone, with other apps closed; the laptop may use the page file.

## 6. Reproduction

See `README.md` for the exact commands that produce the submitted files and `tests/` for the unit tests.

## 8. Changes after v8 (v9 – v11, 27 Sep)

**Why:** local validation on `sample_v2` over-promised the leaderboard by ~1.5 pt for every model.
`rival_density.py` found the cause: in the real test each S2/S3 record is a blocking candidate of
~9.5 S1 businesses (71% have >= 5); `sample_v2` keeps only the sampled S1, so ~2. Stage 2's
competition features and the "best S1 wins" decode were trained and validated with ~5x too little
competition.

- **`sample_v3` / `sample_v3_25`** (`src/make_sample_v3.py`, `src/shrink_sample_v3.py`): sampled S1 (same
  crc32 hash as sample_v2) + every rival S1 that is a real candidate of a kept S2/S3 record +
  `train_pairs.tsv` = the real full-density candidate pairs. `pipeline.build_country` uses
  `<data>/<prefix>_pairs.tsv` when present instead of re-blocking inside the sample (never present
  for test). `shrink_sample_v3.py` produces exactly what a smaller `--frac` would (verified
  identical); `sample_v3_25` = 2.5% of S1 (55,291 sampled S1, 2.1M rival S1, ~41M pairs) fits 16 GB.
- **`train.py`**: only S1 with a ground-truth row are labelled; rival pairs are **context rows**
  (features written to a disk memmap in batches, scored by the final stage-1 model like test rows,
  used only in stage 2 and the decode, never as training labels or in the score).
- **Expected-F0.5 decoder** (`model.fit_ef_decoder` / `decode_ef_assigned`, `fit_decoder.py`):
  isotonic calibration of p2 + lambda (expected unseen matches per S1), fitted on the ES half; per S1
  keeps the top-k (k = 0 allowed) that maximises expected F0.5. Saved as `<models>/decoder.json`
  only if it beats thr/margin on ES; `predict.py` uses it when present. Used by v9
  (+0.1 local, +0.5 LB); v10 did not need it (thr/margin was better on ES).
- **feat_v3** (`--feat-v3`, default off; `F1_V3` = 42 features): `nspan_jac` (digit-run Jaccard),
  `sk_eq` (exact skeleton equality), IDF-weighted name/address similarity `n_idf_cos`, `n_idf_max`,
  `a_idf_cos`, `a_idf_max` (word rarity counted over the country's FULL S1 file, words seen < 3
  times share one "rare" weight); stage 2 adds log sibling density (S1 count per 100k sharing the
  record's exact normalised address / name skeleton). `--density-src` must point to the FULL
  `train_source1.tsv` when training on a sample. `--learn-suffix` exists but is not used (it learns
  real words like "technologies").
- **Speed (no output change):** per-record name/address parts, phonetic skeletons and digit checks
  are cached (`functools.lru_cache`), ASCII text skips accent stripping. Verified bitwise identical
  (normalised strings, keys, candidates, IDF tables, 36- and 42-column features); ~30% faster
  blocking, ~25% faster features single-process.
- **`blend_scores.py`** (repo root): averages the saved stage-2 scores of two models; the weight and
  thr/margin are chosen on the ES half, and the blend is used only if it beats both models alone there.
- **`eval_full.py --val-split`** scores a model on exactly train.py's validation entities of any
  data folder, so models trained on the same sample compare fairly.

| model | trained on | changes | clean-half F0.5 (sample_v3_25) | LB |
|---|---|---|---|---|
| v9 | sample_v2 | v8 + EF decoder | 0.9554 | 0.945 |
| v10 | sample_v3_25 | rival context rows, thr 0.98 / margin 0.32, 1549 / 262 trees | 0.9621 | [FILL] |
| v11 | sample_v3_25 | v10 + feat_v3 (IDF, density), EF decoder, 1792 / 240 trees | **0.9642** (ES 0.9640; EF ES 0.9646) | [FILL] |

**keys_v3 — extra candidates at predict time (`--keys-v3`, used for the final submission).**
- `M` = unordered pairs among the 3 longest distinct core-name words (len ≥ 4); `L` = unordered pairs among the
  4 longest non-generic address words (len ≥ 5). A word PAIR is much sharper than one word (so it survives the
  block caps), and several pairs per record survive one extra / missing / typo'd word; `M` works when the
  address is empty — the two biggest blocking-miss types we measured.
- Kept in separate key arrays: the normal candidates and their weights are untouched (tested byte-identical);
  up to `--keys-v3-topk` (default 5) NEW pairs per S1 are appended. Nothing is learned from labels; the same
  rule applies to every country.
- It is applied at predict time only (the training sample stores its candidate pairs), so it was validated
  separately on labelled data that re-blocks (`sample_v2`, v11, its own decoder, held-out entities only):
  blocking recall 0.954 → 0.964; ES 0.9302 → 0.9325, REP 0.9310 → 0.9336, better in India and US. On the
  small sample with a flags-off model: ES 0.9809 → 0.9813, REP 0.9817 → 0.9823. It costs ~5% more candidates.

**Final clean-up.** One-off scripts moved to `experiments/`; `predict_ajusbyjus.py` → `predict.py`; no country
name appears in the pipeline code except the normalisation dictionaries; `tests/` added (20 fast unit tests);
`requirements.txt` pinned to the exact versions used.
