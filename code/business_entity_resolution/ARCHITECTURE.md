# Business Entity Resolution — Architecture & Dataset Reference

Companion to `README.md` (quick-start commands). This file goes deeper: exact dataset
shape, what every source file does internally, and the relations/gotchas that aren't
obvious from reading one file at a time. Last verified against the code and data on
disk on 2026-09-27 (commit `58946ae`, `models_v8`: val F0.5 0.9537 on sample_v2,
submitted as `output_v8`).

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
`sample_v2` (built by `src/make_sample_v2_fixed.py`) takes 5% of S1 entities and keeps
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

## 6. Reproduction (see also `README.md`)

```
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/make_sample_v2_fixed.py --data dataset/train --out sample_v2 --frac 0.05 --addr-stop-frac 0.01
cd code/business_entity_resolution
python src/train.py --data ../../sample_v2 --models ../../models_v8 --workers 10 --addr-stop-frac 0.01 --keys-v2
python src/predict_ajusbyjus.py --data ../../dataset/test --models ../../models_v8 --out ../../output_v8 --workers 10
cd ../..
python utils/validate_submission.py --matching output_v8/matching_results.tsv --candidate output_v8/candidate_pairs.tsv --test-dir dataset/test
```
(v8 run times on the team laptop: train ≈ 25 min, predict several hours on the full
test set.)
