"""Run the trained pipeline on the test set and write output/matching_results.tsv + output/candidate_pairs.tsv.

Normal run (uses thr/margin from config.json):
    python src/predict.py --data dataset/test --models models --out output --workers 4

Override threshold / margin (e.g. for a quick re-score without retraining):
    python src/predict.py --data dataset/test --models models --out output --thr 0.68 --margin 0.18

Lower the stage-1 batch size if you hit RAM errors (default 2 000 000 pairs ≈ ~3 GB RAM):
    python src/predict.py --data dataset/test --models models --out output --batch 500000
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from io_utils import read_tsv, countries_of, write_lists
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, decode, raw2

# ---------------------------------------------------------------------------
# RAM guard: refuse to run if available memory looks dangerously low.
# Import psutil only if present — it is in requirements.txt; if somehow missing
# we skip the check rather than crashing.
# ---------------------------------------------------------------------------
def _available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        return None


def _check_ram(batch_size, warn_threshold_gb=4.0):
    """Warn if available RAM looks too low for the requested batch size."""
    avail = _available_gb()
    if avail is None:
        return   # psutil not installed — skip check silently
    # Rough estimate: each pair in stage-1 batch costs ~1.5 KB (33 float32 features + overhead)
    estimated_gb = batch_size * 1.5e3 / 1024 ** 3
    if estimated_gb > avail - 1.0:
        print(f"[RAM WARNING] batch_size={batch_size:,} may need ~{estimated_gb:.1f} GB but only "
              f"~{avail:.1f} GB available.  Consider re-running with --batch {batch_size // 2:,}",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True,  help="path to dataset/test/ folder")
    ap.add_argument("--models",  required=True,  help="directory with stage1.txt, stage2.txt, config.json")
    ap.add_argument("--out",     required=True,  help="output directory for TSV files")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--thr",     type=float, default=None,
                    help="override threshold from config.json")
    ap.add_argument("--margin",  type=float, default=None,
                    help="override margin from config.json")
    ap.add_argument("--batch",   type=int, default=2_000_000,
                    help="max pairs per stage-1 inference batch (reduce if OOM; default 2 000 000)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ config
    conf   = json.load(open(f"{a.models}/config.json"))
    cfg    = conf["cfg"]
    thr    = conf["thr"]    if a.thr    is None else a.thr
    margin = conf["margin"] if a.margin is None else a.margin
    print(f"using thr={thr:.2f}  margin={margin:.2f}  "
          f"(source: {'config.json' if a.thr is None else 'command-line override'})")
    if conf.get("final_mode"):
        print("NOTE: models were trained in --final mode (all training data, no holdout)")

    # RAM check before loading anything heavy
    _check_ram(a.batch)

    b1 = lgb.Booster(model_file=f"{a.models}/stage1.txt")
    b2 = lgb.Booster(model_file=f"{a.models}/stage2.txt")

    # ------------------------------------------------------------------ inventory
    s1_all = read_tsv(f"{a.data}/test_source1.tsv", usecols=["entity_id", "country"])
    s1_countries = set(s1_all.country.unique())
    print(f"\nS1 rows per country: {s1_all.country.value_counts().to_dict()}")
    for k in (2, 3):
        cs = read_tsv(f"{a.data}/test_source{k}.tsv", usecols=["country"]).country.value_counts().to_dict()
        extra = set(cs) - s1_countries
        print(f"S{k} rows per country: {cs}"
              + (f"  | labels NOT in S1: {extra}" if extra else ""))

    # ------------------------------------------------------------------ France verification
    # France is an unseen country (not in train). Verify it is present in the test set and
    # will be processed.  Blocking is country-prefixed so France gets its own key namespace
    # automatically — but we must confirm it actually appears and produces non-zero output.
    EXPECTED_UNSEEN = {"France"}   # extend this set if future contests add more
    seen_in_test = s1_countries
    for c in EXPECTED_UNSEEN:
        if c in seen_in_test:
            print(f"[France check] '{c}' is present in test S1 — will be processed normally.")
        else:
            # Not necessarily an error — country name may be spelled differently in the actual data.
            # Print a clear warning so D can investigate before submitting.
            print(f"[France check] WARNING: '{c}' not found in test S1 countries: {sorted(seen_in_test)}"
                  f"\n  → If France is in the test set under a different label, this is a non-issue."
                  f"\n  → If it is genuinely missing from S1, every France S1 entity will appear in"
                  f"\n    the output as an empty match (which is still valid, just 0 recall for France).")

    # ------------------------------------------------------------------ per-country inference
    cand_lists, match_lists = {}, {}
    countries_with_zero_matches = []

    for c in sorted(s1_all.country.unique()):
        s1, oth, cand = build_country(a.data, "test", c, cfg, a.workers)
        if cand is None or len(cand) == 0:
            print(f"[{c}] no candidates — all S1 entities will appear as empty matches")
            countries_with_zero_matches.append(c)
            continue

        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        B = a.batch   # configurable batch size (--batch flag)

        p1  = np.zeros(len(r1), dtype=np.float32)
        raw = []
        for s in range(0, len(r1), B):
            end = min(s + B, len(r1))
            X   = pair_features(s1, oth, r1[s:end], ro[s:end], w[s:end], a.workers)
            p1[s:end] = b1.predict(X)
            raw.append(raw2(X))
            # Progress ticker for large countries (useful to see it's not stuck)
            if len(r1) > B:
                print(f"  [{c}] stage-1 batch {s//B + 1}/{(len(r1)-1)//B + 1} "
                      f"({end:,}/{len(r1):,} pairs)")

        raw = np.vstack(raw)
        X2  = stage2_matrix(r1, ro, p1, raw); del raw
        p2  = b2.predict(X2)

        rr, oo   = decode(r1, ro, p2, thr, margin)
        s1ids    = np.array(s1.ids, dtype=object)
        oids     = np.array(oth.ids, dtype=object)

        for i, o in zip(s1ids[r1], oids[ro]):
            cand_lists.setdefault(i, []).append(o)
        for i, o in zip(s1ids[rr], oids[oo]):
            match_lists.setdefault(i, []).append(o)

        n_matched_s1 = len(set(s1ids[rr]))
        print(f"[{c}] S1={len(s1)}  others={len(oth)}  candidates={len(r1)}  "
              f"matched pairs={len(rr)}  S1 with >=1 match={n_matched_s1}")

        # Per-country France sanity check: if France produced 0 matches, flag it loudly.
        if c in EXPECTED_UNSEEN and len(rr) == 0:
            print(f"  [France check] WARNING: '{c}' produced 0 matched pairs. "
                  f"Candidates existed ({len(r1)}) but none passed thr={thr:.2f}/margin={margin:.2f}. "
                  f"Consider lowering --thr for this run to investigate.")

    # ------------------------------------------------------------------ post-run summary
    print(f"\n--- Run summary ---")
    print(f"total S1 entities:       {len(s1_all):,}")
    print(f"S1 with >=1 match:       {len(match_lists):,}")
    print(f"S1 with no match (empty):{len(s1_all) - len(match_lists):,}")
    print(f"total matched pairs:     {sum(len(v) for v in match_lists.values()):,}")
    if countries_with_zero_matches:
        print(f"countries with 0 candidates (all S1 empty): {countries_with_zero_matches}")

    # ------------------------------------------------------------------ write output
    order = s1_all.entity_id.tolist()
    write_lists(
        f"{a.out}/matching_results.tsv",
        ("source1_entity_id", "matched_entity_ids"),
        order, match_lists
    )
    write_lists(
        f"{a.out}/candidate_pairs.tsv",
        ("source1_entity_id", "candidate_entity_ids"),
        order, cand_lists
    )
    print(f"\nwrote {a.out}/matching_results.tsv  and  {a.out}/candidate_pairs.tsv")
    print("Next step: run utils/validate_submission.py to confirm PASS before uploading.")


if __name__ == "__main__":
    main()
