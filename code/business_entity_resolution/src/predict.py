"""Run the trained pipeline on the test set and write output/matching_results.tsv + output/candidate_pairs.tsv.

Normal run (uses thr/margin from config.json, or <models>/decoder.json if present):
    python src/predict.py --data dataset/test --models models --out output --workers 4

Override threshold / margin (e.g. for a quick re-score without retraining):
    python src/predict.py --data dataset/test --models models --out output --thr 0.68 --margin 0.18

Lower the stage-1 batch size if you hit RAM errors (default 2 000 000 pairs - ~3 GB RAM):
    python src/predict.py --data dataset/test --models models --out output --batch 500000
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from io_utils import read_tsv, countries_of, write_lists
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, decode, raw2, decode_ef


# ---------------------------------------------------------------------------
# RAM guard: warn if available memory looks low for the requested batch size.
# Requires psutil (already in requirements.txt). If missing, skips silently.
# ---------------------------------------------------------------------------
def _check_ram(batch_size):
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024 ** 3
        # Rough estimate: each pair in stage-1 batch costs ~1.5 KB
        estimated_gb = batch_size * 1.5e3 / 1024 ** 3
        if estimated_gb > avail - 1.0:
            print(
                f"[RAM WARNING] batch_size={batch_size:,} may need ~{estimated_gb:.1f} GB but only "
                f"~{avail:.1f} GB available.  Re-run with --batch {batch_size // 2:,} if you hit OOM.",
                file=sys.stderr,
            )
    except ImportError:
        pass  # psutil not installed - skip check


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
    ap.add_argument("--keys-v3", action="store_true",
                    help="EXTRA candidates from name-word-pair / address-word-pair keys on top of the model's own "
                         "blocking (normal candidates unchanged). Off = exactly as before.")
    ap.add_argument("--keys-v3-topk", type=int, default=5, help="max extra candidates per S1 (with --keys-v3)")
    ap.add_argument("--batch",   type=int, default=2_000_000,
                    help="max pairs per stage-1 inference batch (reduce if OOM; default 2 000 000)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ config
    conf   = json.load(open(f"{a.models}/config.json"))
    cfg    = conf["cfg"]
    if a.keys_v3:
        cfg = {**cfg, "keys_v3": True, "keys_v3_topk": a.keys_v3_topk}
        print(f"keys_v3 ON: up to {a.keys_v3_topk} extra candidates per S1")
    thr    = conf["thr"]    if a.thr    is None else a.thr
    margin = conf["margin"] if a.margin is None else a.margin
    print(f"using thr={thr:.2f}  margin={margin:.2f}  "
          f"({'config.json' if a.thr is None else 'command-line override'})")
    if conf.get("final_mode"):
        print("NOTE: models were trained in --final mode (all training data, no holdout)")
    # Expected-F0.5 decoder (fit_decoder.py / train.py write <models>/decoder.json). Absent -> the
    # old thr/margin decoder, exactly as before. --thr/--margin overrides force the old decoder.
    dec = None
    if os.path.exists(f"{a.models}/decoder.json") and a.thr is None and a.margin is None:
        dec = json.load(open(f"{a.models}/decoder.json"))
        print(f"using expected-F0.5 decoder from decoder.json (lam={dec['lam']:.3f}); thr/margin not used")

    _check_ram(a.batch)

    b1 = lgb.Booster(model_file=f"{a.models}/stage1.txt")
    b2 = lgb.Booster(model_file=f"{a.models}/stage2.txt")

    # ------------------------------------------------------------------ inventory
    s1_all      = read_tsv(f"{a.data}/test_source1.tsv", usecols=["entity_id", "country"])
    s1_countries = set(s1_all.country.unique())
    print(f"\nS1 rows per country: {s1_all.country.value_counts().to_dict()}")
    for k in (2, 3):
        cs    = read_tsv(f"{a.data}/test_source{k}.tsv", usecols=["country"]).country.value_counts().to_dict()
        extra = set(cs) - s1_countries
        print(f"S{k} rows per country: {cs}"
              + (f"  | labels NOT in S1: {extra}" if extra else ""))

    # Country is an open set: every label found in test S1 is processed the same way (blocking keys are
    # country-scoped, the model never sees the country). Nothing below depends on specific country names.

    # ------------------------------------------------------------------ per-country inference
    cand_lists, match_lists        = {}, {}
    countries_with_zero_candidates = []

    for c in sorted(s1_all.country.unique()):
        s1, oth, cand = build_country(a.data, "test", c, cfg, a.workers)
        if cand is None or len(cand) == 0:
            print(f"[{c}] no candidates - all S1 entities will appear as empty matches")
            countries_with_zero_candidates.append(c)
            continue

        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        B  = a.batch

        p1  = np.zeros(len(r1), dtype=np.float32)
        raw = []
        for s in range(0, len(r1), B):
            end = min(s + B, len(r1))
            X   = pair_features(s1, oth, r1[s:end], ro[s:end], w[s:end], a.workers,
                                feat_v3=bool(cfg.get("feat_v3")))
            p1[s:end] = b1.predict(X)
            raw.append(raw2(X))
            # Progress ticker - useful to confirm it's not stuck on large countries
            if len(r1) > B:
                print(f"  [{c}] stage-1 batch {s // B + 1}/{(len(r1) - 1) // B + 1} "
                      f"({end:,}/{len(r1):,} pairs)")

        raw = np.vstack(raw)
        dens = ({"addr_by_ro": oth.dens_addr[ro], "name_by_ro": oth.dens_name[ro]}
                if cfg.get("feat_v3") else None)   # counted in this test folder's full source1
        X2  = stage2_matrix(r1, ro, p1, raw, density=dens); del raw, dens
        p2  = b2.predict(X2); del X2

        rr, oo = decode_ef(r1, ro, p2, dec) if dec is not None else decode(r1, ro, p2, thr, margin)

        s1ids = np.array(s1.ids, dtype=object)
        oids  = np.array(oth.ids, dtype=object)
        # save the scores so a different decoder can be applied later WITHOUT re-running predict
        np.savez(f"{a.out}/scores_{c}.npz", r1=r1.astype(np.int32), ro=ro.astype(np.int32),
                 p2=p2.astype(np.float32), s1_ids=np.array(s1.ids, dtype=str), oth_ids=np.array(oth.ids, dtype=str))
        for i, o in zip(s1ids[r1], oids[ro]):
            cand_lists.setdefault(i, []).append(o)
        for i, o in zip(s1ids[rr], oids[oo]):
            match_lists.setdefault(i, []).append(o)

        n_matched_s1 = len(set(s1ids[rr]))
        print(f"[{c}] S1={len(s1)}  others={len(oth)}  candidates={len(r1)}  "
              f"matched pairs={len(rr)}  S1 with >=1 match={n_matched_s1}")

        # Sanity check for ANY country (incl. ones unseen in training): 0 matches despite candidates is suspicious
        if len(rr) == 0 and len(r1) > 0:
            print(f"  WARNING: '{c}' produced 0 matched pairs despite {len(r1):,} candidates - check this country.")

    # ------------------------------------------------------------------ post-run summary
    total_pairs   = sum(len(v) for v in match_lists.values())
    print(f"\n--- Run summary ---")
    print(f"S1 total entities        : {len(s1_all):,}")
    print(f"S1 with >=1 match        : {len(match_lists):,}")
    print(f"S1 empty (no match)      : {len(s1_all) - len(match_lists):,}")
    print(f"Total matched pairs      : {total_pairs:,}")
    if countries_with_zero_candidates:
        print(f"Countries with 0 candidates (entirely empty): {countries_with_zero_candidates}")

    # ------------------------------------------------------------------ write output
    order = s1_all.entity_id.tolist()
    write_lists(
        f"{a.out}/matching_results.tsv",
        ("source1_entity_id", "matched_entity_ids"),
        order, match_lists,
    )
    write_lists(
        f"{a.out}/candidate_pairs.tsv",
        ("source1_entity_id", "candidate_entity_ids"),
        order, cand_lists,
    )
    print(f"\nwrote {a.out}/matching_results.tsv  and  {a.out}/candidate_pairs.tsv")
    print("Next step: python utils/validate_submission.py --matching output/matching_results.tsv "
          "--candidate output/candidate_pairs.tsv --test-dir dataset/test")


if __name__ == "__main__":
    main()
