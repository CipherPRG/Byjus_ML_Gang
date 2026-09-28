"""Track B experiment: per-country threshold/margin instead of one global value.

Rationale (from both our own blocking-recall numbers and an independent LLM review):
blocking recall already differs a lot by country (India 0.9284 vs US 0.9720), so the score
distributions p2 produces per country likely differ too. One global thr/margin is a compromise
that may be leaving F0.5 on the table for one or both countries. This script measures whether
sweeping thr/margin independently per country (using the same decode_prep/decode_apply machinery
already built) beats the single global optimum, on the exact same train/val split as train.py.

Does NOT modify model.py or train.py - read-only experiment, same as model_bench.py's spirit.
Whatever wins, you copy the values into train.py/predict.py yourself.

Usage:
    python experiments/per_country_threshold.py --data ../../sample_dense --workers 5
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse, zlib
import numpy as np
from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features
from model import fit_stage1, fit_stage2, stage2_matrix, raw2, decode_prep, decode_apply
from evaluate import f05_macro

CFG = dict(max_block=30, max_s1_block=200, topk=30)
THR_GRID    = np.arange(0.30, 0.99, 0.02)
MARGIN_GRID = np.arange(0.00, 0.45, 0.02)


def sweep(r1, ro, p2, u1, uo, entity_ids, vt):
    """Same sweep as train.py/model_bench.py, but restricted to a given set of val entity ids."""
    prepped = decode_prep(r1, ro, p2)
    best = (-1.0, 0.5, 0.0)
    for thr in THR_GRID:
        for mg in MARGIN_GRID:
            rr, oo = decode_apply(prepped, thr, mg)
            pred = {}
            for i, o in zip(rr, oo):
                s = u1[i]
                if s in entity_ids:
                    pred.setdefault(s, set()).add(uo[o])
            sc = f05_macro(pred, vt)
            if sc > best[0]:
                best = (sc, float(thr), float(mg))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    X, Y, S1ID, OID, S1COUNTRY = [], [], [], [], []
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats); Y.append(y); S1ID.append(s1ids); OID.append(oids)
        S1COUNTRY.append(np.full(len(r1), c, dtype=object))

    X = np.vstack(X); Y = np.concatenate(Y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    S1COUNTRY = np.concatenate(S1COUNTRY)
    print(f"total pairs: {len(Y)}")

    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID, return_inverse=True)
    is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
    tr = ~is_val
    fold = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]
    val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
    vt = {k: v for k, v in truth.items() if k in val_ids}

    # country of each unique S1 id (u1), for splitting the val set by country later
    u1_country = {}
    for s, ctry in zip(S1ID, S1COUNTRY):
        u1_country.setdefault(s, ctry)

    print("Fitting stage1 (OOF)...")
    p1 = np.zeros(len(Y), dtype=np.float32)
    for f in (0, 1):
        fit = tr & (fold != f)
        pred = tr & (fold == f)
        p1[pred] = fit_stage1(X[fit], Y[fit]).predict_proba(X[pred])[:, 1]
    m1 = fit_stage1(X[tr], Y[tr])
    p1[is_val] = m1.predict_proba(X[is_val])[:, 1]

    print("Fitting stage2...")
    X2 = stage2_matrix(r1, ro, p1, raw2(X))
    m2 = fit_stage2(X2[tr], Y[tr])
    p2 = m2.predict_proba(X2)[:, 1]

    # ---- global (baseline, same as train.py) ----
    global_best = sweep(r1, ro, p2, u1, uo, val_ids, vt)
    print(f"\nGLOBAL:  F0.5={global_best[0]:.4f}  thr={global_best[1]:.2f}  margin={global_best[2]:.2f}")

    # ---- per-country ----
    countries = sorted(set(u1_country.values()))
    per_country = {}
    for ctry in countries:
        ids_c = {s for s in val_ids if u1_country.get(s) == ctry}
        vt_c = {k: v for k, v in vt.items() if k in ids_c}
        if not vt_c:
            continue
        best_c = sweep(r1, ro, p2, u1, uo, ids_c, vt_c)
        per_country[ctry] = best_c
        print(f"{ctry:<10} F0.5={best_c[0]:.4f}  thr={best_c[1]:.2f}  margin={best_c[2]:.2f}  "
              f"(n_val_entities={len(vt_c)})")

    # ---- combined score using each country's OWN best thr/margin simultaneously ----
    prepped = decode_prep(r1, ro, p2)
    combined_pred = {}
    for ctry, (sc, thr, mg) in per_country.items():
        rr, oo = decode_apply(prepped, thr, mg)
        ids_c = {s for s in val_ids if u1_country.get(s) == ctry}
        for i, o in zip(rr, oo):
            s = u1[i]
            if s in ids_c:
                combined_pred.setdefault(s, set()).add(uo[o])
    combined_f05 = f05_macro(combined_pred, vt)

    print("\n" + "=" * 60)
    print(f"GLOBAL thr/margin F0.5:        {global_best[0]:.4f}")
    print(f"PER-COUNTRY thr/margin F0.5:   {combined_f05:.4f}")
    print(f"delta: {combined_f05 - global_best[0]:+.4f}")
    if combined_f05 > global_best[0]:
        print("-> per-country thresholds WIN. Values to use in predict.py:")
        for ctry, (sc, thr, mg) in per_country.items():
            print(f"   {ctry}: thr={thr:.2f} margin={mg:.2f}")
    else:
        print("-> global threshold is at least as good - not worth the added complexity.")


if __name__ == "__main__":
    main()
