"""Train the two-stage matcher on a (dense) training sample and tune the decision threshold for macro F0.5.

    python src/train.py --data ../../sample_dense --models ../../models --workers 4
"""
import argparse, json, os, zlib
import numpy as np
import pandas as pd
from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features, F1
from model import fit_stage1, fit_stage2, stage2_matrix, decode, raw2
from evaluate import f05_macro

CFG = dict(max_block=30, max_s1_block=200, topk=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.models, exist_ok=True)
    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m} for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    R1, RO, W, X, Y, S1ID, OID = [], [], [], [], [], [], []
    n_true = n_found = 0
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]; oids = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)), dtype=np.int8, count=len(r1))
        tot = sum(len(truth[i]) for i in s1.ids if i in truth)
        print(f"[{c}] S1={len(s1)} others={len(oth)} pairs={len(r1)} ({len(r1)/len(s1):.1f}/S1) "
              f"blocking recall={y.sum()/max(tot,1):.4f}")
        n_true += tot; n_found += int(y.sum())
        S1ID.append(s1ids); OID.append(oids); X.append(feats); Y.append(y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID); X = np.vstack(X); Y = np.concatenate(Y)
    print(f"overall blocking recall: {n_found/n_true:.4f}; pairs: {len(Y)}")

    # integer row ids for grouping across countries
    u1, r1 = np.unique(S1ID, return_inverse=True); uo, ro = np.unique(OID, return_inverse=True)
    is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
    fold = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]
    tr = ~is_val

    # stage 1: out-of-fold p1 on the training part, full-model p1 on the validation part
    p1 = np.zeros(len(Y), dtype=np.float32)
    for f in (0, 1):
        fit = tr & (fold != f); pred = tr & (fold == f)
        p1[pred] = fit_stage1(X[fit], Y[fit]).predict_proba(X[pred])[:, 1]
    m1 = fit_stage1(X[tr], Y[tr])
    p1[is_val] = m1.predict_proba(X[is_val])[:, 1]

    # stage 2
    X2 = stage2_matrix(r1, ro, p1, raw2(X))
    m2 = fit_stage2(X2[tr], Y[tr])
    p2 = m2.predict_proba(X2)[:, 1]

    val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}  # ALL val S1, incl. ones with no candidates
    vt = {k: v for k, v in truth.items() if k in val_ids}
    best = (-1, 0.5, 0.0)
    for thr in np.arange(0.30, 0.96, 0.05):
        for mg in (0.0, 0.05, 0.1, 0.2):
            rr, oo = decode(r1, ro, p2, thr, mg)
            pred = {}
            for i, o in zip(rr, oo):
                s = u1[i]
                if s in val_ids:
                    pred.setdefault(s, set()).add(uo[o])
            sc = f05_macro(pred, vt)
            if sc > best[0]:
                best = (sc, float(thr), float(mg))
    print(f"validation macro F0.5 = {best[0]:.4f} at thr={best[1]:.2f} margin={best[2]:.2f}  (val S1: {len(vt)})")
    m1.booster_.save_model(f"{a.models}/stage1.txt"); m2.booster_.save_model(f"{a.models}/stage2.txt")
    json.dump(dict(cfg=CFG, thr=best[1], margin=best[2], val_f05=best[0]), open(f"{a.models}/config.json", "w"))


if __name__ == "__main__":
    main()
