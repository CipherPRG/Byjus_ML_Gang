"""Train the two-stage matcher on a (dense) training sample and tune the decision threshold for macro F0.5.

Normal mode  — trains on sample_dense, holds out 30 % for validation, sweeps thr/margin, saves models/:
    python src/train.py --data ../../sample_dense --models ../../models --workers 4

Final mode   — retrains on ALL data (no holdout) using a pre-locked thr/margin from an existing config.json,
               saves to models_final/.  Run this ONLY after thr/margin are confirmed from a normal-mode run:
    python src/train.py --data ../../sample_dense --models ../../models_final --final \
        --thr 0.70 --margin 0.20 --workers 4
"""
import argparse, json, os, zlib
import numpy as np
import pandas as pd
from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features, F1
from model import fit_stage1, fit_stage2, stage2_matrix, decode, decode_prep, decode_apply, raw2
from evaluate import f05_macro

CFG = dict(max_block=30, max_s1_block=200, topk=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    required=True,  help="path to sample_dense/ (or full train/) folder")
    ap.add_argument("--models",  required=True,  help="output directory for model artefacts")
    ap.add_argument("--workers", type=int, default=1)
    # --final: skip val holdout, use all data for training.  Must supply --thr and --margin.
    ap.add_argument("--final",   action="store_true",
                    help="retrain on ALL data (no holdout). Requires --thr and --margin.")
    ap.add_argument("--thr",     type=float, default=None,
                    help="fixed threshold (required with --final, optional otherwise to skip sweep)")
    ap.add_argument("--margin",  type=float, default=None,
                    help="fixed margin  (required with --final, optional otherwise to skip sweep)")
    a = ap.parse_args()

    if a.final and (a.thr is None or a.margin is None):
        ap.error("--final requires --thr and --margin (copy the best values from a normal-mode run's config.json)")

    os.makedirs(a.models, exist_ok=True)

    # ------------------------------------------------------------------ load ground truth
    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    # ------------------------------------------------------------------ blocking + features
    R1, RO, W, X, Y, S1ID, OID = [], [], [], [], [], [], []
    n_true = n_found = 0
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids  = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter(
            (o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
            dtype=np.int8, count=len(r1)
        )
        tot = sum(len(truth[i]) for i in s1.ids if i in truth)
        print(f"[{c}] S1={len(s1)} others={len(oth)} pairs={len(r1)} ({len(r1)/len(s1):.1f}/S1) "
              f"blocking recall={y.sum()/max(tot,1):.4f}")
        n_true += tot; n_found += int(y.sum())
        S1ID.append(s1ids); OID.append(oids); X.append(feats); Y.append(y)

    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    X    = np.vstack(X);          Y   = np.concatenate(Y)
    print(f"overall blocking recall: {n_found/n_true:.4f}; total pairs: {len(Y)}")

    # ------------------------------------------------------------------ val / train split
    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID,  return_inverse=True)

    if a.final:
        # No holdout — every pair is used for training.
        tr     = np.ones(len(Y), dtype=bool)
        is_val = np.zeros(len(Y), dtype=bool)
        print("--final mode: training on ALL data (no validation holdout)")
    else:
        is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
        tr     = ~is_val

    fold = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]

    # ------------------------------------------------------------------ stage 1
    # Out-of-fold p1 on training portion; direct p1 on val portion.
    p1 = np.zeros(len(Y), dtype=np.float32)
    if a.final:
        # Only one real fold (all data is training); do a single 50/50 internal OOF for p1 quality,
        # then refit on everything for the final stage-1 model.
        for f in (0, 1):
            fit_mask  = fold == f
            pred_mask = fold != f        # predict on the OTHER half
            # Guard: if all positives end up in one fold, skip
            if Y[fit_mask].sum() == 0 or Y[pred_mask].sum() == 0:
                p1[pred_mask] = 0.5
            else:
                p1[pred_mask] = fit_stage1(X[fit_mask], Y[fit_mask]).predict_proba(X[pred_mask])[:, 1]
        m1 = fit_stage1(X, Y)            # final stage-1 on ALL data
    else:
        for f in (0, 1):
            fit  = tr & (fold != f)
            pred = tr & (fold == f)
            p1[pred] = fit_stage1(X[fit], Y[fit]).predict_proba(X[pred])[:, 1]
        m1 = fit_stage1(X[tr], Y[tr], eval_set=(X[is_val], Y[is_val]))
        p1[is_val] = m1.predict_proba(X[is_val])[:, 1]

    # ------------------------------------------------------------------ stage 2
    X2 = stage2_matrix(r1, ro, p1, raw2(X))
    if a.final:
        m2 = fit_stage2(X2, Y)
    else:
        m2 = fit_stage2(X2[tr], Y[tr], eval_set=(X2[is_val], Y[is_val]))
    p2 = m2.predict_proba(X2)[:, 1]

    # ------------------------------------------------------------------ threshold / margin sweep
    if a.final:
        # thr/margin already locked — no sweep needed.
        best = (None, float(a.thr), float(a.margin))
        print(f"--final mode: using fixed thr={best[1]:.2f} margin={best[2]:.2f} (no sweep)")

    elif a.thr is not None and a.margin is not None:
        # Caller supplied fixed values — skip the sweep (useful for quick re-trains).
        best = (None, float(a.thr), float(a.margin))
        print(f"fixed thr/margin supplied: thr={best[1]:.2f} margin={best[2]:.2f} (sweep skipped)")

    else:
        # ---- Fine-grained sweep ----
        # WIDENED again (26 Sep, post is_unbalance=True): a prior run with is_unbalance=True hit
        # thr=0.91 margin=0.30, right at the old grid's edge (thr 0.45-0.92, margin 0.00-0.32) -
        # that's a sign the true optimum is outside the tested range, not that 0.91/0.30 is it.
        # is_unbalance reweights the loss and pushes predicted probabilities more extreme, which
        # shifts where the best thr/margin sits - so widen until the winner stops landing on an edge.
        val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
        vt      = {k: v for k, v in truth.items() if k in val_ids}
        best    = (-1.0, 0.5, 0.0)

        thr_grid    = np.arange(0.30, 0.99, 0.02)
        margin_grid = np.arange(0.00, 0.45, 0.02)
        print(f"sweeping {len(thr_grid)} thresholds x {len(margin_grid)} margins "
              f"= {len(thr_grid)*len(margin_grid)} combos …")

        # decode_prep does the expensive sort/groupby ONCE (thr/margin don't affect it);
        # decode_apply per combo is then just a cheap boolean filter. The old code called
        # decode() (full re-sort) 384 times - this is the same result, much faster.
        prepped = decode_prep(r1, ro, p2)

        for thr in thr_grid:
            for mg in margin_grid:
                rr, oo = decode_apply(prepped, thr, mg)
                pred = {}
                for i, o in zip(rr, oo):
                    s = u1[i]
                    if s in val_ids:
                        pred.setdefault(s, set()).add(uo[o])
                sc = f05_macro(pred, vt)
                if sc > best[0]:
                    best = (sc, float(thr), float(mg))

        print(f"validation macro F0.5 = {best[0]:.4f}  "
              f"at thr={best[1]:.2f} margin={best[2]:.2f}  (val S1: {len(vt)})")

    # ------------------------------------------------------------------ save
    m1.booster_.save_model(f"{a.models}/stage1.txt")
    m2.booster_.save_model(f"{a.models}/stage2.txt")
    val_f05 = best[0] if (best[0] is not None) else "N/A (final mode)"
    json.dump(
        dict(cfg=CFG, thr=best[1], margin=best[2], val_f05=val_f05,
             final_mode=a.final),
        open(f"{a.models}/config.json", "w"), indent=2
    )
    print(f"saved models to {a.models}/")
    if a.final:
        print("NOTE: these models were trained on ALL data. "
              "The thr/margin were locked before training — do NOT retune them on this run.")


if __name__ == "__main__":
    main()
