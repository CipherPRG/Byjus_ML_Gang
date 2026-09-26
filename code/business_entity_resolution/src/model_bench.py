"""B4.5 (WORKPLAN.md) - light-to-heavy stage1 model comparison + a couple of extra LightGBM
hyperparameter combos, all measured on the SAME train/val split and SAME full 2-stage pipeline
(stage2 + decode unchanged) so results are directly comparable by validation macro F0.5.

Does NOT modify model.py or train.py - this is a read-only benchmark. Whatever wins, you copy
its config into model.py's P1 dict yourself (per WORKPLAN.md B4.5: "do not swap the production
model without Track D's sign-off").

Usage (same data/workers args as train.py):
    python src/model_bench.py --data ../../sample_dense --workers 5
    python src/model_bench.py --data ../../sample_dense --workers 5 --candidates logreg,rf200,current
"""
import argparse, time, zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, raw2, fit_stage2, decode_prep, decode_apply, P1 as CURRENT_P1
from evaluate import f05_macro

CFG = dict(max_block=30, max_s1_block=200, topk=30)

# thr/margin sweep - same widened range as train.py's current sweep
THR_GRID    = np.arange(0.30, 0.99, 0.02)
MARGIN_GRID = np.arange(0.00, 0.45, 0.02)


class LGBWrap:
    """Thin wrapper so LightGBM configs share the same .fit/.predict_proba interface as sklearn
    models below, with no eval_set/early stopping (bench = plain fixed-n_estimators fit, matching
    what train.py's OOF loop does)."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.m = lgb.LGBMClassifier(**self.params)
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class Pipeline_LR:
    """StandardScaler + LogisticRegression, same .fit/.predict_proba interface."""
    def fit(self, X, y):
        self.sc = StandardScaler()
        Xs = self.sc.fit_transform(X)
        self.m = LogisticRegression(max_iter=300, class_weight="balanced")
        self.m.fit(Xs, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.sc.transform(X))


CANDIDATES = {
    # name -> zero-arg factory returning a fresh, unfit estimator with .fit/.predict_proba
    "logreg":       lambda: Pipeline_LR(),
    "rf200":        lambda: RandomForestClassifier(
                        n_estimators=200, max_depth=20, n_jobs=-1,
                        class_weight="balanced", random_state=0),
    "current":      lambda: LGBWrap(**CURRENT_P1),
    "lgb_700_127":  lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 700, "num_leaves": 127}),
    "lgb_300_31_lr08":  lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 300, "num_leaves": 31,
                                            "learning_rate": 0.08}),
    "lgb_500_127_lr08": lambda: LGBWrap(**{**CURRENT_P1, "n_estimators": 500, "num_leaves": 127,
                                            "learning_rate": 0.08}),
}


def sweep_f05(r1, ro, p2, u1, uo, val_ids, vt):
    prepped = decode_prep(r1, ro, p2)
    best = (-1.0, 0.5, 0.0)
    for thr in THR_GRID:
        for mg in MARGIN_GRID:
            rr, oo = decode_apply(prepped, thr, mg)
            pred = {}
            for i, o in zip(rr, oo):
                s = u1[i]
                if s in val_ids:
                    pred.setdefault(s, set()).add(uo[o])
            sc = f05_macro(pred, vt)
            if sc > best[0]:
                best = (sc, float(thr), float(mg))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--candidates", default=",".join(CANDIDATES.keys()),
                     help="comma-separated subset of: " + ",".join(CANDIDATES.keys()))
    a = ap.parse_args()
    names = a.candidates.split(",")
    for n in names:
        if n not in CANDIDATES:
            ap.error(f"unknown candidate '{n}'. choices: {','.join(CANDIDATES.keys())}")

    # ------------------------------------------------------------------ same data build as train.py
    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    X, Y, S1ID, OID = [], [], [], []
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids  = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats); Y.append(y); S1ID.append(s1ids); OID.append(oids)

    X = np.vstack(X); Y = np.concatenate(Y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    print(f"total pairs: {len(Y)}")

    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID,  return_inverse=True)
    is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
    tr     = ~is_val
    fold   = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]
    val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
    vt      = {k: v for k, v in truth.items() if k in val_ids}

    RAW2 = raw2(X)
    results = []

    for name in names:
        print(f"\n=== {name} ===")
        t0 = time.time()

        # out-of-fold p1 on the training portion (same scheme as train.py, needed so stage2
        # doesn't learn from an overfit/in-sample p1 on the training rows)
        p1 = np.zeros(len(Y), dtype=np.float32)
        for f in (0, 1):
            fit_mask  = tr & (fold != f)
            pred_mask = tr & (fold == f)
            est = CANDIDATES[name]()
            est.fit(X[fit_mask], Y[fit_mask])
            p1[pred_mask] = est.predict_proba(X[pred_mask])[:, 1]

        # final stage1 fit on all of tr -> used for val p1 and as the reported train/predict time
        est = CANDIDATES[name]()
        t_fit0 = time.time()
        est.fit(X[tr], Y[tr])
        train_time = time.time() - t_fit0

        t_pred0 = time.time()
        p1[is_val] = est.predict_proba(X[is_val])[:, 1]
        predict_time = time.time() - t_pred0

        # stage2 unchanged - current production config, same as train.py
        X2 = stage2_matrix(r1, ro, p1, RAW2)
        m2 = fit_stage2(X2[tr], Y[tr])
        p2 = m2.predict_proba(X2)[:, 1]

        f05, best_thr, best_mg = sweep_f05(r1, ro, p2, u1, uo, val_ids, vt)
        total_time = time.time() - t0

        print(f"val F0.5={f05:.4f}  thr={best_thr:.2f} margin={best_mg:.2f}  "
              f"stage1_train={train_time:.1f}s  stage1_predict={predict_time:.2f}s  "
              f"total={total_time:.1f}s")
        results.append((name, f05, best_thr, best_mg, train_time, predict_time, total_time))

    print("\n" + "=" * 100)
    print(f"{'model':<20}{'val F0.5':>10}{'thr':>7}{'margin':>8}{'train_s':>10}{'predict_s':>11}{'total_s':>9}")
    for name, f05, thr, mg, tt, pt, tot in sorted(results, key=lambda r: -r[1]):
        print(f"{name:<20}{f05:>10.4f}{thr:>7.2f}{mg:>8.2f}{tt:>10.1f}{pt:>11.2f}{tot:>9.1f}")


if __name__ == "__main__":
    main()
