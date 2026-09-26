"""Two-stage matcher: stage 1 = pair features -> p1; stage 2 = context of p1 inside the S1 / S2-S3 neighbourhood -> p2."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import F1

RAW2 = ["nsort", "aset", "ajac", "akey_eq", "a_exact", "anum_first_eq", "w", "ncore_eq", "sk_r", "sk_set",
        "hnum_edit"]
P1 = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0, is_unbalance=True,
          metric="auc", verbose=-1)
P2 = dict(n_estimators=200, learning_rate=0.06, num_leaves=31, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, min_child_samples=20, reg_lambda=1.0, is_unbalance=True,
          metric="auc", verbose=-1)


def raw2(X):
    """the few raw pair features that stage 2 keeps (small memory)"""
    return X[:, [F1.index(c) for c in RAW2]].astype(np.float32)


def stage2_matrix(r1, ro, p1, raw):
    """Context features from p1: rank/gap/top competitor inside each S1 and each S2/S3 record."""
    d = pd.DataFrame({"r1": r1, "ro": ro, "p": p1})
    g1 = d.groupby("r1").p; go = d.groupby("ro").p
    cols = {
        "p1": p1,
        "rk_s1": g1.rank(ascending=False, method="min").values,
        "gap_s1": (d.p - g1.transform("max")).values,
        "rk_o": go.rank(ascending=False, method="min").values,
        "gap_o": (d.p - go.transform("max")).values,
        "max_o_other": (go.transform("max")).values,
        "sum_s1": g1.transform("sum").values - p1,
        "sum_o": go.transform("sum").values - p1,
    }
    out = pd.DataFrame(cols)
    for j, c in enumerate(RAW2):
        out[c] = raw[:, j]
    return out.astype(np.float32)


def fit_stage1(X, y, eval_set=None):
    """eval_set: optional (X_val, y_val) tuple. When given, uses early stopping (50 rounds, AUC)
    instead of training the full fixed n_estimators blind."""
    m = lgb.LGBMClassifier(**P1)
    if eval_set is not None:
        Xv, yv = eval_set
        m.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X, y)
    return m


def fit_stage2(X, y, eval_set=None):
    """eval_set: optional (X_val, y_val) tuple. When given, uses early stopping (50 rounds, AUC)
    instead of training the full fixed n_estimators blind."""
    m = lgb.LGBMClassifier(**P2)
    if eval_set is not None:
        Xv, yv = eval_set
        m.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X, y)
    return m


def decode_prep(r1, ro, p):
    """The expensive, thr/margin-INDEPENDENT part of decode(): sort once, find each 'other'
    record's best and runner-up candidate. Call this ONCE, then decode_apply() many times for
    a threshold/margin sweep instead of re-sorting the whole dataframe per combo (this was the
    actual bottleneck in the 384-combo sweep - each combo used to redo this full sort)."""
    d = pd.DataFrame({"r1": r1, "ro": ro, "p": p}).sort_values(["ro", "p"], ascending=[True, False])
    first = d.groupby("ro").head(1).set_index("ro")
    second = d.groupby("ro").nth(1).set_index("ro").p
    first["p2"] = second.reindex(first.index).fillna(0.0).values
    return first


def decode_apply(first, thr, margin):
    """Cheap filter step, given `first` from decode_prep(). This is the only part that
    actually depends on thr/margin."""
    keep = first[(first.p >= thr) & ((first.p - first.p2) >= margin)]
    return keep.r1.values, keep.index.values


def decode(r1, ro, p, thr, margin):
    """Each S2/S3 record is assigned to its best S1 only, if p>=thr and it beats the runner-up by
    `margin`. Convenience one-shot wrapper (predict.py etc). For a thr/margin SWEEP, call
    decode_prep() once and decode_apply() per combo instead - much faster."""
    return decode_apply(decode_prep(r1, ro, p), thr, margin)
