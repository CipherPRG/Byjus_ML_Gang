"""Two-stage matcher: stage 1 = pair features -> p1; stage 2 = context of p1 inside the S1 / S2-S3 neighbourhood -> p2."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import F1

RAW2 = ["nsort", "aset", "ajac", "akey_eq", "a_exact", "anum_first_eq", "w", "ncore_eq", "sk_r", "sk_set"]
P1 = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, verbose=-1)
P2 = dict(n_estimators=200, learning_rate=0.06, num_leaves=31, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, verbose=-1)


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


def fit_stage1(X, y):
    return lgb.LGBMClassifier(**P1).fit(X, y)


def fit_stage2(X, y):
    return lgb.LGBMClassifier(**P2).fit(X, y)


def decode(r1, ro, p, thr, margin):
    """Each S2/S3 record is assigned to its best S1 only, if p>=thr and it beats the runner-up by `margin`."""
    d = pd.DataFrame({"r1": r1, "ro": ro, "p": p}).sort_values(["ro", "p"], ascending=[True, False])
    first = d.groupby("ro").head(1).set_index("ro")
    second = d.groupby("ro").nth(1).set_index("ro").p
    first["p2"] = second.reindex(first.index).fillna(0.0).values
    keep = first[(first.p >= thr) & ((first.p - first.p2) >= margin)]
    return keep.r1.values, keep.index.values
