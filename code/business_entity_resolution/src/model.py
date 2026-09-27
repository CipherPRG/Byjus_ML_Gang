"""Two-stage matcher and decoders.

Stage 1: LightGBM on pair features -> p1.
Stage 2: LightGBM on the competition around each pair (its rank / gap / rivals among the S1's candidates and
         among the S1 options of the S2/S3 record) + a few raw pair features -> p2.
Decoding: every S2/S3 record is assigned to at most one S1 (its best). Two rules:
  - decode():    keep it if p2 >= thr and p2 - runner-up >= margin (thr/margin chosen on validation);
  - decode_ef(): expected-F0.5 decoder - calibrated probabilities + per-S1 choice of how many matches to keep.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import F1

RAW2 = ["nsort", "aset", "ajac", "akey_eq", "a_exact", "anum_first_eq", "w", "ncore_eq", "sk_r", "sk_set",
        "hnum_edit"]
# Tree caps are upper bounds only: early stopping on the ES half of validation picks the real tree count.
P1 = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0, is_unbalance=True,
          metric="auc", verbose=-1)
P2 = dict(n_estimators=800, learning_rate=0.06, num_leaves=31, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.9, min_child_samples=20, reg_lambda=1.0, is_unbalance=True,
          metric="auc", verbose=-1)


def raw2(X):
    """The few raw pair features that stage 2 keeps (small memory)."""
    return X[:, [F1.index(c) for c in RAW2]].astype(np.float32)


def stage2_matrix(r1, ro, p1, raw, density=None):
    """Stage-2 matrix: context of p1 inside each S1's candidate list and each S2/S3 record's S1 options
    (rank, gap to the best, strongest rival, sum of rivals) + the RAW2 raw pair features -> 19 columns.

    density (feat_v3 only): {"addr_by_ro": array, "name_by_ro": array}, aligned with the rows, giving how many
    Source-1 businesses (per 100k, counted in the FULL S1 file) share the S2/S3 record's exact normalised
    address / name skeleton. High = non-specific (sibling businesses). Adds 2 log1p columns (21 in total)."""
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
    if density is not None:
        out["log1p_addr_density"] = np.log1p(density["addr_by_ro"]).astype(np.float32)
        out["log1p_name_density"] = np.log1p(density["name_by_ro"]).astype(np.float32)
    return out.astype(np.float32)


def fit_stage1(X, y, eval_set=None, n_estimators=None):
    """eval_set: optional (X_val, y_val) tuple. When given, uses early stopping (50 rounds, AUC)
    instead of training the full fixed n_estimators blind. n_estimators: override the tree count
    (used for the OOF fold models so they match the early-stopped main model)."""
    p = dict(P1)
    if n_estimators is not None:
        p["n_estimators"] = int(n_estimators)
    m = lgb.LGBMClassifier(**p)
    if eval_set is not None:
        Xv, yv = eval_set
        m.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X, y)
    return m


def fit_stage2(X, y, eval_set=None, n_estimators=None):
    """eval_set: optional (X_val, y_val) tuple. When given, uses early stopping (50 rounds, AUC)
    instead of training the full fixed n_estimators blind."""
    p = dict(P2)
    if n_estimators is not None:
        p["n_estimators"] = int(n_estimators)
    m = lgb.LGBMClassifier(**p)
    if eval_set is not None:
        Xv, yv = eval_set
        m.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.fit(X, y)
    return m


def decode_prep(r1, ro, p):
    """The expensive, thr/margin-independent part of decode(): sort once and find each S2/S3 record's best
    and runner-up candidate. Call it once, then decode_apply() for every threshold/margin combination."""
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


def best_assignment(r1, ro, p):
    """Each S2/S3 record -> its single best S1 (same rule and tie order as decode_prep).
    Returns (r1, ro, p) of the assigned pairs as numpy arrays."""
    first = decode_prep(r1, ro, p)
    return first.r1.values.astype(np.int64), first.index.values.astype(np.int64), first.p.values.astype(np.float64)


def fit_ef_decoder(p_assigned, y_assigned, n_true_per_s1, n_found_per_s1):
    """Fit the expected-F0.5 decoder on a labelled set (train.py's ES half only).
    - calibration: isotonic regression p2 -> P(true match) on the assigned pairs
      (is_unbalance makes raw p2 over-confident, so it cannot be used as a probability directly)
    - lam: mean number of true matches per S1 that are NOT among its assigned candidates
      (blocking misses etc.), i.e. how many unseen matches to expect.
    Pooled over all countries: no per-country fitting, so it applies unchanged to unseen countries.
    Returns a JSON-serialisable dict."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(np.asarray(p_assigned, dtype=np.float64), np.asarray(y_assigned, dtype=np.float64))
    lam = float(np.mean(np.asarray(n_true_per_s1) - np.asarray(n_found_per_s1))) if len(n_true_per_s1) else 0.0
    return dict(kind="ef", cal_x=[float(v) for v in iso.X_thresholds_],
                cal_y=[float(v) for v in iso.y_thresholds_], lam=max(lam, 0.0), beta2=0.25)


def decode_ef_assigned(r1a, pa, dec):
    """Expected-F0.5 decoding on already-assigned pairs (r1a, pa): per S1, keep the top-k candidates
    (by calibrated probability q) that maximise expected F0.5 ~ (1+b2)*sum(q_1..k) / (b2*(sum q + lam) + k);
    keep none if P(no true match) ~ prod(1-q)*exp(-lam) is higher (a correct empty prediction scores 1.0).
    Returns a boolean keep-mask aligned with r1a."""
    if len(r1a) == 0:
        return np.zeros(0, dtype=bool)
    q = np.interp(pa, dec["cal_x"], dec["cal_y"])
    lam, b2 = dec["lam"], dec.get("beta2", 0.25)
    order = np.lexsort((-q, r1a))                     # by S1, then q descending
    rs, qs = r1a[order], q[order]
    start = np.r_[True, rs[1:] != rs[:-1]]
    gid = np.cumsum(start) - 1
    gstart = np.flatnonzero(start)
    k = np.arange(len(rs)) - gstart[gid] + 1
    cs = np.cumsum(qs); base = np.r_[0.0, cs[gstart[1:] - 1]] if len(gstart) > 1 else np.zeros(1)
    cs = cs - base[gid]
    tot = np.bincount(gid, weights=qs) + lam
    ef = (1 + b2) * cs / (b2 * tot[gid] + k)
    lp0 = np.bincount(gid, weights=np.log1p(-np.minimum(qs, 1 - 1e-6)))
    p0 = np.exp(lp0 - lam)
    # best k per group (first occurrence of the max)
    ng = len(gstart)
    best_ef = np.full(ng, -1.0); np.maximum.at(best_ef, gid, ef)
    is_best = ef >= best_ef[gid]
    kbest = np.full(ng, np.iinfo(np.int64).max); np.minimum.at(kbest, gid, np.where(is_best, k, np.iinfo(np.int64).max))
    kstar = np.where(best_ef > p0, kbest, 0)
    keep_sorted = k <= kstar[gid]
    keep = np.zeros(len(r1a), dtype=bool)
    keep[order] = keep_sorted
    return keep


def decode_ef(r1, ro, p, dec):
    """Full expected-F0.5 decode: best S1 per S2/S3 record, then decode_ef_assigned. Returns (r1, ro) kept."""
    ra, oa, pa = best_assignment(r1, ro, p)
    keep = decode_ef_assigned(ra, pa, dec)
    return ra[keep], oa[keep]


def decode(r1, ro, p, thr, margin):
    """Each S2/S3 record is assigned to its best S1 only, if p>=thr and it beats the runner-up by
    `margin`. Convenience one-shot wrapper (predict.py etc). For a thr/margin SWEEP, call
    decode_prep() once and decode_apply() per combo instead - much faster."""
    return decode_apply(decode_prep(r1, ro, p), thr, margin)
