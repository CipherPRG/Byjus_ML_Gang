"""Decoder experiment on an eval_full cache (read-only, no training).
Run from student_resource/:  python decode_experiment.py eval_cache_v7
Compares, on the SAME scored val entities of sample_v2:
  A) current decoder: every S2/S3 -> best S1, keep if p2 >= thr (thr chosen on ES half)
  B) expected-F0.5 decoder: p2 calibrated (isotonic, fitted on ES half), then per S1 keep the
     top-k candidates maximising expected F0.5, k=0 (predict singleton) included.
All choices (thr, calibrator, lambda) use the ES half only; the REP half is the honest score."""
import sys, os, pickle, hashlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CACHE = sys.argv[1] if len(sys.argv) > 1 else "eval_cache_v7"
THR = np.concatenate([np.arange(0.30, 0.98, 0.02), [0.98, 0.985, 0.99, 0.993, 0.996, 0.998]])


def es_half_of(s):
    return hashlib.md5((s + "e").encode()).digest()[0] % 2 == 0


def f05(n_pred, n_true, tp):
    f = np.zeros(len(n_pred))
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    m = tp > 0
    pr, rc = tp[m] / n_pred[m], tp[m] / n_true[m]
    f[m] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f


G = []
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith(".npz"):
        continue
    c = fn[:-4]
    z = np.load(os.path.join(CACHE, fn))
    ids = pickle.load(open(os.path.join(CACHE, f"{c}_ids.pkl"), "rb"))
    r1, ro, p2, y = z["r1"].astype(np.int64), z["ro"].astype(np.int64), z["p2"], z["y"]
    # current decoder's first step: each other record -> its best S1
    d = pd.DataFrame({"r1": r1, "ro": ro, "p": p2, "y": y}).sort_values(["ro", "p"], ascending=[True, False])
    d = d.groupby("ro", sort=False).head(1)
    s1 = np.array(ids["s1"], dtype=object)
    es = np.array([es_half_of(s) for s in s1])
    G.append(dict(c=c, n1=len(s1), r1=d.r1.values, p=d.p.values.astype(np.float64), y=d.y.values,
                  n_true=ids["n_true"].astype(np.int64), scored=ids["scored"], es=es))
    print(f"[{c}] S1={len(s1):,} assigned pairs={len(d):,} scored={int(ids['scored'].sum()):,}", flush=True)


def score(pick_fn, half):
    """pick_fn(g) -> boolean keep mask over g's assigned pairs. Returns per-country + all mean F0.5 on half."""
    out, allf = {}, []
    for g in G:
        k = pick_fn(g)
        npd = np.bincount(g["r1"][k], minlength=g["n1"])
        tp = np.bincount(g["r1"][k & g["y"]], minlength=g["n1"])
        f = f05(npd, g["n_true"], tp)
        m = g["scored"] & (g["es"] if half == "es" else ~g["es"] if half == "rep" else True)
        out[g["c"]] = f[m].mean(); allf.append(f[m])
    out["ALL"] = np.concatenate(allf).mean()
    return out


def fmt(d):
    return "  ".join(f"{k} {v:.4f}" for k, v in d.items())


# ---- A) current decoder, thr chosen on ES half
best = max(THR, key=lambda t: score(lambda g: g["p"] >= t, "es")["ALL"])
print(f"\nA) threshold decoder, thr={best:.3f} (chosen on ES)")
print("   ES :", fmt(score(lambda g: g["p"] >= best, "es")))
print("   REP:", fmt(score(lambda g: g["p"] >= best, "rep")), " <- honest")

# ---- B) expected-F0.5 decoder
# calibrator + lambda fitted on ES-half scored entities only (all countries pooled: no per-country
# fitting, so it applies unchanged to an unseen country like France)
P = np.concatenate([g["p"][(g["scored"] & g["es"])[g["r1"]]] for g in G])
Yc = np.concatenate([g["y"][(g["scored"] & g["es"])[g["r1"]]] for g in G])
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(P, Yc)
# lambda = expected true matches per S1 that are NOT among its assigned candidates (blocking misses etc.)
miss = np.concatenate([(g["n_true"] - np.bincount(g["r1"][g["y"]], minlength=g["n1"]))[g["scored"] & g["es"]]
                       for g in G])
lam = float(miss.mean())
print(f"\nB) expected-F0.5 decoder: isotonic calibration + lambda={lam:.3f} (both fitted on ES)")


def efdecode(g, lam_=lam, beta2=0.25):
    q = iso.predict(g["p"])
    d = pd.DataFrame({"r1": g["r1"], "q": q, "i": np.arange(len(q))}).sort_values(["r1", "q"],
                                                                                   ascending=[True, False])
    d["k"] = d.groupby("r1").cumcount() + 1
    d["cs"] = d.groupby("r1").q.cumsum()
    tot = d.groupby("r1").q.transform("sum") + lam_
    d["ef"] = (1 + beta2) * d.cs / (beta2 * tot + d.k)
    # expected F of predicting nothing = P(no true match) ~ prod(1-q) * exp(-lambda)
    d["lp0"] = np.log1p(-d.q.clip(upper=1 - 1e-6))
    p0 = np.exp(d.groupby("r1").lp0.transform("sum") - lam_)
    bestk = d.loc[d.groupby("r1").ef.idxmax()].set_index("r1")
    kstar = bestk.k.where(bestk.ef > p0.groupby(d.r1).first().reindex(bestk.index), 0)
    keep = np.zeros(len(q), dtype=bool)
    d["kstar"] = d.r1.map(kstar)
    keep[d.i.values[d.k.values <= d.kstar.values]] = True
    return keep


print("   ES :", fmt(score(efdecode, "es")))
print("   REP:", fmt(score(efdecode, "rep")), " <- honest")

# ---- robustness: lambda chosen on ES from a grid (vs the fitted value); REP still untouched
print("\nlambda sensitivity (ES | REP):")
for L in (0.0, 0.1, lam, 0.3, 0.5):
    e = score(lambda g: efdecode(g, lam_=L), "es")["ALL"]
    r = score(lambda g: efdecode(g, lam_=L), "rep")["ALL"]
    print(f"   lambda={L:.3f}  ES {e:.4f}  REP {r:.4f}")
