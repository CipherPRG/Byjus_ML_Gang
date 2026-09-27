"""Fit the expected-F0.5 decoder for an EXISTING model from its eval_full cache (no retraining).
Run from student_resource/:
    python fit_decoder.py eval_cache_v8 models_v8
Uses only the ES half of the scored val entities to fit; reports threshold vs EF decoder on ES and
on the untouched REP half; writes <models>/decoder.json ONLY if EF beats the threshold decoder on ES
(the choice is made on ES; REP is reported, never used). predict reads decoder.json if present."""
import sys, os, json, pickle, hashlib
sys.path.insert(0, os.path.join("code", "business_entity_resolution", "src"))
import numpy as np
from model import best_assignment, fit_ef_decoder, decode_ef_assigned


def es_half_of(s):  # identical to train.py
    return hashlib.md5((s + "e").encode()).digest()[0] % 2 == 0


def f05(n_pred, n_true, tp):
    f = np.zeros(len(n_pred))
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    m = tp > 0
    pr, rc = tp[m] / n_pred[m], tp[m] / n_true[m]
    f[m] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f


def main():
    cache, models = sys.argv[1], sys.argv[2]
    conf = json.load(open(f"{models}/config.json"))
    thr, margin = conf["thr"], conf["margin"]
    G = []
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(".npz"):
            continue
        c = fn[:-4]
        z = np.load(os.path.join(cache, fn))
        ids = pickle.load(open(os.path.join(cache, f"{c}_ids.pkl"), "rb"))
        r1, ro = z["r1"].astype(np.int64), z["ro"].astype(np.int64)
        ra, oa, pa = best_assignment(r1, ro, z["p2"])
        # label of each assigned pair: look it up among the candidate labels
        key = r1 * (ro.max() + 1) + ro
        srt = np.argsort(key); ks = key[srt]
        ka = ra * (ro.max() + 1) + oa
        ya = z["y"][srt[np.searchsorted(ks, ka)]]
        es = np.array([es_half_of(s) for s in ids["s1"]])
        # margin rule of the threshold decoder needs the runner-up; recompute from candidates
        G.append(dict(c=c, n1=len(ids["s1"]), ra=ra, pa=pa, ya=ya, n_true=ids["n_true"].astype(np.int64),
                      scored=ids["scored"], es=es))
        print(f"[{c}] assigned pairs {len(ra):,}  scored S1 {int(ids['scored'].sum()):,}", flush=True)

    def score(keep_fn, half):
        res, allf = {}, []
        for g in G:
            k = keep_fn(g)
            npd = np.bincount(g["ra"][k], minlength=g["n1"]); tp = np.bincount(g["ra"][k & g["ya"]], minlength=g["n1"])
            f = f05(npd, g["n_true"], tp)
            m = g["scored"] & (g["es"] if half == "es" else ~g["es"])
            res[g["c"]] = f[m].mean(); allf.append(f[m])
        res["ALL"] = np.concatenate(allf).mean()
        return res

    fmt = lambda d: "  ".join(f"{k} {v:.4f}" for k, v in d.items())
    thr_keep = lambda g: g["pa"] >= thr  # margin is 0.00 for v7/v8; see note below
    if margin > 0:
        print(f"NOTE: config margin={margin} > 0; this comparison applies thr only")
    print(f"\nthreshold decoder (thr={thr}):")
    t_es, t_rep = score(thr_keep, "es"), score(thr_keep, "rep")
    print("   ES :", fmt(t_es)); print("   REP:", fmt(t_rep), " <- honest")

    if "--eval-only" in sys.argv:
        # score the EXISTING decoder.json of <models> as shipped (no refit, nothing written)
        dec = json.load(open(f"{models}/decoder.json"))
        ef_keep = lambda g: decode_ef_assigned(g["ra"], g["pa"], dec)
        print(f"\nexisting {models}/decoder.json (lam={dec['lam']:.3f}), as shipped:")
        print("   ES :", fmt(score(ef_keep, "es"))); print("   REP:", fmt(score(ef_keep, "rep")), " <- honest")
        return

    # fit on ES-half scored entities only
    P = np.concatenate([g["pa"][(g["scored"] & g["es"])[g["ra"]]] for g in G])
    Y = np.concatenate([g["ya"][(g["scored"] & g["es"])[g["ra"]]] for g in G])
    nt = np.concatenate([g["n_true"][g["scored"] & g["es"]] for g in G])
    nf = np.concatenate([np.bincount(g["ra"][g["ya"]], minlength=g["n1"])[g["scored"] & g["es"]] for g in G])
    dec = fit_ef_decoder(P, Y, nt, nf)
    ef_keep = lambda g: decode_ef_assigned(g["ra"], g["pa"], dec)
    print(f"\nexpected-F0.5 decoder (fitted on ES: lam={dec['lam']:.3f}, {len(dec['cal_x'])} calibration knots):")
    e_es, e_rep = score(ef_keep, "es"), score(ef_keep, "rep")
    print("   ES :", fmt(e_es)); print("   REP:", fmt(e_rep), " <- honest")

    if e_es["ALL"] > t_es["ALL"]:
        dec.update(fitted_on=cache, es_f05=e_es["ALL"], rep_f05=e_rep["ALL"],
                   thr_es_f05=t_es["ALL"], thr_rep_f05=t_rep["ALL"])
        json.dump(dec, open(f"{models}/decoder.json", "w"), indent=1)
        print(f"\nEF wins on ES -> wrote {models}/decoder.json (predict will use it)")
    else:
        print("\nEF does not beat the threshold decoder on ES -> NOT writing decoder.json")


if __name__ == "__main__":
    main()
