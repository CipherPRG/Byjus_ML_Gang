"""Blend the stage-2 scores of two models (e.g. v10 + v11) -- no retraining, no new features.

Two modes. Run from student_resource/ (like fit_decoder.py).

1) CHOOSE (on labelled data): both models' eval_full caches on the SAME --data folder with --val-split.
     python blend_scores.py choose eval_cache_v10_on_v3 eval_cache_v11_on_v3 blend_v10_v11.json
   For each weight w in 0, 0.25, 0.5, 0.75, 1 (p = w*pA + (1-w)*pB) it sweeps thr/margin (same grid as
   train.py) and picks the best (w, thr, margin) on the ES half ONLY. The REP half is only reported.
   w=1 is model A alone, w=0 is model B alone, so the blend is kept only if it beats both on ES.

2) APPLY (on test): both models' predict output folders (predict saves scores_<country>.npz).
     python blend_scores.py apply output_v10 output_v11 blend_v10_v11.json dataset/test output_blend
   Writes output_blend/matching_results.tsv + candidate_pairs.tsv with the same decoder rule as
   predict (each S2/S3 record -> its best S1 if p >= thr and p - runner-up >= margin).

Both models must have scored the same candidate pairs (same blocking config); the script checks
this and stops if they differ.
"""
import sys, os, json, pickle, hashlib
sys.path.insert(0, os.path.join("code", "business_entity_resolution", "src"))
import numpy as np

THR_GRID = np.concatenate([np.arange(0.30, 0.98, 0.02), [0.98, 0.985, 0.99, 0.993, 0.996, 0.998]])  # = train.py
MARGIN_GRID = np.arange(0.00, 0.45, 0.02)                                                            # = train.py
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def es_half_of(s):  # identical to train.py / fit_decoder.py
    return hashlib.md5((s + "e").encode()).digest()[0] % 2 == 0


def f05(n_pred, n_true, tp):  # identical rules to evaluate.f05_macro
    f = np.zeros(len(n_pred))
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    m = tp > 0
    pr, rc = tp[m] / n_pred[m], tp[m] / n_true[m]
    f[m] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f


def best_per_other(r1, ro, p):
    """For each S2/S3 record: its best S1 (ties: first in input order, like decode_prep's stable sort),
    the best score, and the runner-up score (0 if none). Returns index of the best pair, p_best, p_second."""
    order = np.lexsort((-p, ro))               # by ro, then p descending; lexsort is stable
    ros = ro[order]
    start = np.r_[True, ros[1:] != ros[:-1]]
    first = order[start]
    gstart = np.flatnonzero(start)
    has2 = np.r_[gstart[1:] - gstart[:-1] > 1, len(ros) - gstart[-1] > 1] if len(gstart) else np.zeros(0, bool)
    second = np.zeros(len(first), dtype=np.float64)
    second[has2] = p[order[gstart[has2] + 1]]
    return first, p[first].astype(np.float64), second


def _align(rA, oA, idsA, rB, oB, idsB):
    """Index into B for every pair of A. Fast path when both used identical row indices."""
    if idsA["s1"] == idsB["s1"] and idsA["oth"] == idsB["oth"] and np.array_equal(rA, rB) and np.array_equal(oA, oB):
        return np.arange(len(rA))
    s1map = {s: i for i, s in enumerate(idsB["s1"])}; omap = {s: i for i, s in enumerate(idsB["oth"])}
    s1A = np.array([s1map.get(s, -1) for s in idsA["s1"]], dtype=np.int64)
    oAm = np.array([omap.get(s, -1) for s in idsA["oth"]], dtype=np.int64)
    n = len(idsB["oth"]) + 1
    kA = s1A[rA] * n + oAm[oA]
    kB = rB.astype(np.int64) * n + oB
    srt = np.argsort(kB); ks = kB[srt]
    pos = np.searchsorted(ks, kA); pos[pos >= len(ks)] = 0
    ok = (ks[pos] == kA) & (s1A[rA] >= 0) & (oAm[oA] >= 0)
    if not ok.all() or len(rA) != len(rB):
        raise SystemExit(f"candidate pairs differ between the two models ({int((~ok).sum()):,} of {len(rA):,} "
                         f"A-pairs not in B; sizes {len(rA):,} vs {len(rB):,}) - blend not valid, stopping")
    return srt[pos]


def load_cache(cache):
    out = {}
    for fn in sorted(os.listdir(cache)):
        if fn.endswith(".npz"):
            c = fn[:-4]
            out[c] = (np.load(os.path.join(cache, fn)), pickle.load(open(os.path.join(cache, f"{c}_ids.pkl"), "rb")))
    return out


def choose(cacheA, cacheB, out_json):
    A, B = load_cache(cacheA), load_cache(cacheB)
    if set(A) != set(B):
        raise SystemExit(f"different countries in the caches: {sorted(A)} vs {sorted(B)}")
    G = []
    for c in sorted(A):
        za, ia = A[c]; zb, ib = B[c]
        r1, ro = za["r1"].astype(np.int64), za["ro"].astype(np.int64)
        j = _align(r1, ro, ia, zb["r1"].astype(np.int64), zb["ro"].astype(np.int64), ib)
        es = np.array([es_half_of(s) for s in ia["s1"]])
        G.append(dict(c=c, n1=len(ia["s1"]), r1=r1, ro=ro, y=za["y"].astype(bool),
                      pa=za["p2"].astype(np.float64), pb=zb["p2"].astype(np.float64)[j],
                      n_true=ia["n_true"].astype(np.int64), es=ia["scored"] & es, rep=ia["scored"] & ~es))
        print(f"[{c}] {len(r1):,} pairs aligned; ES S1 {int(G[-1]['es'].sum()):,}  REP S1 {int(G[-1]['rep'].sum()):,}",
              flush=True)

    res = []
    for w in WEIGHTS:
        pre = []
        for g in G:
            p = w * g["pa"] + (1 - w) * g["pb"]
            f, pb, ps = best_per_other(g["r1"], g["ro"], p)
            pre.append((g["r1"][f], pb, pb - ps, g["y"][f]))

        def score(thr, mg, half):
            fs = []
            for g, (fr, pb, gap, fy) in zip(G, pre):
                k = (pb >= thr) & (gap >= mg)
                npd = np.bincount(fr[k], minlength=g["n1"]); tp = np.bincount(fr[k & fy], minlength=g["n1"])
                fs.append(f05(npd, g["n_true"], tp)[g[half]])
            return float(np.concatenate(fs).mean())

        best = (-1.0, 0.0, 0.0)
        for thr in THR_GRID:
            for mg in MARGIN_GRID:
                s = score(thr, mg, "es")
                if s > best[0]:
                    best = (s, float(thr), float(mg))
        rep = score(best[1], best[2], "rep")
        res.append(dict(w=w, thr=best[1], margin=best[2], es=best[0], rep=rep))
        print(f"w={w:.2f} (A weight): ES {best[0]:.4f} at thr={best[1]:.3f} margin={best[2]:.2f} | REP {rep:.4f}",
              flush=True)

    win = max(res, key=lambda r: r["es"])          # chosen on ES only
    a_alone = [r for r in res if r["w"] == 1.0][0]; b_alone = [r for r in res if r["w"] == 0.0][0]
    print(f"\nA alone: ES {a_alone['es']:.4f} REP {a_alone['rep']:.4f} | B alone: ES {b_alone['es']:.4f} "
          f"REP {b_alone['rep']:.4f}")
    print(f"chosen on ES: w={win['w']:.2f} thr={win['thr']:.3f} margin={win['margin']:.2f} -> "
          f"ES {win['es']:.4f} | REP {win['rep']:.4f}  <- honest")
    if win["w"] in (0.0, 1.0):
        print("-> a single model wins on ES: no blend. Submit that model's own output.")
    json.dump(dict(cacheA=cacheA, cacheB=cacheB, chosen=win, all=res), open(out_json, "w"), indent=1)
    print(f"wrote {out_json}")


def apply(outA, outB, blend_json, test_dir, out_dir):
    from io_utils import read_tsv, write_lists
    cfg = json.load(open(blend_json))["chosen"]
    w, thr, mg = cfg["w"], cfg["thr"], cfg["margin"]
    print(f"blend w={w:.2f} (A={outA}) thr={thr:.3f} margin={mg:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    ca = sorted(f for f in os.listdir(outA) if f.startswith("scores_") and f.endswith(".npz"))
    cb = sorted(f for f in os.listdir(outB) if f.startswith("scores_") and f.endswith(".npz"))
    if ca != cb:
        raise SystemExit(f"different score files: {ca} vs {cb}")
    cand_lists, match_lists = {}, {}
    for fn in ca:
        za, zb = np.load(os.path.join(outA, fn)), np.load(os.path.join(outB, fn))
        ia = {"s1": za["s1_ids"].tolist(), "oth": za["oth_ids"].tolist()}
        ib = {"s1": zb["s1_ids"].tolist(), "oth": zb["oth_ids"].tolist()}
        r1, ro = za["r1"].astype(np.int64), za["ro"].astype(np.int64)
        j = _align(r1, ro, ia, zb["r1"].astype(np.int64), zb["ro"].astype(np.int64), ib)
        p = w * za["p2"].astype(np.float64) + (1 - w) * zb["p2"].astype(np.float64)[j]
        f, pb, ps = best_per_other(r1, ro, p)
        k = (pb >= thr) & ((pb - ps) >= mg)
        s1ids = np.array(ia["s1"], dtype=object); oids = np.array(ia["oth"], dtype=object)
        for i, o in zip(s1ids[r1], oids[ro]):
            cand_lists.setdefault(i, []).append(o)
        for i, o in zip(s1ids[r1[f][k]], oids[ro[f][k]]):
            match_lists.setdefault(i, []).append(o)
        print(f"[{fn[7:-4]}] candidates={len(r1):,} matched pairs={int(k.sum()):,}", flush=True)
    order = read_tsv(f"{test_dir}/test_source1.tsv", usecols=["entity_id"]).entity_id.tolist()
    write_lists(f"{out_dir}/matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), order, match_lists)
    write_lists(f"{out_dir}/candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), order, cand_lists)
    print(f"S1 {len(order):,} | with >=1 match {len(match_lists):,} | matched pairs "
          f"{sum(len(v) for v in match_lists.values()):,}\nwrote {out_dir}/matching_results.tsv and candidate_pairs.tsv")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "choose":
        choose(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 7 and sys.argv[1] == "apply":
        apply(*sys.argv[2:7])
    else:
        print(__doc__)
