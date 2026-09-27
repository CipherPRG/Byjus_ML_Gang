"""Full-scale evaluation on the labelled TRAIN set, run exactly like predict.py runs on test.

Why: every validation number so far came from sample_dense (~1/20 of the data). The leaderboard
(86.5) sits ~10 points below our sample_dense val F0.5 (0.967), and local gains don't move it.
This runs the real pipeline at real density on dataset/train (which has ground truth), scores it
with the competition metric, and breaks the lost points down into:
  - blocking misses   (true match never became a candidate)
  - model rejections  (true match was a candidate, decode did not keep it)
  - false positives   (kept a pair that is not a true match)
S1 entities that appear in sample_dense (the model's own training data) are excluded from the
score to avoid leakage (they still take part in blocking/decoding, as they would on test).

Also sweeps thr/margin on the full-scale scores (the sample_dense optimum may not transfer),
and caches per-country arrays to --cache so follow-up analyses don't need a full re-run.

Usage (from code/business_entity_resolution):
    python -u src/eval_full.py --data ../../dataset/train --models ../../models_v3 \
        --exclude ../../sample_dense/train_source1.tsv --cache ../../eval_cache --workers 8
"""
import argparse, json, os, pickle, time, zlib
import numpy as np
import lightgbm as lgb
from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, raw2, decode_prep

THR_GRID = np.concatenate([np.round(np.arange(0.30, 0.98, 0.04), 2), [0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998]])
THR_GRID = np.unique(THR_GRID)
MARGIN_GRID = np.round(np.arange(0.00, 0.41, 0.04), 2)


def in_sorted(keys, sorted_ref):
    """Vectorised membership test of int64 keys against a sorted unique int64 array."""
    if len(sorted_ref) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    idx = np.searchsorted(sorted_ref, keys)
    idx[idx >= len(sorted_ref)] = 0
    return sorted_ref[idx] == keys


def f05_vec(n_pred, n_true, tp):
    """Per-entity F0.5, identical rules to evaluate.f05_macro (empty==empty -> 1, tp==0 -> 0)."""
    f = np.zeros(len(n_pred), dtype=np.float64)
    f[(n_pred == 0) & (n_true == 0)] = 1.0
    m = tp > 0
    pr = tp[m] / n_pred[m]
    rc = tp[m] / n_true[m]
    f[m] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f


def breakdown(label, n_pred, n_true, tp, n_blk, scored):
    f = f05_vec(n_pred, n_true, tp)
    ceil = f05_vec(n_blk, n_true, n_blk)
    N = int(scored.sum())
    fs, loss = f[scored], 1.0 - f[scored]
    npd, ntr, tps, nb = n_pred[scored], n_true[scored], tp[scored], n_blk[scored]
    print(f"\n=== {label}: {N:,} scored S1 entities ===")
    print(f"macro F0.5                                   : {fs.mean():.4f}")
    print(f"ceiling (perfect decode on these candidates) : {ceil[scored].mean():.4f}")
    T, B, P, TP = int(ntr.sum()), int(nb.sum()), int(npd.sum()), int(tps.sum())
    print(f"true pairs {T:,} | in candidates {B:,} (blocking recall {B / max(T, 1):.4f}) | "
          f"predicted {P:,} | TP {TP:,}")
    print(f"pair precision {TP / max(P, 1):.4f} | pair recall {TP / max(T, 1):.4f} | "
          f"recall given blocked {TP / max(B, 1):.4f}")
    print(f"FN from blocking {T - B:,} | FN from model/decode {B - TP:,} | FP {P - TP:,}")
    print(f"truth : singleton rate {(ntr == 0).mean():.2%}, mean matches/entity {ntr.mean():.2f}")
    print(f"ours  : empty rate     {(npd == 0).mean():.2%}, mean matches/entity {npd.mean():.2f}")
    cats = [
        ("singleton, but we matched something", (ntr == 0) & (npd > 0)),
        ("has matches, predicted none: none blocked", (ntr > 0) & (npd == 0) & (nb == 0)),
        ("has matches, predicted none: blocked, rejected", (ntr > 0) & (npd == 0) & (nb > 0)),
        ("has matches, predicted only wrong ones", (ntr > 0) & (npd > 0) & (tps == 0)),
        ("partial (some right, some missed/extra)", (tps > 0) & (fs < 1.0)),
    ]
    print(f"{'where the F0.5 points go':<50}{'entities':>11}{'pts lost':>10}")
    for name, m in cats:
        print(f"{name:<50}{int(m.sum()):>11,}{loss[m].sum() / max(N, 1) * 100:>10.2f}")
    print(f"{'TOTAL':<50}{int((loss > 0).sum()):>11,}{loss.sum() / max(N, 1) * 100:>10.2f}", flush=True)
    return float(fs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset/train folder (must contain train_ground_truth.tsv)")
    ap.add_argument("--models", required=True, help="folder with config.json, stage1.txt, stage2.txt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2_000_000, help="stage-1 batch size (lower if OOM)")
    ap.add_argument("--exclude", default=None,
                    help="TSV whose entity_id column lists S1 ids to leave out of scoring (training sample)")
    ap.add_argument("--cache", default=None, help="folder to save per-country arrays for follow-up analysis")
    ap.add_argument("--countries", nargs="*", default=None, help="e.g. --countries India (default: all)")
    ap.add_argument("--val-split", action="store_true",
                    help="score only train.py's held-out validation entities (crc32(id+'v')%%10<3), so a model "
                         "trained on this --data folder can be compared fairly against another model")
    ap.add_argument("--keys-v3", action="store_true",
                    help="EXTRA candidates from name-word-pair / address-word-pair keys on top of the model's own "
                         "blocking (normal candidates unchanged). Off = exactly as before.")
    ap.add_argument("--keys-v3-topk", type=int, default=5, help="max extra candidates per S1 (with --keys-v3)")
    ap.add_argument("--density-src", default=None,
                    help="feat_v3 models only: FULL train_source1.tsv for density counts when --data is a sample")
    a = ap.parse_args()

    conf = json.load(open(f"{a.models}/config.json"))
    cfg, thr, margin = conf["cfg"], conf["thr"], conf["margin"]
    if a.keys_v3:
        cfg = {**cfg, "keys_v3": True, "keys_v3_topk": a.keys_v3_topk}
    print(f"models={a.models} cfg={cfg} thr={thr:.2f} margin={margin:.2f}", flush=True)
    b1 = lgb.Booster(model_file=f"{a.models}/stage1.txt")
    b2 = lgb.Booster(model_file=f"{a.models}/stage2.txt")

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = dict(zip(gt.source1_entity_id, gt.matched_entity_ids))
    del gt
    excl = set(read_tsv(a.exclude, usecols=["entity_id"]).entity_id) if a.exclude else set()
    print(f"excluding {len(excl):,} S1 ids seen in training", flush=True)
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)

    countries = a.countries or countries_of(f"{a.data}/train_source1.tsv")
    G = []
    for c in countries:
        t0 = time.time()
        s1, oth, cand = build_country(a.data, "train", c, cfg, a.workers, density_src=a.density_src)
        n1 = len(s1)
        # only S1 with a ground-truth row are scored (sample_v3 rival S1 are context without GT rows;
        # scoring them would count them as singletons). Full train / sample_v2: every S1 has a GT row.
        scored = np.fromiter((s in truth and s not in excl
                              and (not a.val_split or zlib.crc32((s + "v").encode()) % 10 < 3)
                              for s in s1.ids), dtype=bool, count=n1)
        n_true = np.zeros(n1, dtype=np.int64)
        for i, s in enumerate(s1.ids):
            m = truth.get(s, "")
            if m:
                n_true[i] = sum(1 for o in m.split(",") if o)
        empty = np.zeros(0, dtype=np.int64)
        if oth is None or cand is None or len(cand) == 0:
            print(f"[{c}] no candidates at all", flush=True)
            G.append(dict(n1=n1, r1=empty, p=empty.astype(np.float32), pp=empty.astype(np.float32),
                          true=empty.astype(bool), n_true=n_true, n_blk=np.zeros(n1, np.int64), scored=scored))
            continue

        n_oth = len(oth)
        o_index = {o: j for j, o in enumerate(oth.ids)}
        tk = []
        for i, s in enumerate(s1.ids):
            m = truth.get(s, "")
            if m:
                for o in m.split(","):
                    j = o_index.get(o) if o else None
                    if j is not None:
                        tk.append(i * n_oth + j)
        del o_index
        true_keys = np.unique(np.array(tk, dtype=np.int64))
        del tk
        n_missing = int(n_true.sum()) - len(true_keys)

        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        del cand
        y = in_sorted(r1.astype(np.int64) * n_oth + ro, true_keys)
        n_blk = np.bincount(r1[y], minlength=n1).astype(np.int64)
        print(f"[{c}] S1={n1:,} others={n_oth:,} candidates={len(r1):,} true pairs={int(n_true.sum()):,} "
              f"(true matches not found in this country's S2/S3: {n_missing:,}) "
              f"blocked in {time.time() - t0:.0f}s", flush=True)

        B = a.batch
        p1 = np.zeros(len(r1), dtype=np.float32)
        raw = []
        for s in range(0, len(r1), B):
            e = min(s + B, len(r1))
            X = pair_features(s1, oth, r1[s:e], ro[s:e], w[s:e], a.workers, feat_v3=bool(cfg.get("feat_v3")))
            p1[s:e] = b1.predict(X)
            raw.append(raw2(X))
            del X
            print(f"  [{c}] stage-1 {e:,}/{len(r1):,}  ({time.time() - t0:.0f}s)", flush=True)
        raw = np.vstack(raw)
        dens = ({"addr_by_ro": oth.dens_addr[ro], "name_by_ro": oth.dens_name[ro]}
                if cfg.get("feat_v3") else None)
        X2 = stage2_matrix(r1, ro, p1, raw, density=dens)
        del dens
        del raw
        p2 = b2.predict(X2).astype(np.float32)
        del X2

        first = decode_prep(r1, ro, p2)
        f_r1 = first.r1.values.astype(np.int64)
        f_ro = first.index.values.astype(np.int64)
        f_p = first.p.values.astype(np.float32)
        f_pp = first.p2.values.astype(np.float32)
        del first
        f_true = in_sorted(f_r1 * n_oth + f_ro, true_keys)

        keep = (f_p >= thr) & ((f_p - f_pp) >= margin)
        n_pred = np.bincount(f_r1[keep], minlength=n1).astype(np.int64)
        tp = np.bincount(f_r1[keep & f_true], minlength=n1).astype(np.int64)
        breakdown(f"{c} @ thr={thr:.2f} margin={margin:.2f}", n_pred, n_true, tp, n_blk, scored)

        if a.cache:
            np.savez(os.path.join(a.cache, f"{c}.npz"), r1=r1, ro=ro, p1=p1, p2=p2, y=y)
            with open(os.path.join(a.cache, f"{c}_ids.pkl"), "wb") as fh:
                pickle.dump({"s1": s1.ids, "oth": oth.ids, "scored": scored, "n_true": n_true,
                             "s1_name": s1.nclean, "s1_addr": s1.aclean,
                             "oth_name": oth.nclean, "oth_addr": oth.aclean}, fh)
        G.append(dict(n1=n1, r1=f_r1, p=f_p, pp=f_pp, true=f_true, n_true=n_true, n_blk=n_blk, scored=scored))
        del r1, ro, w, p1, p2, y
        print(f"[{c}] done in {time.time() - t0:.0f}s", flush=True)

    def assemble(t_, m_):
        npd, tpp = [], []
        for g in G:
            k = (g["p"] >= t_) & ((g["p"] - g["pp"]) >= m_)
            npd.append(np.bincount(g["r1"][k], minlength=g["n1"]))
            tpp.append(np.bincount(g["r1"][k & g["true"]], minlength=g["n1"]))
        return np.concatenate(npd).astype(np.int64), np.concatenate(tpp).astype(np.int64)

    n_true = np.concatenate([g["n_true"] for g in G])
    n_blk = np.concatenate([g["n_blk"] for g in G])
    scored = np.concatenate([g["scored"] for g in G])

    npd, tpp = assemble(thr, margin)
    base = breakdown(f"ALL COUNTRIES @ thr={thr:.2f} margin={margin:.2f}", npd, n_true, tpp, n_blk, scored)

    print(f"\nSweeping {len(THR_GRID)}x{len(MARGIN_GRID)} thr/margin combos on full-scale scores ...", flush=True)
    best = (base, float(thr), float(margin))
    for t_ in THR_GRID:
        for m_ in MARGIN_GRID:
            npd, tpp = assemble(t_, m_)
            sc = float(f05_vec(npd, n_true, tpp)[scored].mean())
            if sc > best[0]:
                best = (sc, float(t_), float(m_))
    print(f"full-scale optimum: F0.5={best[0]:.4f} at thr={best[1]:.2f} margin={best[2]:.2f}  "
          f"(vs {base:.4f} at the sample_dense-tuned thr={thr:.2f} margin={margin:.2f})", flush=True)
    if (best[1], best[2]) != (float(thr), float(margin)):
        npd, tpp = assemble(best[1], best[2])
        breakdown(f"ALL COUNTRIES @ full-scale optimum thr={best[1]:.2f} margin={best[2]:.2f}",
                  npd, n_true, tpp, n_blk, scored)

    if a.cache:
        with open(os.path.join(a.cache, "summary.json"), "w") as fh:
            json.dump({"models": a.models, "cfg": cfg, "thr": thr, "margin": margin, "f05_at_config": base,
                       "best_f05": best[0], "best_thr": best[1], "best_margin": best[2],
                       "countries": countries}, fh, indent=2)
        print(f"\ncached arrays + summary in {a.cache}")


if __name__ == "__main__":
    main()
