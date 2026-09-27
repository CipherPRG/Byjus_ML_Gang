"""Track B experiment: does cross-source corroboration help stage2?

Idea: right now every candidate pair (r1, ro) is scored using only its own features plus
competition context (rank/gap among OTHER candidates for the same r1 or ro). It never asks
"does this S1 entity have independent strong evidence from BOTH source2 AND source3?" - if two
different sources both point strongly to the same entity, that's real corroborating evidence a
per-pair model can't see on its own.

New feature per candidate (r1, ro): n_sources_strong = how many distinct sources (2, 3) have at
least one candidate for this r1 with stage1 p1 >= a threshold. Computed from a per-r1 aggregate,
so it's the same value for every candidate belonging to that r1 (entity-level evidence, not
pair-level) - this is what's genuinely new information stage2 doesn't currently have.

Does NOT touch blocking.py/pipeline.py (Track A's files) - source tags are reconstructed here by
calling Side/candidates directly, mirroring what build_country already does internally.

Usage:
    python src/cross_source_experiment.py --data ../../sample_dense --workers 5
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse, zlib
import numpy as np
import pandas as pd
from io_utils import read_tsv, countries_of, read_country
from pipeline import load_side
from blocking import Side, candidates
from features import pair_features
from model import fit_stage1, fit_stage2, stage2_matrix, raw2, decode_prep, decode_apply
from evaluate import f05_macro

CFG = dict(max_block=30, max_s1_block=200, topk=30)
THR_GRID = np.arange(0.30, 0.99, 0.02)
MARGIN_GRID = np.arange(0.00, 0.45, 0.02)
STRONG_P1 = 0.70  # p1 threshold for "strong independent evidence" from a source


def build_country_with_source(dir_, prefix, country, cfg, workers=1):
    """Like pipeline.build_country, but also returns a `source` array (2 or 3) aligned with
    oth's rows, so cross-source agreement can be computed."""
    s1 = Side(read_country(f"{dir_}/{prefix}_source1.tsv", country))
    parts, sources = [], []
    for k in (2, 3):
        p = load_side(f"{dir_}/{prefix}_source{k}.tsv", country)
        if p is not None:
            parts.append(p)
            sources.append(np.full(len(p), k, dtype=np.int8))
    if not parts:
        return s1, None, None, None
    oth = Side.concat(parts)
    source = np.concatenate(sources)
    cand = candidates(s1, oth, max_block=cfg["max_block"], max_s1_block=cfg["max_s1_block"], topk=cfg["topk"])
    return s1, oth, cand, source


def sweep(r1, ro, p2, u1, uo, val_ids, vt):
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
    a = ap.parse_args()

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    X, Y, S1ID, OID, SOURCE = [], [], [], [], []
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand, source = build_country_with_source(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats); Y.append(y); S1ID.append(s1ids); OID.append(oids)
        SOURCE.append(source[ro])

    X = np.vstack(X); Y = np.concatenate(Y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    SOURCE = np.concatenate(SOURCE)
    print(f"total pairs: {len(Y)}")

    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID, return_inverse=True)
    is_val = np.array([zlib.crc32((s + "v").encode()) % 10 < 3 for s in u1])[r1]
    tr = ~is_val
    fold = np.array([zlib.crc32((s + "f").encode()) % 2 for s in u1])[r1]
    val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
    vt = {k: v for k, v in truth.items() if k in val_ids}

    print("Fitting stage1 (OOF)...")
    p1 = np.zeros(len(Y), dtype=np.float32)
    for f in (0, 1):
        fit = tr & (fold != f)
        pred = tr & (fold == f)
        p1[pred] = fit_stage1(X[fit], Y[fit]).predict_proba(X[pred])[:, 1]
    m1 = fit_stage1(X[tr], Y[tr])
    p1[is_val] = m1.predict_proba(X[is_val])[:, 1]

    # ---- new feature: n_sources_strong, per r1 (entity-level, broadcast to every candidate) ----
    d = pd.DataFrame({"r1": r1, "source": SOURCE, "p1": p1})
    strong = d[d.p1 >= STRONG_P1]
    n_sources = strong.groupby("r1").source.nunique()
    n_sources_by_entity = pd.Series(0, index=np.arange(len(u1)))
    n_sources_by_entity.loc[n_sources.index] = n_sources.values
    cross_src_feat = n_sources_by_entity.reindex(r1).values.astype(np.float32)  # broadcast to every row

    print(f"n_sources_strong distribution: {pd.Series(cross_src_feat).value_counts().to_dict()}")

    RAW2 = raw2(X)

    def run(extra_feat=None, label=""):
        X2 = stage2_matrix(r1, ro, p1, RAW2)
        if extra_feat is not None:
            X2["cross_src"] = extra_feat
        m2 = fit_stage2(X2[tr].values, Y[tr])
        p2 = m2.predict_proba(X2.values)[:, 1]
        best = sweep(r1, ro, p2, u1, uo, val_ids, vt)
        print(f"{label}: F0.5={best[0]:.4f}  thr={best[1]:.2f}  margin={best[2]:.2f}")
        return best

    print("\n=== baseline stage2 (no cross-source feature) ===")
    baseline = run(None, "baseline")

    print("\n=== stage2 + cross-source agreement feature ===")
    with_feat = run(cross_src_feat, "with cross_src")

    print("\n" + "=" * 60)
    print(f"baseline F0.5:          {baseline[0]:.4f}")
    print(f"with cross_src F0.5:    {with_feat[0]:.4f}")
    print(f"delta: {with_feat[0] - baseline[0]:+.4f}")
    if with_feat[0] > baseline[0]:
        print("-> cross-source agreement feature HELPS. Worth adding to model.py's stage2_matrix().")
    else:
        print("-> no improvement - not worth the added complexity.")


if __name__ == "__main__":
    main()
