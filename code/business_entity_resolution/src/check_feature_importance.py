"""B2/Track-B checklist item: check LightGBM stage1 feature importance, flag features
that contribute ~nothing so they can be considered for removal.

Reuses the exact same data-build as train.py/model_bench.py (build_country + pair_features)
and fits stage1 with the current production P1 config on the full training split (no val
split needed here - we just want relative importances, not a score).

Usage:
    python src/check_feature_importance.py --data ../../sample_dense --workers 5
"""
import argparse
import numpy as np

from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features, F1
from model import fit_stage1

CFG = dict(max_block=30, max_s1_block=200, topk=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    X, Y = [], []
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        if cand is None:
            continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        feats = pair_features(s1, oth, r1, ro, w, a.workers)
        s1ids = np.array(s1.ids, dtype=object)[r1]
        oids = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats)
        Y.append(y)

    X = np.vstack(X)
    Y = np.concatenate(Y)
    print(f"total pairs: {len(Y)}, positives: {int(Y.sum())}")
    print(f"features: {len(F1)}")

    m1 = fit_stage1(X, Y)
    imp = m1.booster_.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]

    print("\nFeature importance (gain), sorted high to low:")
    print(f"{'feature':<16}{'importance':>14}{'pct_of_total':>14}")
    total = imp.sum() if imp.sum() > 0 else 1.0
    for i in order:
        print(f"{F1[i]:<16}{imp[i]:>14.1f}{imp[i] / total * 100:>13.2f}%")

    zero_or_near = [F1[i] for i in order if imp[i] <= 0]
    print("\n" + "=" * 60)
    if zero_or_near:
        print(f"Zero/near-zero contribution ({len(zero_or_near)}): {zero_or_near}")
        print("Consider dropping these from F1 / features.py's _chunk() (re-validate F0.5 after removing).")
    else:
        print("No zero-importance features - every feature contributes something.")


if __name__ == "__main__":
    main()
