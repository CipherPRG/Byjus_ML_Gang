"""Check LightGBM stage2 feature importance - specifically to answer whether a_exact
(0.0 gain in stage1, per check_feature_importance.py) is also dead weight in stage2,
where it's fed in directly via model.py's RAW2 list.

Builds X/Y/S1ID/OID across countries exactly like model_bench.py's main() does, fits
stage1 (in-sample, just to get p1 for stage2's context features - fine for an
importance check, not a real training run), builds the stage2 matrix via
stage2_matrix(), fits stage2 with the production P2 config, and prints stage2's own
gain-based feature importance, so a_exact's stage2 contribution can be judged on its
own rather than assumed from its (irrelevant) stage1 importance.

Usage:
    python src/check_stage2_importance.py --data ../../sample_dense --workers 5
"""
import argparse
import numpy as np

from io_utils import read_tsv, countries_of
from pipeline import build_country
from features import pair_features
from model import fit_stage1, fit_stage2, stage2_matrix, raw2

CFG = dict(max_block=30, max_s1_block=200, topk=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

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
        oids = np.array(oth.ids, dtype=object)[ro]
        y = np.fromiter((o in truth.get(s, ()) for s, o in zip(s1ids, oids)),
                         dtype=np.int8, count=len(r1))
        X.append(feats); Y.append(y); S1ID.append(s1ids); OID.append(oids)

    X = np.vstack(X); Y = np.concatenate(Y)
    S1ID = np.concatenate(S1ID); OID = np.concatenate(OID)
    print(f"total pairs: {len(Y)}, positives: {int(Y.sum())}")

    u1, r1 = np.unique(S1ID, return_inverse=True)
    uo, ro = np.unique(OID, return_inverse=True)

    print("Fitting stage1 (in-sample, just to get p1 for stage2 context features)...")
    m1 = fit_stage1(X, Y)
    p1 = m1.predict_proba(X)[:, 1]

    RAW2 = raw2(X)
    X2 = stage2_matrix(r1, ro, p1, RAW2)
    cols = list(X2.columns)
    print(f"stage2 feature matrix: {X2.shape}, columns: {cols}")

    print("Fitting stage2...")
    m2 = fit_stage2(X2.values, Y)
    imp = m2.booster_.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]

    print("\nStage2 feature importance (gain), sorted high to low:")
    print(f"{'feature':<16}{'importance':>14}{'pct_of_total':>14}")
    total = imp.sum() if imp.sum() > 0 else 1.0
    for i in order:
        print(f"{cols[i]:<16}{imp[i]:>14.1f}{imp[i] / total * 100:>13.2f}%")

    zero_or_near = [cols[i] for i in order if imp[i] <= 0]
    print("\n" + "=" * 60)
    if zero_or_near:
        print(f"Zero/near-zero contribution ({len(zero_or_near)}): {zero_or_near}")
    else:
        print("No zero-importance stage2 features.")
    if "a_exact" in cols:
        idx = cols.index("a_exact")
        print(f"\na_exact stage2 importance: {imp[idx]:.1f} ({imp[idx]/total*100:.2f}% of total gain)")


if __name__ == "__main__":
    main()
