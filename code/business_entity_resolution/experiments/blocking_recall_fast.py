"""Fast blocking-recall check. Reuses the SAME build_country()/candidates() code path
that train.py and model_bench.py already use (fast, vectorized-ish blocking), instead of
adithya-sundar's blocking_audit.py, which recomputes keys itself via plain-Python
iterrows() over ~800K records and took 30+ min without finishing.

Gives you the headline number - blocking_recall (fraction of ground-truth S1->match
pairs that actually survive into the candidate set, i.e. the ceiling stage1/stage2 can
never beat) - plus zero-candidate rate, in the time it takes build_country to run
(same as the "total pairs" step you've already seen finish quickly in model_bench.py).

Does NOT reproduce the original script's failure-mode A/B/C/D/E breakdown - if you need
that level of detail, it's worth writing directly rather than debugging the slow version.

Usage:
    python experiments/blocking_recall_fast.py --data ../../../sample_dense --workers 5
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import numpy as np
from io_utils import read_tsv, countries_of
from pipeline import build_country

CFG = dict(max_block=30, max_s1_block=200, topk=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: {m for m in v.split(",") if m}
             for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}

    total_s1 = 0
    total_gt_pairs = 0
    total_recovered = 0
    zero_cand_s1 = 0
    total_candidates = 0

    for c in countries_of(f"{a.data}/train_source1.tsv"):
        s1, oth, cand = build_country(a.data, "train", c, CFG, a.workers)
        s1_ids = np.array(s1.ids, dtype=object)
        total_s1 += len(s1_ids)

        if cand is None:
            for sid in s1_ids:
                total_gt_pairs += len(truth.get(sid, ()))
            zero_cand_s1 += len(s1_ids)
            print(f"  [{c}] no 'other' side data at all - {len(s1_ids)} S1 records, "
                  f"{sum(len(truth.get(sid, ())) for sid in s1_ids)} gt pairs, all missed")
            continue

        oth_ids = np.array(oth.ids, dtype=object)
        r1 = cand.r1.values
        ro = cand.ro.values
        total_candidates += len(r1)

        cand_by_s1 = {}
        for i, o in zip(r1, ro):
            cand_by_s1.setdefault(i, set()).add(oth_ids[o])

        c_gt_pairs = 0
        c_recovered = 0
        c_zero = 0
        for idx, sid in enumerate(s1_ids):
            gt_set = truth.get(sid)
            if not gt_set:
                continue
            c_gt_pairs += len(gt_set)
            got = cand_by_s1.get(idx, set())
            c_recovered += len(gt_set & got)
            if idx not in cand_by_s1:
                c_zero += 1

        total_gt_pairs += c_gt_pairs
        total_recovered += c_recovered
        zero_cand_s1 += c_zero
        c_recall = c_recovered / c_gt_pairs if c_gt_pairs else float("nan")
        print(f"  [{c}] s1={len(s1_ids)} candidates={len(r1)} gt_pairs={c_gt_pairs} "
              f"recovered={c_recovered} recall={c_recall:.4f} zero_cand_s1={c_zero}")

    recall = total_recovered / total_gt_pairs if total_gt_pairs else float("nan")
    zero_rate = zero_cand_s1 / total_s1 if total_s1 else float("nan")
    avg_cand = total_candidates / total_s1 if total_s1 else float("nan")

    print("\n" + "=" * 60)
    print(f"total_s1_entities:     {total_s1}")
    print(f"total_gt_pairs:        {total_gt_pairs}")
    print(f"total_recovered_pairs: {total_recovered}")
    print(f"blocking_recall:       {recall:.4f}")
    print(f"total_candidates:      {total_candidates}")
    print(f"avg_candidates_per_s1: {avg_cand:.2f}")
    print(f"zero_candidate_s1:     {zero_cand_s1}")
    print(f"zero_candidate_rate:   {zero_rate:.4f}")


if __name__ == "__main__":
    main()
