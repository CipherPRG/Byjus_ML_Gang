"""Score a matching_results.tsv against a ground-truth file with the competition metric (macro F0.5).
python score_output.py <matching_results.tsv> <ground_truth.tsv>"""
import sys
import pandas as pd

pred = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv(sys.argv[2], sep="\t", dtype=str, keep_default_na=False)
P = {s: {x for x in m.split(",") if x} for s, m in zip(pred.source1_entity_id, pred.matched_entity_ids)}
tot = 0.0
for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids):
    t = {x for x in m.split(",") if x}; p = P.get(s, set())
    if not t and not p:
        tot += 1.0; continue
    tp = len(t & p)
    if tp == 0:
        continue
    pr, rc = tp / len(p), tp / len(t)
    tot += 1.25 * pr * rc / (0.25 * pr + rc)
print(f"{sys.argv[1]}: macro F0.5 = {tot / len(gt):.4f} over {len(gt):,} S1")
