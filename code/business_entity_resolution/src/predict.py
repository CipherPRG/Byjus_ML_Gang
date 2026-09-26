"""Run the trained pipeline on the test set and write output/matching_results.tsv + output/candidate_pairs.tsv.

    python src/predict.py --data dataset/test --models models --out output --workers 4
"""
import argparse, json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from io_utils import read_tsv, countries_of, write_lists
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, decode, raw2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--thr", type=float, default=None); ap.add_argument("--margin", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conf = json.load(open(f"{a.models}/config.json")); cfg = conf["cfg"]
    thr = conf["thr"] if a.thr is None else a.thr; margin = conf["margin"] if a.margin is None else a.margin
    b1 = lgb.Booster(model_file=f"{a.models}/stage1.txt"); b2 = lgb.Booster(model_file=f"{a.models}/stage2.txt")

    s1_all = read_tsv(f"{a.data}/test_source1.tsv", usecols=["entity_id", "country"])
    print("S1 rows per country:", s1_all.country.value_counts().to_dict())
    for k in (2, 3):
        cs = read_tsv(f"{a.data}/test_source{k}.tsv", usecols=["country"]).country.value_counts().to_dict()
        print(f"S{k} rows per country:", cs, "| labels not present in S1:", set(cs) - set(s1_all.country))
    cand_lists, match_lists = {}, {}
    for c in sorted(s1_all.country.unique()):
        s1, oth, cand = build_country(a.data, "test", c, cfg, a.workers)
        if cand is None or len(cand) == 0:
            print(f"[{c}] no candidates"); continue
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        B = 2_000_000
        p1 = np.zeros(len(r1), dtype=np.float32); raw = []
        for s in range(0, len(r1), B):
            X = pair_features(s1, oth, r1[s:s + B], ro[s:s + B], w[s:s + B], a.workers)
            p1[s:s + B] = b1.predict(X); raw.append(raw2(X))
        raw = np.vstack(raw)
        X2 = stage2_matrix(r1, ro, p1, raw); del raw
        p2 = b2.predict(X2)
        rr, oo = decode(r1, ro, p2, thr, margin)
        s1ids, oids = np.array(s1.ids, dtype=object), np.array(oth.ids, dtype=object)
        for i, o in zip(s1ids[r1], oids[ro]):
            cand_lists.setdefault(i, []).append(o)
        for i, o in zip(s1ids[rr], oids[oo]):
            match_lists.setdefault(i, []).append(o)
        print(f"[{c}] S1={len(s1)} others={len(oth)} candidates={len(r1)} matched pairs={len(rr)} "
              f"S1 with >=1 match={len(set(s1ids[rr]))}")
    order = s1_all.entity_id.tolist()
    write_lists(f"{a.out}/matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), order, match_lists)
    write_lists(f"{a.out}/candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), order, cand_lists)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
