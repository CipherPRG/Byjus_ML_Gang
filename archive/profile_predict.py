"""Time each predict phase on one country (read-only). Run from student_resource/:
   python profile_predict.py sample/dataset/train India models_v8 10"""
import sys, os, time, json
sys.path.insert(0, os.path.join("code", "business_entity_resolution", "src"))
import numpy as np
import lightgbm as lgb
from pipeline import build_country
from features import pair_features
from model import stage2_matrix, raw2, decode


def main():
    DATA, C, M, W = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    cfg = json.load(open(f"{M}/config.json"))["cfg"]
    b1 = lgb.Booster(model_file=f"{M}/stage1.txt"); b2 = lgb.Booster(model_file=f"{M}/stage2.txt")
    T = {}
    t = time.time(); s1, oth, cand = build_country(DATA, "train", C, cfg, W); T["block"] = time.time() - t
    r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
    t = time.time(); X = pair_features(s1, oth, r1, ro, w, W); T["features"] = time.time() - t
    t = time.time(); p1 = b1.predict(X); T["stage1_predict"] = time.time() - t
    t = time.time(); raw = raw2(X); X2 = stage2_matrix(r1, ro, p1, raw); T["stage2_matrix"] = time.time() - t
    t = time.time(); p2 = b2.predict(X2); T["stage2_predict"] = time.time() - t
    t = time.time(); rr, oo = decode(r1, ro, p2, 0.98, 0.0); T["decode"] = time.time() - t
    tot = sum(T.values())
    print(f"{C}: S1={len(s1):,} others={len(oth):,} pairs={len(r1):,}  total {tot:.0f}s")
    for k, v in T.items():
        print(f"  {k:15s} {v:7.1f}s  {v / tot:5.1%}   ({v / len(r1) * 1e6:.1f} us/pair)")


if __name__ == "__main__":
    main()
