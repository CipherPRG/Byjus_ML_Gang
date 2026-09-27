"""Hard zone examples: best-S1 pairs with p2 in [LO, HI) - true (lost to thr) vs false (the FPs that force thr up).
Run: python hard_zone.py eval_cache_v7 India 0.93 0.98"""
import sys, os, pickle
import numpy as np
import pandas as pd

CACHE, C = sys.argv[1], sys.argv[2]
LO, HI = float(sys.argv[3]), float(sys.argv[4])
z = np.load(os.path.join(CACHE, f"{C}.npz"))
ids = pickle.load(open(os.path.join(CACHE, f"{C}_ids.pkl"), "rb"))
d = pd.DataFrame({"r1": z["r1"].astype(np.int64), "ro": z["ro"].astype(np.int64), "p": z["p2"], "y": z["y"]})
d = d.sort_values(["ro", "p"], ascending=[True, False]).groupby("ro").head(1)
d = d[ids["scored"][d.r1.values]]
z_ = d[(d.p >= LO) & (d.p < HI)]
hi = d[d.p >= HI]
print(f"[{C}] best-S1 pairs p in [{LO},{HI}): {len(z_):,}  true share {z_.y.mean():.3f} | "
      f"p>={HI}: {len(hi):,} true share {hi.y.mean():.3f}")
on, oa, sn, sa = ids["oth_name"], ids["oth_addr"], ids["s1_name"], ids["s1_addr"]
for lab, sub in (("TRUE (lost)", z_[z_.y]), ("FALSE (would be FP)", z_[~z_.y])):
    print(f"\n===== {lab} =====")
    for _, r in sub.sample(min(15, len(sub)), random_state=1).iterrows():
        a, b = int(r.r1), int(r.ro)
        print(f"  p={r.p:.3f}  S1: {sn[a][:42]:42s} | {sa[a][:55]}")
        print(f"           OT: {on[b][:42]:42s} | {oa[b][:55]}")
