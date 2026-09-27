"""Blocking-miss diagnosis on sample_v2 (realistic density). Read-only: trains nothing, writes a report.
Run from student_resource/:   python diag_blocking.py > diag_blocking_report.txt
For every TRUE pair missed by v7 blocking (max_block=60, max_s1_block=200, topk=60, stop 0.01) it says WHY:
  none   = S1 and the other record share no blocking key at all        -> needs a new key type
  capped = they share keys, but every shared key is over a size cap    -> caps
  topk   = shared key survives caps, but pair ranked below top-K        -> topk / weights
Then a small sweep of caps/topk: recall vs candidates-per-S1 (the predict RAM/time cost)."""
import os, sys, random, time
from collections import Counter
sys.path.insert(0, os.path.join("code", "business_entity_resolution", "src"))
import numpy as np
import pandas as pd
from io_utils import read_tsv, read_country, countries_of
from blocking import Side, candidates, keys_for, h64
from pipeline import load_side, country_stop
from norm import norm_name, norm_addr

DATA = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "sample_v2"
KV2 = "--kv2" in sys.argv
CFG = dict(max_block=60, max_s1_block=200, topk=60, addr_stop_frac=0.01, keys_v2=KV2)
print(f"DATA={DATA} CFG={CFG}")
SWEEP = [(60, 200, 60), (60, 200, 100), (100, 300, 60), (100, 300, 100), (150, 400, 100), (150, 400, 150)]

gt = read_tsv(f"{DATA}/train_ground_truth.tsv")
truth = {k: [m for m in v.split(",") if m] for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
rng = random.Random(0)

for c in countries_of(f"{DATA}/train_source1.tsv"):
    t0 = time.time()
    s1_df = read_country(f"{DATA}/train_source1.tsv", c)
    stop = country_stop(s1_df, CFG)
    s1 = Side(s1_df, stop, KV2); del s1_df
    parts = [load_side(f"{DATA}/train_source{k}.tsv", c, stop=stop, kv2=KV2) for k in (2, 3)]
    oth = Side.concat([p for p in parts if p is not None])
    i1 = {s: i for i, s in enumerate(s1.ids)}; io = {s: i for i, s in enumerate(oth.ids)}
    tp = [(i1[s], io[o]) for s in s1.ids for o in truth.get(s, ()) if o in io]
    T = pd.DataFrame(tp, columns=["r1", "ro"])
    print(f"\n==================== {c}: S1={len(s1):,} others={len(oth):,} true pairs={len(T):,} "
          f"(built in {time.time()-t0:.0f}s)", flush=True)

    # block sizes per key, on each side
    k1 = pd.DataFrame({"key": s1.key, "r1": s1.krow}).drop_duplicates()
    k2 = pd.DataFrame({"key": oth.key, "ro": oth.krow, "w": oth.kw}).drop_duplicates(["key", "ro"])
    n1 = k1.groupby("key").size().rename("n1"); n2 = k2.groupby("key").size().rename("n2")

    # all keys shared by each true pair
    sh = T.merge(k1, on="r1").merge(k2, on=["key", "ro"])
    sh = sh.join(n1, on="key").join(n2, on="key")
    sh["ok"] = (sh.n2 <= CFG["max_block"]) & (sh.n1 <= CFG["max_s1_block"])

    cand = candidates(s1, oth, CFG["max_block"], CFG["max_s1_block"], CFG["topk"])
    found = set(zip(cand.r1.values, cand.ro.values))
    T["found"] = [(a, b) in found for a, b in zip(T.r1.values, T.ro.values)]
    anykey = sh.groupby(["r1", "ro"]).size()
    okw = sh[sh.ok].groupby(["r1", "ro"]).w.sum()
    idx = pd.MultiIndex.from_frame(T[["r1", "ro"]])
    T["anykey"] = anykey.reindex(idx).fillna(0).values > 0
    T["okw"] = okw.reindex(idx).fillna(0).values
    T["cat"] = np.where(T.found, "found", np.where(~T.anykey, "none", np.where(T.okw == 0, "capped", "topk")))
    cnt = T.cat.value_counts()
    print(f"recall {T.found.mean():.4f} | missed {int((~T.found).sum()):,}: " +
          ", ".join(f"{k}={cnt.get(k,0):,}" for k in ("none", "capped", "topk")))

    # capped: which key types, how big were the blocks
    cp = sh.merge(T[T.cat == "capped"][["r1", "ro"]], on=["r1", "ro"])
    if len(cp):
        mn = cp.groupby(["r1", "ro"]).n2.min()
        print("  capped: smallest shared S2/S3 block size per pair  quantiles 50/75/90%: "
              f"{mn.quantile(.5):.0f}/{mn.quantile(.75):.0f}/{mn.quantile(.9):.0f}; "
              f"<=100: {(mn<=100).mean():.2f} <=150: {(mn<=150).mean():.2f} <=300: {(mn<=300).mean():.2f}")

    # topk: weight of true pair vs this S1's cutoff weight
    if cnt.get("topk", 0):
        cut = cand.groupby("r1").w.min()
        tk = T[T.cat == "topk"]
        print(f"  topk: true-pair weight below cutoff by: "
              f"{(cut.reindex(tk.r1).values - tk.okw.values).mean():.2f} on avg; "
              f"tie with cutoff: {(cut.reindex(tk.r1).values == tk.okw.values).mean():.2f}")

    # examples of 'none' (no shared key): what do they look like?
    nn = T[T.cat == "none"]
    ex = nn.sample(min(25, len(nn)), random_state=0) if len(nn) else nn
    print(f"  --- {len(ex)} examples with NO shared key (S1 || other) ---")
    for a, b in zip(ex.r1.values, ex.ro.values):
        print(f"   N: {s1.nclean[a][:45]:45s} || {oth.nclean[b][:45]}")
        print(f"   A: {s1.aclean[a][:45]:45s} || {oth.aclean[b][:45]}")
    # quick shape stats of 'none' misses
    if len(nn):
        ne = np.array([len(oth.aclean[b].strip()) == 0 for b in nn.ro.values])
        ne1 = np.array([len(s1.aclean[a].strip()) == 0 for a in nn.r1.values])
        same_name = np.array([s1.nclean[a] == oth.nclean[b] for a, b in zip(nn.r1.values, nn.ro.values)])
        print(f"  none: other addr empty {ne.mean():.2f} | S1 addr empty {ne1.mean():.2f} | "
              f"identical clean name {same_name.mean():.2f}")

    # sweep caps/topk: recall vs cost
    print("  sweep (max_block, max_s1_block, topk) -> recall, cand/S1")
    for mb, ms, tk in SWEEP:
        cc = candidates(s1, oth, mb, ms, tk)
        f = set(zip(cc.r1.values, cc.ro.values))
        rec = np.mean([(a, b) in f for a, b in zip(T.r1.values, T.ro.values)])
        print(f"    ({mb:3d},{ms:3d},{tk:3d}) recall {rec:.4f}  cand/S1 {len(cc)/len(s1):.1f}  pairs {len(cc):,}",
              flush=True)
    del s1, oth, k1, k2, sh, cand
