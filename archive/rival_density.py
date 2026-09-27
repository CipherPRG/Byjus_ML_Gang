"""How many S1 businesses compete for each S2/S3 record: sample_v2 (training/val) vs real test candidates.
Read-only, low RAM (tracks a 1-in-20 hash subset of S2/S3 ids). Run from student_resource/:
    python rival_density.py eval_cache_v8 output_v8/candidate_pairs.tsv"""
import sys, os, zlib, pickle
from collections import Counter
import numpy as np


def summarize(label, counts):
    c = np.array(list(counts), dtype=np.int64)
    print(f"{label:28s} records {len(c):>9,} | mean S1 rivals {c.mean():5.2f} | "
          f">=2 rivals {np.mean(c >= 2):.3f} | >=5 {np.mean(c >= 5):.3f} | >=10 {np.mean(c >= 10):.3f}")


cache, test_cand = sys.argv[1], sys.argv[2]
for fn in sorted(os.listdir(cache)):
    if fn.endswith(".npz"):
        z = np.load(os.path.join(cache, fn))
        ro = z["ro"]
        summarize(f"sample_v2 {fn[:-4]}", np.bincount(ro)[np.bincount(ro) > 0])

sub = lambda o: zlib.crc32(o.encode()) % 20 == 0
per_country = {}
s1c = {}
import csv
with open("dataset/test/test_source1.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        s1c[p[0]] = p[-1]
with open(test_cand, encoding="utf-8") as f:
    f.readline()
    for line in f:
        s, _, m = line.rstrip("\r\n").partition("\t")
        if not m:
            continue
        cnt = per_country.setdefault(s1c.get(s, "?"), Counter())
        for o in m.split(","):
            if sub(o):
                cnt[o] += 1
for c, cnt in sorted(per_country.items()):
    summarize(f"TEST {c}", cnt.values())
