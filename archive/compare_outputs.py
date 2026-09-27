"""Label-free comparison of submission files per country (read-only).
Run from student_resource/:  python compare_outputs.py output_v6 output_v7 output_v8
For each output: pairs per S1, empty rate, per country. Then pairwise agreement vs the FIRST output:
share of its pairs kept / pairs added, per country."""
import sys
import pandas as pd

outs = sys.argv[1:] or ["output_v6", "output_v7", "output_v8"]
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], dtype=str)
cmap = dict(zip(s1.entity_id, s1.country))


def load(o):
    df = pd.read_csv(f"{o}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    pairs = set()
    per = {}
    for s, m in zip(df.source1_entity_id, df.matched_entity_ids):
        ms = [x for x in m.split(",") if x]
        per[s] = len(ms)
        pairs.update((s, x) for x in ms)
    return per, pairs


data = {o: load(o) for o in outs}
countries = sorted(set(cmap.values()))
print("per-country: pairs/S1 | empty rate")
for o in outs:
    per, _ = data[o]
    row = []
    for c in countries:
        v = [n for s, n in per.items() if cmap.get(s) == c]
        row.append(f"{c}: {sum(v)/len(v):.3f} | {sum(1 for n in v if n == 0)/len(v):.4f}")
    print(f"  {o:10s} " + "   ".join(row))

base = outs[0]
bp = data[base][1]
print(f"\nagreement vs {base}: share of {base} pairs kept | new pairs as share of {base} pairs")
for o in outs[1:]:
    op = data[o][1]
    row = []
    for c in countries:
        b = {p for p in bp if cmap.get(p[0]) == c}
        n = {p for p in op if cmap.get(p[0]) == c}
        row.append(f"{c}: kept {len(b & n)/max(len(b),1):.3f} | new {len(n - b)/max(len(b),1):.3f}")
    print(f"  {o:10s} " + "   ".join(row))
