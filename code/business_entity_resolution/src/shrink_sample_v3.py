"""Shrink a sample_v3 to a smaller fraction of sampled S1 WITHOUT re-blocking, keeping competition exact.

A smaller --frac of make_sample_v3 would sample the S1 with crc32(id) % 1e6 < frac*1e6 (a subset of the
bigger sample's S1) and keep every real candidate pair of those S1 plus every rival pair on the records
they touch. All of those pairs are already in the big sample's train_pairs.tsv, so this is exactly what
make_sample_v3 would produce at the smaller frac (same S1, same records, same rivals, same weights).

  python src/shrink_sample_v3.py --src ../../sample_v3 --out ../../sample_v3s --max-rows 33000000
Passes 1+2 report the size at several fracs; the largest frac with total rows <= --max-rows is written
(or force one with --frac)."""
import argparse, os, zlib
import numpy as np
import pandas as pd

FRACS = [0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01]
H = lambda s: zlib.crc32(s.encode("utf-8")) % 1_000_000     # identical to make_sample_v3.keep_s1


def chunks(path):
    return pd.read_csv(path, sep="\t", dtype={"s1_id": str, "oth_id": str, "w": np.float32, "country": str},
                       chunksize=2_000_000, keep_default_na=False)


def write_filtered(src, dst, keep):
    n = 0
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as fi, \
            open(dst, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(fi.readline().rstrip("\r\n") + "\n")
        for line in fi:
            s = line.rstrip("\r\n")
            if s and s.split("\t", 1)[0] in keep:
                fo.write(s + "\n"); n += 1
    print(f"  {os.path.basename(dst)}: {n:,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=float, default=33e6); ap.add_argument("--frac", type=float, default=None)
    a = ap.parse_args()
    gt = pd.read_csv(f"{a.src}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    h1 = {s: H(s) for s in gt.source1_entity_id}
    pf = f"{a.src}/train_pairs.tsv"

    # pass 1: sampled pairs -> per record, the smallest hash among its sampled S1
    parts, n_s = [], np.zeros(len(FRACS), dtype=np.int64)
    for ch in chunks(pf):
        hs = ch.s1_id.map(h1)
        m = hs.notna()
        hv = hs[m].values.astype(np.int64)
        n_s += [(hv < f * 1e6).sum() for f in FRACS]
        parts.append(pd.DataFrame({"o": ch.oth_id[m].values, "h": hv}))
    minh = pd.concat(parts).groupby("o").h.min(); del parts
    print(f"pass 1 done: {len(minh):,} records touched by sampled S1", flush=True)

    # pass 2: at frac f a pair is a rival pair iff its S1 is NOT sampled at f (non-GT, or a GT S1 with
    # hash >= f - those become rivals in the smaller sample) and its record is touched by an S1 sampled at f
    n_r = np.zeros(len(FRACS), dtype=np.int64)
    for ch in chunks(pf):
        hv = ch.s1_id.map(h1).fillna(1e9).values
        mv = ch.oth_id.map(minh).fillna(1e9).values
        n_r += [((hv >= f * 1e6) & (mv < f * 1e6)).sum() for f in FRACS]
    print(f"{'frac':>6} {'sampled S1':>11} {'sampled pairs':>14} {'rival pairs':>12} {'total rows':>11}")
    hv_all = np.array(list(h1.values()))
    for f, s_, r_ in zip(FRACS, n_s, n_r):
        print(f"{f:>6} {int((hv_all < f * 1e6).sum()):>11,} {int(s_):>14,} {int(r_):>12,} {int(s_ + r_):>11,}",
              flush=True)
    f = a.frac if a.frac else max([f for f, s_, r_ in zip(FRACS, n_s, n_r) if s_ + r_ <= a.max_rows] or [FRACS[-1]])
    lim = f * 1e6
    print(f"-> writing frac {f} to {a.out}", flush=True)

    # pass 3: write
    os.makedirs(a.out, exist_ok=True)
    keep_s1 = {s for s, h in h1.items() if h < lim}
    keep_o, used_s1 = set(), set()
    with open(f"{a.out}/train_pairs.tsv", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("s1_id\toth_id\tw\tcountry\n")
        for ch in chunks(pf):
            hv = ch.s1_id.map(h1).fillna(1e9).values
            samp = hv < lim
            riv = ~samp & (ch.oth_id.map(minh).fillna(1e9).values < lim)
            sel = ch[samp | riv]
            sel.to_csv(fo, sep="\t", header=False, index=False)
            keep_o.update(sel.oth_id.values); used_s1.update(sel.s1_id.values)
    for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids):   # true matches of kept S1
        if s in keep_s1:
            keep_o.update(x for x in m.split(",") if x)
    write_filtered(f"{a.src}/train_source1.tsv", f"{a.out}/train_source1.tsv", keep_s1 | used_s1)
    for k in (2, 3):
        write_filtered(f"{a.src}/train_source{k}.tsv", f"{a.out}/train_source{k}.tsv", keep_o)
    write_filtered(f"{a.src}/train_ground_truth.tsv", f"{a.out}/train_ground_truth.tsv", keep_s1)
    print(f"done: frac {f}: {len(keep_s1):,} sampled S1, {len(used_s1 - keep_s1):,} rival S1, "
          f"{len(keep_o):,} S2/S3 records", flush=True)


if __name__ == "__main__":
    main()
