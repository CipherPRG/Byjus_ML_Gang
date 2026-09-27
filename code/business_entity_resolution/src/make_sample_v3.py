"""Training sample WITH rival businesses (fixes the train/test competition mismatch of sample_v2).

Measured problem (rival_density.py): in the real test every S2/S3 record is a blocking candidate of
~9.5 S1 businesses on average (71% have >= 5); in sample_v2 only ~2 (3-7% have >= 5), because
sample_v2 keeps only the sampled 5% of S1 and drops every rival S1. Stage 2's competition features
(rank / gap / strongest rival inside each S2/S3 record) and the "best S1 wins" decode were therefore
trained and validated with ~5x too little competition.

This builder runs the REAL blocking on the full train data (same cfg as predict) and writes:
  train_source1.tsv       sampled S1 (same hash as sample_v2: same entities at the same --frac)
                          + every rival S1 that is a real candidate of a kept S2/S3 record
  train_source2/3.tsv     S2/S3 records that are candidates of sampled S1 + true matches of sampled S1
  train_ground_truth.tsv  rows for the SAMPLED S1 only (rivals are context, never trained on or scored)
  train_pairs.tsv         the real full-density candidate pairs to use (s1, other, weight): all pairs of
                          sampled S1 + rival pairs whose record is kept. pipeline.build_country uses this
                          file instead of re-blocking inside the sample, so candidate lists and rival
                          counts are exactly what the full data (and the test) produce.
Usage (from code/business_entity_resolution):
  python src/make_sample_v3.py --data ../../dataset/train --out ../../sample_v3 --frac 0.05 --keys-v2
"""
import argparse, os, time, zlib
import numpy as np
import pandas as pd
from io_utils import read_tsv, countries_of
from pipeline import build_country

CFG_BASE = dict(max_block=60, max_s1_block=200, topk=60)


def keep_s1(eid, frac):  # identical to make_sample_v2_fixed._keep_s1
    return zlib.crc32(eid.encode("utf-8")) % 1_000_000 < int(frac * 1_000_000)


def write_filtered(src, dst, keep):
    n, seen = 0, set()
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as fi, \
            open(dst, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(fi.readline().rstrip("\r\n") + "\n")
        for line in fi:
            s = line.rstrip("\r\n")
            if not s:
                continue
            e = s.split("\t", 1)[0]
            if e in keep and e not in seen:
                fo.write(s + "\n"); seen.add(e); n += 1
    print(f"  {os.path.basename(dst)}: {n:,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--addr-stop-frac", type=float, default=0.01)
    ap.add_argument("--keys-v2", action="store_true", help="must match the model's blocking cfg")
    a = ap.parse_args()
    cfg = {**CFG_BASE, "addr_stop_frac": a.addr_stop_frac}
    if a.keys_v2:
        cfg["keys_v2"] = True
    print(f"cfg={cfg} frac={a.frac}", flush=True)
    os.makedirs(a.out, exist_ok=True)

    gt = read_tsv(f"{a.data}/train_ground_truth.tsv")
    truth = {k: [m for m in v.split(",") if m] for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    del gt

    keep_s1_all, keep_oth_all, sel_all = set(), set(), set()
    pf = open(f"{a.out}/train_pairs.tsv", "w", encoding="utf-8", newline="\n")
    pf.write("s1_id\toth_id\tw\tcountry\n")
    for c in countries_of(f"{a.data}/train_source1.tsv"):
        t0 = time.time()
        s1, oth, cand = build_country(a.data, "train", c, cfg, workers=1)
        if cand is None or len(cand) == 0:
            continue
        s1ids = np.array(s1.ids, dtype=object); oids = np.array(oth.ids, dtype=object)
        sel = np.fromiter((keep_s1(e, a.frac) for e in s1.ids), dtype=bool, count=len(s1))
        r1, ro, w = cand.r1.values, cand.ro.values, cand.w.values
        del cand
        is_sel = sel[r1]
        kept_o = np.zeros(len(oth), dtype=bool); kept_o[ro[is_sel]] = True
        take = is_sel | kept_o[ro]                   # sampled-S1 pairs + rival pairs on kept records
        rivals = np.unique(r1[take & ~is_sel])
        n_sel = int(sel.sum())
        # true matches of sampled S1 (so n_true is complete even if blocking missed them)
        o_index = {o: j for j, o in enumerate(oth.ids)}
        for e in s1ids[sel]:
            for m in truth.get(e, ()):
                j = o_index.get(m)
                if j is not None:
                    kept_o[j] = True
        del o_index
        sel_all.update(s1ids[sel]); keep_s1_all.update(s1ids[sel]); keep_s1_all.update(s1ids[rivals])
        keep_oth_all.update(oids[kept_o])
        tr1, tro, tw = r1[take], ro[take], w[take]
        for i in range(0, len(tr1), 2_000_000):
            j = slice(i, i + 2_000_000)
            pd.DataFrame({"s1_id": s1ids[tr1[j]], "oth_id": oids[tro[j]], "w": tw[j], "country": c}) \
                .to_csv(pf, sep="\t", header=False, index=False)
        riv_per_rec = np.bincount(tro, minlength=len(oth))[kept_o]
        print(f"[{c}] S1 {len(s1):,}: sampled {n_sel:,} + rivals {len(rivals):,} | kept records {int(kept_o.sum()):,} | "
              f"pairs: sampled {int(is_sel.sum()):,} + rival {int((take & ~is_sel).sum()):,} | "
              f"S1 per kept record mean {riv_per_rec[riv_per_rec > 0].mean():.2f} | {time.time() - t0:.0f}s", flush=True)
        del s1, oth, r1, ro, w, sel, is_sel, kept_o, take
    pf.close()

    print("writing sample files ...", flush=True)
    write_filtered(f"{a.data}/train_source1.tsv", f"{a.out}/train_source1.tsv", keep_s1_all)
    for k in (2, 3):
        write_filtered(f"{a.data}/train_source{k}.tsv", f"{a.out}/train_source{k}.tsv", keep_oth_all)
    write_filtered(f"{a.data}/train_ground_truth.tsv", f"{a.out}/train_ground_truth.tsv", sel_all)
    print(f"done: {len(sel_all):,} sampled S1 (trained/scored), {len(keep_s1_all) - len(sel_all):,} rival S1 "
          f"(context only), {len(keep_oth_all):,} S2/S3 records -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
