"""Build a training sample based on what blocking actually produces.

FIXED copy of Aayush's make_sample_v2.py (branch aayush-sample-v2): competing S1 entities are no longer
written (they had no ground-truth rows, which mislabelled their true matches as negatives).

The key difference from make_sample.py:
  Old: keeps 1/mod of S1 by hash of (country+name). Wrong candidates with
       similar-but-different names or the same address are ~mod×  rarer than
       in real data, so the model never learns to reject them.
  New: picks S1 by hash of entity_id, then runs the real blocking to find
       every S2/S3 record those S1s would actually compete with. The hard
       negatives (same street, similar name, different entity) are kept at
       their real density.

Usage
-----
# Quick test on the small sample/dataset/train folder:
python experiments/make_sample_v2.py --data ../../sample/dataset/train --out ../../sample_v2_test --frac 0.2

# Full run (needs ~16 GB RAM; let Pratham do this):
python experiments/make_sample_v2.py --data ../../dataset/train --out ../../sample_v2 --frac 0.05

# India only first (fits in 12 GB):
python experiments/make_sample_v2.py --data ../../dataset/train --out ../../sample_v2_india --frac 0.05 --country India

Then train with:
  python experiments/train.py --data ../../sample_v2 --models ../../models_v7 --workers 8 --addr-stop-frac 0.01
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import argparse
import os
import sys
import zlib
from io_utils import read_tsv, countries_of
from pipeline import build_country

# ------------------------------------------------------------------
# v6 config (same as the best-known setting; addr_stop_frac via CLI)
# ------------------------------------------------------------------
CFG_BASE = dict(max_block=60, max_s1_block=200, topk=60)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _s1_hash(entity_id: str) -> int:
    """Stable [0, 1 000 000) integer for an S1 entity_id."""
    return zlib.crc32(entity_id.encode("utf-8")) % 1_000_000


def _keep_s1(entity_id: str, frac: float) -> bool:
    """Return True for ~frac of S1 entities, deterministically."""
    return _s1_hash(entity_id) < int(frac * 1_000_000)


def _tsv_stream(path):
    """Yield (header_line, row_iterator) streaming the raw TSV."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline().rstrip("\r\n")
        lines = [l.rstrip("\r\n") for l in f]
    return header, iter(lines)


def _read_gt(path):
    """Return dict  source1_entity_id -> frozenset(matched_ids)."""
    df = read_tsv(path)
    result = {}
    for _, row in df.iterrows():
        ids = {m for m in str(row.matched_entity_ids).split(",") if m.strip()}
        result[row.source1_entity_id] = ids
    return result


# ------------------------------------------------------------------
# main logic
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a blocking-realistic training sample (v2)."
    )
    ap.add_argument("--data",  required=True,
                    help="path to train directory (contains train_source1/2/3.tsv + train_ground_truth.tsv)")
    ap.add_argument("--out",   required=True,
                    help="output directory (written in same 4-file format as sample_dense)")
    ap.add_argument("--frac",  type=float, default=0.05,
                    help="fraction of S1 entities to keep, by hash of entity_id (default 0.05 = 5%%)")
    ap.add_argument("--country", default=None,
                    help="restrict to one country (useful on low-memory machines)")
    ap.add_argument("--addr-stop-frac", type=float, default=0.01,
                    help="generic-address-word threshold for blocking (default 0.01, matches v6)")
    a = ap.parse_args()

    if a.frac <= 0 or a.frac > 1:
        ap.error("--frac must be in (0, 1]")

    cfg = {**CFG_BASE, "addr_stop_frac": a.addr_stop_frac}
    os.makedirs(a.out, exist_ok=True)

    # ---- load ground truth ----
    gt_path = os.path.join(a.data, "train_ground_truth.tsv")
    print(f"Loading ground truth: {gt_path}")
    truth = _read_gt(gt_path)
    print(f"  {len(truth)} S1 entities in ground truth")

    # ---- discover countries ----
    s1_path = os.path.join(a.data, "train_source1.tsv")
    all_countries = countries_of(s1_path)
    if a.country:
        if a.country not in all_countries:
            print(f"ERROR: --country '{a.country}' not in {all_countries}", file=sys.stderr)
            sys.exit(1)
        all_countries = [a.country]
    print(f"Countries to process: {all_countries}")

    # ---- per-country blocking pass ----
    # Collect:
    #   keep_s1_ids : set of S1 entity_ids to keep
    #   keep_oth_ids: set of S2/S3 entity_ids to keep (blocking candidates + true matches)
    #   comp_s1_ids : set of competing S1 entity_ids (stretch goal: those that share a candidate)
    keep_s1_ids  = set()
    keep_oth_ids = set()
    comp_s1_ids  = set()

    per_country_stats = {}

    for country in all_countries:
        print(f"\n--- {country} ---")
        print(f"  Running build_country (blocking) …")
        s1, oth, cand = build_country(a.data, "train", country, cfg, workers=1)

        if oth is None or cand is None or len(cand) == 0:
            print(f"  No candidates — skipping")
            continue

        s1_ids  = s1.ids   # list, index = row index in s1
        oth_ids = oth.ids  # list, index = row index in oth

        # Which S1 entities do we keep?
        sel_mask  = [_keep_s1(eid, a.frac) for eid in s1_ids]
        sel_rows  = {i for i, m in enumerate(sel_mask) if m}
        sel_eids  = {s1_ids[i] for i in sel_rows}
        keep_s1_ids.update(sel_eids)

        # Filter candidate df to only rows where r1 is a selected S1
        cand_sel = cand[cand.r1.isin(sel_rows)].copy()

        # Candidates' S2/S3 entity_ids
        cand_oth_eids = {oth_ids[ro] for ro in cand_sel.ro.values}

        # True matches of selected S1s (may not all be in candidate set)
        true_oth_eids = set()
        for eid in sel_eids:
            true_oth_eids.update(truth.get(eid, set()))

        keep_oth_ids.update(cand_oth_eids)
        keep_oth_ids.update(true_oth_eids)

        # Stretch: other S1 entities that compete for the same S2/S3 records
        # i.e. any non-selected S1 row that has a candidate in keep_oth_ids
        cand_all_sel_oth = cand[cand.ro.isin(
            {oth_ids.index(e) for e in keep_oth_ids if e in oth_ids}
            if False else  # use set of row indices instead
            set(cand_sel.ro.unique())
        )]
        # Correct: get all r1 rows that compete for any oth row already selected
        sel_oth_rows = set(cand_sel.ro.unique())
        comp_r1_rows = set(cand[cand.ro.isin(sel_oth_rows)].r1.unique()) - sel_rows
        comp_eids = {s1_ids[i] for i in comp_r1_rows}
        comp_s1_ids.update(comp_eids)

        # Stats
        n_sel      = len(sel_eids)
        n_cand_sel = len(cand_sel)
        cands_per_s1 = n_cand_sel / n_sel if n_sel else 0
        wrong_cand = cand_oth_eids - true_oth_eids
        per_country_stats[country] = dict(
            n_s1_total=len(s1_ids),
            n_s1_kept=n_sel,
            n_oth_cands=len(cand_oth_eids),
            n_true_matches=len(true_oth_eids),
            n_wrong_cands=len(wrong_cand),
            cands_per_s1=cands_per_s1,
            n_comp_s1=len(comp_eids),
        )
        print(f"  S1 total: {len(s1_ids)}  kept: {n_sel}  comp_S1: {len(comp_eids)}")
        print(f"  Cand others: {len(cand_oth_eids)}  true matches: {len(true_oth_eids)}  wrong cands: {len(wrong_cand)}")
        print(f"  Candidates / kept-S1: {cands_per_s1:.1f}")

    print(f"\nTotal S1 kept     : {len(keep_s1_ids)}")
    print(f"Total S1 competing: {len(comp_s1_ids)}")
    print(f"Total S2/S3 kept  : {len(keep_oth_ids)}")

    # All S1 entities to include (selected + competing)
    # FIX (Pratham/Claude, 26 Sep): competing S1s were written to train_source1.tsv WITHOUT ground-truth rows,
    # so train.py labelled all their true matches as negatives (label noise). Version 1 drops them.
    all_s1_ids = keep_s1_ids

    # ---- write output files ----
    print(f"\nWriting output to: {a.out}/")

    # train_source1.tsv — selected + competing S1 rows
    _write_filtered(
        src=os.path.join(a.data, "train_source1.tsv"),
        dst=os.path.join(a.out,  "train_source1.tsv"),
        keep_ids=all_s1_ids,
        id_col=0,
        label="train_source1.tsv",
    )

    # train_source2.tsv and train_source3.tsv — keep_oth_ids only
    for name in ("train_source2.tsv", "train_source3.tsv"):
        _write_filtered(
            src=os.path.join(a.data, name),
            dst=os.path.join(a.out,  name),
            keep_ids=keep_oth_ids,
            id_col=0,
            label=name,
        )

    # train_ground_truth.tsv — only rows for keep_s1_ids (NOT competing S1s — they have no GT row here)
    _write_filtered(
        src=os.path.join(a.data, "train_ground_truth.tsv"),
        dst=os.path.join(a.out,  "train_ground_truth.tsv"),
        keep_ids=keep_s1_ids,
        id_col=0,
        label="train_ground_truth.tsv",
    )

    # ---- self-checks ----
    print("\nRunning self-checks …")
    _self_check(a.out, truth, keep_s1_ids, keep_oth_ids)

    # ---- comparison stats vs sample_dense ----
    print("\n" + "=" * 60)
    print("SAMPLE STATS")
    print("=" * 60)
    print(f"  {'':20s}  {'S1 kept':>10}  {'Others':>10}  {'Cand/S1':>10}  {'Wrong%':>8}")
    for country, st in sorted(per_country_stats.items()):
        wrong_pct = (st["n_wrong_cands"] / st["n_oth_cands"] * 100
                     if st["n_oth_cands"] else 0)
        print(f"  {country:20s}  {st['n_s1_kept']:>10}  {st['n_oth_cands']:>10}  "
              f"{st['cands_per_s1']:>10.1f}  {wrong_pct:>7.1f}%")
    print(f"  {'TOTAL':20s}  {len(keep_s1_ids):>10}  {len(keep_oth_ids):>10}")
    print("=" * 60)
    print(f"\nDone. Output in: {a.out}")
    print("Next step: python experiments/train.py --data ../../<out_dir> --models ../../models_v7 "
          "--workers 8 --addr-stop-frac 0.01")


# ------------------------------------------------------------------
# file utilities
# ------------------------------------------------------------------

def _write_filtered(src, dst, keep_ids, id_col, label):
    """Write src TSV to dst keeping only rows whose id_col field is in keep_ids.
    Also deduplicates by id_col (first occurrence wins).
    """
    kept = 0
    seen = set()
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as fin, \
         open(dst, "w", encoding="utf-8", newline="\n") as fout:
        header = fin.readline()
        fout.write(header.rstrip("\r\n") + "\n")
        for line in fin:
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) <= id_col:
                continue
            eid = parts[id_col]
            if eid in keep_ids and eid not in seen:
                fout.write(stripped + "\n")
                seen.add(eid)
                kept += 1
    print(f"  {label}: {kept} rows written")
    return kept


def _self_check(out_dir, truth, keep_s1_ids, keep_oth_ids):
    """
    Three checks:
    (a) Every true match of every kept S1 is present in the output S2/S3.
    (b) Ground truth has exactly one row per kept S1, and none for others.
    (c) No duplicate entity_ids in any output file.
    """
    errors = []

    # load output ids
    def _load_ids(name, id_col=0):
        path = os.path.join(out_dir, name)
        ids = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()  # skip header
            for ln in f:
                p = ln.rstrip("\r\n").split("\t")
                if len(p) > id_col and p[id_col]:
                    ids.append(p[id_col])
        return ids

    def _load_gt_ids(name):
        path = os.path.join(out_dir, name)
        ids = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for ln in f:
                p = ln.rstrip("\r\n").split("\t")
                if p and p[0]:
                    ids.append(p[0])
        return ids

    s2_ids  = set(_load_ids("train_source2.tsv"))
    s3_ids  = set(_load_ids("train_source3.tsv"))
    oth_out = s2_ids | s3_ids
    s1_out  = set(_load_ids("train_source1.tsv"))
    gt_ids  = _load_gt_ids("train_ground_truth.tsv")

    # (a) true matches present
    missing_matches = 0
    for eid in keep_s1_ids:
        for mid in truth.get(eid, set()):
            if mid and mid not in oth_out:
                missing_matches += 1
    if missing_matches:
        errors.append(f"  (a) FAIL: {missing_matches} true-match IDs missing from output S2/S3")
    else:
        print("  (a) PASS: all true matches of kept S1s are in output S2/S3")

    # (b) GT rows exactly match keep_s1_ids, no extras
    gt_set = set(gt_ids)
    extra_in_gt  = gt_set - keep_s1_ids
    missing_from_gt = keep_s1_ids - gt_set
    if extra_in_gt:
        errors.append(f"  (b) FAIL: {len(extra_in_gt)} extra rows in GT not in keep_s1_ids")
    elif missing_from_gt:
        errors.append(f"  (b) FAIL: {len(missing_from_gt)} kept S1s missing from GT output")
    else:
        print("  (b) PASS: ground truth has exactly one row per kept S1")

    # GT must not contain comp_s1_ids (they aren't in keep_s1_ids by construction, so this is covered above)

    # (c) no duplicates
    for fname in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        ids = _load_ids(fname)
        if len(ids) != len(set(ids)):
            from collections import Counter
            dups = [k for k, v in Counter(ids).items() if v > 1]
            errors.append(f"  (c) FAIL: {fname} has {len(dups)} duplicate entity_ids")
        else:
            print(f"  (c) PASS: no duplicate entity_ids in {fname}")

    gt_dup = len(gt_ids) != len(set(gt_ids))
    if gt_dup:
        errors.append("  (c) FAIL: train_ground_truth.tsv has duplicate source1_entity_id rows")
    else:
        print("  (c) PASS: no duplicate rows in train_ground_truth.tsv")

    if errors:
        print("\nSELF-CHECK FAILURES:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("\nAll self-checks PASSED.")


if __name__ == "__main__":
    main()
