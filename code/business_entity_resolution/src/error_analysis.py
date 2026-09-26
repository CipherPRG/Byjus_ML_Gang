"""Error analysis: categorised false-positive and false-negative report.

For each FP/FN pair, loads the actual name/address strings from source files and
categorises the error:
  - name_collision   : name very similar, address clearly different
  - address_collision: address very similar, name clearly different
  - near_dup         : both name and address are similar (model should have caught this)
  - script_mismatch  : one record has Indic/non-Latin characters (transliteration noise)
  - singleton_fp     : predicted a match for a true singleton
  - other            : doesn't fit the above patterns

Usage (run from student_resource/ root):
    python code/business_entity_resolution/src/error_analysis.py \
        --gt   dataset/train/train_ground_truth.tsv \
        --pred output/matching_results.tsv \
        --src1 dataset/train/train_source1.tsv \
        --src2 dataset/train/train_source2.tsv \
        --src3 dataset/train/train_source3.tsv \
        --n 50

Or with sample_dense/:
    python code/business_entity_resolution/src/error_analysis.py \
        --gt   sample_dense/train_ground_truth.tsv \
        --pred output/matching_results.tsv \
        --src1 sample_dense/train_source1.tsv \
        --src2 sample_dense/train_source2.tsv \
        --src3 sample_dense/train_source3.tsv \
        --n 50

Output is designed to be pasted into the Day 2/Day 3 WORKPLAN.md status update.
"""
import argparse
import os
import sys
import re

import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_TSV = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3,
            encoding="utf-8", encoding_errors="replace")


def _read(path, **kw):
    return pd.read_csv(path, **{**_TSV, **kw})


def _parse_ids(cell):
    if not cell or not cell.strip():
        return frozenset()
    return frozenset(x.strip() for x in cell.split(",") if x.strip())


def _load_records(src1, src2, src3):
    """Returns dict entity_id -> (business_name, business_address, country)."""
    recs = {}
    for path in [src1, src2, src3]:
        df = _read(path, usecols=["entity_id", "business_name", "business_address", "country"])
        for _, row in df.iterrows():
            recs[row.entity_id] = (row.business_name, row.business_address, row.country)
    return recs


# ---------------------------------------------------------------------------
# similarity helpers (works with or without rapidfuzz)
# ---------------------------------------------------------------------------

def _sim(a, b):
    """Token-set ratio in [0,1] — uses rapidfuzz if available, else simple Jaccard."""
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0
    # fallback: token Jaccard
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _has_indic(s):
    return any(c >= "\u0900" for c in s)


def _has_nonlatin(s):
    return any("\u0100" <= c <= "\uFFFF" for c in s)


# ---------------------------------------------------------------------------
# categoriser
# ---------------------------------------------------------------------------

NAME_SIM_HI  = 0.80    # token-set ratio threshold to call names "similar"
NAME_SIM_LO  = 0.40
ADDR_SIM_HI  = 0.70
ADDR_SIM_LO  = 0.25


def _categorise(n1, a1, n2, a2):
    """Return a category label for an FP or FN pair."""
    ns = _sim(n1, n2)
    as_ = _sim(a1, a2)

    # script noise?
    if _has_indic(n1) != _has_indic(n2) or _has_indic(a1) != _has_indic(a2):
        return "script_mismatch"

    if ns >= NAME_SIM_HI and as_ >= ADDR_SIM_HI:
        return "near_dup"
    if ns >= NAME_SIM_HI and as_ < ADDR_SIM_LO:
        return "name_collision"
    if as_ >= ADDR_SIM_HI and ns < NAME_SIM_LO:
        return "address_collision"
    if ns >= NAME_SIM_HI:
        return "near_dup"          # similar name, middling address
    return "other"


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def collect_errors(truth, pred, recs, n_fp=50, n_fn=50):
    """
    truth / pred : dict s1_id -> frozenset
    recs         : dict entity_id -> (name, addr, country)

    Returns:
      fps : list of dicts (up to n_fp false-positive pairs)
      fns : list of dicts (up to n_fn false-negative pairs)
      category_counts_fp : dict category -> int
      category_counts_fn : dict category -> int
    """
    fps, fns = [], []
    cat_fp = {}
    cat_fn = {}

    for s1id, true_set in truth.items():
        p_set = pred.get(s1id, frozenset())
        fp_ids = p_set - true_set
        fn_ids = true_set - p_set

        n1, a1, country = recs.get(s1id, ("?", "?", "?"))

        # false positives
        for oid in fp_ids:
            n2, a2, _ = recs.get(oid, ("?", "?", "?"))
            cat = "singleton_fp" if not true_set else _categorise(n1, a1, n2, a2)
            cat_fp[cat] = cat_fp.get(cat, 0) + 1
            if len(fps) < n_fp:
                fps.append(dict(
                    s1_id=s1id, other_id=oid, country=country,
                    s1_name=n1, s1_addr=a1,
                    other_name=n2, other_addr=a2,
                    category=cat, kind="FP",
                ))

        # false negatives
        for oid in fn_ids:
            n2, a2, _ = recs.get(oid, ("?", "?", "?"))
            cat = _categorise(n1, a1, n2, a2)
            cat_fn[cat] = cat_fn.get(cat, 0) + 1
            if len(fns) < n_fn:
                fns.append(dict(
                    s1_id=s1id, other_id=oid, country=country,
                    s1_name=n1, s1_addr=a1,
                    other_name=n2, other_addr=a2,
                    category=cat, kind="FN",
                ))

    return fps, fns, cat_fp, cat_fn


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _bar(d, total):
    """Print a simple text bar chart."""
    for k, v in sorted(d.items(), key=lambda x: -x[1]):
        pct = v / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        print(f"    {k:<22} {v:>5}  ({pct:5.1f}%)  {bar}")


def _truncate(s, n=60):
    return (s[:n] + "…") if len(s) > n else s


def print_report(fps, fns, cat_fp, cat_fn, n_show=20):
    sep = "=" * 70
    total_fp = sum(cat_fp.values())
    total_fn = sum(cat_fn.values())

    print(sep)
    print("ERROR ANALYSIS REPORT")
    print(sep)
    print(f"\nTotal false positives : {total_fp}")
    print(f"Total false negatives : {total_fn}")

    if not _HAS_RAPIDFUZZ:
        print("\n  NOTE: rapidfuzz not installed — similarity uses token Jaccard (less accurate).")

    # --- FP category breakdown ---
    print(f"\n{sep}")
    print("FALSE POSITIVE CATEGORIES")
    print(sep)
    _bar(cat_fp, total_fp)

    # --- FN category breakdown ---
    print(f"\n{sep}")
    print("FALSE NEGATIVE CATEGORIES")
    print(sep)
    _bar(cat_fn, total_fn)

    # --- FP examples ---
    print(f"\n{sep}")
    print(f"FALSE POSITIVE EXAMPLES (first {min(n_show, len(fps))})")
    print(sep)
    for i, e in enumerate(fps[:n_show], 1):
        print(f"\n  FP #{i}  [{e['country']}]  cat={e['category']}")
        print(f"    S1  {e['s1_id']:<14}  name: {_truncate(e['s1_name'])}")
        print(f"                         addr: {_truncate(e['s1_addr'])}")
        print(f"    OTH {e['other_id']:<14}  name: {_truncate(e['other_name'])}")
        print(f"                         addr: {_truncate(e['other_addr'])}")

    # --- FN examples ---
    print(f"\n{sep}")
    print(f"FALSE NEGATIVE EXAMPLES (first {min(n_show, len(fns))})")
    print(sep)
    for i, e in enumerate(fns[:n_show], 1):
        print(f"\n  FN #{i}  [{e['country']}]  cat={e['category']}")
        print(f"    S1  {e['s1_id']:<14}  name: {_truncate(e['s1_name'])}")
        print(f"                         addr: {_truncate(e['s1_addr'])}")
        print(f"    OTH {e['other_id']:<14}  name: {_truncate(e['other_name'])}")
        print(f"                         addr: {_truncate(e['other_addr'])}")

    # --- actionable summary ---
    print(f"\n{sep}")
    print("ACTIONABLE SUMMARY FOR TRACK A / B")
    print(sep)
    dom_fp = max(cat_fp, key=cat_fp.get) if cat_fp else "N/A"
    dom_fn = max(cat_fn, key=cat_fn.get) if cat_fn else "N/A"
    print(f"  Dominant FP pattern : {dom_fp}  → "
          + _advice_fp(dom_fp))
    print(f"  Dominant FN pattern : {dom_fn}  → "
          + _advice_fn(dom_fn))
    print(sep)


def _advice_fp(cat):
    return {
        "name_collision":    "raise threshold or add address weight to features",
        "address_collision": "raise threshold or add name weight to features",
        "near_dup":          "model confidence issue — raise thr/margin or add more discriminating features",
        "script_mismatch":   "check norm.py translit — FP on cross-script pairs suggests bad normalisation",
        "singleton_fp":      "model is too aggressive on singletons — raise threshold",
        "other":             "inspect examples manually",
    }.get(cat, "inspect examples manually")


def _advice_fn(cat):
    return {
        "name_collision":    "lower threshold slightly or add recall-boosting key types (Track A)",
        "address_collision": "add address-based blocking keys (Track A: PIN/ZIP key)",
        "near_dup":          "blocking recall issue — check if pair is missing from candidates (Track A)",
        "script_mismatch":   "check norm.py translit — blocked pair missed due to key mismatch across scripts",
        "other":             "inspect examples manually",
    }.get(cat, "inspect examples manually")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Categorised FP/FN error analysis for entity resolution predictions."
    )
    ap.add_argument("--gt",   required=True, help="train_ground_truth.tsv")
    ap.add_argument("--pred", required=True, help="matching_results.tsv")
    ap.add_argument("--src1", required=True, help="train_source1.tsv")
    ap.add_argument("--src2", required=True, help="train_source2.tsv")
    ap.add_argument("--src3", required=True, help="train_source3.tsv")
    ap.add_argument("--n",    type=int, default=50,
                    help="max FP/FN examples to collect and show (default 50)")
    ap.add_argument("--show", type=int, default=20,
                    help="how many examples to print per category (default 20)")
    ap.add_argument("--val-only", action="store_true",
                    help="restrict to the 30%% hash-based val split (same as train.py)")
    a = ap.parse_args()

    for p, name in [(a.gt, "--gt"), (a.pred, "--pred"),
                    (a.src1, "--src1"), (a.src2, "--src2"), (a.src3, "--src3")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} path not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Loading source records …")
    recs = _load_records(a.src1, a.src2, a.src3)
    print(f"  {len(recs)} records loaded")

    gt_df = _read(a.gt)
    truth = {row.source1_entity_id: _parse_ids(row.matched_entity_ids)
             for _, row in gt_df.iterrows()}

    pred_df = _read(a.pred)
    pred = {row.source1_entity_id: _parse_ids(row.matched_entity_ids)
            for _, row in pred_df.iterrows()}

    if a.val_only:
        import zlib
        val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
        truth = {k: v for k, v in truth.items() if k in val_ids}
        pred  = {k: v for k, v in pred.items()  if k in val_ids}
        print(f"--val-only: restricted to {len(truth)} val entities")

    print(f"GT entities: {len(truth)}  |  Pred entities: {len(pred)}\n")

    fps, fns, cat_fp, cat_fn = collect_errors(truth, pred, recs,
                                               n_fp=a.n, n_fn=a.n)
    print_report(fps, fns, cat_fp, cat_fn, n_show=a.show)


if __name__ == "__main__":
    main()
