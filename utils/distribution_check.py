"""Distribution check — compare match-count distribution of our output vs. train ground truth.

Run from the student_resource/ folder before every leaderboard submission:

    python utils/distribution_check.py --pred output/matching_results.tsv

Or point at a specific output folder:

    python utils/distribution_check.py \\
        --pred output_v3/matching_results.tsv \\
        --gt   dataset/train/train_ground_truth.tsv

What it prints
--------------
train truth :  empty=XX%  mean=X.XX  median=X  p90=X
our output  :  empty=XX%  mean=X.XX  median=X  p90=X

empty%  = fraction of S1 entities with zero predicted matches (singletons)
mean    = average match-list length across all S1 entities
median  = median match-list length
p90     = 90th-percentile match-list length (how large are the busiest clusters)

How to read the output
----------------------
- empty% in output >> empty% in train truth  →  model is over-conservative (thr too high / margin too large).
  Lower the threshold or margin and re-run.
- empty% in output << empty% in train truth  →  model is merging too aggressively (thr too low).
  Raise the threshold.
- A big gap in mean or p90 while empty% matches  →  a few clusters are exploding; check the
  per-country breakdown or look at blocking candidates for high-degree S1 entities.
- Small differences are expected: the test set distribution won't be identical to train.
  Worry when the gap is >5 pp on empty% or >0.5 on mean.
"""
import argparse
import sys
import os

# ---------------------------------------------------------------------------
# We want this script to work with stdlib only if pandas is absent, but pandas
# is in requirements.txt so it will always be present in the contest env.
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  pip install pandas", file=sys.stderr)
    sys.exit(1)


def stats(series: "pd.Series") -> str:
    """Return a compact stats string for a matched_entity_ids column."""
    n = series.str.split(",").map(lambda x: len([i for i in x if i]))
    return (
        f"empty={(n == 0).mean():.2%}  "
        f"mean={n.mean():.2f}  "
        f"median={int(n.median())}  "
        f"p90={n.quantile(0.9):.1f}"
    )


def load_tsv(path: str, label: str) -> "pd.DataFrame":
    if not os.path.exists(path):
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser(
        description="Compare match-count distribution of our output vs. train ground truth."
    )
    ap.add_argument(
        "--pred",
        required=True,
        help="path to our matching_results.tsv (the file we submit / plan to submit)",
    )
    ap.add_argument(
        "--gt",
        default="dataset/train/train_ground_truth.tsv",
        help="path to train ground truth (default: dataset/train/train_ground_truth.tsv)",
    )
    a = ap.parse_args()

    gt  = load_tsv(a.gt,   "ground truth")
    out = load_tsv(a.pred, "prediction")

    # Validate expected columns exist
    for col in ("matched_entity_ids",):
        if col not in gt.columns:
            print(f"ERROR: ground truth file missing column '{col}'", file=sys.stderr)
            sys.exit(1)
        if col not in out.columns:
            print(f"ERROR: prediction file missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    gt_stats  = stats(gt["matched_entity_ids"])
    out_stats = stats(out["matched_entity_ids"])

    print(f"train truth : {gt_stats}")
    print(f"our output  : {out_stats}")

    # ------------------------------------------------------------------ delta warnings
    def _empty_pct(series):
        n = series.str.split(",").map(lambda x: len([i for i in x if i]))
        return (n == 0).mean()

    gt_empty  = _empty_pct(gt["matched_entity_ids"])
    out_empty = _empty_pct(out["matched_entity_ids"])
    delta     = out_empty - gt_empty

    print()
    if delta > 0.05:
        print(
            f"[WARNING] output is {delta:.1%} more empty than train truth "
            f"— model may be over-conservative (threshold too high / margin too large). "
            f"Consider lowering --thr or --margin."
        )
    elif delta < -0.05:
        print(
            f"[WARNING] output is {abs(delta):.1%} less empty than train truth "
            f"— model may be merging too aggressively (threshold too low). "
            f"Consider raising --thr."
        )
    else:
        print(f"[OK] empty% delta = {delta:+.1%} (within ±5 pp of train truth — looks healthy)")

    # ------------------------------------------------------------------ row count sanity
    print()
    print(f"rows in ground truth : {len(gt):,}")
    print(f"rows in our output   : {len(out):,}")
    if len(out) != len(gt):
        # Not necessarily wrong (test set size differs from train), just informational
        print("  (different row counts are expected — test set ≠ train set size)")


if __name__ == "__main__":
    main()
