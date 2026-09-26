"""Validation harness: per-country F0.5 breakdown, TP/FP/FN counts, worst entities, singleton analysis.

Usage (run from student_resource/ root or the src/ directory):
    python code/business_entity_resolution/src/val_harness.py \
        --gt  dataset/train/train_ground_truth.tsv \
        --pred output/matching_results.tsv \
        --s1   dataset/train/train_source1.tsv

Or with sample_dense/:
    python code/business_entity_resolution/src/val_harness.py \
        --gt  sample_dense/train_ground_truth.tsv \
        --pred output/matching_results.tsv \
        --s1   sample_dense/train_source1.tsv

Track D runs this after every full-data run before deciding whether to submit.
"""
import argparse
import sys
import os
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=3, encoding="utf-8", encoding_errors="replace")


def _parse_id_list(cell):
    """comma-separated id cell -> frozenset (empty string -> empty set)"""
    if not cell or not cell.strip():
        return frozenset()
    return frozenset(x.strip() for x in cell.split(",") if x.strip())


def _load_gt(gt_path):
    """Returns dict s1_id -> frozenset(matched_ids)"""
    df = _read_tsv(gt_path)
    return {row.source1_entity_id: _parse_id_list(row.matched_entity_ids)
            for _, row in df.iterrows()}


def _load_pred(pred_path):
    """Returns dict s1_id -> frozenset(predicted_ids)"""
    df = _read_tsv(pred_path)
    return {row.source1_entity_id: _parse_id_list(row.matched_entity_ids)
            for _, row in df.iterrows()}


def _load_s1_country(s1_path):
    """Returns dict s1_id -> country"""
    df = _read_tsv(s1_path)
    return dict(zip(df.entity_id, df.country))


def _f05(tp, fp, fn):
    """F0.5 from aggregate TP/FP/FN counts (not per-entity average — use _f05_entity for that)."""
    if tp == 0:
        return 0.0
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    return 1.25 * pr * rc / (0.25 * pr + rc)


def _f05_entity(pred_set, true_set):
    """Per-entity F0.5 score (returns 1.0 for double-empty, 0.0 for half-empty)."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    pr = tp / len(pred_set)
    rc = tp / len(true_set)
    return 1.25 * pr * rc / (0.25 * pr + rc)


# ---------------------------------------------------------------------------
# core analysis
# ---------------------------------------------------------------------------

def analyse(truth, pred, id2country):
    """
    Returns a dict with:
      - overall: dict(macro_f05, tp, fp, fn, n_entities)
      - per_country: dict country -> dict(macro_f05, tp, fp, fn, n_entities, n_singletons_correct,
                                          n_singletons_wrong)
      - entity_scores: list of (s1_id, f05, true_set, pred_set, country) sorted by f05 asc
      - singleton_correct: count of singletons correctly left empty
      - singleton_wrong: count of singletons incorrectly given a match
    """
    # Group entities by country
    countries = {}
    for s1id, true_set in truth.items():
        c = id2country.get(s1id, "UNKNOWN")
        countries.setdefault(c, []).append(s1id)

    entity_scores = []
    per_country = {}

    total_tp = total_fp = total_fn = 0
    total_sc = 0.0
    total_n = 0
    singleton_correct = 0
    singleton_wrong = 0

    for c, ids in sorted(countries.items()):
        c_tp = c_fp = c_fn = 0
        c_sc = 0.0
        c_sing_correct = 0
        c_sing_wrong = 0

        for s1id in ids:
            true_set = truth[s1id]
            pred_set = pred.get(s1id, frozenset())

            sc = _f05_entity(pred_set, true_set)
            c_sc += sc
            entity_scores.append((s1id, sc, true_set, pred_set, c))

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            c_tp += tp; c_fp += fp; c_fn += fn

            if not true_set:            # singleton
                if not pred_set:
                    c_sing_correct += 1
                else:
                    c_sing_wrong += 1

        n = len(ids)
        per_country[c] = dict(
            macro_f05=c_sc / n if n else 0.0,
            tp=c_tp, fp=c_fp, fn=c_fn,
            n_entities=n,
            n_singletons_correct=c_sing_correct,
            n_singletons_wrong=c_sing_wrong,
        )
        total_tp += c_tp; total_fp += c_fp; total_fn += c_fn
        total_sc += c_sc; total_n += n
        singleton_correct += c_sing_correct
        singleton_wrong += c_sing_wrong

    # Entities present in pred but not in truth (unexpected rows)
    extra_ids = set(pred) - set(truth)

    overall = dict(
        macro_f05=total_sc / total_n if total_n else 0.0,
        tp=total_tp, fp=total_fp, fn=total_fn,
        n_entities=total_n,
        extra_pred_rows=len(extra_ids),
    )

    entity_scores.sort(key=lambda x: x[1])   # ascending by F0.5

    return dict(
        overall=overall,
        per_country=per_country,
        entity_scores=entity_scores,
        singleton_correct=singleton_correct,
        singleton_wrong=singleton_wrong,
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(result, top_n=20):
    sep = "=" * 70

    # --- overall ---
    ov = result["overall"]
    print(sep)
    print("OVERALL RESULTS")
    print(sep)
    print(f"  Macro F0.5          : {ov['macro_f05']:.4f}")
    print(f"  Total S1 entities   : {ov['n_entities']}")
    print(f"  True  positives     : {ov['tp']}")
    print(f"  False positives     : {ov['fp']}")
    print(f"  False negatives     : {ov['fn']}")
    if ov["extra_pred_rows"]:
        print(f"  WARNING extra pred rows not in GT : {ov['extra_pred_rows']}")
    sc = result["singleton_correct"]
    sw = result["singleton_wrong"]
    stot = sc + sw
    print(f"\n  Singletons (no true match): {stot}")
    print(f"    Correctly left empty : {sc}  ({sc/stot*100:.1f}%)" if stot else "    (none)")
    print(f"    Incorrectly matched  : {sw}  ({sw/stot*100:.1f}%)" if stot else "")

    # --- per country ---
    print()
    print(sep)
    print("PER-COUNTRY BREAKDOWN")
    print(sep)
    for c, d in sorted(result["per_country"].items()):
        print(f"\n  [{c}]")
        print(f"    Macro F0.5  : {d['macro_f05']:.4f}")
        print(f"    Entities    : {d['n_entities']}")
        print(f"    TP / FP / FN: {d['tp']} / {d['fp']} / {d['fn']}")
        sc_c = d["n_singletons_correct"]; sw_c = d["n_singletons_wrong"]
        stot_c = sc_c + sw_c
        if stot_c:
            print(f"    Singletons  : {stot_c}  (correct empty: {sc_c}, wrong match: {sw_c})")

    # --- worst entities ---
    print()
    print(sep)
    print(f"TOP {top_n} WORST-PERFORMING S1 ENTITIES")
    print(sep)
    worst = [e for e in result["entity_scores"] if not (not e[2] and not e[3])]  # skip double-empty
    for rank, (s1id, sc, true_set, pred_set, country) in enumerate(worst[:top_n], 1):
        print(f"\n  #{rank:>2}  {s1id}  [{country}]  F0.5={sc:.4f}")
        tp = pred_set & true_set
        fp = pred_set - true_set
        fn = true_set - pred_set
        print(f"       True matches : {sorted(true_set) or '(none — singleton)'}")
        print(f"       Predicted    : {sorted(pred_set) or '(none)'}")
        print(f"       TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")

    print()
    print(sep)
    print("GO / NO-GO SUMMARY")
    print(sep)
    f05 = ov["macro_f05"]
    if f05 >= 0.95:
        verdict = "GO  — F0.5 >= 0.95"
    elif f05 >= 0.90:
        verdict = "GO  — F0.5 >= 0.90 (acceptable)"
    elif f05 >= 0.85:
        verdict = "CAUTION — F0.5 >= 0.85 but below 0.90"
    else:
        verdict = "NO-GO — F0.5 below 0.85"
    print(f"  {verdict}")
    if result["singleton_wrong"] > 0:
        print(f"  WARNING: {result['singleton_wrong']} singletons incorrectly matched (precision hit)")
    print(sep)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Validation harness: per-country F0.5, TP/FP/FN, worst entities, singleton check."
    )
    ap.add_argument("--gt",   required=True, help="path to train_ground_truth.tsv")
    ap.add_argument("--pred", required=True, help="path to matching_results.tsv to evaluate")
    ap.add_argument("--s1",   required=True, help="path to train_source1.tsv (for country lookup)")
    ap.add_argument("--top",  type=int, default=20, help="how many worst entities to show (default 20)")
    ap.add_argument("--val-only", action="store_true",
                    help="restrict to the 30%% hash-based val split used by train.py (crc32 mod 10 < 3)")
    a = ap.parse_args()

    # validate paths
    for p, name in [(a.gt, "--gt"), (a.pred, "--pred"), (a.s1, "--s1")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} path not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading ground truth  : {a.gt}")
    print(f"Loading predictions   : {a.pred}")
    print(f"Loading S1 for country: {a.s1}")

    truth = _load_gt(a.gt)
    pred  = _load_pred(a.pred)
    id2country = _load_s1_country(a.s1)

    if a.val_only:
        import zlib
        val_ids = {k for k in truth if zlib.crc32((k + "v").encode()) % 10 < 3}
        truth = {k: v for k, v in truth.items() if k in val_ids}
        pred  = {k: v for k, v in pred.items()  if k in val_ids}
        print(f"--val-only: restricted to {len(truth)} entities (30% hash-based val split)")

    print(f"GT entities: {len(truth)}  |  Pred entities: {len(pred)}\n")

    result = analyse(truth, pred, id2country)
    print_report(result, top_n=a.top)

    return result


if __name__ == "__main__":
    main()
