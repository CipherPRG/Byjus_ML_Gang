"""Fast, same-numbers rewrite of adithya-sundar's blocking_audit.py.

The original never finished (killed after 30+ min of CPU-bound work) because of two
hot spots that are effectively quadratic in dataset size, not linear:

  1. compute_record_keys() used df.iterrows() - very slow per-row Series construction.
     Fixed: iterate over .values arrays (same technique blocking.py's Side.__init__
     already uses) instead of iterrows().

  2. For EVERY S1 record that has a ground-truth match, the original did a full
     iterrows() scan over ALL S2 (332K) and ALL S3 (354K) records just to find
     whether each match id was present - O(s1_with_gt x total_other_records).
     Fixed: build an id->row-index dict once per country, so each match lookup is O(1).

  3. get_pair_rank_for_s1() (called once per MISSED pair) scanned every other-side
     record to find which ones share a key with s1 - O(missed_pairs x
     total_other_records). Fixed: build an inverted key->indices index once per
     country, so only records that actually share a key are considered. The
     max_block/max_s1_block filtered key sets were also being recomputed from
     scratch on every single missed-pair call; now computed once per country.

Every filtering rule, failure-mode classification (A/B/C/D/E), and report column is
copied over unchanged from the original - only the algorithmic complexity changed, not
the definitions or the numbers they produce.
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import os
from collections import Counter, defaultdict

import pandas as pd

from blocking import Side, candidates, keys_for, WEIGHT
from norm import norm_name, norm_addr
from io_utils import read_tsv, read_country


def load_ground_truth(gt_path):
    gt_df = read_tsv(gt_path)
    gt_map = {}
    for s1_id, matched_ids in zip(gt_df.source1_entity_id, gt_df.matched_entity_ids):
        matched_ids = matched_ids.strip()
        gt_map[s1_id] = set(matched_ids.split(",")) if matched_ids else set()
    return gt_map


def compute_record_keys_fast(df):
    """Same output shape as the original (dict idx -> [(key, weight)], Counter of key
    frequency), but iterates over .values instead of df.iterrows()."""
    record_keys = defaultdict(list)
    all_keys = []
    names = df["business_name"].values
    addrs = df["business_address"].values
    countries = df["country"].values
    for idx in range(len(df)):
        _, core = norm_name(names[idx])
        _, at = norm_addr(addrs[idx])
        for k in keys_for(countries[idx], core, at):
            record_keys[idx].append((k, WEIGHT[k[0]]))
            all_keys.append(k)
    return record_keys, Counter(all_keys)


def filtered_key_sets(s1_key_freq, other_key_freq, max_block, max_s1_block):
    """The max_block / max_s1_block filtering candidates() does - same for every pair
    in a country, so compute ONCE per country instead of once per missed pair."""
    other_maxblock = {k for k, f in other_key_freq.items() if f <= max_block}
    s1_maxs1block = {k for k, f in s1_key_freq.items() if f <= max_s1_block}
    s1_final = s1_maxs1block & other_maxblock
    return s1_final, other_maxblock


def build_inverted_index(record_keys, allowed_keys):
    """key -> list of record indices whose (filtered) key set contains it."""
    idx_ = defaultdict(list)
    for idx, kw in record_keys.items():
        for k, _ in kw:
            if k in allowed_keys:
                idx_[k].append(idx)
    return idx_


def get_pair_rank_for_s1_fast(s1_idx, other_idx, s1_key_set, filtered_other_keys,
                               other_inverted_index):
    """Same semantics as the original get_pair_rank_for_s1: rank (1-based, best first)
    of `other_idx` among all other-side records sharing >=1 filtered key with s1_idx.
    Uses the inverted index instead of scanning every other-side record."""
    candidate_others = set()
    for k in s1_key_set:
        candidate_others.update(other_inverted_index.get(k, ()))

    weights = []
    for idx2 in candidate_others:
        o_keys = dict(filtered_other_keys[idx2])
        shared = s1_key_set & o_keys.keys()
        w = sum(o_keys[k] for k in shared)
        weights.append((idx2, w))
    weights.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx2, _w) in enumerate(weights, start=1):
        if idx2 == other_idx:
            return rank
    return None


def run_blocking_audit(data_dir, max_block=30, max_s1_block=200, topk=30):
    print(f"Loading training data from {data_dir}")
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    for path, name in [(s1_path, "train_source1.tsv"), (s2_path, "train_source2.tsv"),
                        (s3_path, "train_source3.tsv"), (gt_path, "train_ground_truth.tsv")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            return None

    print("Loading ground truth...")
    gt_map = load_ground_truth(gt_path)

    print("Determining countries...")
    countries = read_tsv(s1_path, usecols=["country"])["country"].unique()
    print(f"Found countries: {list(countries)}")

    total_gt_pairs = 0
    total_recovered_pairs = 0
    total_s1_entities = 0
    total_candidates = 0
    s1_with_zero_candidates = 0
    failure_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    failure_details = []

    for country in countries:
        print(f"\nProcessing country: {country}")
        s1_df = read_country(s1_path, country)
        s2_df = read_country(s2_path, country) if os.path.exists(s2_path) else pd.DataFrame()
        s3_df = read_country(s3_path, country) if os.path.exists(s3_path) else pd.DataFrame()
        other_parts = [d for d in (s2_df, s3_df) if len(d) > 0]
        other_df = pd.concat(other_parts, ignore_index=True) if other_parts else pd.DataFrame(columns=s1_df.columns)

        if len(s1_df) == 0:
            print(f"  No S1 records for {country}")
            continue
        if len(other_df) == 0:
            print(f"  No S2/S3 records for {country}")
            s1_with_zero_candidates += len(s1_df)
            total_s1_entities += len(s1_df)
            continue

        print("  Computing blocking keys for detailed analysis...")
        s1_record_keys, s1_key_freq = compute_record_keys_fast(s1_df)
        other_record_keys, other_key_freq = compute_record_keys_fast(other_df)

        print(f"  Creating Side objects (S1: {len(s1_df)}, S2+S3: {len(other_df)})")
        s1_side = Side(s1_df)
        other_side = Side(other_df)
        total_s1_entities += len(s1_side)

        print(f"  Running blocking (max_block={max_block}, max_s1_block={max_s1_block}, topk={topk})")
        cand_df = candidates(s1_side, other_side, max_block=max_block, max_s1_block=max_s1_block, topk=topk)
        total_candidates += len(cand_df)

        cand_pairs = set(zip(cand_df["r1"].astype(int), cand_df["ro"].astype(int))) if len(cand_df) else set()
        s1_entities_with_cands = set(cand_df["r1"].unique()) if len(cand_df) else set()
        zero_cand_count = len(s1_side) - len(s1_entities_with_cands)
        s1_with_zero_candidates += zero_cand_count
        print(f"  S1 entities with candidates: {len(s1_entities_with_cands)}/{len(s1_side)}")
        print(f"  S1 entities with zero candidates: {zero_cand_count}")
        print(f"  Total candidate pairs: {len(cand_df)}")

        # precompute ONCE per country (was being recomputed per missed pair before)
        s1_final_key_set, other_maxblock_key_set = filtered_key_sets(
            s1_key_freq, other_key_freq, max_block, max_s1_block)
        filtered_other_record_keys = {
            idx: [(k, w) for k, w in kw if k in other_maxblock_key_set]
            for idx, kw in other_record_keys.items()
        }
        other_inverted_index = build_inverted_index(other_record_keys, other_maxblock_key_set)

        def analyze_missed_pair(s1_idx, other_idx):
            s1_keys = [k for k, _ in s1_record_keys[s1_idx]]
            other_keys = [k for k, _ in other_record_keys[other_idx]]
            s1_key_set = set(s1_keys)
            other_key_set = set(other_keys)
            shared_keys = s1_key_set & other_key_set

            if not shared_keys:
                return "A", {}

            survived = shared_keys & s1_final_key_set
            if not survived:
                passing_maxblock = shared_keys & other_maxblock_key_set
                if passing_maxblock:
                    return "C", {}
                return "B", {}

            filtered_s1_key_set = {k for k, w in s1_record_keys[s1_idx] if k in s1_final_key_set}
            rank = get_pair_rank_for_s1_fast(
                s1_idx, other_idx, filtered_s1_key_set,
                filtered_other_record_keys, other_inverted_index)
            if rank is None:
                return "E", {"reason": "rank computation failed"}
            if rank > topk:
                return "D", {"rank": rank}
            return "E", {"reason": f"rank {rank} <= topk but missing from candidates"}

        s2_id_to_idx = {v: i for i, v in enumerate(s2_df["entity_id"].values)} if len(s2_df) else {}
        s3_id_to_idx = {v: i for i, v in enumerate(s3_df["entity_id"].values)} if len(s3_df) else {}
        n_s2 = len(s2_df)

        gt_pairs_in_country = 0
        recovered_pairs_in_country = 0
        s1_ids = s1_df["entity_id"].values

        for s1_idx, s1_id in enumerate(s1_ids):
            true_matches = gt_map.get(s1_id)
            if not true_matches:
                continue
            gt_pairs_in_country += len(true_matches)
            recovered = 0
            for m_id in true_matches:
                if m_id in s2_id_to_idx:
                    other_idx = s2_id_to_idx[m_id]
                    src = "S2"
                elif m_id in s3_id_to_idx:
                    other_idx = n_s2 + s3_id_to_idx[m_id]
                    src = "S3"
                else:
                    continue  # not present in this country's other-side data at all
                if (s1_idx, other_idx) in cand_pairs:
                    recovered += 1
                else:
                    failure_type, diag = analyze_missed_pair(s1_idx, other_idx)
                    failure_counts[failure_type] += 1
                    diag.update({"s1_id": s1_id, "matched_id": m_id, "matched_source": src,
                                  "failure_type": failure_type})
                    failure_details.append(diag)
            recovered_pairs_in_country += recovered

        total_gt_pairs += gt_pairs_in_country
        total_recovered_pairs += recovered_pairs_in_country
        print(f"  Ground truth pairs in {country}: {gt_pairs_in_country}")
        print(f"  Recovered pairs in {country}: {recovered_pairs_in_country}")
        if gt_pairs_in_country > 0:
            print(f"  Blocking recall for {country}: {recovered_pairs_in_country / gt_pairs_in_country:.4f}")

    print("\n" + "=" * 60)
    print("BLOCKING AUDIT RESULTS")
    print("=" * 60)
    print(f"Total S1 entities: {total_s1_entities:,}")
    print(f"Total ground truth pairs: {total_gt_pairs:,}")
    print(f"Total pairs recovered by blocking: {total_recovered_pairs:,}")
    blocking_recall = total_recovered_pairs / total_gt_pairs if total_gt_pairs else 0.0
    print(f"Overall blocking recall: {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")
    avg_candidates_per_s1 = total_candidates / total_s1_entities if total_s1_entities else 0.0
    print(f"Total candidate pairs generated: {total_candidates:,}")
    print(f"Average candidates per S1 entity: {avg_candidates_per_s1:.2f}")
    zero_candidate_rate = s1_with_zero_candidates / total_s1_entities if total_s1_entities else 0.0
    print(f"Percentage of S1 entities with zero candidates: {zero_candidate_rate:.2%}")

    print("\nFailure mode breakdown (of missed pairs):")
    missed_pairs = total_gt_pairs - total_recovered_pairs
    if missed_pairs > 0:
        for mode in ["A", "B", "C", "D"]:
            count = failure_counts[mode]
            print(f"  {mode}: {count:,} ({count / missed_pairs * 100:.1f}% of missed pairs)")
        if failure_counts["E"] > 0:
            print(f"  E: {failure_counts['E']:,} ({failure_counts['E'] / missed_pairs * 100:.1f}% of missed pairs) - indicates implementation issue")
    else:
        print("  No missed pairs to analyze")

    report_path = os.path.join(data_dir, "blocking_audit_report_fast.tsv")
    if failure_details:
        report_df = pd.DataFrame(failure_details)
        cols = ["s1_id", "matched_id", "matched_source", "failure_type", "reason"]
        extra_cols = [c for c in report_df.columns if c not in cols]
        report_df = report_df[[c for c in cols if c in report_df.columns] + extra_cols]
        report_df.to_csv(report_path, sep="\t", index=False)
        print(f"\nDetailed failure report saved to: {report_path}")
        print(f"  Contains {len(failure_details):,} missed pairs")

    summary_path = os.path.join(data_dir, "blocking_audit_summary_fast.tsv")
    summary_data = [
        {"metric": "total_s1_entities", "value": total_s1_entities},
        {"metric": "total_gt_pairs", "value": total_gt_pairs},
        {"metric": "total_recovered_pairs", "value": total_recovered_pairs},
        {"metric": "blocking_recall", "value": blocking_recall},
        {"metric": "total_candidates", "value": total_candidates},
        {"metric": "avg_candidates_per_s1", "value": avg_candidates_per_s1},
        {"metric": "zero_candidate_rate", "value": zero_candidate_rate},
    ]
    for mode in ["A", "B", "C", "D", "E"]:
        summary_data.append({"metric": f"failure_{mode}_count", "value": failure_counts[mode]})
    pd.DataFrame(summary_data).to_csv(summary_path, sep="\t", index=False)
    print(f"Summary metrics saved to: {summary_path}")

    return {"blocking_recall": blocking_recall, "failure_counts": failure_counts}


def main():
    ap = argparse.ArgumentParser(description="Fast rewrite of the blocking audit")
    ap.add_argument("--data", type=str, default="sample_dense")
    ap.add_argument("--max_block", type=int, default=30)
    ap.add_argument("--max_s1_block", type=int, default=200)
    ap.add_argument("--topk", type=int, default=30)
    a = ap.parse_args()
    print("Starting blocking audit (fast rewrite)...")
    results = run_blocking_audit(a.data, a.max_block, a.max_s1_block, a.topk)
    print("\nAudit completed successfully!" if results is not None else "\nAudit failed - see error messages above")


if __name__ == "__main__":
    main()
