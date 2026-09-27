#!/usr/bin/env python3
"""
Sweep blocking parameters to measure recall/cost trade-off.

Loads data once, builds Side objects once, then runs candidates() with
different parameter combinations to measure:
- blocking recall
- total candidates generated
- avg candidates per S1 entity
- zero-candidate rate
- wall-clock time

Does not modify blocking.py, norm.py, or blocking_audit.py - only imports and calls them.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# Import required functions - do not modify these modules
from blocking import Side, candidates
from norm import norm_name, norm_addr
from io_utils import read_tsv


def load_ground_truth(gt_path):
    """Load ground truth and return dict of S1 ID -> set of matching S2/S3 IDs."""
    gt_df = read_tsv(gt_path)
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched_ids = row['matched_entity_ids'].strip()
        if matched_ids:
            matched_set = set(matched_ids.split(','))
        else:
            matched_set = set()
        gt_map[s1_id] = matched_set
    return gt_map


def main():
    parser = argparse.ArgumentParser(description='Sweep blocking parameters')
    parser.add_argument('--data', required=True, help='Directory containing train_source*.tsv and train_ground_truth.tsv')
    args = parser.parse_args()

    # Define file paths
    s1_path = os.path.join(args.data, 'train_source1.tsv')
    s2_path = os.path.join(args.data, 'train_source2.tsv')
    s3_path = os.path.join(args.data, 'train_source3.tsv')
    gt_path = os.path.join(args.data, 'train_ground_truth.tsv')

    # Verify files exist
    for path, name in [(s1_path, 'train_source1.tsv'), (s2_path, 'train_source2.tsv'),
                       (s3_path, 'train_source3.tsv'), (gt_path, 'train_ground_truth.tsv')]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    print("Loading data...")
    load_start = time.time()

    # Load all data once
    s1_df = read_tsv(s1_path)
    s2_df = read_tsv(s2_path)
    s3_df = read_tsv(s3_path)
    gt_map = load_ground_truth(gt_path)

    load_time = time.time() - load_start
    print(f"Data loaded in {load_time:.2f}s")
    print(f"  S1 records: {len(s1_df):,}")
    print(f"  S2 records: {len(s2_df):,}")
    print(f"  S3 records: {len(s3_df):,}")
    print(f"  Ground truth pairs: {sum(len(v) for v in gt_map.values()):,}")

    # Build a fast lookup set of true (s1_id, oth_id) pairs, once, for vectorized recall checks
    true_pairs = set()
    gt_pairs_total = 0
    for s1_id, matched_set in gt_map.items():
        gt_pairs_total += len(matched_set)
        for m_id in matched_set:
            true_pairs.add(s1_id + "|" + m_id)

    # Build Side objects once (key generation doesn't depend on sweep parameters)
    print("\nBuilding Side objects...")
    build_start = time.time()

    s1_side = Side(s1_df)
    s2_side = Side(s2_df)
    s3_side = Side(s3_df)
    oth_side = Side.concat([s2_side, s3_side])  # combined S2+S3

    build_time = time.time() - build_start
    print(f"Side objects built in {build_time:.2f}s")
    print(f"  S1 side: {len(s1_side):,} records")
    print(f"  Oth side (S2+S3): {len(oth_side):,} records")

    # Use numpy arrays instead of dicts for fast vectorized index -> id lookup
    s1_ids_arr = np.array(s1_side.ids, dtype=object)
    oth_ids_arr = np.array(oth_side.ids, dtype=object)

    # Parameter sweep
    max_block_values = [30, 60, 100]
    topk_values = [30, 50, 75]
    max_s1_block = 200  # fixed

    results = []

    total_combinations = len(max_block_values) * len(topk_values)
    combo_count = 0

    print(f"\nStarting parameter sweep ({total_combinations} combinations)...")
    print("-" * 80)

    for max_block in max_block_values:
        for topk in topk_values:
            combo_count += 1
            print(f"[{combo_count}/{total_combinations}] Testing max_block={max_block}, topk={topk}, max_s1_block={max_s1_block}...", end=" ", flush=True)

            # Time the candidates() call
            start_time = time.time()
            cand_df = candidates(s1_side, oth_side,
                                 max_block=max_block,
                                 max_s1_block=max_s1_block,
                                 topk=topk)
            end_time = time.time()

            elapsed = end_time - start_time

            # Compute metrics
            total_candidates = len(cand_df)
            avg_candidates_per_s1 = total_candidates / len(s1_side) if len(s1_side) > 0 else 0

            # Zero candidate rate: fraction of S1 entities with 0 candidates
            if len(cand_df) > 0:
                s1_entities_with_candidates = set(cand_df['r1'].unique())
                zero_candidate_count = len(s1_side) - len(s1_entities_with_candidates)
            else:
                zero_candidate_count = len(s1_side)
            zero_candidate_rate = zero_candidate_count / len(s1_side) if len(s1_side) > 0 else 0

            # Blocking recall: recovered ground-truth pairs / total ground-truth pairs
            gt_pairs = gt_pairs_total

            if len(cand_df) > 0 and gt_pairs > 0:
                s1_ids_for_cand = s1_ids_arr[cand_df['r1'].values]
                oth_ids_for_cand = oth_ids_arr[cand_df['ro'].values]
                pair_strs = np.char.add(np.char.add(s1_ids_for_cand.astype(str), "|"),
                                         oth_ids_for_cand.astype(str))
                recovered_pairs = int(np.isin(pair_strs, list(true_pairs)).sum())
            else:
                recovered_pairs = 0

            recall = recovered_pairs / gt_pairs if gt_pairs > 0 else 0.0

            # Store result
            results.append({
                'max_block': max_block,
                'topk': topk,
                'max_s1_block': max_s1_block,
                'recall': recall,
                'total_candidates': total_candidates,
                'avg_candidates_per_s1': avg_candidates_per_s1,
                'zero_candidate_rate': zero_candidate_rate,
                'time_seconds': elapsed
            })

            # Print progress
            print(f"recall={recall:.4f}, total_candidates={total_candidates:,}, "
                  f"avg_candidates={avg_candidates_per_s1:.2f}, zero_rate={zero_candidate_rate:.2%}, "
                  f"time={elapsed:.1f}s")

    # Sort results by recall descending
    results.sort(key=lambda x: x['recall'], reverse=True)

    # Output summary TSV
    output_path = os.path.join(args.data, 'blocking_sweep_results.tsv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, sep='\t', index=False)

    print("-" * 80)
    print(f"Sweep complete! Results saved to: {output_path}")
    print(f"\nTop 3 combinations by recall:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. max_block={result['max_block']}, topk={result['topk']}: "
              f"recall={result['recall']:.4f}, "
              f"avg_candidates={result['avg_candidates_per_s1']:.2f}, "
              f"time={result['time_seconds']:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())