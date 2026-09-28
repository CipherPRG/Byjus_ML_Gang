"""Blocking audit script to measure baseline performance and identify failure modes.
Loads training data, runs existing blocking unchanged, and analyzes why true matches are missed.
"""
import os as _os, sys as _sys  # experiments/ scripts import the pipeline modules from ../src
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

import pandas as pd
from blocking import Side, candidates, keys_for, WEIGHT
from norm import norm_name, norm_addr
from io_utils import read_tsv, read_country
import os
from collections import Counter, defaultdict


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


def compute_record_keys(df):
    """Compute blocking keys for each record in a dataframe.

    Returns:
        dict: {row_index: list of (key_string, weight)} for each record
        Counter: frequency of each key across all records
    """
    record_keys = defaultdict(list)
    all_keys = []

    for idx, (_, row) in enumerate(df.iterrows()):
        nm, ad, c = row['business_name'], row['business_address'], row['country']
        nc, core = norm_name(nm)
        ac, at = norm_addr(ad)

        # Compute keys for this record
        keys = keys_for(c, core, at)
        for k in keys:
            record_keys[idx].append((k, WEIGHT[k[0]]))
            all_keys.append(k)

    key_freq = Counter(all_keys)
    return record_keys, key_freq


def apply_candidates_filtering(s1_record_keys, s1_key_freq, s2_record_keys, s2_key_freq,
                              max_block, max_s1_block):
    """Apply the exact same filtering as candidates() function.

    Returns:
        tuple: (filtered_s1_keys, filtered_s2_keys) where each is a list of (key, weight) tuples
               that survived the filtering process
    """
    # Filter s2 keys by max_block (line 95 in candidates)
    s2_filtered_by_maxblock = {k: freq for k, freq in s2_key_freq.items() if freq <= max_block}
    s2_key_set = set(s2_filtered_by_maxblock.keys())

    # Filter s1 keys by max_s1_block (line 97 in candidates)
    s1_filtered_by_maxs1block = {k: freq for k, freq in s1_key_freq.items() if freq <= max_s1_block}
    s1_key_set = set(s1_filtered_by_maxs1block.keys())

    # Further filter s1 keys to keep only those that exist in filtered s2 (line 98 in candidates)
    s1_final_key_set = s1_key_set.intersection(s2_key_set)

    # Now return the actual key-weight pairs that survived filtering
    filtered_s1 = []
    for idx, key_weight_list in s1_record_keys.items():
        for key, weight in key_weight_list:
            if key in s1_final_key_set:
                filtered_s1.append((key, weight))

    filtered_s2 = []
    for idx, key_weight_list in s2_record_keys.items():
        for key, weight in key_weight_list:
            if key in s2_key_set:  # Already filtered by max_block above
                filtered_s2.append((key, weight))

    return filtered_s1, filtered_s2, s1_final_key_set, s2_key_set


def compute_pair_weight(s1_idx, s2_idx, s1_record_keys, s2_record_keys):
    """Compute the accumulated weight for a specific pair as done in candidates().

    Returns:
        float: sum of weights of shared keys (weight from other side only)
    """
    # Get keys for each record with weights
    s1_keys_weights = dict(s1_record_keys[s1_idx])
    s2_keys_weights = dict(s2_record_keys[s2_idx])

    # Find shared keys
    s1_key_set = set(s1_keys_weights.keys())
    s2_key_set = set(s2_keys_weights.keys())
    shared_keys = s1_key_set.intersection(s2_key_set)

    # Sum weights from the other side only (as in candidates line 105-106)
    total_weight = 0
    for key in shared_keys:
        total_weight += s2_keys_weights.get(key, 0)

    return total_weight


def get_pair_rank_for_s1(s1_idx, s2_idx, s1_record_keys_filtered, s2_record_keys_filtered):
    """Get the exact rank of a pair for a given S1 entity as determined by candidates().

    Returns:
        int: 1-based rank (1 = best), or None if pair not in candidates
    """
    # Get keys for each record after filtering
    s1_keys_weights = dict(s1_record_keys_filtered[s1_idx])
    s2_keys_weights = dict(s2_record_keys_filtered[s2_idx])

    # Find shared keys
    s1_key_set = set(s1_keys_weights.keys())
    s2_key_set = set(s2_keys_weights.keys())
    shared_keys = s1_key_set.intersection(s2_key_set)

    if not shared_keys:
        return None  # No shared keys, wouldn't be in candidates at all

    # Compute weight for this pair (from other side only)
    s2_weights = dict(s2_record_keys_filtered[s2_idx])
    pair_weight = 0
    for key in shared_keys:
        pair_weight += s2_weights.get(key, 0)

    # To get the rank, we need to compute weights for ALL possible pairs for this s1_idx
    # and see where this pair falls in the sorted list

    # Get all s2 indices that have at least one shared key with this s1_idx
    candidate_s2_indices = []
    for idx2, key_weight_list in s2_record_keys_filtered.items():
        s2_keys = set(kw[0] for kw in key_weight_list)
        if s1_key_set.intersection(s2_keys):
            candidate_s2_indices.append(idx2)

    # Compute weights for all candidate pairs
    pair_weights = []
    for idx2 in candidate_s2_indices:
        s2_keys_weights = dict(s2_record_keys_filtered[idx2])
        s2_key_set = set(s2_keys_weights.keys())
        shared_keys = s1_key_set.intersection(s2_key_set)

        weight = 0
        for key in shared_keys:
            weight += s2_keys_weights.get(key, 0)

        pair_weights.append((idx2, weight))

    # Sort by weight descending (as in candidates line 107)
    # For tie-breaking, candidates() uses sort_values(["r1", "w"], ascending=[True, False])
    # Since we're grouping by r1 first, within each r1 group we sort by w descending
    # For ties in weight, pandas sort is stable, but we don't need to replicate exact tie behavior
    # as long as we sort by weight descending
    pair_weights.sort(key=lambda x: x[1], reverse=True)

    # Find the rank of our specific pair
    for rank, (idx2, weight) in enumerate(pair_weights):
        if idx2 == s2_idx:
            return rank + 1  # 1-based rank

    return None  # Should not happen if we got here


def analyze_missed_pair(s1_idx, s2_idx, s1_record_keys, s2_record_keys,
                       s1_key_freq, s2_key_freq, max_block, max_s1_block, topk):
    """Analyze why a specific S1-S2 pair was missed by blocking.

    Returns:
        str: Failure category ('A', 'B', 'C', 'D', or 'E')
        dict: Diagnostic information
    """
    # Get keys for each record
    s1_keys = [k for k, _ in s1_record_keys[s1_idx]]
    s2_keys = [k for k, _ in s2_record_keys[s2_idx]]

    # Find shared keys
    s1_key_set = set(s1_keys)
    s2_key_set = set(s2_keys)
    shared_keys = s1_key_set.intersection(s2_key_set)

    if len(shared_keys) == 0:
        return 'A', {
            'failure_type': 'A',
            'reason': 'No shared blocking key',
            's1_keys': s1_keys[:5],  # First 5 keys for brevity
            's2_keys': s2_keys[:5],
            'shared_keys': [],
            'num_shared_keys': 0
        }

    # Apply the EXACT same filtering as candidates() function
    # We need to simulate what happens inside candidates() for filtering

    # Step 1: Apply max_block filtering to other side (k2)
    s2_filtered_by_maxblock = {k: freq for k, freq in s2_key_freq.items() if freq <= max_block}
    s2_maxblock_key_set = set(s2_filtered_by_maxblock.keys())

    # Step 2: Apply max_s1_block filtering to s1 side (k1)
    s1_filtered_by_maxs1block = {k: freq for k, freq in s1_key_freq.items() if freq <= max_s1_block}
    s1_maxs1block_key_set = set(s1_filtered_by_maxs1block.keys())

    # Step 3: Further filter k1 to keep only keys that exist in filtered k2
    s1_final_key_set = s1_maxs1block_key_set.intersection(s2_maxblock_key_set)

    # Check if any shared key survived both filters
    shared_keys_survived_filtering = shared_keys.intersection(s1_final_key_set)

    if len(shared_keys_survived_filtering) == 0:
        # All shared keys were filtered out by either max_block or max_s1_block
        # Determine which filter caused the elimination

        # Check if any shared key would survive max_block but not max_s1_block
        shared_keys_passing_maxblock = shared_keys.intersection(s2_maxblock_key_set)
        if len(shared_keys_passing_maxblock) > 0:
            # Some keys pass max_block but fail max_s1_block
            filtered_keys = [k for k in shared_keys_passing_maxblock if k not in s1_maxs1block_key_set]
            return 'C', {
                'failure_type': 'C',
                'reason': f'All shared keys filtered by max_s1_block (>{max_s1_block}) after passing max_block',
                'filtered_keys': filtered_keys[:5],
                'key_frequencies': {k: s1_key_freq[k] for k in filtered_keys[:5]},
                'shared_keys_after_maxblock': list(shared_keys_passing_maxblock)[:5],
                'num_shared_keys_after_maxblock': len(shared_keys_passing_maxblock)
            }
        else:
            # All shared keys fail max_block (other side filtering)
            filtered_keys = [k for k in shared_keys if k not in s2_maxblock_key_set]
            return 'B', {
                'failure_type': 'B',
                'reason': f'All shared keys filtered by max_block (>{max_block})',
                'filtered_keys': filtered_keys[:5],
                'key_frequencies': {k: s2_key_freq[k] for k in filtered_keys[:5]},
                'shared_keys': list(shared_keys)[:5],
                'num_shared_keys': len(shared_keys)
            }

    # If we reach here, at least one shared key survived both filters
    # Now we need to check if the pair would be in top-k

    # Create filtered record keys for weight computation
    filtered_s1_record_keys = defaultdict(list)
    filtered_s2_record_keys = defaultdict(list)

    for idx, key_weight_list in s1_record_keys.items():
        for key, weight in key_weight_list:
            if key in s1_final_key_set:
                filtered_s1_record_keys[idx].append((key, weight))

    for idx, key_weight_list in s2_record_keys.items():
        for key, weight in key_weight_list:
            if key in s2_maxblock_key_set:  # Note: only max_block filtering applied to other side in candidates
                filtered_s2_record_keys[idx].append((key, weight))

    # Now compute the EXACT rank as candidates() would
    rank = get_pair_rank_for_s1(s1_idx, s2_idx, filtered_s1_record_keys, filtered_s2_record_keys)

    if rank is None:
        # This shouldn't happen if we have shared keys that survived filtering
        return 'E', {
            'failure_type': 'E',
            'reason': 'Pair survived filtering but rank computation failed',
            'shared_keys_survived': list(shared_keys_survived_filtering)[:5],
            'num_shared_keys_survived': len(shared_keys_survived_filtering)
        }

    if rank > topk:
        # Genuinely excluded by top-k truncation
        return 'D', {
            'failure_type': 'D',
            'reason': f'Pair survived key filtering but ranked {rank} > topk ({topk})',
            'rank': rank,
            'topk': topk,
            'shared_keys_survived': list(shared_keys_survived_filtering)[:5],
            'num_shared_keys_survived': len(shared_keys_survived_filtering)
        }
    else:
        # Pair should have been in candidates but wasn't - this is unexpected
        return 'E', {
            'failure_type': 'E',
            'reason': f'Pair survived filtering and ranked {rank} <= topk ({topk}) but missing from candidates',
            'rank': rank,
            'topk': topk,
            'shared_keys_survived': list(shared_keys_survived_filtering)[:5],
            'num_shared_keys_survived': len(shared_keys_survived_filtering)
        }


def run_blocking_audit(data_dir, max_block=30, max_s1_block=200, topk=30):
    """Run the blocking audit on the training data."""

    print(f"Loading training data from {data_dir}")

    # Define file paths
    s1_path = os.path.join(data_dir, 'train_source1.tsv')
    s2_path = os.path.join(data_dir, 'train_source2.tsv')
    s3_path = os.path.join(data_dir, 'train_source3.tsv')
    gt_path = os.path.join(data_dir, 'train_ground_truth.tsv')

    # Check if files exist
    for path, name in [(s1_path, 'train_source1.tsv'), (s2_path, 'train_source2.tsv'),
                       (s3_path, 'train_source3.tsv'), (gt_path, 'train_ground_truth.tsv')]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            return None

    # Load ground truth
    print("Loading ground truth...")
    gt_map = load_ground_truth(gt_path)

    # Get countries from source1
    print("Determining countries...")
    s1_df = read_tsv(s1_path, usecols=['country'])
    countries = s1_df['country'].unique()
    print(f"Found countries: {list(countries)}")

    # Initialize counters
    total_gt_pairs = 0
    total_recovered_pairs = 0
    total_s1_entities = 0
    total_candidates = 0
    s1_with_zero_candidates = 0

    # Failure mode counters
    failure_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    failure_details = []  # Store detailed info for missed pairs

    # Process each country
    for country in countries:
        print(f"\nProcessing country: {country}")

        # Load data for this country
        s1_df = read_country(s1_path, country)
        s2_df = read_country(s2_path, country) if os.path.exists(s2_path) else pd.DataFrame()
        s3_df = read_country(s3_path, country) if os.path.exists(s3_path) else pd.DataFrame()

        # Combine S2 and S3
        other_parts = []
        if len(s2_df) > 0:
            other_parts.append(s2_df)
        if len(s3_df) > 0:
            other_parts.append(s3_df)

        if len(other_parts) == 0:
            other_df = pd.DataFrame(columns=s1_df.columns)
        else:
            other_df = pd.concat(other_parts, ignore_index=True)

        if len(s1_df) == 0:
            print(f"  No S1 records for {country}")
            continue

        if len(other_df) == 0:
            print(f"  No S2/S3 records for {country}")
            # All S1 entities will have zero candidates
            s1_with_zero_candidates += len(s1_df)
            total_s1_entities += len(s1_df)
            continue

        # Compute keys for detailed analysis
        print(f"  Computing blocking keys for detailed analysis...")
        s1_record_keys, s1_key_freq = compute_record_keys(s1_df)
        other_record_keys, other_key_freq = compute_record_keys(other_df)

        # Create Side objects for actual blocking
        print(f"  Creating Side objects (S1: {len(s1_df)}, S2+S3: {len(other_df)})")
        s1_side = Side(s1_df)
        other_side = Side(other_df)

        total_s1_entities += len(s1_side)

        # Run actual blocking
        print(f"  Running blocking (max_block={max_block}, max_s1_block={max_s1_block}, topk={topk})")
        cand_df = candidates(s1_side, other_side, max_block=max_block, max_s1_block=max_s1_block, topk=topk)

        total_candidates += len(cand_df)

        # Build lookup for quick candidate checking
        cand_pairs = set()
        if len(cand_df) > 0:
            for _, row in cand_df.iterrows():
                s1_idx = int(row['r1'])
                s2_idx = int(row['ro'])
                cand_pairs.add((s1_idx, s2_idx))

        # Count S1 entities with zero candidates
        s1_entities_with_cands = set()
        if len(cand_df) > 0:
            s1_entities_with_cands = set(cand_df['r1'].unique())

        zero_cand_count = len(s1_side) - len(s1_entities_with_cands)
        s1_with_zero_candidates += zero_cand_count

        print(f"  S1 entities with candidates: {len(s1_entities_with_cands)}/{len(s1_side)}")
        print(f"  S1 entities with zero candidates: {zero_cand_count}")
        print(f"  Total candidate pairs: {len(cand_df)}")

        # Now analyze each ground truth pair
        gt_pairs_in_country = 0
        recovered_pairs_in_country = 0

        # Create dataframes for easier indexing
        s1_df_reset = s1_df.reset_index(drop=True)
        s2_df_reset = s2_df.reset_index(drop=True)
        s3_df_reset = s3_df.reset_index(drop=True)

        # For each S1 entity, check its ground truth matches
        for s1_idx, s1_row in s1_df_reset.iterrows():
            s1_id = s1_row['entity_id']
            if s1_id in gt_map:
                true_matches = gt_map[s1_id]
                gt_pairs_in_country += len(true_matches)

                # Check which of these matches were recovered by blocking
                recovered_for_this_s1 = 0

                # Check S2 matches
                for s2_idx, s2_row in s2_df_reset.iterrows():
                    s2_id = s2_row['entity_id']
                    if s2_id in true_matches:
                        # Check if this pair was in candidates
                        if (s1_idx, s2_idx) in cand_pairs:
                            recovered_for_this_s1 += 1
                        else:
                            # Analyze why this pair was missed
                            failure_type, diag = analyze_missed_pair(
                                s1_idx, s2_idx,
                                s1_record_keys, other_record_keys,
                                s1_key_freq, other_key_freq,
                                max_block, max_s1_block, topk
                            )

                            failure_counts[failure_type] += 1
                            diag.update({
                                's1_id': s1_id,
                                'matched_id': s2_id,
                                'matched_source': 'S2'
                            })
                            failure_details.append(diag)

                # Check S3 matches
                for s3_idx, s3_row in s3_df_reset.iterrows():
                    s3_id = s3_row['entity_id']
                    if s3_id in true_matches:
                        # Check if this pair was in candidates
                        # Note: s3_idx needs to be adjusted for the combined other dataframe
                        s3_idx_in_other = len(s2_df_reset) + s3_idx
                        if (s1_idx, s3_idx_in_other) in cand_pairs:
                            recovered_for_this_s1 += 1
                        else:
                            # Analyze why this pair was missed
                            failure_type, diag = analyze_missed_pair(
                                s1_idx, s3_idx_in_other,
                                s1_record_keys, other_record_keys,
                                s1_key_freq, other_key_freq,
                                max_block, max_s1_block, topk
                            )

                            failure_counts[failure_type] += 1
                            diag.update({
                                's1_id': s1_id,
                                'matched_id': s3_id,
                                'matched_source': 'S3'
                            })
                            failure_details.append(diag)

                recovered_pairs_in_country += recovered_for_this_s1

        total_gt_pairs += gt_pairs_in_country
        total_recovered_pairs += recovered_pairs_in_country

        print(f"  Ground truth pairs in {country}: {gt_pairs_in_country}")
        print(f"  Recovered pairs in {country}: {recovered_pairs_in_country}")
        if gt_pairs_in_country > 0:
            recall = recovered_pairs_in_country / gt_pairs_in_country
            print(f"  Blocking recall for {country}: {recall:.4f}")

    # Calculate overall metrics
    print("\n" + "="*60)
    print("BLOCKING AUDIT RESULTS")
    print("="*60)
    print(f"Total S1 entities: {total_s1_entities:,}")
    print(f"Total ground truth pairs: {total_gt_pairs:,}")
    print(f"Total pairs recovered by blocking: {total_recovered_pairs:,}")

    if total_gt_pairs > 0:
        blocking_recall = total_recovered_pairs / total_gt_pairs
        print(f"Overall blocking recall: {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")
    else:
        print("Overall blocking recall: N/A (no ground truth pairs)")
        blocking_recall = 0.0

    print(f"Total candidate pairs generated: {total_candidates:,}")
    if total_s1_entities > 0:
        avg_candidates_per_s1 = total_candidates / total_s1_entities
        print(f"Average candidates per S1 entity: {avg_candidates_per_s1:.2f}")
    else:
        avg_candidates_per_s1 = 0.0
        print("Average candidates per S1 entity: N/A")

    if total_s1_entities > 0:
        zero_candidate_rate = s1_with_zero_candidates / total_s1_entities
        print(f"Percentage of S1 entities with zero candidates: {zero_candidate_rate:.2%}")
    else:
        zero_candidate_rate = 0.0
        print("Percentage of S1 entities with zero candidates: N/A")

    print("\nFailure mode breakdown (of missed pairs):")
    missed_pairs = total_gt_pairs - total_recovered_pairs
    if missed_pairs > 0:
        for mode in ['A', 'B', 'C', 'D']:  # E should ideally be 0 with proper implementation
            count = failure_counts[mode]
            pct = count / missed_pairs * 100
            print(f"  {mode}: {count:,} ({pct:.1f}% of missed pairs)")

        # Show E separately if any
        e_count = failure_counts['E']
        if e_count > 0:
            pct = e_count / missed_pairs * 100
            print(f"  E: {e_count:,} ({pct:.1f}% of missed pairs) - indicates implementation issue")
    else:
        print("  No missed pairs to analyze")

    # Save detailed report
    report_path = os.path.join(data_dir, 'blocking_audit_report.tsv')
    if failure_details:
        report_df = pd.DataFrame(failure_details)
        # Reorder columns for readability
        cols = ['s1_id', 'matched_id', 'matched_source', 'failure_type', 'reason']
        # Add any additional columns that exist
        extra_cols = [col for col in report_df.columns if col not in cols]
        final_cols = cols + extra_cols
        report_df = report_df[final_cols]
        report_df.to_csv(report_path, sep='\t', index=False)
        print(f"\nDetailed failure report saved to: {report_path}")
        print(f"  Contains {len(failure_details):,} missed pairs")
    else:
        print("\nNo detailed failure data to save")

    # Save summary metrics
    summary_path = os.path.join(data_dir, 'blocking_audit_summary.tsv')
    summary_data = []

    # Add overall metrics
    summary_data.append({'metric': 'total_s1_entities', 'value': total_s1_entities})
    summary_data.append({'metric': 'total_gt_pairs', 'value': total_gt_pairs})
    summary_data.append({'metric': 'total_recovered_pairs', 'value': total_recovered_pairs})
    summary_data.append({'metric': 'blocking_recall', 'value': blocking_recall})
    summary_data.append({'metric': 'total_candidates', 'value': total_candidates})
    summary_data.append({'metric': 'avg_candidates_per_s1', 'value': avg_candidates_per_s1})
    summary_data.append({'metric': 'zero_candidate_rate', 'value': zero_candidate_rate})

    # Add failure counts
    for mode in ['A', 'B', 'C', 'D', 'E']:
        summary_data.append({'metric': f'failure_{mode}_count', 'value': failure_counts[mode]})

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"Summary metrics saved to: {summary_path}")

    # Return summary metrics
    return {
        'total_s1_entities': total_s1_entities,
        'total_gt_pairs': total_gt_pairs,
        'total_recovered_pairs': total_recovered_pairs,
        'blocking_recall': blocking_recall,
        'total_candidates': total_candidates,
        'avg_candidates_per_s1': avg_candidates_per_s1,
        'zero_candidate_rate': zero_candidate_rate,
        'failure_counts': failure_counts,
        'missed_pairs_analyzed': len(failure_details)
    }


def main():
    """Main function to run the blocking audit."""
    import argparse

    parser = argparse.ArgumentParser(description='Run blocking audit on training data')
    parser.add_argument('--data', type=str, default='sample_dense',
                        help='Path to data directory (default: sample_dense)')
    parser.add_argument('--max_block', type=int, default=30,
                        help='Max block parameter (default: 30)')
    parser.add_argument('--max_s1_block', type=int, default=200,
                        help='Max S1 block parameter (default: 200)')
    parser.add_argument('--topk', type=int, default=30,
                        help='Top-k parameter (default: 30)')

    args = parser.parse_args()

    print("Starting blocking audit...")
    print(f"Data directory: {args.data}")
    print(f"Parameters: max_block={args.max_block}, max_s1_block={args.max_s1_block}, topk={args.topk}")

    results = run_blocking_audit(
        data_dir=args.data,
        max_block=args.max_block,
        max_s1_block=args.max_s1_block,
        topk=args.topk
    )

    if results is not None:
        print("\nAudit completed successfully!")
    else:
        print("\nAudit failed - see error messages above")


if __name__ == '__main__':
    main()
