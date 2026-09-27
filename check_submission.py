#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_submission.py  <output_dir>  <test_dir>

Validates the two output TSVs (matching_results.tsv and candidate_pairs.tsv)
against every rule in the problem statement, then prints per-country stats.

Reads all files in chunks so it works on arbitrarily large datasets.

Exit codes
----------
0  — all checks passed
1  — at least one issue found (issues printed to stdout)
"""

import sys
import os
import csv
import argparse
import collections

# Make stdout UTF-8 on Windows (avoids cp1252 UnicodeEncodeError for arrow chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_tsv(path, chunksize=50_000):
    """Yield rows from a TSV as dicts, chunksize rows at a time."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunksize:
                yield from chunk
                chunk = []
        if chunk:
            yield from chunk


def load_entity_set(path, chunksize=50_000):
    """Return the set of entity_ids in a source TSV (chunked read)."""
    ids = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        chunk = []
        for row in reader:
            chunk.append(row["entity_id"])
            if len(chunk) >= chunksize:
                ids.update(chunk)
                chunk = []
        ids.update(chunk)
    return ids


def load_country_map(path, chunksize=50_000):
    """Return {entity_id: country} for a source TSV."""
    m = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        chunk = []
        for row in reader:
            chunk.append((row["entity_id"], row["country"]))
            if len(chunk) >= chunksize:
                m.update(chunk)
                chunk = []
        m.update(chunk)
    return m


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate(output_dir: str, test_dir: str):
    issues = []
    def err(msg):
        issues.append(msg)

    # ── paths ────────────────────────────────────────────────────────────────
    matching_path   = os.path.join(output_dir, "matching_results.tsv")
    candidate_path  = os.path.join(output_dir, "candidate_pairs.tsv")
    s1_path         = os.path.join(test_dir,   "test_source1.tsv")
    s2_path         = os.path.join(test_dir,   "test_source2.tsv")
    s3_path         = os.path.join(test_dir,   "test_source3.tsv")

    for label, p in [
        ("matching_results.tsv", matching_path),
        ("candidate_pairs.tsv",  candidate_path),
        ("test_source1.tsv",     s1_path),
        ("test_source2.tsv",     s2_path),
        ("test_source3.tsv",     s3_path),
    ]:
        if not os.path.isfile(p):
            err(f"MISSING FILE: {label} not found at {p}")

    if issues:
        return issues  # can't proceed without files

    # ── load reference sets (chunked) ────────────────────────────────────────
    print("Loading test S1 …")
    s1_country = load_country_map(s1_path)
    s1_ids     = set(s1_country)

    print("Loading test S2 + S3 …")
    s2_ids = load_entity_set(s2_path)
    s3_ids = load_entity_set(s3_path)
    valid_match_ids = s2_ids | s3_ids

    print(f"  S1: {len(s1_ids):,}  S2: {len(s2_ids):,}  S3: {len(s3_ids):,}")

    # ── pass 1: matching_results.tsv ─────────────────────────────────────────
    print("\nChecking matching_results.tsv …")
    seen_s1_matching  = {}          # s1_id -> line number (duplicate check)
    matched_id_set    = {}          # s1_id -> frozenset of matched IDs

    # per-country accumulators
    country_pairs   = collections.Counter()   # country -> total matched pairs
    country_empty   = collections.Counter()   # country -> empty-match rows
    country_s1_seen = collections.Counter()   # country -> rows seen

    matching_row_count = 0
    for lineno, row in enumerate(iter_tsv(matching_path), start=2):
        matching_row_count += 1
        s1id = row.get("source1_entity_id", "").strip()
        raw  = row.get("matched_entity_ids", "").strip()

        # duplicate S1 row
        if s1id in seen_s1_matching:
            err(f"DUPLICATE source1_entity_id '{s1id}' (first seen line {seen_s1_matching[s1id]}, again line {lineno})")
        else:
            seen_s1_matching[s1id] = lineno

        # S1 not in test set
        if s1id not in s1_ids:
            err(f"UNKNOWN source1_entity_id '{s1id}' (line {lineno}) — not in test_source1.tsv")

        # parse matched list
        matched = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

        # duplicate IDs within the list
        seen_local = {}
        for mid in matched:
            if mid in seen_local:
                err(f"DUPLICATE matched ID '{mid}' inside list for '{s1id}' (line {lineno})")
            seen_local[mid] = True

        # IDs must be from S2/S3
        for mid in matched:
            if mid not in valid_match_ids:
                err(f"INVALID matched ID '{mid}' for '{s1id}' (line {lineno}) — not in test S2/S3")

        matched_id_set[s1id] = frozenset(matched)

        # stats
        country = s1_country.get(s1id, "UNKNOWN")
        country_s1_seen[country] += 1
        if matched:
            country_pairs[country] += len(matched)
        else:
            country_empty[country] += 1

    # every test S1 must appear
    missing_s1 = s1_ids - set(seen_s1_matching)
    if missing_s1:
        sample = sorted(missing_s1)[:10]
        err(f"MISSING {len(missing_s1):,} S1 entities from matching_results.tsv (sample: {sample})")

    # ── pass 2: candidate_pairs.tsv ──────────────────────────────────────────
    print("Checking candidate_pairs.tsv …")
    seen_s1_cand    = {}
    cand_id_set     = {}          # s1_id -> frozenset of candidate IDs
    candidate_row_count = 0

    country_cand_pairs  = collections.Counter()
    country_cand_empty  = collections.Counter()

    for lineno, row in enumerate(iter_tsv(candidate_path), start=2):
        candidate_row_count += 1
        s1id = row.get("source1_entity_id", "").strip()
        raw  = row.get("candidate_entity_ids", "").strip()

        if s1id in seen_s1_cand:
            err(f"DUPLICATE source1_entity_id '{s1id}' in candidate_pairs.tsv (first line {seen_s1_cand[s1id]}, again {lineno})")
        else:
            seen_s1_cand[s1id] = lineno

        if s1id not in s1_ids:
            err(f"UNKNOWN source1_entity_id '{s1id}' in candidate_pairs.tsv (line {lineno})")

        cands = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

        seen_local = {}
        for cid in cands:
            if cid in seen_local:
                err(f"DUPLICATE candidate ID '{cid}' inside list for '{s1id}' in candidate_pairs.tsv (line {lineno})")
            seen_local[cid] = True

        for cid in cands:
            if cid not in valid_match_ids:
                err(f"INVALID candidate ID '{cid}' for '{s1id}' in candidate_pairs.tsv (line {lineno}) — not in test S2/S3")

        cand_id_set[s1id] = frozenset(cands)

        country = s1_country.get(s1id, "UNKNOWN")
        country_cand_pairs[country] += len(cands)
        if not cands:
            country_cand_empty[country] += 1

    missing_s1_cand = s1_ids - set(seen_s1_cand)
    if missing_s1_cand:
        sample = sorted(missing_s1_cand)[:10]
        err(f"MISSING {len(missing_s1_cand):,} S1 entities from candidate_pairs.tsv (sample: {sample})")

    # ── cross-check: every matched ID must appear in candidates ──────────────
    print("Cross-checking matched ⊆ candidates …")
    leaked = 0
    for s1id, mids in matched_id_set.items():
        cands = cand_id_set.get(s1id, frozenset())
        diff  = mids - cands
        if diff:
            leaked += len(diff)
            sample_diff = sorted(diff)[:3]
            suffix = " (+more)" if len(diff) > 3 else ""
            err(f"MATCHED IDs not in candidates for '{s1id}': {sample_diff}{suffix}")
    if leaked:
        err(f"  → {leaked:,} matched ID(s) total are absent from candidate_pairs.tsv")

    # ── print per-country stats ──────────────────────────────────────────────
    print("\n" + "="*68)
    print("  PER-COUNTRY STATS — matching_results.tsv")
    print("="*68)
    countries = sorted(country_s1_seen)
    header = f"  {'Country':<12}  {'S1 rows':>9}  {'Matched pairs':>14}  {'Avg pairs/S1':>13}  {'Empty rate':>10}"
    print(header)
    print("  " + "-"*64)
    for c in countries:
        n      = country_s1_seen[c]
        pairs  = country_pairs[c]
        empty  = country_empty[c]
        avg    = pairs / n if n else 0.0
        erate  = empty / n if n else 0.0
        print(f"  {c:<12}  {n:>9,}  {pairs:>14,}  {avg:>13.2f}  {erate:>9.1%}")

    print("\n" + "="*68)
    print("  PER-COUNTRY STATS — candidate_pairs.tsv")
    print("="*68)
    all_cand_countries = sorted(set(list(country_cand_pairs) + list(country_cand_empty)))
    header = f"  {'Country':<12}  {'S1 rows':>9}  {'Cand pairs':>11}  {'Avg cands/S1':>13}  {'Empty rate':>10}"
    print(header)
    print("  " + "-"*64)
    for c in all_cand_countries:
        n     = country_s1_seen.get(c, 0)
        pairs = country_cand_pairs[c]
        empty = country_cand_empty[c]
        avg   = pairs / n if n else 0.0
        erate = empty / n if n else 0.0
        print(f"  {c:<12}  {n:>9,}  {pairs:>11,}  {avg:>13.2f}  {erate:>9.1%}")

    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate submission output files against the problem-statement rules."
    )
    parser.add_argument("output_dir", help="Directory containing matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("test_dir",   help="Directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    issues = validate(args.output_dir, args.test_dir)

    print("\n" + "="*68)
    if issues:
        print(f"  FAIL — {len(issues)} issue(s) found:")
        for i, msg in enumerate(issues, 1):
            print(f"  [{i:02d}] {msg}")
        sys.exit(1)
    else:
        print("  PASS — all checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
