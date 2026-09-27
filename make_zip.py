#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_zip.py  <output_dir>  <zip_path>

Builds the final submission zip exactly as required by the problem statement:

    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/               ← all .py source files (no __pycache__)
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Exclusions (never enter the zip)
---------------------------------
- models*/        — trained model artefacts (size + fair-play rules)
- sample*/        — dataset samples used during dev
- eval_cache*/    — cached eval artefacts
- __pycache__/    — compiled bytecode
- *.pyc / *.pyo
- catboost_info/  — CatBoost training artefacts
- *.log           — log files
- *.txt           — log-style .txt files (output.txt etc.)

Usage
-----
    python make_zip.py output/ Byjus_ML_Gang_submission.zip
    python make_zip.py output/ ../Byjus_ML_Gang_submission.zip

The zip_path may be absolute or relative (relative to the current working directory).

Print
-----
- Full file tree with in-zip paths and sizes
- Total uncompressed and compressed size
"""

import sys
import os
import zipfile
import argparse

# Windows cp1252 terminals can't print box-drawing chars; force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Paths (relative to repo root — this script lives in the repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories / files to include, mapped to their in-zip destination
# Each entry: (source_abspath, zip_prefix)
#   zip_prefix is the directory inside the zip that the source lands in.

SRC_ROOT = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
CODE_ROOT = os.path.join(REPO_ROOT, "code", "business_entity_resolution")


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------

EXCLUDE_DIR_PREFIXES = (
    "models",
    "sample",
    "eval_cache",
    "__pycache__",
    "catboost_info",
)

EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".log"}

EXCLUDE_BASENAMES = {
    # generated test/scratch artefacts that live in src/ but shouldn't ship
    "output.txt",
}


def is_excluded(rel_path: str) -> bool:
    """Return True if a file or directory should be excluded from the zip."""
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if any(part.startswith(pfx) for pfx in EXCLUDE_DIR_PREFIXES):
            return True
    base = os.path.basename(rel_path)
    _, ext = os.path.splitext(base)
    if ext in EXCLUDE_EXTENSIONS:
        return True
    if ext == ".txt" and base != "requirements.txt":
        # exclude *.txt logs except requirements.txt
        return True
    if base in EXCLUDE_BASENAMES:
        return True
    return False


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_src_files():
    """Walk src/ and return list of (abs_path, zip_arc_path) pairs."""
    result = []
    zip_root = "code/business_entity_resolution/src"
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        # prune excluded subdirs in-place so os.walk skips them
        dirnames[:] = [
            d for d in dirnames
            if not any(d.startswith(pfx) for pfx in EXCLUDE_DIR_PREFIXES)
        ]
        for fname in sorted(filenames):
            abs_path = os.path.join(dirpath, fname)
            rel_to_src = os.path.relpath(abs_path, SRC_ROOT)
            arc = zip_root + "/" + rel_to_src.replace("\\", "/")
            if is_excluded(arc):
                continue
            result.append((abs_path, arc))
    return result


def collect_output_files(output_dir: str):
    required = ["matching_results.tsv", "candidate_pairs.tsv"]
    result = []
    for fname in required:
        abs_path = os.path.join(output_dir, fname)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"Required output file missing: {abs_path}\n"
                f"Run predict.py first to generate the output files."
            )
        result.append((abs_path, f"output/{fname}"))
    return result


def collect_fixed_files():
    """
    Collect README.md, requirements.txt, and Documentation_template.md
    from the repo root / code directory.
    """
    entries = [
        (
            os.path.join(CODE_ROOT, "README.md"),
            "code/business_entity_resolution/README.md",
        ),
        (
            os.path.join(CODE_ROOT, "requirements.txt"),
            "code/business_entity_resolution/requirements.txt",
        ),
        (
            os.path.join(REPO_ROOT, "Documentation_template.md"),
            "Documentation_template.md",
        ),
    ]
    result = []
    for abs_path, arc in entries:
        if not os.path.isfile(abs_path):
            print(f"  WARNING: expected file not found, skipping: {abs_path}")
            continue
        result.append((abs_path, arc))
    return result


# ---------------------------------------------------------------------------
# Build zip
# ---------------------------------------------------------------------------

def fmt_size(n_bytes: int) -> str:
    for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n_bytes >= threshold:
            return f"{n_bytes / threshold:.1f} {unit}"
    return f"{n_bytes} B"


def build_zip(output_dir: str, zip_path: str):
    # ── collect everything ───────────────────────────────────────────────────
    output_files = collect_output_files(output_dir)
    src_files    = collect_src_files()
    fixed_files  = collect_fixed_files()

    all_files = output_files + src_files + fixed_files
    all_files.sort(key=lambda t: t[1])  # sort by arc path for readable tree

    if not all_files:
        print("ERROR: no files to zip — nothing written.")
        sys.exit(1)

    # ── write zip ────────────────────────────────────────────────────────────
    zip_path = os.path.abspath(zip_path)
    print(f"Writing {zip_path} …\n")

    total_uncompressed = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for abs_path, arc in all_files:
            zf.write(abs_path, arc)
            info = zf.getinfo(arc)
            total_uncompressed += info.file_size

    compressed_size = os.path.getsize(zip_path)

    # ---- print file tree (flat sorted list, ASCII-safe) --------------------
    print("Contents:")
    print("-" * 72)

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = sorted(zf.infolist(), key=lambda i: i.filename)
        for info in entries:
            parts = info.filename.split("/")
            depth = len(parts) - 1
            indent = "  " * depth
            print(f"{indent}{parts[-1]}  ({fmt_size(info.file_size)})")

    print("-" * 72)
    print(f"Files        : {len(all_files)}")
    print(f"Uncompressed : {fmt_size(total_uncompressed)}  ({total_uncompressed:,} bytes)")
    print(f"Compressed   : {fmt_size(compressed_size)}  ({compressed_size:,} bytes)")
    print(f"Ratio        : {compressed_size / total_uncompressed:.1%}")
    print(f"\nZip written  : {zip_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the final submission zip per the problem-statement spec. "
            "Run from the repo root (the directory that contains problem_statement.md)."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory with matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "zip_path",
        help="Where to write the zip, e.g. Byjus_ML_Gang_submission.zip",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"ERROR: output_dir does not exist: {output_dir}")
        sys.exit(1)

    try:
        build_zip(output_dir, args.zip_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
