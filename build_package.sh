#!/usr/bin/env bash
# ============================================================
# build_package.sh — C5: assemble submission_package/ dir
# Run from the student_resource/ root:
#   bash build_package.sh [output_dir] [models_dir]
#
# Arguments (both optional):
#   $1  source output dir  (default: output)
#   $2  source models dir  (default: models_v4)
#
# Creates:
#   submission_package/
#     output/
#       matching_results.tsv
#       candidate_pairs.tsv
#     code/business_entity_resolution/
#       src/    (all .py files)
#       README.md
#       requirements.txt
#     Documentation_template.md
#
# Track D then reviews this folder, adds the final output
# files from the best run, zips it, and uploads.
# ============================================================

set -euo pipefail

OUT_SRC="${1:-output}"
MODELS_SRC="${2:-models_v4}"
PKG="submission_package"

echo ""
echo "============================================================"
echo " BUILD SUBMISSION PACKAGE"
echo " Source output dir : $OUT_SRC"
echo " Source models dir : $MODELS_SRC"
echo " Destination       : $PKG/"
echo "============================================================"

# ---- clean slate ----
if [[ -d "$PKG" ]]; then
    echo "Removing existing $PKG/ ..."
    rm -rf "$PKG"
fi

# ---- create directory tree ----
mkdir -p "$PKG/output"
mkdir -p "$PKG/code/business_entity_resolution/src"

MISSING_OUTPUT=0

# ---- copy output files ----
echo ""
echo "[1/4] Copying output files from $OUT_SRC/ ..."
for f in matching_results.tsv candidate_pairs.tsv; do
    if [[ -f "$OUT_SRC/$f" ]]; then
        cp "$OUT_SRC/$f" "$PKG/output/$f"
        echo "  OK  $OUT_SRC/$f"
    else
        echo "  WARN: $OUT_SRC/$f not found -- leaving placeholder"
        echo "[placeholder - copy final output here before zipping]" > "$PKG/output/$f"
        MISSING_OUTPUT=1
    fi
done

# ---- copy source code ----
echo ""
echo "[2/4] Copying source code ..."
for f in code/business_entity_resolution/src/*.py; do
    cp "$f" "$PKG/code/business_entity_resolution/src/"
    echo "  OK  $f"
done

# ---- copy README and requirements ----
echo ""
echo "[3/4] Copying README.md and requirements.txt ..."
for f in README.md requirements.txt; do
    src="code/business_entity_resolution/$f"
    if [[ -f "$src" ]]; then
        cp "$src" "$PKG/code/business_entity_resolution/$f"
        echo "  OK  $src"
    else
        echo "  WARN: $src not found"
    fi
done

# ---- copy Documentation_template.md ----
echo ""
echo "[4/4] Copying Documentation_template.md ..."
if [[ -f "Documentation_template.md" ]]; then
    cp "Documentation_template.md" "$PKG/Documentation_template.md"
    echo "  OK  Documentation_template.md"
else
    echo "  WARN: Documentation_template.md not found"
fi

# ---- summary ----
echo ""
echo "============================================================"
echo " PACKAGE CONTENTS:"
echo "============================================================"
find "$PKG" -type f | sort | while read -r f; do echo "  $f"; done
echo ""

if [[ $MISSING_OUTPUT -ne 0 ]]; then
    echo " WARNING: One or more output files were missing."
    echo " Copy the final matching_results.tsv and candidate_pairs.tsv"
    echo " into $PKG/output/ before zipping."
    echo ""
fi

echo " To create the zip:"
echo "   zip -r Byjus_ML_Gang_submission.zip $PKG/"
echo "   # or: cd $PKG && zip -r ../Byjus_ML_Gang_submission.zip ."
echo ""
echo " DONE."
echo "============================================================"
