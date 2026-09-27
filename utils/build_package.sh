#!/usr/bin/env bash
# =============================================================================
#  build_package.sh  —  Assemble the final submission zip for the contest.
#
#  Run from the student_resource/ folder (the one containing dataset/, code/, utils/):
#      bash utils/build_package.sh
#
#  What it does:
#    1. Creates submission_package/ with the required directory structure.
#    2. Copies output/matching_results.tsv + candidate_pairs.tsv.
#    3. Copies code/business_entity_resolution/ (src, README.md, requirements.txt).
#    4. Copies Documentation_template.md.
#    5. Runs validate_submission.py against the COPIED files to confirm PASS.
#    6. Zips into <team_name>_submission.zip.
#
#  IMPORTANT: Run this only after you have confirmed which output/ run is your BEST
#  leaderboard submission.  The script copies from output/ — make sure that folder
#  contains the files from the best run, not just the most recent one.
# =============================================================================

set -euo pipefail   # exit on any error, undefined var, or pipe failure

# ---- EDIT THIS to your team name (no spaces) ---------------------------------
TEAM_NAME="Byjus_ML_Gang"
# ------------------------------------------------------------------------------

PKG="submission_package"
ZIP="${TEAM_NAME}_submission.zip"

echo ""
echo "[build_package] Starting submission package assembly..."
echo "[build_package] Team name : ${TEAM_NAME}"
echo "[build_package] Output dir: ${PKG}/"
echo ""

# ---- 1. Verify required source files exist -----------------------------------
echo "[1/7] Checking required source files..."

for f in \
    "output/matching_results.tsv" \
    "output/candidate_pairs.tsv" \
    "Documentation_template.md" \
    "code/business_entity_resolution/src" \
    "code/business_entity_resolution/README.md" \
    "code/business_entity_resolution/requirements.txt"
do
    if [ ! -e "${f}" ]; then
        echo "ERROR: '${f}' not found."
        echo "       Make sure you are running from student_resource/ and predict.py has been run."
        exit 1
    fi
done
echo "   OK — all source files present."

# ---- 2. Clean and create package directory -----------------------------------
echo "[2/7] Creating ${PKG}/ directory structure..."

rm -rf "${PKG}"
mkdir -p "${PKG}/output"
mkdir -p "${PKG}/code/business_entity_resolution/src"
echo "   Done."

# ---- 3. Copy output files ----------------------------------------------------
echo "[3/7] Copying output files..."

cp "output/matching_results.tsv" "${PKG}/output/matching_results.tsv"
cp "output/candidate_pairs.tsv"  "${PKG}/output/candidate_pairs.tsv"

# Print file sizes so D can confirm these are the expected files
wc -l "${PKG}/output/matching_results.tsv" | awk '{print "   matching_results.tsv : " $1 " lines"}'
wc -l "${PKG}/output/candidate_pairs.tsv"  | awk '{print "   candidate_pairs.tsv  : " $1 " lines"}'
echo "   Done."

# ---- 4. Copy code ------------------------------------------------------------
echo "[4/7] Copying code/business_entity_resolution/..."

cp code/business_entity_resolution/src/*.py "${PKG}/code/business_entity_resolution/src/"
cp "code/business_entity_resolution/README.md"        "${PKG}/code/business_entity_resolution/README.md"
cp "code/business_entity_resolution/requirements.txt" "${PKG}/code/business_entity_resolution/requirements.txt"

echo "   Copied src/*.py files:"
ls "${PKG}/code/business_entity_resolution/src/" | awk '{print "     " $0}'
echo "   Done."

# ---- 5. Copy documentation ---------------------------------------------------
echo "[5/7] Copying Documentation_template.md..."
cp "Documentation_template.md" "${PKG}/Documentation_template.md"
echo "   Done."

# ---- 6. Validate the COPIED output files ------------------------------------
echo "[6/7] Running validate_submission.py against package output files..."
echo ""

python utils/validate_submission.py \
    --matching "${PKG}/output/matching_results.tsv" \
    --candidate "${PKG}/output/candidate_pairs.tsv" \
    --test-dir dataset/test

echo ""
echo "[build_package] Validation PASSED."

# ---- 7. Zip the package ------------------------------------------------------
echo ""
echo "[7/7] Zipping into ${ZIP} ..."

rm -f "${ZIP}"
cd "${PKG}"
zip -r "../${ZIP}" .
cd ..

# Print zip size
du -sh "${ZIP}" | awk '{print "   " $2 " created (" $1 ")"}'

echo ""
echo "============================================================"
echo " DONE.  Upload ${ZIP} to the contest portal."
echo " Verify the zip contains:"
echo "   ${ZIP}/output/matching_results.tsv"
echo "   ${ZIP}/output/candidate_pairs.tsv"
echo "   ${ZIP}/code/business_entity_resolution/src/*.py"
echo "   ${ZIP}/code/business_entity_resolution/README.md"
echo "   ${ZIP}/code/business_entity_resolution/requirements.txt"
echo "   ${ZIP}/Documentation_template.md"
echo "============================================================"
echo ""
