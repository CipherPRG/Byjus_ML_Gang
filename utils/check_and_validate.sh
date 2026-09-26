#!/usr/bin/env bash
# ============================================================
# check_and_validate.sh  —  Track C pre-submission wrapper
# Run from the student_resource/ root:
#   bash utils/check_and_validate.sh <matching_tsv> <candidate_tsv> <test_dir> <gt_tsv> <s1_train_tsv>
#
# Arguments:
#   $1  path to matching_results.tsv      (required)
#   $2  path to candidate_pairs.tsv       (required)
#   $3  path to test dataset dir          (required, e.g. dataset/test)
#   $4  path to train_ground_truth.tsv    (required for val_harness)
#   $5  path to train_source1.tsv         (required for val_harness country lookup)
#
# Example:
#   bash utils/check_and_validate.sh \
#       output/matching_results.tsv \
#       output/candidate_pairs.tsv \
#       dataset/test \
#       dataset/train/train_ground_truth.tsv \
#       dataset/train/train_source1.tsv
# ============================================================

set -euo pipefail

MATCHING="${1:-}"
CANDIDATE="${2:-}"
TEST_DIR="${3:-}"
GT="${4:-}"
S1="${5:-}"

usage() {
    echo ""
    echo "Usage:"
    echo "  bash utils/check_and_validate.sh <matching_tsv> <candidate_tsv> <test_dir> <gt_tsv> <s1_train_tsv>"
    echo ""
    exit 1
}

# ---- argument check ----
[[ -z "$MATCHING"  ]] && { echo "ERROR: missing argument 1 (matching_results.tsv)";  usage; }
[[ -z "$CANDIDATE" ]] && { echo "ERROR: missing argument 2 (candidate_pairs.tsv)";   usage; }
[[ -z "$TEST_DIR"  ]] && { echo "ERROR: missing argument 3 (test dataset dir)";       usage; }
[[ -z "$GT"        ]] && { echo "ERROR: missing argument 4 (train_ground_truth.tsv)"; usage; }
[[ -z "$S1"        ]] && { echo "ERROR: missing argument 5 (train_source1.tsv)";      usage; }

OVERALL="GO"

echo ""
echo "============================================================"
echo " STEP 1 / 2 — validate_submission.py"
echo "============================================================"

# Disable set -e temporarily so we can capture the exit code
set +e
python utils/validate_submission.py \
    --matching  "$MATCHING" \
    --candidate "$CANDIDATE" \
    --test-dir  "$TEST_DIR"
VALIDATE_EXIT=$?
set -e

if [[ $VALIDATE_EXIT -ne 0 ]]; then
    echo ""
    echo "[FAIL] validate_submission.py exited with code $VALIDATE_EXIT"
    echo "       Fix the issues above before submitting."
    OVERALL="NO-GO"
else
    echo ""
    echo "[PASS] validate_submission.py"
fi

echo ""
echo "============================================================"
echo " STEP 2 / 2 — val_harness.py  (train val split)"
echo "============================================================"

set +e
python code/business_entity_resolution/src/val_harness.py \
    --gt   "$GT" \
    --pred "$MATCHING" \
    --s1   "$S1" \
    --val-only
HARNESS_EXIT=$?
set -e

if [[ $HARNESS_EXIT -ne 0 ]]; then
    echo ""
    echo "[WARN] val_harness.py exited with code $HARNESS_EXIT"
fi

# ---- final verdict ----
echo ""
echo "============================================================"
if [[ "$OVERALL" == "NO-GO" ]]; then
    echo " FINAL VERDICT:  NO-GO  (validate_submission.py FAILED)"
    echo " Do NOT upload — fix format issues first."
else
    echo " FINAL VERDICT:  GO — format validated, F0.5 breakdown printed above."
    echo " Review per-country F0.5 and singleton counts before uploading."
fi
echo "============================================================"
echo ""

exit $VALIDATE_EXIT
