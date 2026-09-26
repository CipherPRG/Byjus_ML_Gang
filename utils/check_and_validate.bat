@echo off
:: ============================================================
:: check_and_validate.bat  —  Track C pre-submission wrapper
:: Run from the student_resource/ root:
::   utils\check_and_validate.bat <matching_tsv> <candidate_tsv> <test_dir> <gt_tsv> <s1_train_tsv>
::
:: Arguments:
::   %1  path to matching_results.tsv      (required)
::   %2  path to candidate_pairs.tsv       (required)
::   %3  path to test dataset dir          (required, e.g. dataset/test)
::   %4  path to train_ground_truth.tsv    (required for val_harness)
::   %5  path to train_source1.tsv         (required for val_harness country lookup)
::
:: Example:
::   utils\check_and_validate.bat output\matching_results.tsv output\candidate_pairs.tsv dataset\test dataset\train\train_ground_truth.tsv dataset\train\train_source1.tsv
:: ============================================================

setlocal enabledelayedexpansion

set MATCHING=%~1
set CANDIDATE=%~2
set TEST_DIR=%~3
set GT=%~4
set S1=%~5

:: ---- argument check ----
if "%MATCHING%"=="" (
    echo ERROR: missing argument 1 ^(matching_results.tsv^)
    goto :usage
)
if "%CANDIDATE%"=="" (
    echo ERROR: missing argument 2 ^(candidate_pairs.tsv^)
    goto :usage
)
if "%TEST_DIR%"=="" (
    echo ERROR: missing argument 3 ^(test dataset dir^)
    goto :usage
)
if "%GT%"=="" (
    echo ERROR: missing argument 4 ^(train_ground_truth.tsv^)
    goto :usage
)
if "%S1%"=="" (
    echo ERROR: missing argument 5 ^(train_source1.tsv^)
    goto :usage
)

echo.
echo ============================================================
echo  STEP 1 / 2 — validate_submission.py
echo ============================================================
python utils\validate_submission.py ^
    --matching "%MATCHING%" ^
    --candidate "%CANDIDATE%" ^
    --test-dir "%TEST_DIR%"

set VALIDATE_EXIT=%ERRORLEVEL%

if %VALIDATE_EXIT% NEQ 0 (
    echo.
    echo [FAIL] validate_submission.py exited with code %VALIDATE_EXIT%
    echo        Fix the issues above before submitting.
    set OVERALL=NO-GO
    goto :harness
)

echo.
echo [PASS] validate_submission.py

:harness
echo.
echo ============================================================
echo  STEP 2 / 2 — val_harness.py  ^(train val split^)
echo ============================================================
python code\business_entity_resolution\src\val_harness.py ^
    --gt "%GT%" ^
    --pred "%MATCHING%" ^
    --s1 "%S1%" ^
    --val-only

set HARNESS_EXIT=%ERRORLEVEL%

if %HARNESS_EXIT% NEQ 0 (
    echo.
    echo [WARN] val_harness.py exited with code %HARNESS_EXIT%
)

:: ---- final verdict ----
echo.
echo ============================================================
if "%OVERALL%"=="NO-GO" (
    echo  FINAL VERDICT:  NO-GO  ^(validate_submission.py FAILED^)
    echo  Do NOT upload — fix format issues first.
) else (
    echo  FINAL VERDICT:  GO — format validated, F0.5 breakdown printed above.
    echo  Review the per-country F0.5 and singleton counts before uploading.
)
echo ============================================================
echo.

endlocal
exit /b %VALIDATE_EXIT%

:usage
echo.
echo Usage:
echo   utils\check_and_validate.bat ^<matching_tsv^> ^<candidate_tsv^> ^<test_dir^> ^<gt_tsv^> ^<s1_train_tsv^>
echo.
endlocal
exit /b 1
