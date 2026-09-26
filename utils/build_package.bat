@echo off
REM ============================================================================
REM  build_package.bat  —  Assemble the final submission zip for the contest.
REM
REM  Run from the student_resource/ folder (the one containing dataset/, code/, utils/):
REM      utils\build_package.bat
REM
REM  What it does:
REM    1. Creates submission_package\ with the required directory structure.
REM    2. Copies output\matching_results.tsv + candidate_pairs.tsv.
REM    3. Copies code\business_entity_resolution\ (src, README.md, requirements.txt).
REM    4. Copies Documentation_template.md.
REM    5. Runs validate_submission.py against the COPIED files to confirm PASS.
REM    6. Zips into <team_name>_submission.zip (requires PowerShell 5+, present on Win10+).
REM
REM  IMPORTANT: Run this only after you have confirmed which output/ run is your BEST
REM  leaderboard submission.  The script copies from output\ — make sure that folder
REM  contains the files from the best run, not just the most recent one.
REM ============================================================================

setlocal EnableDelayedExpansion

REM ---- EDIT THIS to your team name (no spaces) --------------------------------
set TEAM_NAME=Byjus_ML_Gang
REM -----------------------------------------------------------------------------

set PKG=submission_package
set ZIP=%TEAM_NAME%_submission.zip

echo.
echo [build_package] Starting submission package assembly...
echo [build_package] Team name : %TEAM_NAME%
echo [build_package] Output dir: %PKG%\
echo.

REM ---- 1. Verify required source files exist ----------------------------------
echo [1/6] Checking required source files...

if not exist "output\matching_results.tsv" (
    echo ERROR: output\matching_results.tsv not found.
    echo        Run predict.py first, then re-run this script.
    exit /b 1
)
if not exist "output\candidate_pairs.tsv" (
    echo ERROR: output\candidate_pairs.tsv not found.
    echo        Run predict.py first, then re-run this script.
    exit /b 1
)
if not exist "Documentation_template.md" (
    echo ERROR: Documentation_template.md not found.
    echo        Make sure you are running from student_resource\.
    exit /b 1
)
if not exist "code\business_entity_resolution\src" (
    echo ERROR: code\business_entity_resolution\src not found.
    echo        Make sure you are running from student_resource\.
    exit /b 1
)
echo    OK — all source files present.

REM ---- 2. Clean and create package directory ----------------------------------
echo [2/6] Creating %PKG%\ directory structure...

if exist "%PKG%" (
    echo    Removing existing %PKG%\ ...
    rmdir /s /q "%PKG%"
)

mkdir "%PKG%\output"
mkdir "%PKG%\code\business_entity_resolution\src"
echo    Done.

REM ---- 3. Copy output files ---------------------------------------------------
echo [3/6] Copying output files...

copy /y "output\matching_results.tsv" "%PKG%\output\matching_results.tsv" >nul
copy /y "output\candidate_pairs.tsv"  "%PKG%\output\candidate_pairs.tsv"  >nul

REM Show file sizes so D can confirm these are the expected files
for %%f in ("%PKG%\output\matching_results.tsv") do echo    matching_results.tsv : %%~zf bytes
for %%f in ("%PKG%\output\candidate_pairs.tsv")  do echo    candidate_pairs.tsv  : %%~zf bytes
echo    Done.

REM ---- 4. Copy code -----------------------------------------------------------
echo [4/6] Copying code\business_entity_resolution\...

REM Copy all .py files from src/
for %%f in ("code\business_entity_resolution\src\*.py") do (
    copy /y "%%f" "%PKG%\code\business_entity_resolution\src\" >nul
    echo    copied src\%%~nxf
)
copy /y "code\business_entity_resolution\README.md"       "%PKG%\code\business_entity_resolution\README.md"       >nul
copy /y "code\business_entity_resolution\requirements.txt" "%PKG%\code\business_entity_resolution\requirements.txt" >nul
echo    Done.

REM ---- 5. Copy documentation --------------------------------------------------
echo [5/6] Copying Documentation_template.md...
copy /y "Documentation_template.md" "%PKG%\Documentation_template.md" >nul
echo    Done.

REM ---- 6. Validate the COPIED output files ------------------------------------
echo [6/6] Running validate_submission.py against package output files...
echo.

python utils\validate_submission.py ^
    --matching "%PKG%\output\matching_results.tsv" ^
    --candidate "%PKG%\output\candidate_pairs.tsv" ^
    --test-dir dataset\test

if errorlevel 1 (
    echo.
    echo [build_package] FAIL — validate_submission.py returned an error.
    echo                 DO NOT zip or upload until this is resolved.
    exit /b 1
)

echo.
echo [build_package] Validation PASSED.

REM ---- 7. Zip the package -----------------------------------------------------
echo.
echo [7/7] Zipping into %ZIP% ...

if exist "%ZIP%" del /q "%ZIP%"

REM PowerShell Compress-Archive (available on Windows 10 / Server 2016+)
powershell -NoProfile -Command ^
    "Compress-Archive -Path '%PKG%\*' -DestinationPath '%ZIP%' -Force"

if errorlevel 1 (
    echo ERROR: PowerShell Compress-Archive failed.
    echo        Manually zip the contents of %PKG%\ into %ZIP%.
    exit /b 1
)

for %%f in ("%ZIP%") do echo    %ZIP% created (%%~zf bytes)

echo.
echo ============================================================
echo  DONE.  Upload %ZIP% to the contest portal.
echo  Verify the zip contains:
echo    %ZIP%\output\matching_results.tsv
echo    %ZIP%\output\candidate_pairs.tsv
echo    %ZIP%\code\business_entity_resolution\src\*.py
echo    %ZIP%\code\business_entity_resolution\README.md
echo    %ZIP%\code\business_entity_resolution\requirements.txt
echo    %ZIP%\Documentation_template.md
echo ============================================================
echo.

endlocal
