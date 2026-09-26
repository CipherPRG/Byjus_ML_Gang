@echo off
:: ============================================================
:: build_package.bat — C5: assemble submission_package/ dir
:: Run from the student_resource/ root:
::   build_package.bat [output_dir] [models_dir]
::
:: Arguments (both optional):
::   %1  source output dir  (default: output)
::   %2  source models dir  (default: models_v4)
::
:: Creates:
::   submission_package/
::     output/
::       matching_results.tsv
::       candidate_pairs.tsv
::     code/business_entity_resolution/
::       src/    (all .py files)
::       README.md
::       requirements.txt
::     Documentation_template.md
::
:: Track D then reviews this folder, adds the final output
:: files from the best run, zips it, and uploads.
:: ============================================================
setlocal enabledelayedexpansion

set OUT_SRC=%~1
if "%OUT_SRC%"=="" set OUT_SRC=output

set MODELS_SRC=%~2
if "%MODELS_SRC%"=="" set MODELS_SRC=models_v4

set PKG=submission_package

echo.
echo ============================================================
echo  BUILD SUBMISSION PACKAGE
echo  Source output dir : %OUT_SRC%
echo  Source models dir : %MODELS_SRC%
echo  Destination       : %PKG%\
echo ============================================================

:: ---- clean slate ----
if exist "%PKG%" (
    echo Removing existing %PKG%\ ...
    rmdir /s /q "%PKG%"
)

:: ---- create directory tree ----
mkdir "%PKG%\output"
mkdir "%PKG%\code\business_entity_resolution\src"

:: ---- copy output files ----
echo.
echo [1/4] Copying output files from %OUT_SRC%\ ...

set MISSING_OUTPUT=0
for %%F in (matching_results.tsv candidate_pairs.tsv) do (
    if exist "%OUT_SRC%\%%F" (
        copy /y "%OUT_SRC%\%%F" "%PKG%\output\%%F" >nul
        echo   OK  %OUT_SRC%\%%F
    ) else (
        echo   WARN: %OUT_SRC%\%%F not found -- leaving placeholder
        echo [placeholder - copy final output here before zipping] > "%PKG%\output\%%F"
        set MISSING_OUTPUT=1
    )
)

:: ---- copy source code ----
echo.
echo [2/4] Copying source code ...
for %%F in (code\business_entity_resolution\src\*.py) do (
    copy /y "%%F" "%PKG%\code\business_entity_resolution\src\" >nul
    echo   OK  %%F
)

:: ---- copy README and requirements ----
echo.
echo [3/4] Copying README.md and requirements.txt ...
for %%F in (README.md requirements.txt) do (
    if exist "code\business_entity_resolution\%%F" (
        copy /y "code\business_entity_resolution\%%F" "%PKG%\code\business_entity_resolution\%%F" >nul
        echo   OK  code\business_entity_resolution\%%F
    ) else (
        echo   WARN: code\business_entity_resolution\%%F not found
    )
)

:: ---- copy Documentation_template.md ----
echo.
echo [4/4] Copying Documentation_template.md ...
if exist "Documentation_template.md" (
    copy /y "Documentation_template.md" "%PKG%\Documentation_template.md" >nul
    echo   OK  Documentation_template.md
) else (
    echo   WARN: Documentation_template.md not found
)

:: ---- summary ----
echo.
echo ============================================================
echo  PACKAGE CONTENTS:
echo ============================================================
for /r "%PKG%" %%F in (*) do echo   %%F
echo.
if %MISSING_OUTPUT%==1 (
    echo  WARNING: One or more output files were missing.
    echo  Copy the final matching_results.tsv and candidate_pairs.tsv
    echo  into submission_package\output\ before zipping.
)
echo.
echo  To create the zip (PowerShell):
echo    Compress-Archive -Path submission_package\* -DestinationPath Byjus_ML_Gang_submission.zip
echo.
echo  DONE.
echo ============================================================
endlocal
exit /b 0
