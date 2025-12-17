@echo off
REM ============================================================================
REM GeoAnomalyMapper - Automated Validation Runner
REM Run this script before bed, check results in the morning
REM ============================================================================

echo.
echo ============================================================================
echo GEOANOMALYMAPPER - AUTOMATED VALIDATION
echo ============================================================================
echo.
echo This will:
echo   1. Download USGS MRDS database (~50k deposits)
echo   2. Calculate precision at multiple radii
echo   3. Generate statistical analysis
echo   4. Create visualization plots
echo   5. Generate KML for Google Earth
echo   6. Produce comprehensive validation report
echo.
echo Estimated time: 10-30 minutes
echo.
pause

REM Create timestamp for log file
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set timestamp=%mydate%_%mytime%

REM Create log file
set logfile=validation_results\validation_log_%timestamp%.txt

echo.
echo Starting validation...
echo Log file: %logfile%
echo.

REM Run validation
python auto_validation.py data\outputs\usa_targets_FINAL.csv > %logfile% 2>&1

REM Check if successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================================
    echo VALIDATION COMPLETE!
    echo ============================================================================
    echo.
    echo Results saved to: validation_results\
    echo.
    echo Key files:
    echo   - validation_report_*.txt   ^(Read this first^)
    echo   - targets_validated.csv     ^(Enhanced target list^)
    echo   - top_targets_inspection.kml ^(Open in Google Earth^)
    echo   - validation_analysis.png   ^(Charts and graphs^)
    echo.
    echo Next steps:
    echo   1. Read the validation report
    echo   2. Open KML in Google Earth to inspect top targets
    echo   3. Check the precision numbers
    echo.
    pause
) else (
    echo.
    echo ============================================================================
    echo ERROR OCCURRED
    echo ============================================================================
    echo.
    echo Check the log file for details: %logfile%
    echo.
    pause
)