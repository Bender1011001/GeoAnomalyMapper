@echo off
REM Batch script to process Sentinel-1 interferograms using SNAP GPT
REM Hardware optimized: 16GB cache, 8 threads

echo ========================================
echo SNAP InSAR Batch Processing
echo ========================================
echo.
echo Processing 5 interferometric pairs...
echo Each pair takes 30-60 minutes with optimization
echo Total time: 2.5-5 hours
echo.
echo Press Ctrl+C to cancel, or
pause

REM Create output directory
if not exist "C:\Users\admin\Downloads\SAR-project\data\processed\insar" mkdir "C:\Users\admin\Downloads\SAR-project\data\processed\insar"

REM Set SNAP GPT path
set GPT="C:\Program Files\esa-snap\bin\gpt.exe"
set GRAPH="C:\Users\admin\Downloads\SAR-project\GeoAnomalyMapper\snap_interferogram_graph.xml"
set INDIR="C:\Users\admin\Downloads\SAR-project\data\raw\insar\sentinel1"
set OUTDIR="C:\Users\admin\Downloads\SAR-project\data\processed\insar"

echo.
echo ========================================
echo Processing Pair 1/5
echo ========================================
%GPT% %GRAPH% ^
  -Pmaster=%INDIR%\S1A_IW_SLC__1SDV_20251005T140219_20251005T140246_061291_07A55D_B681.SAFE.zip ^
  -Pslave=%INDIR%\S1A_IW_SLC__1SDV_20251006T130356_20251006T130423_061305_07A5EF_3B84.SAFE.zip ^
  -Poutput=%OUTDIR%\interferogram_01.tif ^
  -c 16G -q 8

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Interferogram 1 completed
) else (
    echo [ERROR] Interferogram 1 failed
)

echo.
echo ========================================
echo Processing Pair 2/5
echo ========================================
%GPT% %GRAPH% ^
  -Pmaster=%INDIR%\S1A_IW_SLC__1SDV_20251006T130356_20251006T130423_061305_07A5EF_3B84.SAFE.zip ^
  -Pslave=%INDIR%\S1C_IW_SLC__1SDV_20251008T013143_20251008T013210_004464_008D6F_FD00.SAFE.zip ^
  -Poutput=%OUTDIR%\interferogram_02.tif ^
  -c 16G -q 8

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Interferogram 2 completed
) else (
    echo [ERROR] Interferogram 2 failed
)

echo.
echo ========================================
echo Processing Pair 3/5
echo ========================================
%GPT% %GRAPH% ^
  -Pmaster=%INDIR%\S1C_IW_SLC__1SDV_20251008T013143_20251008T013210_004464_008D6F_FD00.SAFE.zip ^
  -Pslave=%INDIR%\S1C_IW_SLC__1SDV_20251008T013439_20251008T013506_004464_008D6F_B190.SAFE.zip ^
  -Poutput=%OUTDIR%\interferogram_03.tif ^
  -c 16G -q 8

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Interferogram 3 completed
) else (
    echo [ERROR] Interferogram 3 failed
)

echo.
echo ========================================
echo Processing Pair 4/5
echo ========================================
%GPT% %GRAPH% ^
  -Pmaster=%INDIR%\S1C_IW_SLC__1SDV_20251008T013439_20251008T013506_004464_008D6F_B190.SAFE.zip ^
  -Pslave=%INDIR%\S1A_IW_SLC__1SDV_20251008T142348_20251008T142415_061335_07A723_19F9.SAFE.zip ^
  -Poutput=%OUTDIR%\interferogram_04.tif ^
  -c 16G -q 8

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Interferogram 4 completed
) else (
    echo [ERROR] Interferogram 4 failed
)

echo.
echo ========================================
echo Processing Pair 5/5
echo ========================================
%GPT% %GRAPH% ^
  -Pmaster=%INDIR%\S1A_IW_SLC__1SDV_20251008T142348_20251008T142415_061335_07A723_19F9.SAFE.zip ^
  -Pslave=%INDIR%\S1A_IW_SLC__1SDV_20251008T142503_20251008T142520_061335_07A723_4BC6.SAFE.zip ^
  -Poutput=%OUTDIR%\interferogram_05.tif ^
  -c 16G -q 8

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Interferogram 5 completed
) else (
    echo [ERROR] Interferogram 5 failed
)

echo.
echo ========================================
echo PROCESSING COMPLETE
echo ========================================
echo.
echo Outputs saved to:
echo %OUTDIR%
echo.
echo Next steps:
echo 1. python multi_resolution_fusion.py --output california_with_insar
echo 2. python validate_against_known_features.py california_with_insar.tif
echo 3. python create_visualization.py california_with_insar.tif
echo.
pause