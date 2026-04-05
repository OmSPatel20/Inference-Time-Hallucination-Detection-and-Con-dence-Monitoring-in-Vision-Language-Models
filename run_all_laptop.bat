@echo off
REM run_all_laptop.bat — Run on Windows laptop with RTX 4060 (8GB VRAM)
REM
REM Usage:
REM   run_all_laptop.bat          Full run (~4-6 hours)
REM   run_all_laptop.bat quick    Quick test (~20 min)

set MODE=%1
if "%MODE%"=="" set MODE=full

echo ==============================================
echo VLM Hallucination Detection - Laptop Mode
echo Hardware: RTX 4060 (8GB) + INT4 quantization
echo Mode: %MODE%
echo ==============================================

if "%MODE%"=="quick" goto quick
if "%MODE%"=="full" goto full
echo Unknown mode: %MODE%
echo Usage: run_all_laptop.bat [quick^|full]
goto end

:quick
echo.
echo === QUICK TEST (~20 min) ===
echo.

echo [1/4] Downloading data...
python scripts/01_download_data.py
if errorlevel 1 goto fail

echo [2/4] Testing basic inference...
python scripts/02_basic_inference.py
if errorlevel 1 goto fail

echo [3/4] Testing entropy detector...
python scripts/04_entropy_detector.py
if errorlevel 1 goto fail

echo [4/4] Quick POPE eval (200 samples)...
python scripts/07_pope_evaluation.py --split random --max_samples 200 --no_contrastive
if errorlevel 1 goto fail

echo.
echo === QUICK TEST DONE ===
echo If everything passed, run: run_all_laptop.bat full
goto end

:full
echo.
echo === FULL RUN (~4-6 hours) ===
echo.

echo [1/7] Downloading data...
python scripts/01_download_data.py
if errorlevel 1 goto fail

echo [2/7] Verifying inference...
python scripts/02_basic_inference.py
if errorlevel 1 goto fail

echo [3/7] Benchmarking throughput...
python scripts/03_batched_pipeline.py
if errorlevel 1 goto fail

echo [4/7] POPE evaluation (1000 samples x 3 splits)...
python scripts/10_experiment_runner.py
if errorlevel 1 goto fail

echo [5/7] CHAIR evaluation (200 captions)...
python scripts/08_chair_evaluation.py
if errorlevel 1 goto fail

echo [6/7] Confidence drift monitoring (1000 inferences)...
python scripts/09_confidence_monitor.py
if errorlevel 1 goto fail

echo [7/7] Generating analysis plots...
python scripts/11_analysis_plots.py
if errorlevel 1 goto fail

echo.
echo ==============================================
echo FULL RUN COMPLETE
echo Results:  outputs\
echo Plots:    outputs\plots\
echo ==============================================
goto end

:fail
echo.
echo ERROR: Last command failed. Check the output above.
pause
goto end

:end
