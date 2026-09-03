@echo off
setlocal

REM ===== User-configurable parameters =====
set "DATASET=wtq"
set "START_INDEX=0"
set "END_INDEX=2"
REM Relative output path below second_paper\outputs\.
set "OUTPUT_SUBDIR=windows_local_0_5/wtq_baseline"

if "%PARATERA_API_KEY%"=="" (
  echo PARATERA_API_KEY is not set in this terminal session.
  exit /b 2
)

if /I not "%DATASET%"=="wtq" if /I not "%DATASET%"=="tabfact" (
  echo DATASET must be wtq or tabfact.
  exit /b 2
)

set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%"
set "PYTHONPATH=%PROJECT_DIR%\..;%PYTHONPATH%"
set "MODEL_ID=DeepSeek-V3.2"

conda run -n work2 --no-capture-output python "%PROJECT_DIR%\scripts\run_paper1.py" ^
  --dataset "%DATASET%" --mode baseline --experiment "%OUTPUT_SUBDIR%" ^
  --start "%START_INDEX%" --end "%END_INDEX%" --all-questions --model "%MODEL_ID%" ^
  --temperature 0.0 --timeout 180 --overwrite
