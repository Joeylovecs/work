@echo off
setlocal

if "%PARATERA_API_KEY%"=="" (
  echo PARATERA_API_KEY is not set in this terminal session.
  exit /b 2
)

set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%"
set "PYTHONPATH=%PROJECT_DIR%\..;%PYTHONPATH%"
set "MODEL_ID=DeepSeek-V3.2"

conda run -n work2 --no-capture-output python "%PROJECT_DIR%\scripts\run_paper1.py" ^
  --dataset wtq --mode baseline --experiment windows_local_0_5/wtq_baseline ^
  --start 0 --end 5 --all-questions --model "%MODEL_ID%" ^
  --temperature 0.0 --timeout 180 --overwrite
if errorlevel 1 exit /b %errorlevel%

conda run -n work2 --no-capture-output python "%PROJECT_DIR%\scripts\run_paper1.py" ^
  --dataset tabfact --mode baseline --experiment windows_local_0_5/tabfact_baseline ^
  --start 0 --end 5 --all-questions --model "%MODEL_ID%" ^
  --temperature 0.0 --timeout 180 --overwrite
