@echo off
call "%~dp0config.bat"
if errorlevel 1 exit /b %errorlevel%

if "%PARATERA_API_KEY%"=="" (
  echo PARATERA_API_KEY is not set in this terminal session.
  exit /b 2
)
if /I not "%DATASET%"=="wtq" if /I not "%DATASET%"=="tabfact" (
  echo DATASET must be wtq or tabfact.
  exit /b 2
)
if not exist "%PROJECT_DIR%\scripts\run_paper1.py" (
  echo Project runner was not found: %PROJECT_DIR%\scripts\run_paper1.py
  exit /b 2
)
where conda >nul 2>&1
if errorlevel 1 (
  echo conda was not found on PATH.
  exit /b 2
)
conda run -n "%CONDA_ENV%" --no-capture-output python --version
if errorlevel 1 exit /b %errorlevel%
exit /b 0
