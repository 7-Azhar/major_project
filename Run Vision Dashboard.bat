@echo off
setlocal
cd /d "%~dp0"
set "AURA_PYTHON=ML\.venv\Scripts\python.exe"
if not exist "%AURA_PYTHON%" set "AURA_PYTHON=ML\.venv-portable\Scripts\python.exe"
if not exist "%AURA_PYTHON%" (
  echo Run Setup AURA.cmd first.
  pause
  exit /b 1
)
echo Starting Vision dashboard only. Keep this window open.
"%AURA_PYTHON%" "ML\launch_dashboard.py" %*
if errorlevel 1 (
  pause
  exit /b 1
)
