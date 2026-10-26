@echo off
setlocal
cd /d "%~dp0"
set "AURA_PYTHON=ML\.venv\Scripts\python.exe"
if not exist "%AURA_PYTHON%" set "AURA_PYTHON=ML\.venv-portable\Scripts\python.exe"
if not exist "%AURA_PYTHON%" (
  echo AURA's Python environment is missing. See ML\PORTABLE_SETUP.md.
  pause
  exit /b 1
)
echo Starting AURA. Keep this window open. Ctrl+C stops the server.
"%AURA_PYTHON%" "ML\launch_dashboard.py" %*
if errorlevel 1 pause
