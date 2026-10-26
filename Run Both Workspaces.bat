@echo off
setlocal
cd /d "%~dp0"
if not exist "ground_station\.venv\Scripts\python.exe" (
  echo Run Setup AURA.cmd and then Setup Ground Station.cmd first.
  pause
  exit /b 1
)
echo Starting Detection workspace and Vision dashboard. Keep this window open.
"ground_station\.venv\Scripts\python.exe" "ground_station\launch.py" --open-both %*
if errorlevel 1 (
  pause
  exit /b 1
)
