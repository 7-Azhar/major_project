@echo off
setlocal
cd /d "%~dp0"
if not exist "ground_station\.venv\Scripts\python.exe" (
  echo Run Setup Ground Station.cmd first.
  pause
  exit /b 1
)
"ground_station\.venv\Scripts\python.exe" "ground_station\launch.py"
if errorlevel 1 pause
