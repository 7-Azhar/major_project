@echo off
setlocal
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Install 64-bit Python 3.11 with the Python launcher from python.org, then try again.
  pause
  exit /b 1
)
py -3.11 setup_aura.py
if errorlevel 1 (
  echo Setup did not finish. Read the error above and the troubleshooting section in README.md.
  pause
  exit /b 1
)
echo Setup complete. Double-click Start AURA.cmd to open the dashboard.
pause
