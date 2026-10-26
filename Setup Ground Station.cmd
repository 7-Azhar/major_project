@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv "ground_station\.venv"
if errorlevel 1 goto failed
"ground_station\.venv\Scripts\python.exe" -m pip install -r "ground_station\requirements-lock.txt"
if errorlevel 1 goto failed
"ground_station\.venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo Ground station setup complete. Run Start Ground Station.cmd.
pause
exit /b 0
:failed
echo Setup failed. See ground_station\README.md.
pause
exit /b 1
