param(
    [string]$Source = (Join-Path $PSScriptRoot '..\feeed enhancement\samples\fpv\real_fpv_user_flight.mp4'),
    [ValidateSet(1, 2, 3)][int]$Mode = 1,
    [string]$OsdProfile = (Join-Path $PSScriptRoot 'profiles\user_flight_osd.json')
)
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'experiment_viewer.py') --source $Source --mode $Mode --profile $OsdProfile
if ($LASTEXITCODE -ne 0) { throw 'AURA Experiment 1 failed. See the error above.' }
