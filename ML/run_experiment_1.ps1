param(
    [string]$Source = (Join-Path $PSScriptRoot '..\feeed enhancement\samples\fpv\real_fpv_user_flight.mp4'),
    [ValidateSet(1, 2, 3)][int]$Mode = 1,
    [string]$OsdProfile = (Join-Path $PSScriptRoot 'profiles\user_flight_osd.json'),
    [string]$Device = 'auto',
    [string]$PythonPath = ''
)
$auraPython = & (Join-Path $PSScriptRoot 'resolve_python.ps1') -PythonPath $PythonPath
& $auraPython (Join-Path $PSScriptRoot 'experiment_viewer.py') --source $Source --mode $Mode --profile $OsdProfile --device $Device
if ($LASTEXITCODE -ne 0) { throw 'AURA Experiment 1 failed. See the error above.' }
