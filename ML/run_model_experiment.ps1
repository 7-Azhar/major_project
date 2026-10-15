param(
    [ValidateSet(1, 2, 3, 4)][int]$Experiment = 1,
    [string]$Source = (Join-Path $PSScriptRoot '..\feeed enhancement\samples\fpv\real_fpv_user_flight.mp4'),
    [ValidateSet(1, 2, 3)][int]$Mode = 1,
    [string]$OsdProfile = (Join-Path $PSScriptRoot 'profiles\user_flight_osd.json')
)
$auraRegistry = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'experiments.json') -Raw | ConvertFrom-Json
$auraExperiment = $auraRegistry.PSObject.Properties[[string]$Experiment].Value
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'experiment_viewer.py') --source $Source --mode $Mode --profile $OsdProfile --weights (Join-Path $PSScriptRoot $auraExperiment.weights) --title $auraExperiment.name
if ($LASTEXITCODE -ne 0) { throw 'AURA experiment failed. See the error above.' }
