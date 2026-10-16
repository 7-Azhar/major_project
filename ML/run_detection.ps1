param(
    [string]$Source = (Join-Path $PSScriptRoot '..\feeed enhancement\samples\fpv\real_fpv_user_flight.mp4'),
    [int]$ImageSize = 640,
    [int]$MaxFrames = 0,
    [double]$Confidence = 0.25,
    [string]$OsdProfile = '',
    [switch]$Show,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$PythonPath = ''
)
$auraPython = & (Join-Path $PSScriptRoot 'resolve_python.ps1') -PythonPath $PythonPath
$auraOutput = Join-Path $PSScriptRoot ('outputs\review_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$auraArguments = @((Join-Path $PSScriptRoot 'detect_video.py'), '--source', $Source,
    '--output', $auraOutput, '--imgsz', $ImageSize, '--max-frames', $MaxFrames, '--conf', $Confidence, '--device', $Device)
if ($OsdProfile) { $auraArguments += @('--osd-profile', $OsdProfile) }
if ($Show) { $auraArguments += '--show' }
& $auraPython @auraArguments
if ($LASTEXITCODE -ne 0) { throw 'AURA detection failed. See the error above.' }
Write-Host "Review files saved to $auraOutput"
