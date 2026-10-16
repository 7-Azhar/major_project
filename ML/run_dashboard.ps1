param(
    [int]$Port = 8765,
    [string]$Device = 'auto',
    [ValidateSet('auto', 'fp32')][string]$Precision = 'auto',
    [string]$PythonPath = '',
    [switch]$OpenBrowser
)
$PythonPath = & (Join-Path $PSScriptRoot 'resolve_python.ps1') -PythonPath $PythonPath
Write-Host "Open http://127.0.0.1:$Port in your browser. Keep this launcher running; Ctrl+C stops it."
$browserArguments = @()
if ($OpenBrowser) { $browserArguments = @('--open-browser') }
& $PythonPath (Join-Path $PSScriptRoot 'dashboard.py') --port $Port --device $Device --precision $Precision @browserArguments
if ($LASTEXITCODE -ne 0) { throw 'AURA dashboard stopped with an error. See the message above.' }
