param(
    [int]$Port = 8765,
    [string]$Device = 'auto',
    [ValidateSet('auto', 'fp32')][string]$Precision = 'auto',
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Python environment missing. See ML/PORTABLE_SETUP.md.' }
Write-Host "Open http://127.0.0.1:$Port in your browser. Keep this launcher running; Ctrl+C stops it."
& $PythonPath (Join-Path $PSScriptRoot 'dashboard.py') --port $Port --device $Device --precision $Precision
if ($LASTEXITCODE -ne 0) { throw 'AURA dashboard stopped with an error. See the message above.' }
