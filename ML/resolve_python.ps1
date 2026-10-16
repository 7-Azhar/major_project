param([string]$PythonPath = '')
if ($PythonPath) {
    if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Python executable not found: $PythonPath" }
    (Resolve-Path -LiteralPath $PythonPath).Path
    return
}
foreach ($candidate in @('.venv\Scripts\python.exe', '.venv-portable\Scripts\python.exe')) {
    $fullPath = Join-Path $PSScriptRoot $candidate
    if (Test-Path -LiteralPath $fullPath) { $fullPath; return }
}
throw 'AURA environment missing. Double-click Setup AURA.cmd in the project folder first.'
