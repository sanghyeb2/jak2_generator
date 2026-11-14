# Explicit dependency installation only. Does not train or generate.
$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $packageRoot '.venv/Scripts/python.exe'))) {
    py -3.11 -m venv (Join-Path $packageRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python3.11 venv creation failed' }
}
$packagePython = Join-Path $packageRoot '.venv/Scripts/python.exe'
& $packagePython -m pip install -r (Join-Path $packageRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Host 'Installed. The local data/assets/runtime bundle is also required.'
