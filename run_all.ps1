# Single entry point: environment -> fixed assets -> preflight -> full training pipeline.
param(
    [string]$RunName = 'final_reproduction_20261120',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
& (Join-Path $packageRoot 'setup.ps1')
$packagePython = Join-Path $packageRoot '.venv/Scripts/python.exe'
$prepareArgs = @((Join-Path $packageRoot 'prepare_assets.py'), '--download')
$privateArchive = Join-Path $packageRoot 'private_inputs.zip'
if (Test-Path -LiteralPath $privateArchive) { $prepareArgs += @('--data-archive', $privateArchive) }
& $packagePython @prepareArgs
if ($LASTEXITCODE -ne 0) { throw 'Data or initial asset preparation failed' }
& $packagePython (Join-Path $packageRoot 'preflight.py')
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; no training started' }
if ($CheckOnly) { Write-Host 'Preflight passed. No training or generation started.'; exit 0 }
& $packagePython (Join-Path $packageRoot 'main.py') --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw 'Pipeline stopped. Inspect the error before restarting.' }
Write-Host (Join-Path $packageRoot "runs/$RunName/selection/neural_submission.csv")
