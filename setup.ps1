<#
.SYNOPSIS
    One-time setup: creates the .venv virtual environment, installs the
    packages and runs the tests.

.DESCRIPTION
    Safe to run again (for example after pulling updates): an existing .venv is
    reused and the packages are brought up to date.

.PARAMETER Locked
    Install the exact versions from requirements-lock.txt instead of the
    minimum versions in requirements.txt.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param([switch]$Locked)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PythonVersion = "3.14"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Invoke-Step([string]$Description, [scriptblock]$Command) {
    Write-Host "`n==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit code $LASTEXITCODE)." }
}

if (-not (Test-Path $VenvPython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "The Python launcher 'py' was not found. Install Python $PythonVersion from python.org " +
              "(tick 'Add python.exe to PATH'), open a new terminal and run this script again."
    }
    Invoke-Step "Creating the virtual environment with Python $PythonVersion" { py "-$PythonVersion" -m venv .venv }
}

$Requirements = if ($Locked) { "requirements-lock.txt" } else { "requirements.txt" }
Invoke-Step "Updating pip" { & $VenvPython -m pip install --upgrade pip --quiet }
Invoke-Step "Installing packages from $Requirements" { & $VenvPython -m pip install -r $Requirements --quiet }
Invoke-Step "Running the tests" { & $VenvPython -m pytest -q -p no:cacheprovider }

Write-Host "`nSetup complete. Start the app with:  powershell -ExecutionPolicy Bypass -File .\run.ps1" -ForegroundColor Green
