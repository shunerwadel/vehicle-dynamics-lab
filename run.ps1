<#
.SYNOPSIS
    Starts the Vehicle Dynamics Lab web app and opens it in the browser.

.PARAMETER Port
    Port to serve the app on (default 8501). Use another one if 8501 is busy.

.PARAMETER NoBrowser
    Don't open the browser automatically.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run.ps1
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -Port 8502

    Stop the app with Ctrl+C.
#>
param(
    [int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "No .venv found. Run setup first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
}

$Url = "http://localhost:$Port"
if (-not $NoBrowser) {
    # .streamlit/config.toml runs Streamlit headless, so open the browser once the app answers.
    Start-Job -ArgumentList $Url -ScriptBlock {
        param($Url)
        for ($i = 0; $i -lt 120; $i++) {
            try {
                Invoke-WebRequest "$Url/_stcore/health" -UseBasicParsing -TimeoutSec 1 | Out-Null
                Start-Process $Url
                return
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } | Out-Null
}

Write-Host "Starting Vehicle Dynamics Lab at $Url (press Ctrl+C to stop)..." -ForegroundColor Green
& $VenvPython -m streamlit run app.py --server.port $Port
