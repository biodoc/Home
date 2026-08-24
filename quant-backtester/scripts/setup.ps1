<#
.SYNOPSIS
    Sets up the Quant Signal Backtester on Windows.

.DESCRIPTION
    The PowerShell equivalent of `make setup`. Creates a virtual environment,
    installs dependencies, installs the package, and runs the health check.

    Every step is a plain command you could type yourself -- nothing is hidden.

.EXAMPLE
    .\scripts\setup.ps1

.EXAMPLE
    # If PowerShell refuses to run the script ("running scripts is disabled"),
    # allow it for this window only -- this does not change machine settings:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\scripts\setup.ps1
#>

[CmdletBinding()]
param(
    # Python launcher to use. The default asks the Windows py launcher for the
    # newest 3.x it knows about.
    [string]$Python = "py -3"
)

$ErrorActionPreference = "Stop"

# Run from the project root regardless of where this was invoked.
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    Write-Host "Setting up in $projectRoot" -ForegroundColor Cyan
    Write-Host ""

    # -- locate a usable Python -------------------------------------------
    $pythonExe, $pythonArgs = $Python.Split(" ", 2)
    if (-not (Get-Command $pythonExe -ErrorAction SilentlyContinue)) {
        # The py launcher ships with the python.org installer but not with
        # Microsoft Store builds, so fall back before giving up.
        if (Get-Command "python" -ErrorAction SilentlyContinue) {
            $pythonExe = "python"; $pythonArgs = ""
        } else {
            throw "No Python found. Install Python 3.11+ from https://python.org/downloads (tick 'Add python.exe to PATH')."
        }
    }

    $versionOutput = & $pythonExe $pythonArgs --version 2>&1
    Write-Host "Using $versionOutput"

    if ($versionOutput -match "(\d+)\.(\d+)") {
        $major = [int]$Matches[1]; $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
            throw "Python $major.$minor is too old. This project needs 3.11+ because the pinned pandas and numpy do not build on older versions. Install a newer Python from https://python.org/downloads and re-run."
        }
    }

    # -- virtual environment ----------------------------------------------
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Host "Reusing existing .venv"
    } else {
        Write-Host "Creating .venv ..."
        & $pythonExe $pythonArgs -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
    }

    # -- dependencies ------------------------------------------------------
    # Calling .venv\Scripts\python.exe directly avoids needing to "activate",
    # which sidesteps PowerShell execution-policy problems entirely.
    Write-Host "Installing dependencies ..."
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the pip output above." }

    Write-Host "Installing the qsb package ..."
    & $venvPython -m pip install -e . --quiet
    if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }

    # -- health check ------------------------------------------------------
    Write-Host ""
    & $venvPython backtest.py --doctor

    Write-Host ""
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host "Run a backtest with:" -ForegroundColor Green
    Write-Host "    .venv\Scripts\python.exe backtest.py --tickers UAMY SMR --signals mean_reversion"
    Write-Host ""
    Write-Host "Or activate the environment first, and drop the long path:"
    Write-Host "    .venv\Scripts\Activate.ps1"
    Write-Host "    python backtest.py --tickers UAMY SMR --signals mean_reversion"
}
catch {
    Write-Host ""
    Write-Host "Setup failed: $_" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
