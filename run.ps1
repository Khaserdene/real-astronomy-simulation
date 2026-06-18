# Astronomy Simulator launcher (PowerShell)
#   Right-click -> Run with PowerShell, or:  ./run.ps1
#   First run sets up the venv. Pass --cli ... to run the headless CLI.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[setup] Creating Python 3.10 virtual environment..."
    py -3.10 -m venv .venv
    if (-not (Test-Path $py)) {
        Write-Host "ERROR: Python 3.10 not found. Install it from python.org and retry." -ForegroundColor Red
        Read-Host "Press Enter to exit"; exit 1
    }
    Write-Host "[setup] Installing dependencies (a few minutes)..."
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}

if ($args.Count -gt 0 -and $args[0] -eq "--cli") {
    Write-Host "[run] Astronomy Simulator (CLI)..."
    & $py cli.py @($args[1..($args.Count - 1)])
} else {
    Write-Host "[run] Launching Astronomy Simulator GUI..."
    & $py -m gui.main @args
}
