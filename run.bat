@echo off
REM ====================================================================
REM  Astronomy Simulator launcher (Windows)
REM  Double-click to start the GUI. First run sets up the environment.
REM  Pass --cli ... to run the headless CLI instead of the GUI.
REM ====================================================================
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

REM --- first-run setup: create the venv and install dependencies ---
if not exist "%PY%" (
    echo [setup] Creating Python 3.10 virtual environment...
    py -3.10 -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.10 was not found.
        echo Install Python 3.10 from https://www.python.org/downloads/
        echo then run this launcher again.
        echo.
        pause
        exit /b 1
    )
    echo [setup] Installing dependencies ^(this can take a few minutes^)...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: dependency installation failed. See messages above.
        pause
        exit /b 1
    )
)

REM --- launch: GUI by default, or pass through CLI args ---
if "%~1"=="--cli" goto cli
echo [run] Launching Astronomy Simulator GUI...
"%PY%" -m gui.main %*
goto done

:cli
shift
echo [run] Astronomy Simulator - CLI mode...
"%PY%" cli.py %1 %2 %3 %4 %5 %6 %7 %8 %9

:done
if errorlevel 1 (
    echo.
    echo The program exited with an error. See the messages above.
    pause
)
endlocal
