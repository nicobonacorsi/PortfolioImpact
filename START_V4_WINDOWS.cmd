@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 is required. Install it and try again.
        pause
        exit /b 1
    )
)

.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements-lock.txt
if errorlevel 1 (
    echo Could not install the pinned dependencies.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m castle_risk.codespaces_server --open
if errorlevel 1 (
    echo Portfolio Impact could not start. Review the error above.
    pause
    exit /b 1
)
