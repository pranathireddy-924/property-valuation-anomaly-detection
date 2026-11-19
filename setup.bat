@echo off
cd /d "%~dp0"
echo Setting up the project. This takes a few minutes the first time.
echo.
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if not exist .venv\Scripts\python.exe (
    echo Python was not found.
    echo Install Python 3.12 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during the install. Then run this file again.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installing the packages failed. Check the internet connection and run this file again.
    pause
    exit /b 1
)
echo.
echo Setup finished. Double-click start_dashboard.bat to open the dashboard.
pause
