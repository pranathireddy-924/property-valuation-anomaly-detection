@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Run setup.bat first.
    pause
    exit /b 1
)
echo The dashboard will open in your browser. Keep this window open while you use it.
.venv\Scripts\python.exe -m streamlit run app.py
pause
