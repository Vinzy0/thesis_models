@echo off
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0launch_webcam.pyw"
    exit /b
)
pyw -3.11 "%~dp0launch_webcam.pyw"
if errorlevel 1 pause
