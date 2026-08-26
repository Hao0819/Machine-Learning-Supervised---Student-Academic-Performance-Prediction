@echo off
REM ===================================================================
REM  Student Academic Performance Prediction - web interface launcher
REM  Double-click this file to start the app, then open the address
REM  shown in the window (usually http://127.0.0.1:5000).
REM  Keep this window open while using the app.
REM ===================================================================

cd /d "%~dp0"

set PYTHON=python
python --version >nul 2>&1
if errorlevel 1 set PYTHON="%USERPROFILE%\anaconda3\python.exe"

echo Starting the Student Performance Prediction web app...
echo.

%PYTHON% app.py

echo.
echo The server has stopped.
echo If you saw an error above, install the requirements first:
echo     pip install -r requirements.txt
echo.
pause
