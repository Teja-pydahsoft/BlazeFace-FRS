@echo off
echo Starting BlazeFace-FRS System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the application
echo Starting application...
python main.py

pause
