@echo off
echo ========================================
echo Document Preprocessing Script
echo ========================================
echo.
echo This script will preprocess documents using LLM
echo and save them to the preprocessed folder.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found
    echo Please create a .env file with your OPENROUTER_API_KEY
    echo You can copy .env.example and modify it.
    echo.
    pause
)

echo Starting document preprocessing...
echo.

REM Run the preprocessing script with verbose output
python preprocess_documents.py --verbose %*

echo.
echo ========================================
echo Preprocessing completed!
echo Check the 'preprocessed' folder for output files.
echo ========================================
pause
