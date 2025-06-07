@echo off
echo ========================================
echo TTS Processing Script
echo ========================================
echo.
echo This script will convert preprocessed documents
echo from the preprocessed folder to audio files.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if preprocessed folder exists and has files
if not exist preprocessed (
    echo ERROR: preprocessed folder not found
    echo Please run preprocessing first using run_preprocessing.bat
    echo or python src\preprocess_documents.py
    pause
    exit /b 1
)

REM Count files in preprocessed folder
set count=0
for %%f in (preprocessed\*.*) do set /a count+=1
if %count%==0 (
    echo ERROR: No files found in preprocessed folder
    echo Please run preprocessing first using run_preprocessing.bat
    echo or python src\preprocess_documents.py
    pause
    exit /b 1
)

echo Found %count% file(s) in preprocessed folder
echo Starting TTS processing...
echo.

REM Run the TTS script with verbose output
python src\run_tts.py --verbose %*

echo.
echo ========================================
echo TTS processing completed!
echo Check the 'audios' folder for output files.
echo ========================================
pause
