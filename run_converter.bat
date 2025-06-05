@echo off
echo Document to Audiobook Converter
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM Check if the main script exists
if not exist "document_to_audiobook.py" (
    echo Error: document_to_audiobook.py not found
    echo Please make sure you're running this from the correct directory
    pause
    exit /b 1
)

REM Check if documents directory exists
if not exist "documents" (
    echo Creating documents directory...
    mkdir documents
)

REM Check if audios directory exists
if not exist "audios" (
    echo Creating audios directory...
    mkdir audios
)

REM Check if there are any documents to process
dir /b documents\*.txt documents\*.md documents\*.rtf >nul 2>&1
if errorlevel 1 (
    echo.
    echo No documents found in the 'documents' folder.
    echo Please add some .txt, .md, or .rtf files to the 'documents' folder.
    echo.
    pause
    exit /b 1
)

echo Starting conversion...
echo.
python document_to_audiobook.py

echo.
echo Conversion completed!
echo Check the 'audios' folder for your audiobook files.
echo.
pause
