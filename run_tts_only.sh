#!/bin/bash

echo "========================================"
echo "TTS Processing Script"
echo "========================================"
echo
echo "This script will convert preprocessed documents"
echo "from the preprocessed folder to audio files."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python and try again."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if preprocessed folder exists and has files
if [ ! -d "preprocessed" ]; then
    echo "ERROR: preprocessed folder not found"
    echo "Please run preprocessing first using ./run_preprocessing.sh"
    echo "or python preprocess_documents.py"
    exit 1
fi

# Count files in preprocessed folder
count=$(find preprocessed -type f | wc -l)
if [ $count -eq 0 ]; then
    echo "ERROR: No files found in preprocessed folder"
    echo "Please run preprocessing first using ./run_preprocessing.sh"
    echo "or python preprocess_documents.py"
    exit 1
fi

echo "Found $count file(s) in preprocessed folder"
echo "Starting TTS processing..."
echo

# Run the TTS script with verbose output
$PYTHON_CMD run_tts.py --verbose "$@"

echo
echo "========================================"
echo "TTS processing completed!"
echo "Check the 'audios' folder for output files."
echo "========================================"
