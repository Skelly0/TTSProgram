#!/bin/bash

echo "========================================"
echo "Document Preprocessing Script"
echo "========================================"
echo
echo "This script will preprocess documents using LLM"
echo "and save them to the preprocessed folder."
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

# Check if .env file exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found"
    echo "Please create a .env file with your OPENROUTER_API_KEY"
    echo "You can copy .env.example and modify it."
    echo
    read -p "Press Enter to continue..."
fi

echo "Starting document preprocessing..."
echo

# Run the preprocessing script with verbose output
$PYTHON_CMD src/preprocess_documents.py --verbose "$@"

echo
echo "========================================"
echo "Preprocessing completed!"
echo "Check the 'preprocessed' folder for output files."
echo "========================================"
