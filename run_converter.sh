#!/bin/bash

echo "Document to Audiobook Converter"
echo "================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.7 or higher"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.7 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if the main script exists
if [ ! -f "document_to_audiobook.py" ]; then
    echo "Error: document_to_audiobook.py not found"
    echo "Please make sure you're running this from the correct directory"
    exit 1
fi

# Create directories if they don't exist
if [ ! -d "documents" ]; then
    echo "Creating documents directory..."
    mkdir documents
fi

if [ ! -d "audios" ]; then
    echo "Creating audios directory..."
    mkdir audios
fi

# Check if there are any documents to process
if ! ls documents/*.txt documents/*.md documents/*.rtf 1> /dev/null 2>&1; then
    echo
    echo "No documents found in the 'documents' folder."
    echo "Please add some .txt, .md, or .rtf files to the 'documents' folder."
    echo
    exit 1
fi

echo "Starting conversion..."
echo
$PYTHON_CMD document_to_audiobook.py

echo
echo "Conversion completed!"
echo "Check the 'audios' folder for your audiobook files."
echo
