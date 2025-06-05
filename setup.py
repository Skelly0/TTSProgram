#!/usr/bin/env python3
"""
Setup script for Document to Audiobook Converter
Helps users install dependencies and set up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def check_espeak():
    """Check if espeak is available."""
    try:
        subprocess.run(["espeak", "--version"], capture_output=True, check=True)
        print("✓ eSpeak is installed and available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ eSpeak is not installed or not in PATH")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["documents", "audios"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")
    return True

def main():
    """Main setup function."""
    print("Document to Audiobook Converter - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependency installation failed. You may need to install them manually:")
        print("pip install kokoro==0.9.4 soundfile torch numpy")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check eSpeak
    if not check_espeak():
        print("\n⚠️  eSpeak is required but not found. Please install it:")
        print("\nWindows:")
        print("  - Download from: http://espeak.sourceforge.net/download.html")
        print("  - Or use chocolatey: choco install espeak")
        print("\nmacOS:")
        print("  brew install espeak")
        print("\nLinux (Ubuntu/Debian):")
        print("  sudo apt-get install espeak espeak-data")
        print("\nAfter installing eSpeak, you can run the converter.")
    else:
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now:")
        print("1. Add text documents to the 'documents' folder")
        print("2. Run: python document_to_audiobook.py")
        print("3. Find your audiobooks in the 'audios' folder")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
