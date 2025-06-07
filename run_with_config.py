#!/usr/bin/env python3
"""
Main entry point for Document to Audiobook Converter
This script provides easy access to the converter using config.py settings.
"""

import os
import sys
from pathlib import Path

def main():
    """Main entry point that delegates to the actual script in src/"""
    # Add src directory to Python path
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))
    
    # Change to the project root directory
    os.chdir(Path(__file__).parent)
    
    # Import and run the actual script
    try:
        from run_with_config import main as run_main
        run_main()
    except ImportError as e:
        print(f"❌ Error importing run_with_config from src/: {e}")
        print("💡 Make sure all required dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
