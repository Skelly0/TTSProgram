#!/usr/bin/env python3
"""
Run document preprocessing and TTS using settings from config.py
"""

import os
import sys
import subprocess
from pathlib import Path

def load_config():
    """Load configuration from config.py"""
    try:
        import config
        return config
    except ImportError:
        print("❌ config.py not found. Please create it from config.example.py")
        sys.exit(1)

def main():
    """Main function to run preprocessing and TTS using config.py settings."""
    print("🎯 Document to Audiobook Converter - Using config.py")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Display current configuration
    print(f"🤖 LLM Model: {config.LLM_MODEL}")
    print(f"🗣️  Voice: {config.VOICE}")
    print(f"🌍 Language: {'British English' if config.LANG_CODE == 'b' else 'American English'}")
    print(f"⚡ Speed: {config.SPEED}x")
    print(f"📂 Documents Dir: {config.DOCUMENTS_DIR}")
    print(f"🎵 Audios Dir: {config.AUDIOS_DIR}")
    print(f"🔧 LLM Preprocessing: {'Enabled' if config.ENABLE_LLM_PREPROCESSING else 'Disabled'}")
    print()
    
    # Check if API key is set
    api_key = getattr(config, 'OPENROUTER_API_KEY', None) or os.getenv('OPENROUTER_API_KEY')
    if config.ENABLE_LLM_PREPROCESSING and not api_key:
        print("❌ OpenRouter API key is required for LLM preprocessing")
        print("💡 Set OPENROUTER_API_KEY in config.py or as environment variable")
        sys.exit(1)
    
    try:
        if config.ENABLE_LLM_PREPROCESSING:
            # Step 1: Run preprocessing
            print("🔄 Step 1: Running document preprocessing...")
            preprocess_cmd = [
                "python", "preprocess_documents.py",
                "--documents-dir", config.DOCUMENTS_DIR,
                "--preprocessed-dir", "preprocessed",
                "--llm-model", config.LLM_MODEL,
                "--max-chunk-size", str(config.MAX_CHUNK_SIZE)
            ]
            
            if config.VERBOSE_LOGGING:
                preprocess_cmd.append("--verbose")
            
            if api_key:
                preprocess_cmd.extend(["--openrouter-api-key", api_key])
            
            print(f"Running: {' '.join(preprocess_cmd)}")
            result = subprocess.run(preprocess_cmd, check=True)
            print("✅ Preprocessing completed successfully!")
            print()
        
        # Step 2: Run TTS
        print("🔄 Step 2: Running TTS conversion...")
        tts_cmd = [
            "python", "run_tts.py",
            "--preprocessed-dir", "preprocessed" if config.ENABLE_LLM_PREPROCESSING else config.DOCUMENTS_DIR,
            "--audios-dir", config.AUDIOS_DIR,
            "--voice", config.VOICE,
            "--lang-code", config.LANG_CODE,
            "--speed", str(config.SPEED)
        ]
        
        if config.VERBOSE_LOGGING:
            tts_cmd.append("--verbose")
        
        print(f"Running: {' '.join(tts_cmd)}")
        result = subprocess.run(tts_cmd, check=True)
        print("✅ TTS conversion completed successfully!")
        
        print()
        print("🎉 Document to audiobook conversion completed!")
        print(f"📁 Check the '{config.AUDIOS_DIR}' directory for your audiobooks")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
