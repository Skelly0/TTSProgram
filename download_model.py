#!/usr/bin/env python3
"""
Script to manually download and cache the Kokoro TTS model.
Run this if you're having issues with automatic model downloading.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required dependencies:")
    print("pip install huggingface_hub torch")
    sys.exit(1)

def download_kokoro_model():
    """Download the Kokoro TTS model to local cache."""
    model_repos = [
        "hexgrad/Kokoro-82M",
        "onnx-community/Kokoro-82M-v1.0-ONNX"
    ]
    
    print("🔧 Downloading Kokoro TTS models...")
    
    for repo_id in model_repos:
        try:
            print(f"📥 Downloading {repo_id}...")
            cache_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=None,  # Use default cache directory
                resume_download=True
            )
            print(f"✅ Successfully downloaded {repo_id} to {cache_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")
            continue
    
    print("❌ Failed to download any Kokoro models")
    return False

def check_model_cache():
    """Check if models are already cached."""
    try:
        from transformers import AutoModel
        model_repos = ["hexgrad/Kokoro-82M", "onnx-community/Kokoro-82M-v1.0-ONNX"]
        
        for repo_id in model_repos:
            try:
                # Try to load model info without downloading
                print(f"🔍 Checking cache for {repo_id}...")
                # This will use cached version if available
                cache_dir = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=None,
                    local_files_only=True  # Only use cached files
                )
                print(f"✅ Found cached model: {repo_id} at {cache_dir}")
                return True
            except Exception:
                continue
        
        print("ℹ️  No cached models found")
        return False
    except Exception as e:
        print(f"⚠️  Error checking cache: {e}")
        return False

def clear_model_cache():
    """Clear the Hugging Face model cache."""
    try:
        from huggingface_hub import scan_cache_dir
        
        print("🧹 Scanning cache directory...")
        cache_info = scan_cache_dir()
        
        kokoro_repos = [repo for repo in cache_info.repos if "kokoro" in repo.repo_id.lower()]
        
        if not kokoro_repos:
            print("ℹ️  No Kokoro models found in cache")
            return
        
        print(f"Found {len(kokoro_repos)} Kokoro model(s) in cache:")
        for repo in kokoro_repos:
            print(f"  - {repo.repo_id} ({repo.size_on_disk_str})")
        
        response = input("Do you want to clear these cached models? (y/N): ")
        if response.lower() in ['y', 'yes']:
            for repo in kokoro_repos:
                try:
                    repo.delete()
                    print(f"✅ Cleared cache for {repo.repo_id}")
                except Exception as e:
                    print(f"❌ Failed to clear {repo.repo_id}: {e}")
        else:
            print("Cache clearing cancelled")
            
    except Exception as e:
        print(f"❌ Error managing cache: {e}")

def main():
    """Main function with menu options."""
    print("Kokoro TTS Model Manager")
    print("=" * 30)
    print("1. Check cached models")
    print("2. Download models")
    print("3. Clear model cache")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                check_model_cache()
            elif choice == "2":
                if download_kokoro_model():
                    print("🎉 Model download completed successfully!")
                else:
                    print("❌ Model download failed")
            elif choice == "3":
                clear_model_cache()
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid option. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
