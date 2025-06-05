#!/usr/bin/env python3
"""
Example usage script for Document to Audiobook Converter with OpenRouter integration
"""

import os
import subprocess
import sys

def main():
    print("🎯 Document to Audiobook Converter - OpenRouter Integration Examples")
    print("=" * 70)
    
    # Check if API key is available
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    print("\n📋 Available Usage Examples:")
    print("-" * 40)
    
    examples = [
        {
            "name": "Basic conversion with LLM preprocessing",
            "description": "Uses Claude 3.5 Sonnet for optimal quality",
            "command": "python document_to_audiobook.py",
            "requires_api": True
        },
        {
            "name": "Budget-friendly conversion",
            "description": "Uses Claude Haiku for cost-effective processing",
            "command": "python document_to_audiobook.py --llm-model anthropic/claude-3-haiku",
            "requires_api": True
        },
        {
            "name": "Fantasy content optimization",
            "description": "Perfect for your Region documents with complex names",
            "command": "python document_to_audiobook.py --llm-model anthropic/claude-3.5-sonnet --voice af_bella --speed 0.95",
            "requires_api": True
        },
        {
            "name": "Basic conversion (no LLM)",
            "description": "Traditional text processing without AI enhancement",
            "command": "python document_to_audiobook.py --disable-llm-preprocessing",
            "requires_api": False
        },
        {
            "name": "Test OpenRouter integration",
            "description": "Test the LLM preprocessing without full conversion",
            "command": "python test_openrouter.py",
            "requires_api": True
        }
    ]
    
    for i, example in enumerate(examples, 1):
        status = "✅" if not example["requires_api"] or api_key else "🔑"
        print(f"\n{i}. {example['name']} {status}")
        print(f"   {example['description']}")
        print(f"   Command: {example['command']}")
        
        if example["requires_api"] and not api_key:
            print("   ⚠️  Requires OPENROUTER_API_KEY environment variable")
    
    print("\n" + "=" * 70)
    print("🔧 Setup Instructions:")
    print("-" * 25)
    
    if not api_key:
        print("1. Get your OpenRouter API key from: https://openrouter.ai/keys")
        print("2. Set environment variable:")
        print("   export OPENROUTER_API_KEY='your_key_here'")
        print("3. Run any of the examples above")
    else:
        print("✅ OpenRouter API key detected!")
        print("You can run any of the examples above.")
    
    print("\n📚 Documentation:")
    print("- Full documentation: README_OPENROUTER.md")
    print("- Configuration: config.example.py")
    print("- Test integration: python test_openrouter.py")
    
    print("\n💡 Quick Start for Fantasy Documents:")
    print("1. Place your 'Region 1a - Dytikratia.txt' in documents/ folder")
    if api_key:
        print("2. Run: python document_to_audiobook.py")
    else:
        print("2. Set OPENROUTER_API_KEY and run: python document_to_audiobook.py")
    print("3. Find enhanced audiobook in audios/ folder")
    
    print("\n🎉 Ready to convert your fantasy world-building documents!")

if __name__ == "__main__":
    main()
