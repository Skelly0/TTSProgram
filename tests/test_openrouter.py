#!/usr/bin/env python3
"""
Test script for OpenRouter LLM integration
Tests the LLM preprocessing functionality without running full TTS conversion.
"""

import os
import sys
from document_to_audiobook import OpenRouterLLMProcessor

def test_openrouter_integration():
    """Test the OpenRouter LLM processor with a sample text."""
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        print("❌ No OpenRouter API key found.")
        print("💡 Set OPENROUTER_API_KEY environment variable or use:")
        print("   export OPENROUTER_API_KEY='your_key_here'")
        return False
    
    print("🤖 Testing OpenRouter LLM integration...")
    print(f"🔑 API key found: {api_key[:8]}...")
    
    # Sample fantasy text (similar to your Region documents)
    sample_text = """
    Eastern Crinia – The Dytikratia
    
    Also known as Thoesis ke Aleta, or simply Lydion, this region spans Thursia and Aletia at minimum. Its politics revolve around four hegemons and an uneasy caste system.
    
    Economy
    The region's economy is dominated by:
    • Agricultural production in the fertile valleys
    • Mining operations in the Thursian mountains
    • Trade routes connecting the eastern provinces
    
    Notable Locations
    1. Launinrach - The ancient capital [1]
    2. Aletia - The modern commercial center
    3. Thursian March - Military frontier region
    
    [1] See historical records for founding details
    """
    
    try:
        # Initialize the LLM processor
        processor = OpenRouterLLMProcessor(
            api_key=api_key,
            model="anthropic/claude-3.5-sonnet",
            site_title="OpenRouter Test"
        )
        
        print("✅ LLM processor initialized successfully")
        print("🔄 Processing sample text...")
        
        # Process the sample text
        processed_text = processor.preprocess_text_chunk(sample_text)
        
        print("✅ Text processing completed!")
        print("\n" + "="*60)
        print("📝 ORIGINAL TEXT:")
        print("="*60)
        print(sample_text)
        print("\n" + "="*60)
        print("🤖 LLM PROCESSED (SSML):")
        print("="*60)
        print(processed_text)
        print("="*60)
        
        # Basic validation
        if "<speak>" in processed_text and "</speak>" in processed_text:
            print("✅ SSML structure validated")
        else:
            print("⚠️  Warning: SSML structure may be incomplete")
        
        if "Launinrach" in processed_text:
            print("✅ Fantasy names preserved")
        
        if "see note" in processed_text.lower():
            print("✅ Footnotes converted")
        
        print("\n🎉 OpenRouter integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your API key is valid")
        print("2. Verify you have credits in your OpenRouter account")
        print("3. Check your internet connection")
        print("4. Try a different model (e.g., anthropic/claude-3-haiku)")
        return False

if __name__ == "__main__":
    print("🧪 OpenRouter LLM Integration Test")
    print("="*50)
    
    success = test_openrouter_integration()
    
    if success:
        print("\n✅ All tests passed! You can now use LLM preprocessing.")
        print("💡 Run the full converter with:")
        print("   python document_to_audiobook.py")
    else:
        print("\n❌ Tests failed. Please check the issues above.")
        print("💡 You can still use the converter without LLM preprocessing:")
        print("   python document_to_audiobook.py --disable-llm-preprocessing")
    
    sys.exit(0 if success else 1)
