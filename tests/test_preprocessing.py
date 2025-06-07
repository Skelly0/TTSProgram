#!/usr/bin/env python3
"""
Test script to verify that preprocessing saves documents to a new folder.
"""

import os
import tempfile
import shutil
from pathlib import Path
from document_to_audiobook import DocumentToAudiobookConverter

def test_preprocessing_save():
    """Test that preprocessing saves documents to the preprocessed folder."""
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "test_documents"
        audios_dir = temp_path / "test_audios"
        preprocessed_dir = temp_path / "test_preprocessed"
        
        # Create directories
        docs_dir.mkdir()
        
        # Create a test document
        test_doc = docs_dir / "test_document.txt"
        test_content = """# Test Document

This is a test document with some **bold text** and *italic text*.

It has multiple paragraphs to test the preprocessing functionality.

- List item 1
- List item 2
- List item 3

The end."""
        
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"Created test document: {test_doc}")
        
        # Initialize converter with preprocessing enabled but without LLM (to avoid API calls)
        converter = DocumentToAudiobookConverter(
            documents_dir=str(docs_dir),
            audios_dir=str(audios_dir),
            preprocessed_dir=str(preprocessed_dir),
            save_preprocessed=True,
            enable_llm_preprocessing=False,  # Disable LLM to avoid API calls
            openrouter_api_key=None
        )
        
        # Read and preprocess the document
        text = converter.read_document(test_doc)
        preprocessed_text = converter.preprocess_text(text)
        
        # Save preprocessed document
        saved_path = converter.save_preprocessed_document(test_doc, preprocessed_text)
        
        # Verify the preprocessed file was created
        if saved_path and saved_path.exists():
            print(f"Preprocessed document saved: {saved_path}")
            
            # Read and display the preprocessed content
            with open(saved_path, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            
            print(f"Original content length: {len(test_content)} characters")
            print(f"Preprocessed content length: {len(saved_content)} characters")
            print(f"Preprocessed content preview:")
            print("-" * 50)
            print(saved_content[:200] + "..." if len(saved_content) > 200 else saved_content)
            print("-" * 50)
            
            return True
        else:
            print("Failed to save preprocessed document")
            return False

if __name__ == "__main__":
    print("Testing preprocessing save functionality...")
    success = test_preprocessing_save()
    if success:
        print("Test passed! Preprocessing saves documents to new folder.")
    else:
        print("Test failed!")
