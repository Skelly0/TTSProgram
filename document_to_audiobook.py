#!/usr/bin/env python3
"""
Document to Audiobook Converter using Kokoro-82M TTS
Converts text documents to audio files using the Kokoro TTS model.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List, Optional
import re

try:
    from kokoro import KPipeline
    import soundfile as sf
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required dependencies:")
    print("pip install kokoro==0.9.4 soundfile torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentToAudiobookConverter:
    """Main converter class for processing documents to audiobooks."""
    
    def __init__(self, documents_dir: str = "documents", audios_dir: str = "audios", 
                 voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0):
        """
        Initialize the converter.
        
        Args:
            documents_dir: Directory containing input documents
            audios_dir: Directory for output audio files
            voice: Voice model to use (af_heart, af_bella, af_sarah, am_adam, am_michael)
            lang_code: Language code ('a' for American English, 'b' for British English)
            speed: Speech speed multiplier (1.0 = normal speed)
        """
        self.documents_dir = Path(documents_dir)
        self.audios_dir = Path(audios_dir)
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(exist_ok=True)
        self.audios_dir.mkdir(exist_ok=True)
        
        # Initialize Kokoro pipeline
        try:
            self.pipeline = KPipeline(lang_code=lang_code)
            logger.info(f"Initialized Kokoro pipeline with language code: {lang_code}")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro pipeline: {e}")
            raise
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported document file extensions."""
        return ['.txt', '.md', '.rtf']
    
    def read_document(self, file_path: Path) -> str:
        """
        Read text content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Text content of the document
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        logger.info(f"Successfully read {file_path} with {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try binary mode and decode with errors='ignore'
            with open(file_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='ignore')
                logger.warning(f"Read {file_path} with error handling - some characters may be lost")
                return content
                
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TTS output.
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Clean up special characters that might cause issues
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace(''', "'")
        text = text.replace(''', "'")
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def split_text_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into manageable chunks for TTS processing.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum number of words per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk.split()) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                # If current chunk is not empty, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If the sentence itself is too long, split it further
                if len(sentence.split()) > max_chunk_size:
                    words = sentence.split()
                    for i in range(0, len(words), max_chunk_size):
                        chunk_words = words[i:i + max_chunk_size]
                        chunks.append(" ".join(chunk_words))
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def generate_audio_for_chunk(self, text_chunk: str) -> tuple:
        """
        Generate audio for a single text chunk.
        
        Args:
            text_chunk: Text to convert to audio
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            generator = self.pipeline(
                text_chunk,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r'\n+'
            )
            
            # Collect all audio segments
            audio_segments = []
            sample_rate = None
            
            for i, (gs, ps, audio) in enumerate(generator):
                if sample_rate is None:
                    sample_rate = 24000  # Kokoro default sample rate
                audio_segments.append(audio)
            
            if not audio_segments:
                logger.warning(f"No audio generated for chunk: {text_chunk[:50]}...")
                return None, None
            
            # Concatenate audio segments
            import numpy as np
            full_audio = np.concatenate(audio_segments)
            
            return full_audio, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to generate audio for chunk: {e}")
            return None, None
    
    def convert_document_to_audio(self, document_path: Path) -> bool:
        """
        Convert a single document to an audio file.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Read and preprocess the document
            text = self.read_document(document_path)
            text = self.preprocess_text(text)
            
            if not text.strip():
                logger.warning(f"Document {document_path} is empty or contains no readable text")
                return False
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Generate audio for each chunk
            all_audio_segments = []
            sample_rate = None
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                audio, sr = self.generate_audio_for_chunk(chunk)
                
                if audio is not None:
                    if sample_rate is None:
                        sample_rate = sr
                    all_audio_segments.append(audio)
                else:
                    logger.warning(f"Failed to generate audio for chunk {i+1}")
            
            if not all_audio_segments:
                logger.error(f"No audio generated for document {document_path}")
                return False
            
            # Concatenate all audio segments
            import numpy as np
            full_audio = np.concatenate(all_audio_segments)
            
            # Create output filename
            output_filename = document_path.stem + ".wav"
            output_path = self.audios_dir / output_filename
            
            # Save audio file
            sf.write(str(output_path), full_audio, sample_rate)
            logger.info(f"Successfully created audiobook: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {document_path}: {e}")
            return False
    
    def convert_all_documents(self) -> None:
        """Convert all supported documents in the documents directory."""
        supported_extensions = self.get_supported_file_types()
        
        # Find all supported document files
        document_files = []
        for ext in supported_extensions:
            document_files.extend(self.documents_dir.glob(f"*{ext}"))
        
        if not document_files:
            logger.warning(f"No supported documents found in {self.documents_dir}")
            logger.info(f"Supported file types: {', '.join(supported_extensions)}")
            return
        
        logger.info(f"Found {len(document_files)} document(s) to process")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for doc_file in document_files:
            try:
                if self.convert_document_to_audio(doc_file):
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {doc_file}: {e}")
                failed_conversions += 1
        
        logger.info(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed")

def main():
    """Main function to run the document to audiobook converter."""
    parser = argparse.ArgumentParser(
        description="Convert documents to audiobooks using Kokoro-82M TTS"
    )
    parser.add_argument(
        "--documents-dir", 
        default="documents",
        help="Directory containing input documents (default: documents)"
    )
    parser.add_argument(
        "--audios-dir",
        default="audios", 
        help="Directory for output audio files (default: audios)"
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        choices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael"],
        help="Voice model to use (default: af_heart)"
    )
    parser.add_argument(
        "--lang-code",
        default="a",
        choices=["a", "b"],
        help="Language code: 'a' for American English, 'b' for British English (default: a)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = DocumentToAudiobookConverter(
            documents_dir=args.documents_dir,
            audios_dir=args.audios_dir,
            voice=args.voice,
            lang_code=args.lang_code,
            speed=args.speed
        )
        
        converter.convert_all_documents()
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
