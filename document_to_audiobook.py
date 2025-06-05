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
import time
from datetime import datetime, timedelta

try:
    from kokoro import KPipeline
    import soundfile as sf
    import torch
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required dependencies:")
    print("pip install kokoro==0.9.4 soundfile torch")
    sys.exit(1)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Helper class to track and log conversion progress."""
    
    def __init__(self):
        self.start_time = None
        self.current_document = None
        self.total_documents = 0
        self.processed_documents = 0
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_start_time = None
        
    def start_conversion(self, total_docs: int):
        """Start tracking overall conversion progress."""
        self.start_time = time.time()
        self.total_documents = total_docs
        self.processed_documents = 0
        logger.info(f"🚀 Starting conversion of {total_docs} document(s)")
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_document(self, doc_name: str, total_chunks: int):
        """Start tracking progress for a specific document."""
        self.current_document = doc_name
        self.total_chunks = total_chunks
        self.current_chunk = 0
        logger.info(f"📄 Processing document: {doc_name}")
        logger.info(f"📊 Document {self.processed_documents + 1}/{self.total_documents}")
        logger.info(f"🧩 Split into {total_chunks} chunks for processing")
        
    def start_chunk(self, chunk_num: int, chunk_preview: str):
        """Start tracking progress for a specific chunk."""
        self.current_chunk = chunk_num
        self.chunk_start_time = time.time()
        preview = chunk_preview[:50] + "..." if len(chunk_preview) > 50 else chunk_preview
        logger.info(f"🔄 Processing chunk {chunk_num}/{self.total_chunks}: '{preview}'")
        
    def finish_chunk(self, chunk_num: int, success: bool):
        """Log completion of a chunk."""
        if self.chunk_start_time:
            duration = time.time() - self.chunk_start_time
            status = "✅ Completed" if success else "❌ Failed"
            logger.info(f"{status} chunk {chunk_num}/{self.total_chunks} in {duration:.2f}s")
        
    def finish_document(self, doc_name: str, success: bool, output_path: str = None):
        """Log completion of a document."""
        self.processed_documents += 1
        status = "✅ Successfully converted" if success else "❌ Failed to convert"
        logger.info(f"{status} document: {doc_name}")
        if success and output_path:
            logger.info(f"💾 Saved audiobook: {output_path}")
        
        # Calculate and log progress
        progress_pct = (self.processed_documents / self.total_documents) * 100
        logger.info(f"📈 Overall progress: {self.processed_documents}/{self.total_documents} ({progress_pct:.1f}%)")
        
        # Estimate remaining time
        if self.start_time and self.processed_documents > 0:
            elapsed = time.time() - self.start_time
            avg_time_per_doc = elapsed / self.processed_documents
            remaining_docs = self.total_documents - self.processed_documents
            estimated_remaining = avg_time_per_doc * remaining_docs
            
            if remaining_docs > 0:
                eta = datetime.now() + timedelta(seconds=estimated_remaining)
                logger.info(f"⏱️  Estimated completion: {eta.strftime('%H:%M:%S')} ({estimated_remaining:.0f}s remaining)")
        
    def finish_conversion(self, successful: int, failed: int):
        """Log completion of entire conversion process."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"🏁 Conversion completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        logger.info(f"📊 Final results: {successful} successful, {failed} failed")
        if successful > 0:
            logger.info(f"🎉 Successfully created {successful} audiobook(s)!")
        if failed > 0:
            logger.warning(f"⚠️  {failed} document(s) failed to convert")

# Global progress tracker instance
progress_tracker = ProgressTracker()

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
            logger.info(f"🔧 Initializing Kokoro TTS pipeline...")
            self.pipeline = KPipeline(lang_code=lang_code)
            logger.info(f"✅ Kokoro pipeline initialized successfully")
            logger.info(f"🗣️  Voice: {voice}, Language: {'American English' if lang_code == 'a' else 'British English'}, Speed: {speed}x")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro pipeline: {e}")
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
                        file_size = len(content)
                        word_count = len(content.split())
                        logger.info(f"📖 Read {file_path.name} ({file_size:,} chars, ~{word_count:,} words) using {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try binary mode and decode with errors='ignore'
            with open(file_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='ignore')
                logger.warning(f"⚠️  Read {file_path.name} with error handling - some characters may be lost")
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
        logger.debug(f"🔪 Splitting text into chunks (max {max_chunk_size} words per chunk)")
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.debug(f"📝 Found {len(sentences)} sentences to process")
        
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
        
        final_chunks = [chunk for chunk in chunks if chunk.strip()]
        
        # Log chunk statistics
        total_words = sum(len(chunk.split()) for chunk in final_chunks)
        avg_words = total_words / len(final_chunks) if final_chunks else 0
        logger.info(f"🧩 Created {len(final_chunks)} chunks (avg {avg_words:.1f} words per chunk)")
        
        return final_chunks
    
    def generate_audio_for_chunk(self, text_chunk: str) -> tuple:
        """
        Generate audio for a single text chunk.
        
        Args:
            text_chunk: Text to convert to audio
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.debug(f"🎵 Generating audio for chunk ({len(text_chunk.split())} words)")
            
            generator = self.pipeline(
                text_chunk,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r'\n+'
            )
            
            # Collect all audio segments
            audio_segments = []
            sample_rate = None
            segment_count = 0
            
            for i, (gs, ps, audio) in enumerate(generator):
                if sample_rate is None:
                    sample_rate = 24000  # Kokoro default sample rate
                audio_segments.append(audio)
                segment_count += 1
            
            if not audio_segments:
                logger.warning(f"⚠️  No audio generated for chunk: {text_chunk[:50]}...")
                return None, None
            
            # Concatenate audio segments
            import numpy as np
            full_audio = np.concatenate(audio_segments)
            
            duration = len(full_audio) / sample_rate
            logger.debug(f"🎶 Generated {duration:.2f}s of audio from {segment_count} segments")
            
            return full_audio, sample_rate
            
        except Exception as e:
            logger.error(f"❌ Failed to generate audio for chunk: {e}")
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
            # Read and preprocess the document
            text = self.read_document(document_path)
            
            logger.info(f"🔄 Preprocessing text...")
            original_length = len(text)
            text = self.preprocess_text(text)
            processed_length = len(text)
            logger.debug(f"📝 Text preprocessing: {original_length:,} → {processed_length:,} characters")
            
            if not text.strip():
                logger.warning(f"⚠️  Document {document_path.name} is empty or contains no readable text")
                return False
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            
            # Start document progress tracking
            progress_tracker.start_document(document_path.name, len(chunks))
            
            # Generate audio for each chunk
            all_audio_segments = []
            sample_rate = None
            successful_chunks = 0
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                progress_tracker.start_chunk(i + 1, chunk)
                
                audio, sr = self.generate_audio_for_chunk(chunk)
                
                if audio is not None:
                    if sample_rate is None:
                        sample_rate = sr
                    all_audio_segments.append(audio)
                    successful_chunks += 1
                    progress_tracker.finish_chunk(i + 1, True)
                else:
                    failed_chunks += 1
                    progress_tracker.finish_chunk(i + 1, False)
                    logger.warning(f"⚠️  Failed to generate audio for chunk {i+1}")
            
            if not all_audio_segments:
                logger.error(f"❌ No audio generated for document {document_path.name}")
                progress_tracker.finish_document(document_path.name, False)
                return False
            
            # Log chunk processing summary
            logger.info(f"📊 Chunk processing complete: {successful_chunks} successful, {failed_chunks} failed")
            
            # Concatenate all audio segments
            logger.info(f"🔗 Combining {len(all_audio_segments)} audio segments...")
            import numpy as np
            full_audio = np.concatenate(all_audio_segments)
            
            # Calculate final audio statistics
            total_duration = len(full_audio) / sample_rate
            logger.info(f"🎵 Generated {total_duration:.2f}s ({total_duration/60:.1f} minutes) of audio")
            
            # Create output filename
            output_filename = document_path.stem + ".wav"
            output_path = self.audios_dir / output_filename
            
            # Save audio file
            logger.info(f"💾 Saving audiobook to {output_path.name}...")
            sf.write(str(output_path), full_audio, sample_rate)
            
            # Calculate file size
            file_size = output_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"📁 Audiobook saved: {file_size_mb:.1f} MB")
            
            progress_tracker.finish_document(document_path.name, True, str(output_path))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {document_path.name}: {e}")
            progress_tracker.finish_document(document_path.name, False)
            return False
    
    def convert_all_documents(self) -> None:
        """Convert all supported documents in the documents directory."""
        supported_extensions = self.get_supported_file_types()
        
        logger.info(f"🔍 Scanning {self.documents_dir} for documents...")
        logger.info(f"📋 Supported file types: {', '.join(supported_extensions)}")
        
        # Find all supported document files
        document_files = []
        for ext in supported_extensions:
            found_files = list(self.documents_dir.glob(f"*{ext}"))
            if found_files:
                logger.debug(f"Found {len(found_files)} {ext} file(s)")
            document_files.extend(found_files)
        
        if not document_files:
            logger.warning(f"❌ No supported documents found in {self.documents_dir}")
            logger.info(f"💡 Place your documents in the '{self.documents_dir}' folder")
            logger.info(f"📋 Supported formats: {', '.join(supported_extensions)}")
            return
        
        # Sort files for consistent processing order
        document_files.sort(key=lambda x: x.name)
        
        # Log discovered files
        logger.info(f"📚 Found {len(document_files)} document(s) to process:")
        for i, doc_file in enumerate(document_files, 1):
            file_size = doc_file.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"  {i}. {doc_file.name} ({file_size_kb:.1f} KB)")
        
        # Start overall progress tracking
        progress_tracker.start_conversion(len(document_files))
        
        successful_conversions = 0
        failed_conversions = 0
        
        for doc_file in document_files:
            try:
                logger.info(f"\n{'='*60}")
                if self.convert_document_to_audio(doc_file):
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            except KeyboardInterrupt:
                logger.info(f"⏹️  Conversion interrupted by user")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error processing {doc_file.name}: {e}")
                failed_conversions += 1
                progress_tracker.finish_document(doc_file.name, False)
        
        logger.info(f"\n{'='*60}")
        progress_tracker.finish_conversion(successful_conversions, failed_conversions)

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
        logger.info(f"🎯 Document to Audiobook Converter")
        logger.info(f"📂 Documents directory: {args.documents_dir}")
        logger.info(f"🎵 Audio output directory: {args.audios_dir}")
        logger.info(f"🗣️  Voice: {args.voice}")
        logger.info(f"🌍 Language: {'American English' if args.lang_code == 'a' else 'British English'}")
        logger.info(f"⚡ Speed: {args.speed}x")
        
        converter = DocumentToAudiobookConverter(
            documents_dir=args.documents_dir,
            audios_dir=args.audios_dir,
            voice=args.voice,
            lang_code=args.lang_code,
            speed=args.speed
        )
        
        converter.convert_all_documents()
        
    except KeyboardInterrupt:
        logger.info(f"\n⏹️  Conversion interrupted by user")
        logger.info(f"👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
