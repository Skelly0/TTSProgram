#!/usr/bin/env python3
"""
TTS-Only Script
Runs TTS processing on preprocessed documents from the preprocessed folder.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List
import time
from datetime import datetime, timedelta
import urllib.request
import socket

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file at startup
load_env_file()

try:
    from kokoro import KPipeline
    import soundfile as sf
    import torch
    import numpy as np
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required dependencies:")
    print("pip install kokoro==0.9.4 soundfile torch")
    sys.exit(1)

def check_network_connectivity() -> bool:
    """Check if we can connect to Hugging Face Hub."""
    try:
        # Try to connect to Hugging Face Hub
        socket.create_connection(("huggingface.co", 443), timeout=10)
        return True
    except (socket.error, socket.timeout):
        return False

def check_huggingface_hub() -> bool:
    """Check if Hugging Face Hub is accessible."""
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        return True
    except Exception:
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TTSProgressTracker:
    """Helper class to track TTS processing progress."""
    
    def __init__(self):
        self.start_time = None
        self.total_documents = 0
        self.processed_documents = 0
        self.current_document = None
        self.current_chunk = 0
        self.total_chunks = 0
        self.chunk_start_time = None
        
    def start_tts_processing(self, total_docs: int):
        """Start tracking overall TTS processing progress."""
        self.start_time = time.time()
        self.total_documents = total_docs
        self.processed_documents = 0
        logger.info(f"🚀 Starting TTS processing of {total_docs} preprocessed document(s)")
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_document(self, doc_name: str, total_chunks: int):
        """Start tracking progress for a specific document."""
        self.current_document = doc_name
        self.total_chunks = total_chunks
        self.current_chunk = 0
        logger.info(f"📄 Processing document: {doc_name}")
        logger.info(f"📊 Document {self.processed_documents + 1}/{self.total_documents}")
        logger.info(f"🧩 Split into {total_chunks} chunks for TTS processing")
        
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
        
    def finish_tts_processing(self, successful: int, failed: int):
        """Log completion of entire TTS processing."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"🏁 TTS processing completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        logger.info(f"📊 Final results: {successful} successful, {failed} failed")
        if successful > 0:
            logger.info(f"🎉 Successfully created {successful} audiobook(s)!")
        if failed > 0:
            logger.warning(f"⚠️  {failed} document(s) failed to convert")

class TTSProcessor:
    """Handles TTS processing of preprocessed documents."""
    
    def __init__(self, preprocessed_dir: str = "preprocessed", audios_dir: str = "audios",
                 voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0,
                 max_chunk_size: int = 500):
        """
        Initialize the TTS processor.
        
        Args:
            preprocessed_dir: Directory containing preprocessed documents
            audios_dir: Directory for output audio files
            voice: Voice model to use (af_heart, af_bella, af_sarah, am_adam, am_michael)
            lang_code: Language code ('a' for American English, 'b' for British English)
            speed: Speech speed multiplier (1.0 = normal speed)
            max_chunk_size: Maximum words per chunk for TTS processing
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.audios_dir = Path(audios_dir)
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.max_chunk_size = max_chunk_size
        
        # Create directories if they don't exist
        self.preprocessed_dir.mkdir(exist_ok=True)
        self.audios_dir.mkdir(exist_ok=True)
        
        # Initialize Kokoro pipeline
        self.pipeline = None
        self._initialize_pipeline()
        
        # Progress tracker
        self.progress_tracker = TTSProgressTracker()
        
    def _initialize_pipeline(self):
        """Initialize the Kokoro TTS pipeline with fallback options."""
        logger.info(f"🔧 Initializing Kokoro TTS pipeline...")
        logger.info(f"🗣️  Voice: {self.voice}, Language: {'American English' if self.lang_code == 'a' else 'British English'}, Speed: {self.speed}x")
        
        # Check network connectivity first
        logger.info("🌐 Checking network connectivity...")
        if not check_network_connectivity():
            logger.error("❌ No internet connection detected")
            raise RuntimeError("No internet connection. Please check your network and try again.")
        
        if not check_huggingface_hub():
            logger.error("❌ Cannot reach Hugging Face Hub")
            raise RuntimeError("Cannot connect to Hugging Face Hub. Please check your connection and try again.")
        
        logger.info("✅ Network connectivity confirmed")
        
        # List of model repositories to try in order of preference
        model_repos = [
            "hexgrad/Kokoro-82M",
            "onnx-community/Kokoro-82M-v1.0-ONNX"
        ]
        
        for repo_id in model_repos:
            try:
                logger.info(f"🔄 Attempting to load model from: {repo_id}")
                self.pipeline = KPipeline(lang_code=self.lang_code, repo_id=repo_id)
                logger.info(f"✅ Kokoro pipeline initialized successfully with {repo_id}")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {repo_id}: {e}")
                continue
        
        # If all models fail, try without specifying repo_id (use default)
        try:
            logger.info(f"🔄 Attempting to load default model...")
            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info(f"✅ Kokoro pipeline initialized successfully with default model")
            return
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kokoro pipeline with default model: {e}")
        
        # Final fallback: provide helpful error message
        error_msg = (
            "Failed to initialize Kokoro TTS pipeline. This could be due to:\n"
            "1. Model files are corrupted or missing from Hugging Face Hub\n"
            "2. Authentication issues with Hugging Face\n"
            "3. Insufficient disk space for model download\n"
            "4. Temporary server issues\n\n"
            "Solutions to try:\n"
            "- Clear Hugging Face cache: huggingface-cli delete-cache\n"
            "- Login to Hugging Face: huggingface-cli login\n"
            "- Try again in a few minutes\n"
            "- Check available disk space"
        )
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported preprocessed file extensions."""
        return ['.ssml', '.txt', '.md']
    
    def read_preprocessed_document(self, file_path: Path) -> str:
        """Read text content from a preprocessed document file."""
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
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS processing."""
        import re
        
        logger.debug(f"🔪 Splitting text into chunks (max {self.max_chunk_size} words per chunk)")
        
        # For SSML files, try to split on SSML boundaries first
        if '<speak>' in text and '</speak>' in text:
            # Extract content from SSML speak tags
            speak_matches = re.findall(r'<speak[^>]*>(.*?)</speak>', text, re.DOTALL)
            if speak_matches:
                # Process each speak block separately
                all_chunks = []
                for speak_content in speak_matches:
                    chunks = self._split_ssml_content(speak_content)
                    all_chunks.extend(chunks)
                
                logger.info(f"🧩 Created {len(all_chunks)} chunks from SSML content")
                return all_chunks
        
        # Fallback to sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.debug(f"📝 Found {len(sentences)} sentences to process")
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk.split()) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                # If current chunk is not empty, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If the sentence itself is too long, split it further
                if len(sentence.split()) > self.max_chunk_size:
                    words = sentence.split()
                    for i in range(0, len(words), self.max_chunk_size):
                        chunk_words = words[i:i + self.max_chunk_size]
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
    
    def _split_ssml_content(self, ssml_content: str) -> List[str]:
        """Split SSML content into chunks while preserving SSML structure."""
        import re
        
        # Split on paragraph breaks or major SSML breaks
        parts = re.split(r'<break\s+time="[^"]*"\s*/?>|</p>|<p[^>]*>', ssml_content)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            test_chunk = current_chunk + " " + part if current_chunk else part
            
            if len(test_chunk.split()) <= self.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(f"<speak>{current_chunk}</speak>")
                current_chunk = part
        
        if current_chunk:
            chunks.append(f"<speak>{current_chunk}</speak>")
        
        return chunks
    
    def generate_audio_for_chunk(self, text_chunk: str) -> tuple:
        """Generate audio for a single text chunk."""
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
            full_audio = np.concatenate(audio_segments)
            
            duration = len(full_audio) / sample_rate
            logger.debug(f"🎶 Generated {duration:.2f}s of audio from {segment_count} segments")
            
            return full_audio, sample_rate
            
        except Exception as e:
            logger.error(f"❌ Failed to generate audio for chunk: {e}")
            return None, None
    
    def process_document(self, document_path: Path) -> bool:
        """Process a single preprocessed document to audio."""
        try:
            # Read the preprocessed document
            text = self.read_preprocessed_document(document_path)
            
            if not text.strip():
                logger.warning(f"⚠️  Document {document_path.name} is empty or contains no readable text")
                return False
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            
            # Start document progress tracking
            self.progress_tracker.start_document(document_path.name, len(chunks))
            
            # Generate audio for each chunk
            all_audio_segments = []
            sample_rate = None
            successful_chunks = 0
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                self.progress_tracker.start_chunk(i + 1, chunk)
                
                audio, sr = self.generate_audio_for_chunk(chunk)
                
                if audio is not None:
                    if sample_rate is None:
                        sample_rate = sr
                    all_audio_segments.append(audio)
                    successful_chunks += 1
                    self.progress_tracker.finish_chunk(i + 1, True)
                else:
                    failed_chunks += 1
                    self.progress_tracker.finish_chunk(i + 1, False)
                    logger.warning(f"⚠️  Failed to generate audio for chunk {i+1}")
            
            if not all_audio_segments:
                logger.error(f"❌ No audio generated for document {document_path.name}")
                self.progress_tracker.finish_document(document_path.name, False)
                return False
            
            # Log chunk processing summary
            logger.info(f"📊 Chunk processing complete: {successful_chunks} successful, {failed_chunks} failed")
            
            # Concatenate all audio segments
            logger.info(f"🔗 Combining {len(all_audio_segments)} audio segments...")
            full_audio = np.concatenate(all_audio_segments)
            
            # Calculate final audio statistics
            total_duration = len(full_audio) / sample_rate
            logger.info(f"🎵 Generated {total_duration:.2f}s ({total_duration/60:.1f} minutes) of audio")
            
            # Create output filename (remove _preprocessed suffix if present)
            base_name = document_path.stem
            if base_name.endswith('_preprocessed'):
                base_name = base_name[:-12]  # Remove '_preprocessed'
            output_filename = base_name + ".wav"
            output_path = self.audios_dir / output_filename
            
            # Save audio file
            logger.info(f"💾 Saving audiobook to {output_path.name}...")
            sf.write(str(output_path), full_audio, sample_rate)
            
            # Calculate file size
            file_size = output_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"📁 Audiobook saved: {file_size_mb:.1f} MB")
            
            self.progress_tracker.finish_document(document_path.name, True, str(output_path))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {document_path.name}: {e}")
            self.progress_tracker.finish_document(document_path.name, False)
            return False
    
    def process_all_documents(self) -> None:
        """Process all supported documents in the preprocessed directory."""
        supported_extensions = self.get_supported_file_types()
        
        logger.info(f"🔍 Scanning {self.preprocessed_dir} for preprocessed documents...")
        logger.info(f"📋 Supported file types: {', '.join(supported_extensions)}")
        
        # Find all supported document files
        document_files = []
        for ext in supported_extensions:
            found_files = list(self.preprocessed_dir.glob(f"*{ext}"))
            if found_files:
                logger.debug(f"Found {len(found_files)} {ext} file(s)")
            document_files.extend(found_files)
        
        if not document_files:
            logger.warning(f"❌ No supported documents found in {self.preprocessed_dir}")
            logger.info(f"💡 Place your preprocessed documents in the '{self.preprocessed_dir}' folder")
            logger.info(f"📋 Supported formats: {', '.join(supported_extensions)}")
            logger.info(f"🔧 Run preprocess_documents.py first to create preprocessed files")
            return
        
        # Sort files for consistent processing order
        document_files.sort(key=lambda x: x.name)
        
        # Log discovered files
        logger.info(f"📚 Found {len(document_files)} preprocessed document(s) to process:")
        for i, doc_file in enumerate(document_files, 1):
            file_size = doc_file.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"  {i}. {doc_file.name} ({file_size_kb:.1f} KB)")
        
        # Start overall progress tracking
        self.progress_tracker.start_tts_processing(len(document_files))
        
        successful_conversions = 0
        failed_conversions = 0
        
        for doc_file in document_files:
            try:
                logger.info(f"\n{'='*60}")
                if self.process_document(doc_file):
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            except KeyboardInterrupt:
                logger.info(f"⏹️  Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error processing {doc_file.name}: {e}")
                failed_conversions += 1
                self.progress_tracker.finish_document(doc_file.name, False)
        
        logger.info(f"\n{'='*60}")
        self.progress_tracker.finish_tts_processing(successful_conversions, failed_conversions)

def main():
    """Main function to run the TTS processor."""
    parser = argparse.ArgumentParser(
        description="Process preprocessed documents to audiobooks using Kokoro-82M TTS"
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="preprocessed",
        help="Directory containing preprocessed documents (default: preprocessed)"
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
        "--max-chunk-size",
        type=int,
        default=500,
        help="Maximum words per chunk for TTS processing (default: 500)"
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
        logger.info(f"🎯 TTS Processor for Preprocessed Documents")
        logger.info(f"📝 Preprocessed documents directory: {args.preprocessed_dir}")
        logger.info(f"🎵 Audio output directory: {args.audios_dir}")
        logger.info(f"🗣️  Voice: {args.voice}")
        logger.info(f"🌍 Language: {'American English' if args.lang_code == 'a' else 'British English'}")
        logger.info(f"⚡ Speed: {args.speed}x")
        logger.info(f"🧩 Max chunk size: {args.max_chunk_size} words")
        
        processor = TTSProcessor(
            preprocessed_dir=args.preprocessed_dir,
            audios_dir=args.audios_dir,
            voice=args.voice,
            lang_code=args.lang_code,
            speed=args.speed,
            max_chunk_size=args.max_chunk_size
        )
        
        processor.process_all_documents()
        
    except KeyboardInterrupt:
        logger.info(f"\n⏹️  Processing interrupted by user")
        logger.info(f"👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
