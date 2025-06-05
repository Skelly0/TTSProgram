#!/usr/bin/env python3
"""
Document Preprocessing Script
Preprocesses documents using LLM and saves them to the preprocessed folder.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List
import time
from datetime import datetime

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
    from openai import OpenAI
    import requests
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required dependencies:")
    print("pip install openai requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PreprocessingProgressTracker:
    """Helper class to track preprocessing progress."""
    
    def __init__(self):
        self.start_time = None
        self.total_documents = 0
        self.processed_documents = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        
    def start_preprocessing(self, total_docs: int):
        """Start tracking overall preprocessing progress."""
        self.start_time = time.time()
        self.total_documents = total_docs
        self.processed_documents = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        logger.info(f"🚀 Starting preprocessing of {total_docs} document(s)")
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_document(self, doc_name: str, chunks: int):
        """Start tracking progress for a specific document."""
        self.total_chunks += chunks
        logger.info(f"📄 Processing document: {doc_name}")
        logger.info(f"📊 Document {self.processed_documents + 1}/{self.total_documents}")
        logger.info(f"🧩 Split into {chunks} chunks for LLM processing")
        
    def finish_chunk(self):
        """Mark a chunk as completed."""
        self.processed_chunks += 1
        progress_pct = (self.processed_chunks / self.total_chunks) * 100 if self.total_chunks > 0 else 0
        logger.info(f"📈 Chunk progress: {self.processed_chunks}/{self.total_chunks} ({progress_pct:.1f}%)")
        
    def finish_document(self, doc_name: str, success: bool, output_path: str = None):
        """Log completion of a document."""
        self.processed_documents += 1
        status = "✅ Successfully preprocessed" if success else "❌ Failed to preprocess"
        logger.info(f"{status} document: {doc_name}")
        if success and output_path:
            logger.info(f"💾 Saved preprocessed document: {output_path}")
        
    def finish_preprocessing(self, successful: int, failed: int):
        """Log completion of entire preprocessing process."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"🏁 Preprocessing completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        logger.info(f"📊 Final results: {successful} successful, {failed} failed")
        if successful > 0:
            logger.info(f"🎉 Successfully preprocessed {successful} document(s)!")
        if failed > 0:
            logger.warning(f"⚠️  {failed} document(s) failed to preprocess")

# Import the LLM processor from the main script
sys.path.append(str(Path(__file__).parent))
from document_to_audiobook import OpenRouterLLMProcessor

class DocumentPreprocessor:
    """Handles document preprocessing using LLM."""
    
    def __init__(self, documents_dir: str = "documents", preprocessed_dir: str = "preprocessed",
                 openrouter_api_key: str = None, llm_model: str = "anthropic/claude-3.5-sonnet",
                 site_url: str = "", site_title: str = "Document Preprocessor",
                 max_chunk_size: int = 600):
        """
        Initialize the preprocessor.
        
        Args:
            documents_dir: Directory containing input documents
            preprocessed_dir: Directory for preprocessed output files
            openrouter_api_key: OpenRouter API key for LLM preprocessing
            llm_model: LLM model to use for preprocessing
            site_url: Optional site URL for OpenRouter rankings
            site_title: Optional site title for OpenRouter rankings
            max_chunk_size: Maximum words per chunk for LLM processing
        """
        self.documents_dir = Path(documents_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.max_chunk_size = max_chunk_size
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(exist_ok=True)
        self.preprocessed_dir.mkdir(exist_ok=True)
        
        # Initialize LLM processor
        if not openrouter_api_key:
            raise ValueError("OpenRouter API key is required for preprocessing")
            
        logger.info(f"🤖 Initializing LLM preprocessor...")
        self.llm_processor = OpenRouterLLMProcessor(
            api_key=openrouter_api_key,
            model=llm_model,
            site_url=site_url,
            site_title=site_title
        )
        logger.info(f"✅ LLM preprocessor initialized")
        
        # Progress tracker
        self.progress_tracker = PreprocessingProgressTracker()
        
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported document file extensions."""
        return ['.txt', '.md', '.rtf']
    
    def read_document(self, file_path: Path) -> str:
        """Read text content from a document file."""
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
        """Split text into manageable chunks for LLM processing."""
        import re
        
        logger.debug(f"🔪 Splitting text into chunks (max {self.max_chunk_size} words per chunk)")
        
        # Split by sentences first
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
    
    def preprocess_document(self, document_path: Path) -> bool:
        """Preprocess a single document."""
        try:
            # Read the document
            text = self.read_document(document_path)
            
            if not text.strip():
                logger.warning(f"⚠️  Document {document_path.name} is empty or contains no readable text")
                return False
            
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            
            # Start document progress tracking
            self.progress_tracker.start_document(document_path.name, len(chunks))
            
            # Process each chunk with LLM
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                processed_chunk = self.llm_processor.preprocess_text_chunk(chunk)
                processed_chunks.append(processed_chunk)
                self.progress_tracker.finish_chunk()
            
            # Combine all processed chunks
            logger.info(f"🔗 Combining {len(processed_chunks)} processed chunks...")
            full_processed_text = "\n\n".join(processed_chunks)
            
            # Create output filename
            output_filename = document_path.stem + "_preprocessed.ssml"
            output_path = self.preprocessed_dir / output_filename
            
            # Save preprocessed document
            logger.info(f"💾 Saving preprocessed document to {output_filename}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_processed_text)
            
            # Calculate file size
            file_size = output_path.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"📁 Preprocessed document saved: {file_size_kb:.1f} KB")
            
            self.progress_tracker.finish_document(document_path.name, True, str(output_path))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess {document_path.name}: {e}")
            self.progress_tracker.finish_document(document_path.name, False)
            return False
    
    def preprocess_all_documents(self) -> None:
        """Preprocess all supported documents in the documents directory."""
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
        logger.info(f"📚 Found {len(document_files)} document(s) to preprocess:")
        for i, doc_file in enumerate(document_files, 1):
            file_size = doc_file.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"  {i}. {doc_file.name} ({file_size_kb:.1f} KB)")
        
        # Start overall progress tracking
        self.progress_tracker.start_preprocessing(len(document_files))
        
        successful_preprocessing = 0
        failed_preprocessing = 0
        
        for doc_file in document_files:
            try:
                logger.info(f"\n{'='*60}")
                if self.preprocess_document(doc_file):
                    successful_preprocessing += 1
                else:
                    failed_preprocessing += 1
            except KeyboardInterrupt:
                logger.info(f"⏹️  Preprocessing interrupted by user")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error preprocessing {doc_file.name}: {e}")
                failed_preprocessing += 1
                self.progress_tracker.finish_document(doc_file.name, False)
        
        logger.info(f"\n{'='*60}")
        self.progress_tracker.finish_preprocessing(successful_preprocessing, failed_preprocessing)

def main():
    """Main function to run the document preprocessor."""
    parser = argparse.ArgumentParser(
        description="Preprocess documents using OpenRouter LLM for TTS optimization"
    )
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Directory containing input documents (default: documents)"
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="preprocessed",
        help="Directory for preprocessed output files (default: preprocessed)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key for LLM preprocessing (can also use OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--llm-model",
        default="anthropic/claude-3.5-sonnet",
        help="LLM model to use for preprocessing (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--site-url",
        default="",
        help="Optional site URL for OpenRouter rankings"
    )
    parser.add_argument(
        "--site-title",
        default="Document Preprocessor",
        help="Optional site title for OpenRouter rankings"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=600,
        help="Maximum words per chunk for LLM processing (default: 600)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get OpenRouter API key from args or environment
    openrouter_api_key = args.openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
    
    if not openrouter_api_key:
        logger.error("❌ OpenRouter API key is required for preprocessing")
        logger.info("💡 Set OPENROUTER_API_KEY environment variable or use --openrouter-api-key")
        sys.exit(1)
    
    try:
        logger.info(f"🎯 Document Preprocessor with OpenRouter Integration")
        logger.info(f"📂 Documents directory: {args.documents_dir}")
        logger.info(f"📝 Preprocessed output directory: {args.preprocessed_dir}")
        logger.info(f"🤖 LLM model: {args.llm_model}")
        logger.info(f"🧩 Max chunk size: {args.max_chunk_size} words")
        
        preprocessor = DocumentPreprocessor(
            documents_dir=args.documents_dir,
            preprocessed_dir=args.preprocessed_dir,
            openrouter_api_key=openrouter_api_key,
            llm_model=args.llm_model,
            site_url=args.site_url,
            site_title=args.site_title,
            max_chunk_size=args.max_chunk_size
        )
        
        preprocessor.preprocess_all_documents()
        
    except KeyboardInterrupt:
        logger.info(f"\n⏹️  Preprocessing interrupted by user")
        logger.info(f"👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
