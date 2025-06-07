# Enhanced Features Summary

## What's New

This update adds comprehensive LLM preprocessing capabilities and separate command workflows to the Document to Audiobook Converter.

## 🚀 New Features

### 1. Enhanced LLM Logging
- **Detailed API Tracking**: Complete visibility into OpenRouter API calls
- **Token Usage Monitoring**: Track input/output tokens and estimated costs
- **Performance Metrics**: API response times and processing durations
- **Error Context**: Detailed error reporting with HTTP status codes
- **SSML Validation**: Comprehensive validation of generated SSML content
- **Connection Testing**: Automatic API connectivity verification

### 2. Separate Command Workflow
- **`preprocess_documents.py`**: Standalone preprocessing with LLM enhancement
- **`run_tts.py`**: TTS-only processing for preprocessed documents
- **Independent Operation**: Run preprocessing and TTS separately for better control
- **Batch Scripts**: Easy-to-use `.bat` and `.sh` scripts for both workflows

### 3. Batch Processing Scripts
- **Windows**: `run_preprocessing.bat` and `run_tts_only.bat`
- **Linux/Mac**: `run_preprocessing.sh` and `run_tts_only.sh`
- **Error Checking**: Automatic validation of prerequisites and file availability
- **User-Friendly**: Clear status messages and error handling

## 📊 Enhanced Logging Details

### LLM Processing Logs
```
🤖 Starting LLM preprocessing for chunk (1,234 chars, ~200 words)
🔧 Using model: anthropic/claude-3.5-sonnet
📡 Sending request to OpenRouter API...
📡 API response received in 2.34s
📊 Token usage - Input: 150, Output: 200, Total: 350
💰 Estimated cost: $0.001234
📝 LLM generated 1,456 characters of SSML
🔍 Validating and cleaning SSML output...
📊 SSML elements found - Breaks: 5, Phonemes: 3, Emphasis: 2
✅ LLM preprocessing complete in 2.67s (API: 2.34s, Processing: 0.33s)
```

### Progress Tracking
```
🚀 Starting preprocessing of 3 document(s)
📄 Processing document: my_book.txt
📊 Document 1/3
🧩 Split into 25 chunks for LLM processing
📈 Chunk progress: 15/25 (60.0%)
⏱️  Estimated completion: 14:32:15 (120s remaining)
```

## 🛠️ Usage Examples

### Two-Step Workflow (Recommended)

**Step 1: Preprocessing**
```bash
# Windows
run_preprocessing.bat

# Linux/Mac
./run_preprocessing.sh

# Direct Python
python preprocess_documents.py --verbose
```

**Step 2: TTS Processing**
```bash
# Windows
run_tts_only.bat

# Linux/Mac
./run_tts_only.sh

# Direct Python
python run_tts.py --verbose
```

### Advanced Options

**Custom LLM Model**
```bash
python preprocess_documents.py --llm-model "anthropic/claude-3-haiku" --verbose
```

**Custom Chunk Sizes**
```bash
python preprocess_documents.py --max-chunk-size 800 --verbose
python run_tts.py --max-chunk-size 400 --verbose
```

**Custom Directories**
```bash
python preprocess_documents.py --documents-dir "books" --preprocessed-dir "enhanced"
python run_tts.py --preprocessed-dir "enhanced" --audios-dir "audiobooks"
```

## 📁 File Structure

```
TTSProgram/
├── document_to_audiobook.py      # Original all-in-one script (enhanced)
├── preprocess_documents.py       # NEW: Preprocessing-only script
├── run_tts.py                    # NEW: TTS-only script
├── run_preprocessing.bat         # NEW: Windows preprocessing batch
├── run_tts_only.bat             # NEW: Windows TTS batch
├── run_preprocessing.sh          # NEW: Linux/Mac preprocessing script
├── run_tts_only.sh              # NEW: Linux/Mac TTS script
├── documents/                    # Input documents
├── preprocessed/                 # LLM-enhanced documents (SSML)
└── audios/                      # Output audiobooks
```

## 🎯 Benefits

### For Users
- **Better Control**: Run preprocessing and TTS independently
- **Cost Management**: Preview LLM costs before TTS processing
- **Debugging**: Inspect preprocessed SSML before audio generation
- **Flexibility**: Use different settings for preprocessing vs TTS

### For Developers
- **Comprehensive Logging**: Full visibility into all operations
- **Error Tracking**: Detailed error context and recovery information
- **Performance Monitoring**: Timing and resource usage metrics
- **API Management**: Token usage and cost tracking

## 🔧 Configuration

All existing configuration options remain available, plus new options for:
- LLM model selection
- Chunk size control
- Directory customization
- Verbose logging levels
- API key management

## 🚀 Getting Started

1. **Set up your API key** (if using LLM preprocessing):
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

2. **Try the two-step workflow**:
   ```bash
   # Step 1: Preprocess with LLM
   python preprocess_documents.py --verbose
   
   # Step 2: Generate audio
   python run_tts.py --verbose
   ```

3. **Or use the convenient batch scripts**:
   ```bash
   # Windows
   run_preprocessing.bat
   run_tts_only.bat
   
   # Linux/Mac
   ./run_preprocessing.sh
   ./run_tts_only.sh
   ```

The enhanced system provides complete visibility into the document-to-audiobook conversion process, making it easy to track progress, identify issues, and optimize results!
