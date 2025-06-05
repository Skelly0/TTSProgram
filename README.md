# Document to Audiobook Converter

A simple Python program that converts text documents into audiobooks using the Kokoro-82M TTS (Text-to-Speech) model.

## Features

- **Multiple Document Formats**: Supports `.txt`, `.md`, and `.rtf` files
- **High-Quality TTS**: Uses the Kokoro-82M model with 82 million parameters
- **LLM Preprocessing**: Optional OpenRouter integration for advanced text preprocessing and SSML generation
- **Multiple Voices**: Choose from 5 different voice models
- **Language Options**: American or British English
- **Automatic Processing**: Batch convert all documents in a folder
- **Smart Text Processing**: Handles markdown formatting and special characters
- **Chunked Processing**: Efficiently processes long documents by splitting them into manageable chunks
- **Separate Commands**: Run preprocessing and TTS independently for better workflow control
- **Enhanced Logging**: Comprehensive progress tracking with detailed statistics and timing information

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install kokoro==0.9.4 soundfile torch numpy
   ```

3. **Install espeak** (required for Kokoro):
   
   **Windows**:
   - Download and install eSpeak from: http://espeak.sourceforge.net/download.html
   - Or use chocolatey: `choco install espeak`
   
   **macOS**:
   ```bash
   brew install espeak
   ```
   
   **Linux (Ubuntu/Debian)**:
   ```bash
   sudo apt-get install espeak espeak-data
   ```

## Usage

### Quick Start (All-in-One)

1. **Create the documents folder** (if it doesn't exist):
   ```bash
   mkdir documents
   ```

2. **Add your text documents** to the `documents` folder:
   - Supported formats: `.txt`, `.md`, `.rtf`
   - Example: `documents/my_book.txt`, `documents/article.md`

3. **Run the converter**:
   ```bash
   python document_to_audiobook.py
   ```

4. **Find your audiobooks** in the `audios` folder:
   - Output format: `.wav` files
   - Same filename as input: `my_book.wav`, `article.wav`

### Two-Step Workflow (Recommended)

For better control and to leverage LLM preprocessing, use the two-step approach:

#### Step 1: Preprocessing (with LLM Enhancement)

**Using batch scripts (Windows):**
```bash
run_preprocessing.bat
```

**Using shell scripts (Linux/Mac):**
```bash
./run_preprocessing.sh
```

**Using Python directly:**
```bash
python preprocess_documents.py --verbose
```

This will:
- Process documents with OpenRouter LLM for enhanced SSML generation
- Save preprocessed files to the `preprocessed` folder
- Provide detailed logging of the LLM processing

#### Step 2: TTS Processing

**Using batch scripts (Windows):**
```bash
run_tts_only.bat
```

**Using shell scripts (Linux/Mac):**
```bash
./run_tts_only.sh
```

**Using Python directly:**
```bash
python run_tts.py --verbose
```

This will:
- Convert preprocessed documents to audio
- Use the enhanced SSML for better speech quality
- Save audiobooks to the `audios` folder

### LLM Preprocessing Setup

To use LLM preprocessing, you need an OpenRouter API key:

1. **Get an API key** from [OpenRouter](https://openrouter.ai/keys)

2. **Set up your environment**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API key
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Choose your model** (optional, defaults to Claude 3.5 Sonnet):
   ```bash
   # In .env file
   LLM_MODEL=anthropic/claude-3.5-sonnet
   ```

### Advanced Usage

#### Custom Directories
```bash
# All-in-one converter
python document_to_audiobook.py --documents-dir "my_docs" --audios-dir "my_audiobooks"

# Preprocessing only
python preprocess_documents.py --documents-dir "my_docs" --preprocessed-dir "my_preprocessed"

# TTS only
python run_tts.py --preprocessed-dir "my_preprocessed" --audios-dir "my_audiobooks"
```

#### Different Voice Models
```bash
# Female voices
python document_to_audiobook.py --voice af_heart    # Default female voice
python document_to_audiobook.py --voice af_bella    # Alternative female voice
python document_to_audiobook.py --voice af_sarah    # Another female voice

# Male voices
python document_to_audiobook.py --voice am_adam     # Male voice
python document_to_audiobook.py --voice am_michael  # Alternative male voice
```

#### Language Options
```bash
python document_to_audiobook.py --lang-code a  # American English (default)
python document_to_audiobook.py --lang-code b  # British English
```

#### Speech Speed
```bash
python document_to_audiobook.py --speed 0.8  # Slower speech
python document_to_audiobook.py --speed 1.2  # Faster speech
python document_to_audiobook.py --speed 1.0  # Normal speed (default)
```

#### LLM Model Selection
```bash
# Use different LLM models for preprocessing
python preprocess_documents.py --llm-model "anthropic/claude-3-haiku"  # Faster, cheaper
python preprocess_documents.py --llm-model "openai/gpt-4o"            # Alternative model
python preprocess_documents.py --llm-model "anthropic/claude-3.5-sonnet"  # Default, best quality
```

#### Chunk Size Control
```bash
# Adjust chunk sizes for processing
python preprocess_documents.py --max-chunk-size 800    # Larger chunks for LLM
python run_tts.py --max-chunk-size 400                 # Smaller chunks for TTS
```

#### Verbose Output
```bash
python document_to_audiobook.py --verbose
python preprocess_documents.py --verbose
python run_tts.py --verbose
```

### Complete Examples

**All-in-one with LLM preprocessing:**
```bash
python document_to_audiobook.py \
    --documents-dir "books" \
    --audios-dir "audiobooks" \
    --voice af_bella \
    --lang-code b \
    --speed 0.9 \
    --llm-model "anthropic/claude-3.5-sonnet" \
    --verbose
```

**Two-step workflow:**
```bash
# Step 1: Preprocess with LLM
python preprocess_documents.py \
    --documents-dir "books" \
    --preprocessed-dir "processed" \
    --llm-model "anthropic/claude-3.5-sonnet" \
    --max-chunk-size 600 \
    --verbose

# Step 2: Generate audio
python run_tts.py \
    --preprocessed-dir "processed" \
    --audios-dir "audiobooks" \
    --voice af_bella \
    --lang-code b \
    --speed 0.9 \
    --verbose
```

## Command Line Options

### document_to_audiobook.py (All-in-One)

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--documents-dir` | Input documents directory | `documents` | Any valid path |
| `--audios-dir` | Output audio directory | `audios` | Any valid path |
| `--preprocessed-dir` | Preprocessed documents directory | `preprocessed` | Any valid path |
| `--voice` | Voice model to use | `af_heart` | `af_heart`, `af_bella`, `af_sarah`, `am_adam`, `am_michael` |
| `--lang-code` | Language variant | `a` | `a` (American), `b` (British) |
| `--speed` | Speech speed multiplier | `1.0` | Any positive number |
| `--openrouter-api-key` | OpenRouter API key for LLM | None | API key string |
| `--llm-model` | LLM model for preprocessing | `anthropic/claude-3.5-sonnet` | Any OpenRouter model |
| `--disable-llm-preprocessing` | Disable LLM preprocessing | `False` | Flag |
| `--save-preprocessed` | Save preprocessed documents | `True` | Flag |
| `--no-save-preprocessed` | Disable saving preprocessed docs | `False` | Flag |
| `--site-url` | Site URL for OpenRouter rankings | Empty | URL string |
| `--site-title` | Site title for OpenRouter rankings | `Document to Audiobook Converter` | Any string |
| `--verbose` | Enable detailed logging | `False` | Flag |

### preprocess_documents.py (Preprocessing Only)

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--documents-dir` | Input documents directory | `documents` | Any valid path |
| `--preprocessed-dir` | Output preprocessed directory | `preprocessed` | Any valid path |
| `--openrouter-api-key` | OpenRouter API key (required) | None | API key string |
| `--llm-model` | LLM model for preprocessing | `anthropic/claude-3.5-sonnet` | Any OpenRouter model |
| `--site-url` | Site URL for OpenRouter rankings | Empty | URL string |
| `--site-title` | Site title for OpenRouter rankings | `Document Preprocessor` | Any string |
| `--max-chunk-size` | Max words per chunk for LLM | `600` | Positive integer |
| `--verbose` | Enable detailed logging | `False` | Flag |

### run_tts.py (TTS Only)

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--preprocessed-dir` | Input preprocessed directory | `preprocessed` | Any valid path |
| `--audios-dir` | Output audio directory | `audios` | Any valid path |
| `--voice` | Voice model to use | `af_heart` | `af_heart`, `af_bella`, `af_sarah`, `am_adam`, `am_michael` |
| `--lang-code` | Language variant | `a` | `a` (American), `b` (British) |
| `--speed` | Speech speed multiplier | `1.0` | Any positive number |
| `--max-chunk-size` | Max words per chunk for TTS | `500` | Positive integer |
| `--verbose` | Enable detailed logging | `False` | Flag |

## Supported File Formats

- **`.txt`**: Plain text files
- **`.md`**: Markdown files (formatting will be removed)
- **`.rtf`**: Rich Text Format files

## How It Works

### All-in-One Mode
1. **Document Discovery**: Scans the documents directory for supported file types
2. **Text Extraction**: Reads and preprocesses text content
3. **LLM Preprocessing** (optional): Uses OpenRouter API to enhance text with SSML formatting
4. **Text Cleaning**: Removes markdown formatting and normalizes special characters
5. **Chunking**: Splits long documents into manageable chunks for processing
6. **TTS Generation**: Converts each chunk to audio using Kokoro-82M
7. **Audio Assembly**: Concatenates all chunks into a single audio file
8. **File Output**: Saves the final audiobook as a WAV file

### Two-Step Mode

**Step 1: Preprocessing (`preprocess_documents.py`)**
1. **Document Discovery**: Scans the documents directory for supported file types
2. **Text Extraction**: Reads text content with multiple encoding support
3. **Text Chunking**: Splits documents into optimal chunks for LLM processing
4. **LLM Enhancement**: Sends each chunk to OpenRouter for SSML generation
5. **SSML Validation**: Validates and cleans the generated SSML
6. **File Output**: Saves enhanced documents to the preprocessed directory

**Step 2: TTS Processing (`run_tts.py`)**
1. **Preprocessed File Discovery**: Scans the preprocessed directory for enhanced documents
2. **SSML Processing**: Reads and validates SSML content
3. **Smart Chunking**: Splits SSML while preserving structure
4. **TTS Generation**: Converts SSML chunks to audio using Kokoro-82M
5. **Audio Assembly**: Concatenates all audio segments
6. **File Output**: Saves the final audiobook as a WAV file

### LLM Preprocessing Benefits
- **Enhanced Speech Quality**: SSML tags improve pronunciation and pacing
- **Better Narrative Flow**: Intelligent break placement and emphasis
- **Pronunciation Guides**: IPA phonetic notation for difficult words
- **Structure Preservation**: Maintains document hierarchy and formatting
- **Fantasy/Technical Content**: Specialized handling of proper nouns and terminology

## Troubleshooting

### Quick Fixes for Common Issues

**"Failed to initialize Kokoro pipeline" or "Cannot find files on Hub"**
1. **Check internet connection**: The program needs to download the TTS model on first run
2. **Use the model download helper**:
   ```bash
   python download_model.py
   ```
3. **Clear model cache if corrupted**:
   ```bash
   huggingface-cli delete-cache
   ```
4. **Try manual model download**:
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('hexgrad/Kokoro-82M')"
   ```

**"No supported documents found"**
- Verify documents are in the correct directory
- Check file extensions are supported (`.txt`, `.md`, `.rtf`)

**"UnicodeDecodeError"**
- The program automatically handles various text encodings
- If issues persist, try converting your document to UTF-8 encoding

**Memory Issues with Large Documents**
- The program automatically chunks large documents
- For very large files, consider splitting them manually

### Detailed Troubleshooting

For comprehensive troubleshooting information, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) which includes:
- Network connectivity issues
- Model download problems
- Alternative TTS solutions
- Corporate firewall/proxy configurations
- Performance optimization tips

### Performance Tips

- **GPU Acceleration**: If you have a CUDA-compatible GPU, PyTorch will automatically use it for faster processing
- **Chunk Size**: The default chunk size (500 words) balances quality and performance
- **Multiple Files**: Process multiple smaller files rather than one very large file for better progress tracking

## Technical Details

- **Model**: Kokoro-82M (82 million parameter TTS model)
- **Output Format**: WAV files at 24kHz sample rate
- **Processing**: Automatic text chunking and audio concatenation
- **Languages**: English (American and British variants)
- **Voices**: 5 different voice models available

## License

This project uses the Kokoro-82M model, which is licensed under Apache-2.0. Please refer to the [Kokoro repository](https://github.com/hexgrad/kokoro) for more details.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## Acknowledgments

- [Kokoro-82M](https://github.com/hexgrad/kokoro) - The TTS model used in this project
- [eSpeak](http://espeak.sourceforge.net/) - Text-to-speech synthesis engine
