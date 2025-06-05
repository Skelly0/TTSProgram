# Document to Audiobook Converter

A simple Python program that converts text documents into audiobooks using the Kokoro-82M TTS (Text-to-Speech) model.

## Features

- **Multiple Document Formats**: Supports `.txt`, `.md`, and `.rtf` files
- **High-Quality TTS**: Uses the Kokoro-82M model with 82 million parameters
- **Multiple Voices**: Choose from 5 different voice models
- **Language Options**: American or British English
- **Automatic Processing**: Batch convert all documents in a folder
- **Smart Text Processing**: Handles markdown formatting and special characters
- **Chunked Processing**: Efficiently processes long documents by splitting them into manageable chunks

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

### Basic Usage

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

### Advanced Usage

#### Custom Directories
```bash
python document_to_audiobook.py --documents-dir "my_docs" --audios-dir "my_audiobooks"
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

#### Verbose Output
```bash
python document_to_audiobook.py --verbose
```

### Complete Example
```bash
python document_to_audiobook.py \
    --documents-dir "books" \
    --audios-dir "audiobooks" \
    --voice af_bella \
    --lang-code b \
    --speed 0.9 \
    --verbose
```

## Command Line Options

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--documents-dir` | Input documents directory | `documents` | Any valid path |
| `--audios-dir` | Output audio directory | `audios` | Any valid path |
| `--voice` | Voice model to use | `af_heart` | `af_heart`, `af_bella`, `af_sarah`, `am_adam`, `am_michael` |
| `--lang-code` | Language variant | `a` | `a` (American), `b` (British) |
| `--speed` | Speech speed multiplier | `1.0` | Any positive number |
| `--verbose` | Enable detailed logging | `False` | Flag |

## Supported File Formats

- **`.txt`**: Plain text files
- **`.md`**: Markdown files (formatting will be removed)
- **`.rtf`**: Rich Text Format files

## How It Works

1. **Document Discovery**: Scans the documents directory for supported file types
2. **Text Extraction**: Reads and preprocesses text content
3. **Text Cleaning**: Removes markdown formatting and normalizes special characters
4. **Chunking**: Splits long documents into manageable chunks for processing
5. **TTS Generation**: Converts each chunk to audio using Kokoro-82M
6. **Audio Assembly**: Concatenates all chunks into a single audio file
7. **File Output**: Saves the final audiobook as a WAV file

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
