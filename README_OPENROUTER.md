# Document to Audiobook Converter with OpenRouter LLM Integration

Convert text documents to high-quality audiobooks using Kokoro-82M TTS with optional OpenRouter LLM preprocessing for enhanced text formatting and SSML generation.

## 🚀 New Features

- **🤖 LLM-Enhanced Processing**: Optional OpenRouter integration for intelligent text preprocessing
- **📝 SSML Generation**: Automatic conversion to Speech Synthesis Markup Language for better audio quality
- **🎭 Fantasy Content Optimized**: Special handling for fantasy names, footnotes, and complex world-building documents
- **🧠 Smart Text Processing**: Advanced text optimization using state-of-the-art language models

## Features

- **🗣️ High-Quality TTS**: Uses the advanced Kokoro-82M model for natural-sounding speech
- **🎵 Multiple Voice Options**: Choose from 5 different voices (3 female, 2 male)
- **📚 Batch Processing**: Convert multiple documents at once
- **📊 Progress Tracking**: Real-time progress updates with detailed logging
- **📄 Flexible Input**: Supports .txt, .md, and .rtf files
- **⚡ Easy Setup**: Simple installation and usage

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up OpenRouter API** (optional but recommended):
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
   Get your API key from [OpenRouter](https://openrouter.ai/keys)

3. **Place your documents** in the `documents/` folder

4. **Run the converter**:
   ```bash
   # With LLM preprocessing (recommended)
   python document_to_audiobook.py
   
   # Without LLM preprocessing (basic mode)
   python document_to_audiobook.py --disable-llm-preprocessing
   ```

5. **Find your audiobooks** in the `audios/` folder

## OpenRouter LLM Integration

The converter can use OpenRouter's API to preprocess text with advanced language models, providing:

### Enhanced Text Processing
- **SSML Generation**: Converts raw text to Speech Synthesis Markup Language
- **Fantasy Name Pronunciation**: Adds phonetic guides for complex fantasy names
- **Intelligent Formatting**: Handles footnotes, lists, and complex document structures
- **Narrative Flow**: Optimizes text for audiobook listening experience

### Supported LLM Models

| Model | Cost (per 1M tokens) | Best For |
|-------|---------------------|----------|
| `anthropic/claude-3.5-sonnet` | $3-15 | **Recommended** - Best quality |
| `anthropic/claude-3-haiku` | $0.25-1.25 | Fast and economical |
| `openai/gpt-4o` | $5-15 | High quality alternative |
| `openai/gpt-3.5-turbo` | $0.50-2 | Budget-friendly option |
| `meta-llama/llama-3.1-8b-instruct` | $0.10-0.80 | Open source model |

### What the LLM Does

The LLM preprocessing transforms your raw text according to the detailed guide you provided:

1. **Triage and Clean**: Removes table-of-contents, collapses blank lines, converts headings
2. **Smart Chunking**: Splits text into 600-800 word chunks with intelligent boundaries
3. **SSML Conversion**: Applies the specialized prompt template for fantasy content:
   - Preserves lore but keeps sentences ≤ 30 words
   - Inserts proper `<break>` tags for pacing
   - Converts lists to spoken format ("First... Second... Third...")
   - Handles fantasy names with pronunciation guides
   - Converts footnotes to inline parentheticals
   - Summarizes tabular data

## Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection (for model download)
- OpenRouter API key (optional, for LLM features)

### Setup
```bash
# Clone or download this repository
git clone <repository-url>
cd TTSProgram

# Install required packages
pip install -r requirements.txt

# Create directories (optional - they'll be created automatically)
mkdir documents audios

# Set up configuration (optional)
cp config.example.py config.py
# Edit config.py with your settings
```

## Usage

### Basic Usage
```bash
# With LLM preprocessing (requires API key)
python document_to_audiobook.py

# Without LLM preprocessing
python document_to_audiobook.py --disable-llm-preprocessing
```

### Advanced Options
```bash
python document_to_audiobook.py \
    --openrouter-api-key "your_key_here" \
    --llm-model "anthropic/claude-3.5-sonnet" \
    --voice af_bella \
    --lang-code b \
    --speed 1.2 \
    --documents-dir my_docs \
    --audios-dir my_audiobooks \
    --verbose
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--documents-dir` | `documents` | Input documents directory |
| `--audios-dir` | `audios` | Output audio files directory |
| `--voice` | `af_heart` | Voice model (see Voice Options) |
| `--lang-code` | `a` | Language: `a` (American) or `b` (British) |
| `--speed` | `1.0` | Speech speed multiplier |
| `--openrouter-api-key` | `None` | OpenRouter API key (or use env var) |
| `--llm-model` | `anthropic/claude-3.5-sonnet` | LLM model for preprocessing |
| `--disable-llm-preprocessing` | `False` | Disable LLM preprocessing |
| `--site-url` | `""` | Site URL for OpenRouter rankings |
| `--site-title` | `"Document to Audiobook Converter"` | Site title for rankings |
| `--verbose` | `False` | Enable detailed logging |

### Voice Options

| Voice Code | Description |
|------------|-------------|
| `af_heart` | Female voice (default) |
| `af_bella` | Female voice (alternative) |
| `af_sarah` | Female voice (alternative) |
| `am_adam` | Male voice |
| `am_michael` | Male voice (alternative) |

## Configuration

### Environment Variables
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Configuration File
Copy `config.example.py` to `config.py` and customize:
```python
OPENROUTER_API_KEY = "your_api_key_here"
LLM_MODEL = "anthropic/claude-3.5-sonnet"
ENABLE_LLM_PREPROCESSING = True
VOICE = "af_heart"
SPEED = 1.0
```

## Examples

### Convert Fantasy Documents (Recommended)
```bash
# Perfect for the "Region 1a – Dytikratia" type documents
export OPENROUTER_API_KEY="your_key_here"

python document_to_audiobook.py \
    --llm-model "anthropic/claude-3.5-sonnet" \
    --voice af_bella \
    --speed 0.95 \
    --verbose
```

### Budget-Friendly Conversion
```bash
# Use Claude Haiku for cost-effective processing
python document_to_audiobook.py \
    --llm-model "anthropic/claude-3-haiku" \
    --voice af_heart
```

### High-Speed Basic Conversion
```bash
# Skip LLM preprocessing for simple documents
python document_to_audiobook.py \
    --disable-llm-preprocessing \
    --voice am_adam \
    --speed 1.1
```

### Batch Convert with Custom Settings
```bash
python document_to_audiobook.py \
    --documents-dir "fantasy_docs" \
    --audios-dir "fantasy_audiobooks" \
    --llm-model "openai/gpt-4o" \
    --voice af_sarah \
    --speed 0.9
```

## LLM Preprocessing Features

When enabled, the LLM preprocessing provides exactly what your guide specified:

### Text Enhancement
- **SSML Generation**: Converts text to Speech Synthesis Markup Language
- **Pronunciation Guides**: Adds phonetic notation for complex names like "Launinrach" → `<phoneme alphabet="ipa" ph="laʊˈniːnɾax">`
- **Narrative Flow**: Optimizes text structure for audio consumption

### Fantasy Content Support (Your Use Case)
- **Name Pronunciation**: Handles fantasy character and place names
- **Footnote Integration**: Converts `[1]` to `(see note 1 for background)`
- **World-building**: Optimizes complex lore documents like your Region files
- **Province Lists**: Converts tabular data to "There are seven plausible province splits..."

### Document Structure
- **Heading Conversion**: `### Thursia` becomes narrative signposts with `<emphasis>`
- **List Processing**: Bullet points become "First... Second... Third..."
- **Table Summarization**: Collapses complex tables to spoken summaries
- **Break Insertion**: Adds `<break time="600ms"/>` at headings, `<break time="300ms"/>` at sections

## Cost Considerations

LLM preprocessing uses OpenRouter API credits. Typical costs:

| Document Size | Tokens Used | Cost (Claude 3.5) | Cost (Claude Haiku) |
|---------------|-------------|-------------------|---------------------|
| 5,000 words | ~7,000-10,000 | $0.02-0.15 | $0.002-0.01 |
| 20,000 words | ~28,000-40,000 | $0.08-0.60 | $0.007-0.05 |
| 50,000 words | ~70,000-100,000 | $0.21-1.50 | $0.018-0.13 |

Your "Region 1a – Dytikratia" document would likely cost $0.05-0.30 with Claude 3.5 Sonnet.

## Supported File Formats

- **Text files** (`.txt`)
- **Markdown files** (`.md`) 
- **Rich Text Format** (`.rtf`)

The converter automatically handles:
- Multiple text encodings (UTF-8, Latin-1, etc.)
- Markdown formatting removal
- Text cleaning and optimization
- Smart sentence splitting
- Fantasy name pronunciation (with LLM)
- Footnote conversion
- List formatting

## Output

- **Format**: WAV audio files
- **Quality**: 24kHz sample rate
- **Naming**: Same as input file with `.wav` extension
- **Location**: `audios/` directory (or specified output directory)
- **Enhancement**: SSML-optimized audio (with LLM preprocessing)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenRouter API errors**: Check your API key and model availability
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

3. **No documents found**: Check that your files are in the correct directory

4. **Network issues**: Required for both TTS model and LLM API access

5. **LLM preprocessing fails**: The system automatically falls back to basic processing

### Getting Help

- Check the console output for detailed error messages
- Use `--verbose` flag for more detailed logging
- Verify your OpenRouter API key and credits
- Ensure you have sufficient disk space for model files and output audio

## Technical Details

- **TTS Model**: Kokoro-82M (high-quality neural TTS)
- **LLM Integration**: OpenRouter API with multiple model support
- **Audio Format**: 24kHz WAV
- **Text Processing**: LLM-enhanced SSML generation or basic cleaning
- **Memory Management**: Efficient chunk-based processing
- **Error Handling**: Graceful fallbacks and detailed error reporting
- **API Management**: Automatic retry and fallback mechanisms

## Implementation Details

This implementation follows your detailed guide:

1. **Triage Phase**: Regex-based cleaning before LLM processing
2. **Intelligent Chunking**: 600-800 word chunks with smart boundaries
3. **Specialized Prompt**: Uses your exact AudioBookFormatter-v1 prompt
4. **SSML Validation**: Ensures proper XML structure
5. **Fallback System**: Graceful degradation to basic processing if LLM fails

The system essentially implements your "minimal working pre-processor" but with a robust, production-ready framework around it.

## License

This project is open source. Please check individual component licenses for the TTS model and dependencies.
