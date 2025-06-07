# OpenRouter Setup Guide

Quick setup guide for using OpenRouter LLM integration with the Document to Audiobook Converter.

## 🚀 Quick Setup

### 1. Get OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Copy your API key

### 2. Configure API Key

**Option A: Using .env file (Recommended)**
```bash
# Copy the template
cp .env.example .env

# Edit .env file and add your API key
# Replace 'your_openrouter_api_key_here' with your actual key
```

**Option B: Using environment variable**
```bash
# Linux/Mac
export OPENROUTER_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set OPENROUTER_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="your_api_key_here"
```

**Option C: Using command line**
```bash
python document_to_audiobook.py --openrouter-api-key "your_api_key_here"
```

### 3. Test Integration
```bash
# Test the LLM integration
python test_openrouter.py

# If successful, run full conversion
python document_to_audiobook.py
```

## 📁 File Locations

Place these files in your project root directory (same folder as `document_to_audiobook.py`):

```
TTSProgram/
├── document_to_audiobook.py
├── .env                    ← Your API key goes here
├── .env.example           ← Template file
├── config.py              ← Optional: copy from config.example.py
├── documents/             ← Your input documents
└── audios/               ← Generated audiobooks
```

## 🔧 Configuration Priority

The system checks for your API key in this order:
1. Command line argument (`--openrouter-api-key`)
2. Environment variable (`OPENROUTER_API_KEY`)
3. `.env` file in project root
4. `config.py` file (if you create one)

## 💡 Usage Examples

```bash
# Basic usage (uses .env file)
python document_to_audiobook.py

# Specify model
python document_to_audiobook.py --llm-model anthropic/claude-3-haiku

# Without LLM preprocessing
python document_to_audiobook.py --disable-llm-preprocessing
```

## 🛠️ Troubleshooting

**"No OpenRouter API key found"**
- Check your `.env` file exists and has the correct format
- Verify the API key is valid on OpenRouter website
- Make sure `.env` file is in the same directory as the script

**"LLM preprocessing failed"**
- Check your OpenRouter account has sufficient credits
- Try a different model (e.g., `anthropic/claude-3-haiku`)
- The system will automatically fall back to basic processing

**"Permission denied" or file not found**
- Ensure `.env` file is in the project root directory
- Check file permissions (should be readable)

## 📚 More Information

- Full documentation: `README_OPENROUTER.md`
- Configuration options: `config.example.py`
- Usage examples: `python example_usage.py`
