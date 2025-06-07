# Preprocessing Feature Documentation

## Overview

The Document to Audiobook Converter now includes a preprocessing feature that saves processed versions of your documents to a separate folder before converting them to audio. This allows you to:

- Keep original documents unchanged
- Review how documents are processed before TTS conversion
- Compare original vs. preprocessed content
- Debug preprocessing issues
- Reuse preprocessed documents

## How It Works

### Basic Preprocessing (Default)
When LLM preprocessing is disabled, the system performs basic text cleaning:
- Removes markdown formatting (headers, bold, italic, links)
- Normalizes whitespace and special characters
- Cleans up sentence structure
- Converts footnotes to inline references

### LLM Preprocessing (With OpenRouter API)
When LLM preprocessing is enabled with an OpenRouter API key:
- Converts text to SSML (Speech Synthesis Markup Language)
- Optimizes content for Kokoro-82M TTS model
- Adds pronunciation guides for fantasy names
- Structures content with appropriate breaks and emphasis
- Saves as `.ssml` files

## Usage

### Command Line Options

```bash
# Enable preprocessing (default behavior)
python document_to_audiobook.py --save-preprocessed

# Disable preprocessing
python document_to_audiobook.py --no-save-preprocessed

# Custom preprocessed directory
python document_to_audiobook.py --preprocessed-dir my_processed_docs

# Example with all options
python document_to_audiobook.py \
    --documents-dir documents \
    --audios-dir audios \
    --preprocessed-dir preprocessed \
    --save-preprocessed \
    --disable-llm-preprocessing
```

### Directory Structure

```
your-project/
├── documents/           # Original documents
│   ├── chapter1.txt
│   └── chapter2.md
├── preprocessed/        # Processed documents (new!)
│   ├── chapter1_preprocessed.txt
│   └── chapter2_preprocessed.txt
└── audios/             # Generated audio files
    ├── chapter1.wav
    └── chapter2.wav
```

### File Naming Convention

- **Basic preprocessing**: `original_name_preprocessed.ext`
- **LLM preprocessing**: `original_name_preprocessed.ssml`

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--save-preprocessed` | `True` | Enable saving preprocessed documents |
| `--no-save-preprocessed` | `False` | Disable saving preprocessed documents |
| `--preprocessed-dir` | `"preprocessed"` | Directory for preprocessed documents |

## Examples

### Example 1: Basic Usage
```bash
python document_to_audiobook.py
```
- Processes documents with basic preprocessing
- Saves to `preprocessed/` folder
- Creates audio files in `audios/` folder

### Example 2: LLM Preprocessing
```bash
python document_to_audiobook.py --openrouter-api-key your_key_here
```
- Uses LLM to create SSML-formatted documents
- Saves as `.ssml` files in `preprocessed/` folder
- Optimized for better TTS output

### Example 3: Custom Directories
```bash
python document_to_audiobook.py \
    --documents-dir my_books \
    --preprocessed-dir processed_books \
    --audios-dir audiobooks
```

## Benefits

1. **Transparency**: See exactly how your documents are processed
2. **Debugging**: Identify preprocessing issues before audio generation
3. **Reusability**: Preprocessed documents can be reused or manually edited
4. **Comparison**: Compare original vs. processed content
5. **Backup**: Original documents remain untouched

## File Size Comparison

Preprocessing typically results in:
- **Basic preprocessing**: 5-10% reduction in file size (formatting removal)
- **LLM preprocessing**: Variable size change depending on SSML markup added

## Troubleshooting

### Preprocessed folder not created
- Ensure you have write permissions in the target directory
- Check that `--save-preprocessed` is enabled (default)

### Empty preprocessed files
- Check that original documents contain readable text
- Verify file encoding (UTF-8 recommended)

### SSML files not generated
- Ensure OpenRouter API key is provided and valid
- Check that LLM preprocessing is enabled (default when API key provided)

## Technical Details

The preprocessing feature is implemented in the [`DocumentToAudiobookConverter`](document_to_audiobook.py:300) class:

- [`save_preprocessed_document()`](document_to_audiobook.py:498) method handles file saving
- [`preprocess_text()`](document_to_audiobook.py:458) method performs text processing
- [`preprocess_text_chunk()`](document_to_audiobook.py:196) method handles LLM processing

## Testing

You can test the preprocessing feature using the included test script:

```bash
python test_preprocessing.py
```

This will create temporary documents and verify that preprocessing works correctly.
