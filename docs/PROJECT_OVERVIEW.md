# Document to Audiobook Converter - Project Overview

## What This Program Does

This program converts text documents (like books, articles, or stories) into high-quality audiobooks using the Kokoro-82M text-to-speech model. Simply put your text files in the `documents` folder, run the program, and get professional-sounding audiobooks in the `audios` folder.

## Quick Start

### Option 1: Easy Setup (Recommended)
1. Run the setup script: `python setup.py`
2. Add your text files to the `documents` folder
3. Double-click `run_converter.bat` (Windows) or run `./run_converter.sh` (Mac/Linux)

### Option 2: Manual Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install eSpeak (see README.md for instructions)
3. Add documents to `documents` folder
4. Run: `python document_to_audiobook.py`

## File Structure

```
TTSProgram/
├── document_to_audiobook.py    # Main program
├── requirements.txt            # Python dependencies
├── setup.py                   # Setup helper script
├── run_converter.bat          # Windows runner script
├── run_converter.sh           # Unix/Linux/Mac runner script
├── README.md                  # Detailed documentation
├── PROJECT_OVERVIEW.md        # This file
├── documents/                 # Put your text files here
│   └── sample_story.txt       # Example document
└── audios/                    # Generated audiobooks appear here
```

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown files
- `.rtf` - Rich text format files

## Voice Options

The program includes 5 different voices:
- **af_heart** (default) - Female voice
- **af_bella** - Alternative female voice
- **af_sarah** - Another female voice
- **am_adam** - Male voice
- **am_michael** - Alternative male voice

## Example Usage

### Basic conversion:
```bash
python document_to_audiobook.py
```

### With custom voice:
```bash
python document_to_audiobook.py --voice am_adam
```

### With British accent:
```bash
python document_to_audiobook.py --lang-code b --voice af_bella
```

### Slower speech:
```bash
python document_to_audiobook.py --speed 0.8
```

## What Happens During Conversion

1. **Scans** the `documents` folder for text files
2. **Reads** and cleans the text (removes formatting, fixes encoding)
3. **Splits** long documents into manageable chunks
4. **Converts** each chunk to speech using Kokoro-82M
5. **Combines** all audio chunks into a single file
6. **Saves** the final audiobook as a WAV file in `audios` folder

## Performance Notes

- **First run**: May take longer as the AI model downloads and initializes
- **Large files**: Automatically split into chunks for efficient processing
- **GPU acceleration**: Uses GPU if available for faster conversion
- **Memory usage**: Optimized to handle large documents without excessive memory use

## Troubleshooting

### "No supported documents found"
- Make sure your files are in the `documents` folder
- Check that files have supported extensions (.txt, .md, .rtf)

### "Failed to initialize Kokoro pipeline"
- Install eSpeak (see README.md for platform-specific instructions)
- Ensure all Python dependencies are installed

### "Permission denied" or script won't run
- On Mac/Linux: Make the script executable with `chmod +x run_converter.sh`
- On Windows: Right-click `run_converter.bat` and "Run as administrator" if needed

## Tips for Best Results

1. **Clean text**: Remove excessive formatting before conversion
2. **Reasonable length**: Very long documents work fine but take longer
3. **Proper punctuation**: Helps the TTS model with natural speech rhythm
4. **Chapter breaks**: Consider splitting very long books into chapters

## Technical Requirements

- **Python**: 3.7 or higher
- **eSpeak**: Text-to-speech engine (platform-specific installation)
- **Storage**: ~500MB for the AI model (downloaded automatically)
- **RAM**: 2GB+ recommended for large documents

## Getting Help

- Check `README.md` for detailed documentation
- Run with `--verbose` flag for detailed logging
- Ensure all dependencies are properly installed
- Verify eSpeak is working: `espeak "test"` should produce audio

## License & Credits

- Uses the Kokoro-82M model (Apache-2.0 license)
- Built with Python, PyTorch, and eSpeak
- See README.md for full acknowledgments
