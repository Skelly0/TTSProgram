# Troubleshooting Guide for TTS Program

## Common Issues and Solutions

### 1. Kokoro Model Download Failures

**Error:** `An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache`

**Possible Causes:**
- Network connectivity issues
- Hugging Face Hub temporary unavailability
- Corrupted model cache
- Authentication issues
- Insufficient disk space

**Solutions:**

#### Quick Fix
1. **Check Internet Connection:**
   ```bash
   ping huggingface.co
   ```

2. **Try the Model Download Script:**
   ```bash
   python download_model.py
   ```

3. **Clear Hugging Face Cache:**
   ```bash
   huggingface-cli delete-cache
   ```

4. **Login to Hugging Face (if needed):**
   ```bash
   huggingface-cli login
   ```

#### Manual Model Download
If automatic download fails, you can manually download the model:

```python
from huggingface_hub import snapshot_download

# Download the main Kokoro model
cache_dir = snapshot_download(
    repo_id="hexgrad/Kokoro-82M",
    cache_dir=None,  # Uses default cache
    resume_download=True
)
print(f"Model downloaded to: {cache_dir}")
```

#### Alternative Models
If the main model doesn't work, try these alternatives:
- `onnx-community/Kokoro-82M-v1.0-ONNX` (ONNX version)
- Use local TTS alternatives like `espeak` or `festival`

### 2. Import Errors

**Error:** `ImportError: No module named 'kokoro'`

**Solution:**
```bash
pip install kokoro==0.9.4 soundfile torch
```

### 3. Audio Output Issues

**Error:** Audio files are not being created or are corrupted

**Solutions:**
1. Check output directory permissions
2. Ensure sufficient disk space
3. Try different audio formats
4. Check if `soundfile` is properly installed

### 4. Performance Issues

**Symptoms:** Very slow processing or high memory usage

**Solutions:**
1. **Reduce chunk size** in the script
2. **Use GPU acceleration** if available:
   ```python
   # Check if CUDA is available
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```
3. **Process smaller documents** first
4. **Close other applications** to free memory

### 5. Voice and Language Issues

**Error:** Unsupported voice or language codes

**Available Options:**
- **Voices:** `af_heart`, `af_bella`, `af_sarah`, `am_adam`, `am_michael`
- **Language Codes:** `a` (American English), `b` (British English)

### 6. Network-Specific Issues

#### Corporate Networks/Firewalls
If you're behind a corporate firewall:

1. **Configure proxy settings:**
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=https://your-proxy:port
   ```

2. **Use alternative download methods:**
   ```bash
   # Download via git (if git is allowed)
   git lfs clone https://huggingface.co/hexgrad/Kokoro-82M
   ```

#### VPN Issues
If using a VPN:
1. Try disconnecting the VPN temporarily
2. Use a different VPN server location
3. Check if the VPN blocks Hugging Face domains

### 7. Disk Space Issues

**Check available space:**
```bash
# Windows
dir C:\ 

# Linux/Mac
df -h
```

**Hugging Face cache location:**
- Windows: `C:\Users\{username}\.cache\huggingface`
- Linux/Mac: `~/.cache/huggingface`

### 8. Alternative TTS Solutions

If Kokoro continues to fail, consider these alternatives:

#### 1. Google Text-to-Speech (gTTS)
```bash
pip install gtts
```

#### 2. Microsoft Speech Platform
```bash
pip install pyttsx3
```

#### 3. Amazon Polly
```bash
pip install boto3
```

#### 4. Local TTS (espeak)
```bash
# Windows (via chocolatey)
choco install espeak

# Linux
sudo apt-get install espeak

# Mac
brew install espeak
```

### 9. Debug Mode

Run the script with additional debugging:

```bash
# Enable verbose logging
python document_to_audiobook.py --lang-code b --voice af_bella --debug

# Check Python environment
python -c "import sys; print(sys.path)"
python -c "import kokoro; print(kokoro.__file__)"
```

### 10. Getting Help

If none of these solutions work:

1. **Check the GitHub Issues:** Look for similar problems
2. **Create a detailed bug report** including:
   - Operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce
   - Network environment details

3. **Contact Support:** Include logs and system information

## Prevention Tips

1. **Regular Updates:**
   ```bash
   pip install --upgrade kokoro soundfile torch
   ```

2. **Stable Internet:** Ensure stable connection during model downloads

3. **Sufficient Storage:** Keep at least 2GB free space for models

4. **Environment Management:** Use virtual environments to avoid conflicts

5. **Backup Cache:** Backup working model cache for offline use
