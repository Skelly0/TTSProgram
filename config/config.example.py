#!/usr/bin/env python3
"""
Example configuration file for Document to Audiobook Converter
Copy this to config.py and customize your settings.
"""

# OpenRouter API Configuration
OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # Get from https://openrouter.ai/keys

# LLM Model Selection
# Popular models for text preprocessing:
# - "anthropic/claude-3.5-sonnet" (recommended for quality)
# - "anthropic/claude-3-haiku" (faster, cheaper)
# - "openai/gpt-4o" (good quality)
# - "openai/gpt-3.5-turbo" (faster, cheaper)
# - "meta-llama/llama-3.1-8b-instruct" (open source)
LLM_MODEL = "qwen/qwen3-235b-a22b"

# Site information for OpenRouter rankings (optional)
SITE_URL = "https://your-site.com"  # Optional: your site URL
SITE_TITLE = "Document to Audiobook Converter"  # Optional: your app name

# TTS Configuration
VOICE = "af_heart"  # af_heart, af_bella, af_sarah, am_adam, am_michael
LANG_CODE = "a"     # 'a' for American English, 'b' for British English
SPEED = 1.0         # Speech speed multiplier

# Processing Configuration
ENABLE_LLM_PREPROCESSING = True  # Set to False to disable LLM preprocessing
MAX_CHUNK_SIZE = 600            # Words per chunk for LLM processing (600-800 recommended)

# Directory Configuration
DOCUMENTS_DIR = "documents"
AUDIOS_DIR = "audios"

# Logging Configuration
VERBOSE_LOGGING = False  # Set to True for debug output
