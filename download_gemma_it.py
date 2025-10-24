#!/usr/bin/env python3
"""
Download gemma-2-9b-it model

Usage:
  export HF_TOKEN=your_token_here
  python3 download_gemma_it.py

Or run from notebook/script that has already authenticated
"""

import os
import sys
from huggingface_hub import snapshot_download, login, HfFolder

# Try to get token from multiple sources
hf_token = None

# 1. Check environment variable
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

# 2. Check if already logged in via HfFolder
if not hf_token:
    hf_token = HfFolder.get_token()

if hf_token:
    login(token=hf_token)
    print("✓ Logged in with HF token")
else:
    print("=" * 60)
    print("ERROR: No HF token found")
    print("=" * 60)
    print("\nPlease authenticate first:")
    print("  Option 1: export HF_TOKEN=your_token_here")
    print("  Option 2: huggingface-cli login")
    print("  Option 3: Run login cell in notebook first")
    print("\nGet your token at: https://huggingface.co/settings/tokens")
    print("Accept license at: https://huggingface.co/google/gemma-2-9b-it")
    sys.exit(1)

print("\n" + "=" * 60)
print("DOWNLOADING: google/gemma-2-9b-it")
print("=" * 60)
print("This will take several minutes (~18GB download)...\n")

try:
    cache_dir = snapshot_download(
        "google/gemma-2-9b-it",
        cache_dir="/workspace/.hf_home",
        resume_download=True,
        max_workers=4,
    )
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Model cached at: {cache_dir}")
    print("\nYou can now load the model with:")
    print('  model = LanguageModel("google/gemma-2-9b-it", device_map="cuda")')
    
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nMake sure you have accepted the license at:")
    print("  https://huggingface.co/google/gemma-2-9b-it")
    sys.exit(1)
