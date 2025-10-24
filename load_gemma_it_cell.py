# REPLACE YOUR NOTEBOOK CELL 3 WITH THIS CODE
# This loads the instruction-tuned Gemma-2-9B-IT model

import os
os.environ['HF_HOME'] = '/workspace/.hf_home'
# Set HF_TOKEN from your environment or use: huggingface-cli login
# os.environ['HF_TOKEN'] = 'your_token_here'

from nnsight import LanguageModel
import torch

print("Loading Gemma-2-9B-IT (instruction-tuned) model from cache...")

model = LanguageModel(
    "google/gemma-2-9b-it",  # <-- Now using the IT version
    device_map="cuda",
)

print(f"\n✓ Model loaded successfully!")
print(f"Model: Gemma-2-9B-IT (Instruction-Tuned)")
print(f"Total parameters: {sum(p.numel() for p in model.model.parameters()):,} ({sum(p.numel() for p in model.model.parameters()) / 1e9:.2f}B)")
print(f"Device: cuda (RTX 5090)")
print(f"\nThis is the instruction-tuned version - optimized for chat and instructions!")

