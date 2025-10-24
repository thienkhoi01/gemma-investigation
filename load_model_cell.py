from nnsight import LanguageModel
import torch
import os

# Set cache location to where we downloaded the model
os.environ['HF_HOME'] = '/workspace/.hf_home'

print("Loading Gemma-2-9B model from cache... (this should be fast now!)")

model = LanguageModel(
    "google/gemma-2-9b",
    device_map="cuda",  # Load directly to GPU
    local_files_only=True  # Use only cached files, don't try to download
)

print(f"✓ Model loaded successfully!")
print(f"Model type: {type(model.model).__name__}")
print(f"Number of parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print(f"Device: {next(model.model.parameters()).device}")

