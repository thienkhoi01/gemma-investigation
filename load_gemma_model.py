"""
Load Gemma-2-9B model using nnsight for investigation

REQUIREMENTS:
1. Accept the license at: https://huggingface.co/google/gemma-2-9b
2. Set your HF token as an environment variable or provide it when prompted

To set your HF token:
    export HF_TOKEN=your_token_here
    
Or run:
    huggingface-cli login
"""

import os
from nnsight import LanguageModel
from huggingface_hub import login

# Check for HF token
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

if hf_token:
    print("Found HF token in environment, logging in...")
    login(token=hf_token)
    print("✓ Logged in successfully!")
else:
    print("=" * 60)
    print("AUTHENTICATION REQUIRED")
    print("=" * 60)
    print("\nThe Gemma-2-9B model requires authentication.")
    print("\nOption 1: Set environment variable")
    print("  export HF_TOKEN=your_token_here")
    print("  python3 load_gemma_model.py")
    print("\nOption 2: Login via CLI")
    print("  huggingface-cli login")
    print("  python3 load_gemma_model.py")
    print("\nOption 3: Enter token now")
    token_input = input("\nEnter your HF token (or press Enter to skip): ").strip()
    
    if token_input:
        login(token=token_input)
        print("✓ Logged in successfully!")
    else:
        print("\n⚠ No token provided. Please authenticate and try again.")
        print("\nGet your token at: https://huggingface.co/settings/tokens")
        print("Accept the license at: https://huggingface.co/google/gemma-2-9b")
        exit(1)

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)
print("\nLoading Gemma-2-9B model from Hugging Face using nnsight...")
print("This may take a few minutes on first load...")

# Load the model using nnsight
model = LanguageModel(
    "google/gemma-2-9b",
    device_map="auto",
    torch_dtype="auto"
)

print(f"\n✓ Model loaded successfully!")
print(f"\nModel: {model.model_name}")
print(f"Tokenizer vocab size: {len(model.tokenizer)}")
print(f"Device map: {model.model.hf_device_map if hasattr(model.model, 'hf_device_map') else 'N/A'}")

# Print model architecture info
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(f"\nModel type: {type(model.model).__name__}")
print(f"\nKey config parameters:")
config = model.model.config
print(f"  num_hidden_layers: {config.num_hidden_layers}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
print(f"  intermediate_size: {config.intermediate_size}")
print(f"  vocab_size: {config.vocab_size}")
print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")

# Show the model structure
print("\n" + "=" * 60)
print("MODEL STRUCTURE (top level modules)")
print("=" * 60)
for name, module in model.model.named_children():
    print(f"\n{name}: {type(module).__name__}")

# Show layer structure
print("\n" + "=" * 60)
print("LAYER STRUCTURE")
print("=" * 60)
if hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
    print(f"\nNumber of transformer layers: {len(model.model.model.layers)}")
    if len(model.model.model.layers) > 0:
        print(f"\nFirst layer modules:")
        for name, module in model.model.model.layers[0].named_children():
            print(f"  {name}: {type(module).__name__}")

print("\n" + "=" * 60)
print("READY FOR INVESTIGATION")
print("=" * 60)
print("\nThe model is now loaded and ready for investigation!")
print("\nYou can:")
print("  - Use model.trace() to trace model execution")
print("  - Access layers via model.model.model.layers[i]")
print("  - Inspect activations, attention patterns, etc.")
print("\nExample usage:")
print('''
# Trace model execution with a prompt
with model.trace("Hello, how are you?") as tracer:
    # Access hidden states from first layer
    hidden_states = model.model.model.layers[0].output[0].save()
    
    # Access attention outputs
    attn_output = model.model.model.layers[0].self_attn.output.save()

# Get the saved values
print("Hidden states shape:", hidden_states.value.shape)
print("Attention output shape:", attn_output.value.shape)

# Generate text
output = model.generate("The capital of France is", max_new_tokens=10)
print("Generated:", output)
''')

print("\nModel object is available as 'model' variable")
print("You can now investigate the model interactively in Python!\n")
