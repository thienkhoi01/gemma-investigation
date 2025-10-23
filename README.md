# Gemma-2-9B Investigation with nnsight

This project provides tools to load and investigate the Gemma-2-9B model from Hugging Face using nnsight.

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face

The Gemma-2-9B model is gated and requires authentication:

1. **Get access to the model**: Visit https://huggingface.co/google/gemma-2-9b and accept the license
2. **Get your HF token**: Visit https://huggingface.co/settings/tokens and create a token (read access is sufficient)
3. **Authenticate** using one of these methods:

**Option A: Environment variable (recommended)**
```bash
export HF_TOKEN=your_token_here
```

**Option B: CLI login**
```bash
huggingface-cli login
```

**Option C: In script/notebook**
```python
from huggingface_hub import login
login(token="your_token_here")
```

### 3. Load and Investigate the Model

**Using the Python script:**
```bash
python3 load_gemma_model.py
```

**Using the Jupyter notebook:**
```bash
jupyter notebook investigate_gemma.ipynb
```

## Files

- `requirements.txt` - Python dependencies
- `load_gemma_model.py` - Python script to load and explore the model
- `investigate_gemma.ipynb` - Interactive Jupyter notebook with examples

## What You Can Investigate

With nnsight, you can:

1. **Trace model execution** - Capture intermediate activations at any layer
2. **Inspect attention patterns** - Examine attention weights and outputs
3. **Analyze hidden states** - Track how representations evolve through layers
4. **Examine embeddings** - Investigate token and position embeddings
5. **Study MLP activations** - Look at feed-forward network internals
6. **Generate text** - Run inference and analyze the generation process

## Example Usage

```python
from nnsight import LanguageModel

# Load model
model = LanguageModel("google/gemma-2-9b", device_map="auto")

# Trace execution and capture hidden states
with model.trace("The capital of France is") as tracer:
    hidden_states = model.model.model.layers[0].output[0].save()
    
# Inspect the captured activations
print(hidden_states.value.shape)
```

## Model Architecture

**Gemma-2-9B Specifications:**
- Parameters: ~9 billion
- Architecture: Decoder-only transformer
- Layers: 42
- Hidden size: 3584
- Attention heads: 16
- KV heads: 8 (Grouped Query Attention)
- Vocabulary size: 256,000

## System Requirements

- **RAM**: Minimum 32GB recommended
- **GPU**: NVIDIA GPU with at least 20GB VRAM (for full model)
  - Alternatively, use CPU (slower) or model quantization
- **Storage**: ~20GB for model weights

## Troubleshooting

**401 Unauthorized Error:**
- Make sure you've accepted the model license at https://huggingface.co/google/gemma-2-9b
- Verify your HF token is valid and properly set

**Out of Memory:**
- Use `torch_dtype="float16"` or `torch_dtype="bfloat16"` for reduced memory
- Consider using model quantization (4-bit or 8-bit)
- Reduce batch size or sequence length

**Slow Loading:**
- First load downloads ~20GB of weights - this is normal
- Subsequent loads use cached weights and are much faster

## Resources

- [nnsight Documentation](https://nnsight.net/)
- [Gemma Model Card](https://huggingface.co/google/gemma-2-9b)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## License

This investigation code is provided as-is. The Gemma-2-9B model has its own license which you must accept to use it.

