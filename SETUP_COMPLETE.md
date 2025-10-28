# Setup Complete! ✅

## What Was Fixed

### 1. Fixed `uv sync` Issue
**Problem:** The `pyproject.toml` had a build system configuration with `packages = []` that caused `uv sync` to fail.

**Solution:** Removed the `[build-system]` and `[tool.hatch.build.targets.wheel]` sections since this is a collection of scripts/notebooks, not an installable package.

**File Changed:** `pyproject.toml`

### 2. Fixed Notebook Import Issue
**Problem:** Cell 4 used `BitsAndBytesConfig` without importing it.

**Solution:** Removed the unused quantization config code from the model loading cell.

**File Changed:** `investigate_qwen.ipynb` (Cell 4)

### 3. Downloaded Qwen3-32B Model
**Model:** Qwen/Qwen3-32B (62GB)
**Location:** `/workspace/models/Qwen3-32B`
**Method:** HuggingFace fast transfer (hf-transfer)

## Current Status

✅ All dependencies installed (152 packages)
✅ Qwen3-32B model downloaded (62GB)
✅ Notebook configured correctly
✅ Model path: `/workspace/models/Qwen3-32B`

## Dependencies Installed

- **Core:** nnsight 0.5.9
- **PyTorch:** 2.9.0 with CUDA 12.8
- **Transformers:** 4.57.1
- **Accelerate:** 1.11.0
- **Fast Transfer:** hf-transfer 0.1.9
- **Jupyter:** Full stack (notebook, lab, ipykernel)
- **Quantization:** bitsandbytes 0.48.1
- **Plus:** 140+ other dependencies

## Your Notebook is Ready!

The `investigate_qwen.ipynb` notebook is configured to load the model from:
```python
model = LanguageModel(
    "/workspace/models/Qwen3-32B",  # Local model path
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    dispatch=True
)
```

### To Use the Notebook:

1. **Start Jupyter:**
   ```bash
   cd /root/gemma-investigation
   uv run jupyter notebook
   ```

2. **Or use JupyterLab:**
   ```bash
   cd /root/gemma-investigation
   uv run jupyter lab
   ```

3. **Open:** `investigate_qwen.ipynb`

4. **Run cells in order:**
   - Cell 1: Authentication (optional)
   - Cell 2: Model loading - will use the downloaded model
   - Cell 3+: Analysis functions and experiments

## Model Information

**Model:** Qwen3-32B (Envoy architecture)
- **Parameters:** 31.98B
- **Layers:** 64
- **Hidden Size:** 5120
- **Attention Heads:** 64
- **KV Heads:** 8
- **Vocab Size:** 151,936
- **Max Position:** 40,960 tokens

## Memory Requirements

- **BF16 (current config):** ~64GB VRAM
- **Expected on A100:** Should fit comfortably on 80GB A100

## Functions Available in Notebook

1. **`talk_to_model(prompt, max_new_tokens, system_prompt, enable_thinking)`**
   - Generate responses from the model

2. **`get_logit_lens(prompt, system_prompt, token_lookback, enable_thinking)`**
   - Analyze internal layer predictions

3. **`talk_to_model_prefilled(user_message, prefilled_response, ...)`**
   - Generate with pre-filled assistant response

4. **`get_logit_lens_prefilled(user_message, prefilled_response, ...)`**
   - Logit lens with pre-filled context

5. **`get_token_probability(user_message, prefilled_response, top_k)`**
   - Get probability distribution over next tokens

## Notes

- The model uses Qwen's chat template with thinking mode support
- Set `enable_thinking=True` to use chain-of-thought reasoning
- Set `enable_thinking=False` for direct answers
- All functions support custom system prompts for behavior steering

## If You Need to Re-download

```bash
cd /root/gemma-investigation
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
snapshot_download('Qwen/Qwen3-32B', local_dir='/workspace/models/Qwen3-32B')
"
```

## Troubleshooting

**If model doesn't load:**
- Check GPU memory: `nvidia-smi`
- Kill existing processes: `kill -9 <PID>`
- Restart Jupyter kernel

**If packages are missing:**
```bash
cd /root/gemma-investigation
uv sync
```

**If you need to add dependencies:**
```bash
uv add <package-name>
```

