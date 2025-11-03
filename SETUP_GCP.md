# GCP Setup Instructions

This guide walks you through setting up the Kimi Linear implementation on a GCP GPU instance.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- GitHub account

## Step 1: Clone the Repository

Once you've SSH'd into your GCP instance, clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/recursive-kimi-linear.git
cd recursive-kimi-linear
```

## Step 2: Set Up Python Environment

```bash
# Install Python 3.10+ (if not already installed)
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio  # CPU version (no CUDA needed)
pip install flash-linear-attention transformers huggingface-hub einops
```

**Note**: Since this is a CPU-only instance, we're installing CPU versions of PyTorch. This is fine for model loading, weight conversion, and testing. For actual training/inference with GPU, you'll need a GPU instance.

## Step 3: Download Hugging Face Weights

Download the Kimi Linear model weights from Hugging Face:

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Login to Hugging Face (you'll need a token from https://huggingface.co/settings/tokens)
huggingface-cli login

# Download the model
python - << EOF
from huggingface_hub import snapshot_download

model_id = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
local_dir = "./models/kimi-linear-48b"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f"Model downloaded to {local_dir}")
EOF
```

Alternatively, using `git-lfs`:

```bash
# Install git-lfs
sudo apt install -y git-lfs
git lfs install

# Clone the model repository
git clone https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct models/kimi-linear-48b
```

## Step 4: Copy Parameters to Custom Implementation

The Hugging Face model uses a specific configuration format. To use it with our custom KDA implementation, you need to:

1. **Load the original model state dict**
2. **Map parameters to our custom implementation**
3. **Save the converted checkpoint**

Create a conversion script:

```bash
cat > convert_weights.py << 'EOF'
"""Convert Hugging Face weights to our custom KDA implementation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Paths
hf_model_path = "./models/kimi-linear-48b"
output_path = "./checkpoints/kda_custom.bin"

# Load original model
print(f"Loading model from {hf_model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    hf_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Extract state dict
state_dict = model.state_dict()

# Create mapping from HF format to our custom format
# The KDA layers are in model.model.layers.{i}.attn
custom_state_dict = {}

for key, value in state_dict.items():
    # Map attention layers (KDA)
    if 'attn' in key and 'kda' in key.lower() or 'attn' in key:
        # Rename to match our custom implementation
        new_key = key.replace('model.model.layers.', 'layers.')
        # If it's a KDA layer, ensure proper naming
        if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key:
            custom_state_dict[new_key] = value
        elif 'A_log' in key or 'dt_bias' in key:
            custom_state_dict[new_key] = value
        elif 'f_proj' in key or 'b_proj' in key or 'g_proj' in key:
            custom_state_dict[new_key] = value
        elif 'o_proj' in key:
            custom_state_dict[new_key] = value
        else:
            custom_state_dict[new_key] = value
    else:
        # Keep other layers as-is (embeddings, norm, MLP, etc.)
        new_key = key.replace('model.', '')
        custom_state_dict[new_key] = value

# Save converted checkpoint
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(custom_state_dict, output_path)
print(f"Converted weights saved to {output_path}")
print(f"Total parameters: {sum(p.numel() for p in custom_state_dict.values()):,}")
EOF

python convert_weights.py
```

## Step 5: Verify Installation

Test the setup:

```bash
python - << EOF
import torch
from fla.layers.kda import KimiDeltaAttention

# Test KDA layer
attn = KimiDeltaAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    mode='chunk'
)

x = torch.randn(1, 128, 2048)
out, _, _ = attn(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print("✓ KDA layer working correctly!")
EOF
```

## Step 6: Run Inference

Create an inference script:

```bash
cat > inference.py << 'EOF'
"""Run inference with the converted model."""
import torch
from transformers import AutoTokenizer
from pathlib import Path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/kimi-linear-48b", trust_remote_code=True)

# Load converted model
checkpoint = torch.load("./checkpoints/kda_custom.bin")

# Example inference
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

print("Model loaded successfully!")
print(f"Prompt: {prompt}")
# Add your inference code here
EOF

python inference.py
```

## Troubleshooting

### CUDA Out of Memory
- Use smaller batch sizes
- Enable gradient checkpointing
- Use mixed precision (BF16)

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt` (if created)
- Check Python version: `python --version` (should be 3.10+)

### Model Conversion Issues
- Verify the model structure matches: `print(model)`
- Check layer names in state dict: `print(list(state_dict.keys())[:10])`

## Additional Resources

- [KDA Algorithm Documentation](./KDA_ALGORITHM_DOCUMENTATION.md)
- [Implementation Notes](./IMPLEMENTATION_NOTES.md)
- [Architecture Summary](./ARCHITECTURE_SUMMARY.md)

