# Lambda Cloud GH200 Setup Instructions

## Instance Details

- **IP**: 192.222.58.183
- **SSH**: `ssh lambda-gh200`
- **GPU**: NVIDIA GH200 480GB (97GB memory available)
- **CUDA**: 12.8
- **Architecture**: ARM64 (aarch64)

## Step 1: Clone Repository

```bash
ssh lambda-gh200
cd ~
git clone https://github.com/JacobFV/recursive-kimi-linear.git
cd recursive-kimi-linear
```

## Step 2: Set Up Python Environment

```bash
# Update system packages
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip

# Install PyTorch for ARM64 with CUDA support
# Note: GH200 is ARM64, check PyTorch compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install flash-linear-attention transformers huggingface-hub einops accelerate
```

**Important**: GH200 is ARM64 architecture. Verify PyTorch has ARM64 CUDA builds. If not available, you may need to build from source or use CPU version for weight conversion.

## Step 3: Download Hugging Face Weights

```bash
# Login to Hugging Face (if needed)
huggingface-cli login

# Download the Kimi Linear model
python3 - << EOF
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

Or using git-lfs:
```bash
sudo apt install -y git-lfs
git lfs install
git clone https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct models/kimi-linear-48b
```

## Step 4: Convert Weights to Custom Implementation

Create a conversion script:

```bash
cat > convert_weights.py << 'EOF'
"""Convert Hugging Face weights to custom KDA implementation."""
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

# Paths
hf_model_path = "./models/kimi-linear-48b"
output_path = "./checkpoints/kda_custom.bin"

print(f"Loading model from {hf_model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    hf_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Extract state dict
state_dict = model.state_dict()

# Create mapping from HF format to our custom format
custom_state_dict = {}

for key, value in state_dict.items():
    # Map attention layers (KDA)
    if 'attn' in key:
        new_key = key.replace('model.model.layers.', 'layers.')
        custom_state_dict[new_key] = value
    else:
        # Keep other layers as-is
        new_key = key.replace('model.', '')
        custom_state_dict[new_key] = value

# Save converted checkpoint
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(custom_state_dict, output_path)
print(f"Converted weights saved to {output_path}")
print(f"Total parameters: {sum(p.numel() for p in custom_state_dict.values()):,}")
EOF

python3 convert_weights.py
```

## Step 5: Verify Setup

```bash
# Test KDA layer import
python3 - << EOF
try:
    from fla.layers.kda import KimiDeltaAttention
    print("✓ KDA layer imported successfully")
    
    # Test with small tensor
    import torch
    attn = KimiDeltaAttention(hidden_size=2048, num_heads=16, head_dim=128, mode='chunk')
    x = torch.randn(1, 128, 2048)
    out, _, _ = attn(x)
    print(f"✓ KDA layer working! Input: {x.shape}, Output: {out.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOF
```

## Step 6: Test GPU

```bash
# Verify GPU is accessible
python3 - << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

## Troubleshooting

### ARM64 PyTorch Issues
GH200 uses ARM64. If PyTorch doesn't have ARM64 CUDA builds:
- Use CPU version for weight conversion: `pip install torch` (CPU)
- For inference, you may need to build PyTorch from source or use alternative frameworks

### Memory Issues
- The model is 48B parameters (even with sparse 3B activation)
- Use CPU offloading or quantization if needed
- Consider using bfloat16 or int8 quantization

### Flash Linear Attention
- Verify `flash-linear-attention` works on ARM64
- May need to install from source if no ARM64 wheels available

## Next Steps

Once weights are converted and verified:
1. Load the converted checkpoint
2. Test inference with a sample prompt
3. Benchmark performance on the GH200

## Managing the Instance

### Terminate instance (to save costs):
```bash
# Via Lambda Cloud dashboard or API
# Always terminate when not in use - GH200 instances are expensive!
```

### Check instance status:
Visit https://cloud.lambdalabs.com/instances

