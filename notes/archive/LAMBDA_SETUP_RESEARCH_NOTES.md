# Lambda Cloud GH200 Setup - Researcher Notes

**Date**: November 2025  
**Instance**: Lambda Cloud GH200 480GB (ARM64/aarch64)  
**Objective**: Set up PyTorch with CUDA support and KDA implementation for Kimi Linear 48B model

## Initial Setup Attempt

### Environment Discovery
- **OS**: Ubuntu 22.04 (jammy)
- **Architecture**: ARM64 (aarch64)
- **GPU**: NVIDIA GH200 480GB (97GB memory available)
- **CUDA Version**: 12.8 (confirmed via `nvcc --version`)
- **Python**: 3.10.12 (system default)

### First Challenge: ARM64 CUDA Support in PyTorch

**Initial Assumption**: PyTorch does not support CUDA on ARM64 architectures.

**Initial Setup**:
1. Created Python 3.10 virtual environment
2. Installed PyTorch from standard pip repository
   - Result: CPU-only build (`torch-2.9.0+cpu`)
   - `torch.cuda.is_available()` returned `False`
   - GPU accessible via `nvidia-smi` but PyTorch couldn't use it

**Key Finding**: Standard PyTorch installation methods default to CPU-only builds for ARM64, even when CUDA is available on the system.

## Discovery: PyTorch CUDA Builds for ARM64

### Research Findings

Through investigation, discovered that **PyTorch does provide CUDA-enabled builds for ARM64**, but they require:
1. Using the specific CUDA index URL: `https://download.pytorch.org/whl/cu124`
2. Explicitly specifying the architecture-compatible wheel

**Available Builds**:
- PyTorch 2.4.0+ with CUDA 12.4 support for ARM64
- PyTorch 2.5.1 with CUDA 12.4 support (tested)
- Compatible with CUDA 12.8 (backward compatible)

### Solution Implementation

**Step 1: Uninstall CPU-only PyTorch**
```bash
pip uninstall -y torch torchvision torchaudio
```

**Step 2: Install CUDA-enabled PyTorch**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Result**: 
- PyTorch 2.5.1 with CUDA 12.4 support installed
- `torch.cuda.is_available()` returned `True`
- GPU successfully accessible: NVIDIA GH200 480GB
- CUDA version: 12.4 (compatible with system CUDA 12.8)

## Python Version Upgrade

### Motivation

Flash Linear Attention (`flash-linear-attention`) recommended Python 3.11+ for best experience. Warnings appeared during execution:
- `torch.compile` not available in Python 3.10
- Performance optimizations require Python 3.11+

### Implementation

**Step 1: Install Python 3.12**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

**Step 2: Recreate Virtual Environment**
```bash
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
```

**Step 3: Reinstall Dependencies**
- PyTorch 2.5.1 with CUDA 12.4 (ARM64 build)
- All dependencies reinstalled for Python 3.12

**Result**:
- Python 3.12.12 active
- No more Python version warnings
- `torch.compile` available (future optimization)

## Final Verification

### System Status

**Python Environment**:
```
Python 3.12.12
PyTorch: 2.5.1
CUDA available: True
CUDA version: 12.4
```

**GPU Access**:
```
GPU: NVIDIA GH200 480GB
GPU Memory: 101.47 GB available
Device: cuda:0
```

**KDA Layer Test**:
```python
from fla.layers.kda import KimiDeltaAttention
import torch

attn = KimiDeltaAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    mode='chunk'
).cuda()

x = torch.randn(1, 128, 2048).cuda()
out, _, _ = attn(x)
# ✓ Success: KDA layer working on GPU
```

## Key Learnings

### 1. ARM64 CUDA Support in PyTorch

**Finding**: PyTorch does support CUDA on ARM64, but requires explicit installation from CUDA-specific index.

**Implication**: Standard `pip install torch` defaults to CPU-only on ARM64, even when CUDA is available.

**Solution**: Use `--index-url https://download.pytorch.org/whl/cu124` for ARM64 CUDA builds.

### 2. CUDA Version Compatibility

**System CUDA**: 12.8  
**PyTorch CUDA**: 12.4  
**Result**: Compatible (CUDA maintains backward compatibility)

**Note**: PyTorch 12.4 build works with system CUDA 12.8.

### 3. GH200 Architecture

The NVIDIA GH200 is an ARM64-based GPU, which required:
- ARM64-compatible PyTorch builds
- ARM64-compatible Triton kernels
- Proper CUDA driver installation (pre-installed on Lambda Cloud)

### 4. Triton on ARM64

**Initial Concern**: Triton kernels might not work on ARM64.

**Finding**: 
- Triton installs successfully on ARM64
- Kernels compile and execute correctly
- No platform-specific issues encountered

### 5. Flash Linear Attention Compatibility

Flash Linear Attention (`flash-linear-attention`) works correctly on ARM64 with CUDA, including:
- KDA (Kimi Delta Attention) layer implementation
- Custom Triton kernels
- GPU acceleration

## Technical Specifications

### Final Environment

```
OS: Ubuntu 22.04 (ARM64/aarch64)
Python: 3.12.12
PyTorch: 2.5.1+cu124
CUDA: 12.4 (PyTorch) / 12.8 (System)
GPU: NVIDIA GH200 480GB (101.47 GB available)
Architecture: ARM64 (aarch64)
```

### Installed Packages

**Core**:
- torch==2.5.1 (CUDA 12.4)
- torchvision==0.20.1
- torchaudio==2.5.1
- transformers==4.57.1
- transformers>=4.45.0
- einops==0.8.1
- accelerate==1.11.0
- triton==3.5.0
- flash-linear-attention==0.4.0

**Model Support**:
- huggingface-hub==0.36.0
- safetensors==0.6.2
- tokenizers==0.22.1

## Model Setup

### Weight Conversion

**Source**: `moonshotai/Kimi-Linear-48B-A3B-Instruct`  
**Download Location**: `./models/kimi-linear-48b/` (92GB)  
**Converted Checkpoint**: `./checkpoints/kda_custom.bin` (92GB)  
**Parameters**: 49,122,681,728 (~49B parameters)

**Conversion Script**: `convert_weights.py`
- Loads Hugging Face model
- Maps attention layers to custom KDA format
- Saves state dict for custom implementation

## Performance Considerations

### GPU Memory

- **Available**: 101.47 GB
- **Model Size**: ~92 GB (FP16)
- **Buffer**: ~9 GB available for activations, batch processing

### Inference Capabilities

With 101GB GPU memory:
- Can load full 48B parameter model
- Sufficient memory for inference batches
- Room for gradient checkpointing if needed
- Potential for quantized inference modes

## Future Optimizations

### 1. torch.compile Support

Now that Python 3.12 is installed, `torch.compile` is available for:
- Kernel fusion
- Optimization passes
- Potential performance improvements

### 2. Quantization

Consider int8/bfloat16 quantization for:
- Faster inference
- Lower memory usage
- Enabling larger batch sizes

### 3. Tensor Parallelism

With 97GB GPU memory, may be able to:
- Use model parallelism if needed
- Split model across multiple GPUs (if available)
- Optimize for throughput

## Troubleshooting Reference

### Issue: PyTorch CUDA Not Available on ARM64

**Symptoms**:
- `torch.cuda.is_available()` returns `False`
- GPU visible in `nvidia-smi` but not accessible from PyTorch

**Solution**:
1. Uninstall CPU-only PyTorch: `pip uninstall torch torchvision torchaudio`
2. Install from CUDA index: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

### Issue: Python Version Warnings

**Symptoms**:
- Warnings about Python 3.10 being below recommended 3.11+
- `torch.compile` not available

**Solution**:
1. Install Python 3.12: `sudo apt install python3.12 python3.12-venv`
2. Recreate virtual environment: `python3.12 -m venv .venv`
3. Reinstall dependencies

### Issue: Triton Platform Warnings

**Symptoms**:
- Warnings about Triton not supported on platform

**Resolution**: These warnings resolved once CUDA-enabled PyTorch was installed. Triton kernels compile and execute correctly on ARM64 with CUDA.

## Conclusion

Successfully established a working environment for the Kimi Linear 48B model on Lambda Cloud GH200 with:
- ✅ GPU acceleration enabled
- ✅ Python 3.12 for optimal performance
- ✅ All dependencies installed and verified
- ✅ KDA layer tested and working on GPU
- ✅ Model weights downloaded and converted

The setup demonstrates that ARM64 GPUs (GH200) are fully supported for PyTorch CUDA workloads, contrary to initial assumptions. The key was using the correct PyTorch installation method for ARM64 CUDA builds.

## References

- [PyTorch ARM64 CUDA Support Issue](https://github.com/pytorch/pytorch/issues/134117)
- PyTorch CUDA wheels: `https://download.pytorch.org/whl/cu124`
- Lambda Cloud Documentation
- Flash Linear Attention Repository

---

**Researcher**: Setup and documentation  
**Date**: November 3, 2025  
**Environment**: Lambda Cloud GH200 Instance

