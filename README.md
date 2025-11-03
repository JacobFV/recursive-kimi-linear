# Recursive Kimi-Linear

This repository contains comprehensive documentation and implementation of the **Kimi Delta Attention (KDA)** algorithm from the Kimi Linear architecture.

## Overview

**Kimi Linear** is a hybrid linear attention architecture that outperforms traditional full attention methods. It features:

- **Kimi Delta Attention (KDA)**: A refined linear attention mechanism with fine-grained gating
- **Hybrid Architecture**: 3:1 ratio of KDA to MLA layers
- **High Efficiency**: Up to 75% KV cache reduction, 6× faster decoding at 1M tokens
- **Superior Performance**: Outperforms full attention on various benchmarks

## Repository Contents

### Documentation

- **[KDA_ALGORITHM_DOCUMENTATION.md](./KDA_ALGORITHM_DOCUMENTATION.md)**: Comprehensive technical documentation of the KDA algorithm, including:
  - Mathematical formulation
  - Algorithm details
  - Implementation specifics
  - Performance optimizations

### Implementation

The KDA implementation has been copied from the [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention) repository:

- `fla/ops/kda/`: Core KDA kernels
  - `chunk.py`: Chunkwise computation for training
  - `fused_recurrent.py`: Recurrent computation for inference
  - `naive.py`: Reference implementations
  - `gate.py`: Fine-grained gating mechanism
  - `chunk_intra.py`: Intra-chunk attention computation
  - `chunk_inter.py`: Inter-chunk computation
  - `wy_fast.py`: WY representation utilities

- `fla/layers/kda.py`: High-level KDA layer implementation

- `fla/models/kda/`: Complete model implementations

## Key Algorithm Details

### Core Innovation: Fine-Grained Gating

KDA introduces **fine-grained gating** where each attention head and key dimension has independent forget gates:

```
g[t] = -exp(A_log) ⊙ softplus(f_proj(hidden_states) + dt_bias)
```

This is a key improvement over Gated DeltaNet which applies gating uniformly per head.

### State Update Rule

The recurrent state update follows the delta rule:

```
S[t] = S[t-1] ⊙ exp(g[t]) + β[t] · k[t] ⊗ (v[t] - k[t]^T · S[t-1])
```

Where:
- `S[t]`: State matrix `[H, K, V]`
- `g[t]`: Fine-grained forget gate `[H, K]`
- `β[t]`: Beta update gate `[H]`
- `k[t]`, `v[t]`: Key and value vectors

### Chunkwise Algorithm

For efficient parallel computation, sequences are processed in chunks of 64 tokens:

1. **Intra-chunk**: Compute attention matrices Aqk and Akk within chunks
2. **Inter-chunk**: Maintain recurrent state between chunks
3. **WY representation**: Efficient matrix decomposition for computation

## Model Specifications

- **Total Parameters**: 48B
- **Activated Parameters**: 3B (sparse MoE activation)
- **Context Length**: Up to 1M tokens
- **Architecture**: Hybrid KDA + MLA (3:1 ratio)

## Performance

Kimi Linear achieves:

- **MMLU-Pro (4k context)**: 51.0 performance with similar speed as full attention
- **RULER (128k context)**: 84.3 performance, 3.98× speedup
- **Long sequences (1M tokens)**: 6.3× faster TPOT compared to MLA
- **KV Cache**: Up to 75% reduction

## Quick Setup

### On GCP

1. **Create GCP Instance**: See [GCP_INSTANCE_SETUP.md](./GCP_INSTANCE_SETUP.md) for manual instance creation instructions (GPU availability varies by zone)
2. **SSH into instance**: `ssh kimi-gcp` (after updating SSH config per instructions)
3. **Follow Setup Guide**: See [SETUP_GCP.md](./SETUP_GCP.md) for complete setup instructions including:
   - Cloning this repository
   - Downloading Hugging Face weights
   - Converting weights to custom implementation

## Resources

### Papers

- **Primary Paper**: [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
  - arXiv ID: 2510.26692
  - Year: 2025

### Code Repositories

- **Official GitHub**: https://github.com/MoonshotAI/Kimi-Linear
- **FLA Implementation**: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
- **Hugging Face Model**: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct

### Usage

The KDA implementation can be used via the FLA library:

```python
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.layers.kda import KimiDeltaAttention

# For training (chunk mode)
attention = KimiDeltaAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    mode='chunk'
)
```

See [KDA_ALGORITHM_DOCUMENTATION.md](./KDA_ALGORITHM_DOCUMENTATION.md) for detailed usage examples.

## Citation

If you use this implementation, please cite:

```bibtex
@misc{team2025kimi,
    title={Kimi Linear: An Expressive, Efficient Attention Architecture},
    author={Zhang, Yu and Lin, Zongyu and Yao, Xingcheng and others},
    year={2025},
    eprint={2510.26692},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

The KDA implementation code is copied from the FLA repository, which is subject to its own license terms. Please refer to the original repository for licensing information.

