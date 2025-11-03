# Kimi Linear Architecture Summary

## Quick Reference

### What is Kimi Linear?

Kimi Linear is a **hybrid linear attention architecture** that achieves a historic milestone: **it is the first linear attention method to outperform full attention under fair comparisons** across short-context, long-context, and RL scaling regimes.

### Core Innovation: Kimi Delta Attention (KDA)

KDA extends Gated DeltaNet with a **finer-grained gating mechanism**:

```
Traditional: g[t] ∈ [H]           (one gate per head)
KDA:         g[t] ∈ [H, K]        (one gate per head × key_dim)
```

This enables more precise control over memory retention and decay, making better use of limited finite-state RNN memory.

### Algorithm Essence

**State Update**:
```
S[t] = S[t-1] ⊙ exp(g[t]) + β[t] · k[t] ⊗ (v[t] - k[t]^T · S[t-1])
```

**Output**:
```
o[t] = q[t]^T · S[t]
```

Where:
- `g[t]`: Fine-grained forget gate `[H, K]` (computed via `-exp(A_log) ⊙ softplus(...)`)
- `β[t]`: Update strength gate `[H]` (computed via `sigmoid(...)`)
- `S[t]`: Recurrent state `[H, K, V]`

### Key Features

1. **Fine-Grained Gating**: Independent gates for each `[head, key_dim]` pair
2. **Chunkwise Algorithm**: Processes sequences in 64-token chunks for parallelization
3. **DPLR Variant**: Specialized Diagonal-Plus-Low-Rank formulation
4. **Hybrid Architecture**: 3:1 ratio of KDA to MLA layers

### Performance

| Metric | Value |
|--------|-------|
| Total Parameters | 48B |
| Activated Parameters | 3B (MoE) |
| Context Length | Up to 1M tokens |
| KV Cache Reduction | Up to 75% |
| Decoding Speedup (1M tokens) | 6× faster |
| MMLU-Pro (4k context) | 51.0 |
| RULER (128k context) | 84.3, 3.98× speedup |

### Model Architecture

```
Input → Embedding → [KDABlock × 3] → MLABlock → [KDABlock × 3] → ... → Output
                         ↑                ↑
                    KDA Layer        MLA Layer (global attention)
```

**Hybrid Ratio**: 3 KDA layers for every 1 MLA layer

### Implementation Components

1. **Core Kernels** (`fla/ops/kda/`):
   - `chunk.py`: Chunkwise computation (training)
   - `fused_recurrent.py`: Recurrent computation (inference)
   - `gate.py`: Fine-grained gating
   - `chunk_intra.py`: Intra-chunk attention
   - `chunk_inter.py`: Inter-chunk gradients
   - `wy_fast.py`: WY representation utilities
   - `naive.py`: Reference implementations

2. **High-Level Layer** (`fla/layers/kda.py`):
   - `KimiDeltaAttention`: Complete attention layer

3. **Model** (`fla/models/kda/`):
   - `KDAModel`: Base model
   - `KDAForCausalLM`: Language model

### Differences from Gated DeltaNet

| Aspect | Gated DeltaNet | KDA |
|--------|----------------|-----|
| Gate Granularity | Per head `[H]` | Per head×key `[H, K]` |
| Gate Function | Simple activation | Softplus with learned params |
| Initialization | Standard | Log-space `[log(1), log(16)]` |
| DPLR Variant | General | Specialized, more efficient |

### Mathematical Formulation

#### Gate Computation
```
g[t] = -exp(A_log) ⊙ softplus(f_proj(hidden_states) + dt_bias)
```
- `A_log`: Learnable per-head parameter (initialized in `[log(1), log(16)]`)
- `f_proj`: Two-layer MLP
- `dt_bias`: Learnable bias for fine-tuning

#### Beta Gate
```
β[t] = sigmoid(b_proj(hidden_states))
```

#### Chunkwise Attention Matrices
- **Aqk**: Query-key attention (lower triangular, causal)
- **Akk**: Key-key attention (strictly lower triangular)

Both use DPLR formulation for efficiency.

### Usage

```python
from fla.layers.kda import KimiDeltaAttention

attention = KimiDeltaAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    mode='chunk'  # 'chunk' for training, 'fused_recurrent' for short sequences
)
```

### Key Papers

- **Primary**: [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- **Related**: Gated DeltaNet, DeltaNet, Mamba, RetNet

### Resources

- **GitHub**: https://github.com/MoonshotAI/Kimi-Linear
- **Hugging Face**: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- **FLA Implementation**: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

## Document Guide

- **[KDA_ALGORITHM_DOCUMENTATION.md](./KDA_ALGORITHM_DOCUMENTATION.md)**: Comprehensive technical documentation
- **[IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md)**: Implementation details and code structure
- **[README.md](./README.md)**: Repository overview and quick start

