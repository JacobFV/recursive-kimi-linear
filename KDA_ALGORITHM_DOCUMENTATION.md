# Kimi Delta Attention (KDA) - Comprehensive Algorithm Documentation

## Overview

**Kimi Linear** is a hybrid linear attention architecture that, for the first time, **outperforms full attention under fair comparisons** across various scenarios—including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core is **Kimi Delta Attention (KDA)**—a refined version of Gated DeltaNet that introduces a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory.

### Key Achievement

Kimi Linear is the first linear attention architecture to outperform full attention with identical training recipes, demonstrating superior performance across:
- Short-context tasks
- Long-context tasks (up to 1M tokens)
- Reinforcement learning scaling regimes

## Key Characteristics

- **Total Parameters**: 48B
- **Activated Parameters**: 3B (sparse activation via MoE)
- **Context Length**: Up to 1M tokens
- **Performance**: Outperforms full attention while achieving:
  - Up to 75% reduction in KV cache usage
  - Up to 6× faster decoding throughput for 1M token contexts
  - Superior performance on MMLU-Pro (4k context) and RULER (128k context)

## Architecture Overview

Kimi Linear uses a **hybrid architecture** with a **3:1 ratio** of KDA to global MLA (Multi-Head Latent Attention) layers. This hybrid design reduces memory usage while maintaining or surpassing the quality of full attention.

## Core Algorithm: Kimi Delta Attention (KDA)

### 1. Mathematical Foundation

KDA is based on the **Delta Rule** for sequence modeling, which maintains a recurrent state `S` that gets updated at each timestep. The key innovation is the **fine-grained gating mechanism** that allows per-element control over memory retention and decay.

#### State Update Rule

The core recurrence relation for KDA is:

```
S[t] = S[t-1] ⊙ exp(g[t]) + β[t] · k[t] ⊗ (v[t] - k[t]^T · S[t-1])
```

Where:
- `S[t]`: State matrix of shape `[H, K, V]` (heads, key dimension, value dimension)
- `g[t]`: Forget gate in log space, shape `[H, K]` (fine-grained per head and key dimension)
- `β[t]`: Beta gate, shape `[H]` (per head)
- `k[t]`: Key vector, shape `[H, K]`
- `v[t]`: Value vector, shape `[H, V]`
- `⊙`: Element-wise multiplication
- `⊗`: Outer product

#### Output Computation

The output at timestep `t` is computed as:

```
o[t] = q[t]^T · S[t]
```

Where `q[t]` is the query vector of shape `[H, K]`.

### 2. Fine-Grained Gating Mechanism

#### Gate Computation

The forget gate `g` is computed using a specialized gate function:

```
g = -exp(A_log) ⊙ softplus(f_proj(hidden_states) + dt_bias)
```

Where:
- `A_log`: Learnable parameter per head, initialized uniformly in log space `[log(1), log(16)]`
- `f_proj`: Two-layer MLP that projects hidden states to gate values
- `dt_bias`: Learnable bias term for fine-tuning the gate dynamics
- `softplus`: Smooth activation function with learnable beta and threshold parameters

**Key Innovation**: Unlike previous models (Gated DeltaNet, Mamba) that apply gating uniformly across attention heads, KDA introduces **fine-grained gating** where each head and key dimension can have different forgetting rates. This enables more precise control over memory retention.

#### Beta Gate

The beta gate `β` controls the update strength:

```
β[t] = sigmoid(b_proj(hidden_states))
```

This is similar to traditional attention mechanisms but operates at the head level.

### 3. Chunkwise Algorithm

For hardware efficiency, KDA employs a **bespoke chunkwise algorithm** that processes sequences in chunks of 64 tokens. This enables parallel computation while maintaining the recurrent semantics.

**Key Innovation**: The algorithm uses a **specialized variant of Diagonal-Plus-Low-Rank (DPLR) transition matrices** that:
- Substantially reduces computation compared to the general DPLR formulation
- Remains more consistent with the classical delta rule
- Enables efficient parallel processing across chunks

#### Intra-Chunk Computation

Within each chunk, the algorithm computes attention matrices:

1. **Aqk**: Query-Key attention matrix
   ```
   Aqk[i,j] = q[i]^T · k[j] · exp(g[j] - g[i])  (for i >= j)
   ```

2. **Akk**: Key-Key attention matrix  
   ```
   Akk[i,j] = β[j] · k[i]^T · k[j] · exp(g[j] - g[i])  (for i > j)
   ```

These matrices use a **Diagonal-Plus-Low-Rank (DPLR) formulation** that reduces computational overhead while aligning with the classical delta rule.

#### Inter-Chunk Recurrence

Between chunks, the algorithm maintains a recurrent state `h`:

```
h[chunk_i] = h[chunk_i-1] ⊙ exp(g_final) + update_term
```

Where the update term incorporates information from all tokens in the current chunk.

### 4. Naive Recurrent Implementation

For reference, the naive sequential implementation (for short sequences ≤ 64 tokens) is:

```python
def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    
    S = initial_state or zeros(B, H, K, V)
    if initial_state is not None:
        S += initial_state
    
    o = zeros_like(v)
    
    for i in range(T):
        q_i, k_i, v_i, g_i, beta_i = q[:,i], k[:,i], v[:,i], g[:,i], beta[:,i]
        
        # Update state: decay + delta update
        S = S * exp(g_i[..., None])  # Element-wise decay
        delta = v_i - (k_i[..., None] * S).sum(-2)  # Compute delta
        S = S + beta_i[..., None] * k_i[..., None] * delta  # Delta rule update
        
        # Compute output
        o[:, i] = (q_i[..., None] * S).sum(-2)
    
    return o, S
```

### 5. Key Components

#### Short Convolutions

Optional short 1D convolutions (kernel size 4) are applied to Q, K, V before attention:

```python
q = SiLU(conv1d(q_proj(hidden_states)))
k = SiLU(conv1d(k_proj(hidden_states)))
v = SiLU(conv1d(v_proj(hidden_states)))
```

This provides local context modeling, similar to Mamba.

#### Output Gating

The output is gated using a projection and normalization:

```python
o = o_norm(o, g_proj(hidden_states))
o = o_proj(o)
```

Where `o_norm` is a fused RMSNorm with sigmoid gating.

## Implementation Details

### Forward Pass (Chunk Mode)

1. **Project and process inputs**:
   - Project Q, K, V from hidden states
   - Apply short convolutions (if enabled)
   - Compute gates: `g = fused_kda_gate(...)` and `β = sigmoid(b_proj(...))`

2. **Compute cumulative gates**:
   - `g_cumsum = chunk_local_cumsum(g, chunk_size=64)`

3. **Intra-chunk attention matrices**:
   - Compute `Aqk` and `Akk` using chunkwise kernels

4. **WY representation**:
   - Decompose `Akk` into WY form for efficient computation
   - Compute `w = Akk @ (exp(g) * k)` and `u = Akk @ v`

5. **Recurrent state update**:
   - Update hidden state `h` using gated delta rule
   - Compute new values `v_new`

6. **Output computation**:
   - `o = chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h, scale)`

### Backward Pass

The backward pass computes gradients for:
- `dq`, `dk`, `dv`: Gradients w.r.t. queries, keys, values
- `dg`: Gradients w.r.t. gate values
- `dβ`: Gradients w.r.t. beta gates
- `dh0`: Gradients w.r.t. initial state

The implementation uses automatic differentiation with custom kernels for efficiency.

## Gate Function Details

The gate function is implemented as a specialized Triton kernel that computes:

```python
def kda_gate(g, A_log, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
    """
    Computes: g_out = -exp(A_log) ⊙ softplus(g + g_bias)
    
    Args:
        g: [..., H*head_k_dim] - gate input
        A_log: [H] - log-space parameters
        g_bias: [H*head_k_dim] - optional bias
        beta: softplus beta parameter
        threshold: softplus threshold for linear approximation
    
    Returns:
        g_out: [..., H, head_k_dim] - gated values
    """
    g = rearrange(g, '... (h d) -> ... h d', d=head_k_dim)
    if g_bias is not None:
        g = g + g_bias
    
    A_exp = -exp(A_log).unsqueeze(-1)  # [H, 1]
    g_softplus = softplus(g, beta=beta, threshold=threshold)
    
    return A_exp * g_softplus
```

**Key Features**:
- Uses `softplus` with configurable beta and threshold for numerical stability
- Applies negative exponential of A_log for decay behavior
- Fine-grained: operates per head and per key dimension

## Differences from Gated DeltaNet

1. **Fine-grained gating**: Gating is applied per `[head, key_dim]` rather than per `[head]` or globally
2. **Refined gate computation**: Uses softplus with learned parameters instead of simpler activations
3. **Better initialization**: A_log is initialized in log space `[1, 16]` for stable training
4. **Enhanced output gating**: Fused RMSNorm with sigmoid gating for output projection

## Performance Optimizations

1. **Chunkwise processing**: Enables parallel computation across chunks
2. **WY representation**: Efficient representation of attention matrices using Woodbury matrix identity
3. **Fused kernels**: Combined operations reduce memory traffic
4. **L2 normalization**: Optional L2 normalization of Q and K for numerical stability
5. **Mixed precision**: Forward pass uses BF16, certain computations use FP32 for accuracy

## Hybrid Architecture

The Kimi Linear model uses a **layerwise hybrid of KDA and MLA**:

- **3:1 ratio**: 3 KDA layers for every 1 MLA layer
- **MLA layers**: Standard Multi-Head Latent Attention for global context
- **Benefit**: Reduces KV cache by up to 75% while maintaining or surpassing full attention quality
- **Training**: Uses identical training recipe as full attention for fair comparison

## Model Configuration

Key configuration parameters:

```python
KDAConfig(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    expand_v=1.0,  # Value dimension expansion
    num_v_heads=None,  # Grouped Value Attention if > num_heads
    use_short_conv=True,  # Enable short convolutions
    conv_size=4,  # Convolution kernel size
    allow_neg_eigval=False,  # Allow negative eigenvalues (for state tracking)
    attn_mode="chunk",  # "chunk" or "fused_recurrent"
    # ... other standard transformer configs
)
```

## References

- **Paper**: [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) (arXiv:2510.26692)
- **GitHub**: https://github.com/MoonshotAI/Kimi-Linear
- **Hugging Face**: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- **FLA Implementation**: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

## Citation

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

