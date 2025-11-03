# KDA Implementation Notes

## File Structure

```
fla/
├── ops/kda/           # Core KDA kernels
│   ├── __init__.py    # Public API exports
│   ├── chunk.py       # Chunkwise computation (training)
│   ├── fused_recurrent.py  # Recurrent computation (inference)
│   ├── naive.py       # Reference implementations
│   ├── gate.py        # Fine-grained gating kernels
│   ├── chunk_intra.py # Intra-chunk attention computation
│   ├── chunk_inter.py # Inter-chunk computation
│   └── wy_fast.py     # WY representation utilities
├── layers/
│   └── kda.py         # High-level KimiDeltaAttention layer
└── models/kda/        # Complete model implementations
    ├── __init__.py
    ├── configuration_kda.py
    └── modeling_kda.py
```

## Key Implementation Files

### 1. `fla/ops/kda/chunk.py`

Main chunkwise forward and backward implementations for training.

**Key Functions**:
- `chunk_kda_fwd()`: Forward pass computing attention matrices and output
- `chunk_kda_bwd()`: Backward pass computing gradients
- `chunk_kda()`: Public API with autograd support

**Key Operations**:
1. Compute cumulative gates using `chunk_local_cumsum`
2. Compute intra-chunk attention matrices `Aqk` and `Akk`
3. Decompose into WY representation for efficient computation
4. Update recurrent state using gated delta rule
5. Compute output using GLA-style attention

### 2. `fla/ops/kda/fused_recurrent.py`

Recurrent implementation for inference (sequences ≤ 64 tokens).

**Key Function**:
- `fused_recurrent_kda()`: Wrapper around `fused_recurrent_gated_delta_rule`

This uses the underlying gated delta rule implementation optimized for short sequences.

### 3. `fla/ops/kda/gate.py`

Fine-grained gating mechanism implementation.

**Key Components**:
- `kda_gate_fwd_kernel`: Triton kernel for forward pass
- `kda_gate_bwd_kernel`: Triton kernel for backward pass
- `fused_kda_gate()`: Public API with autograd

**Computation**:
```
g_out = -exp(A_log) ⊙ softplus(g + g_bias)
```

Uses configurable `beta` and `threshold` parameters for numerical stability in softplus.

### 4. `fla/ops/kda/chunk_intra.py`

Intra-chunk attention matrix computation using Triton kernels.

**Key Kernels**:
- `chunk_kda_fwd_kernel_intra_sub_inter`: Compute attention between different sub-chunks
- `chunk_kda_fwd_kernel_intra_sub_intra`: Compute attention within sub-chunks
- `chunk_kda_bwd_kernel_intra`: Backward pass for intra-chunk computation

**Matrices Computed**:
- `Aqk[i,j]`: Query-key attention (for i >= j, causal mask)
- `Akk[i,j]`: Key-key attention (for i > j, strictly lower triangular)

### 5. `fla/ops/kda/chunk_inter.py`

Inter-chunk computation for gradients.

**Key Kernel**:
- `chunk_kda_bwd_kernel_inter`: Computes gradients for q, k, v, g across chunks

### 6. `fla/ops/kda/wy_fast.py`

WY representation utilities for efficient matrix operations.

**Key Functions**:
- `recompute_w_u_fwd()`: Recompute W and U matrices from A matrix
- `prepare_wy_repr_bwd()`: Prepare WY representation for backward pass

**WY Representation**:
The algorithm uses the Woodbury matrix identity to efficiently represent and compute with attention matrices.

### 7. `fla/ops/kda/naive.py`

Reference implementations for understanding and testing.

**Functions**:
- `naive_recurrent_kda()`: Simple sequential implementation
- `naive_chunk_kda()`: Chunkwise implementation without optimizations

These are useful for:
- Understanding the algorithm logic
- Debugging optimized implementations
- Testing correctness

### 8. `fla/layers/kda.py`

High-level PyTorch layer implementing `KimiDeltaAttention`.

**Key Components**:
- Projections: `q_proj`, `k_proj`, `v_proj`, `f_proj`, `b_proj`, `g_proj`, `o_proj`
- Short convolutions: Optional 1D convs for Q, K, V
- Gate parameters: `A_log` (learnable per head), `dt_bias` (learnable bias)
- Normalization: `o_norm` (fused RMSNorm with sigmoid gating)

**Forward Pass**:
1. Project hidden states to Q, K, V
2. Apply short convolutions (if enabled)
3. Compute gates: `g = fused_kda_gate(...)`, `β = sigmoid(b_proj(...))`
4. Call chunk or recurrent kernel
5. Apply output gating and projection

### 9. `fla/models/kda/`

Complete model implementations compatible with Hugging Face Transformers.

**Files**:
- `configuration_kda.py`: Model configuration class
- `modeling_kda.py`: Model implementations (`KDAModel`, `KDAForCausalLM`)

**Features**:
- Compatible with Hugging Face API
- Supports generation with `model.generate()`
- Includes proper weight initialization
- Supports gradient checkpointing

## Dependencies

The implementation depends on:
- **PyTorch** >= 2.5
- **Triton** >= 3.0 (for custom kernels)
- **einops** (for tensor reshaping)
- **transformers** >= 4.45.0 (for model compatibility)

## Performance Considerations

### Memory Efficiency

1. **Chunkwise Processing**: Reduces memory footprint by processing in chunks
2. **Fused Operations**: Combines multiple operations to reduce memory traffic
3. **Mixed Precision**: Uses BF16 for most operations, FP32 for critical computations

### Computational Efficiency

1. **Triton Kernels**: Custom GPU kernels for optimal performance
2. **WY Representation**: Efficient matrix decomposition
3. **Autotuning**: Kernels are autotuned for different hardware configurations

### Numerical Stability

1. **L2 Normalization**: Optional L2 norm of Q and K
2. **Softplus Thresholding**: Linear approximation for large values
3. **Cumulative Gates**: Stable computation of exponential gates

## Usage Examples

### Basic Usage

```python
from fla.layers.kda import KimiDeltaAttention

attention = KimiDeltaAttention(
    hidden_size=2048,
    num_heads=16,
    head_dim=128,
    mode='chunk'  # 'chunk' for training, 'fused_recurrent' for inference
)

output = attention(hidden_states)
```

### Using the Kernels Directly

```python
from fla.ops.kda import chunk_kda

o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    scale=1.0 / sqrt(head_dim),
    initial_state=None,
    output_final_state=True
)
```

## Notes on Code Organization

1. **Separation of Concerns**: 
   - Low-level kernels in `ops/kda/`
   - High-level layers in `layers/kda.py`
   - Complete models in `models/kda/`

2. **Reusability**: 
   - Kernels are reused across different components
   - Common utilities shared via `fla.ops.utils`

3. **Extensibility**:
   - New kernels can be added without changing layer interface
   - Configuration allows easy experimentation

## Testing and Verification

The reference implementations in `naive.py` should produce identical results to optimized versions (within numerical precision). Use them for:

- Verification of optimized implementations
- Understanding algorithm behavior
- Debugging numerical issues

## Future Improvements

Potential areas for enhancement:

1. **Variable-Length Sequences**: Already supported via `cu_seqlens`
2. **Multi-GPU**: Can be extended with tensor parallelism
3. **Quantization**: Could benefit from INT8/FP8 quantization
4. **Sparsity**: Could leverage sparse attention patterns

