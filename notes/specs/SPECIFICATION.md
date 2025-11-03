# Recursive Kimi-Linear Implementation Specification

This document provides a comprehensive specification for the recursive chunked generation implementation on top of Kimi-Linear-48B-A3B-Instruct.

## Architecture Overview

### High-Level Flow

```
1. Prefix → Seed Chunk (W tokens, greedy/sample)
2. Inner Loop (K steps):
   a. Full transformer forward (all 32 layers)
   b. Per-layer refine cells update hidden states
   c. Boundary head predicts commit/length
   d. Update tokens (masked positions only)
   e. If p_commit > threshold: break
3. Commit B' ≤ W tokens (strip trailing blanks)
4. Append to output, advance to next chunk
```

### Key Design Decisions

1. **Full Depth Recursion**: All transformer layers participate in each refinement step
2. **Minimal Surgery**: Only adds tiny refine cells + boundary head (≪1% params)
3. **Zero-Init Residuals**: Refine cell outputs zero-init to preserve baseline behavior
4. **KDA State Persistence**: Leverages KDA recurrent states across inner steps

## Component Specifications

### 1. RefineCell (`kimi_linear/recursive/refine_cell.py`)

**Purpose**: Per-layer tiny MLP that refines hidden states via residual updates.

**Architecture**:
```python
Input: h [B, T, d_model], s [B, d_state]
  ↓
Pool: pooled = h.mean(dim=1)  # [B, d_model]
  ↓
State Update: s_next = GRUCell(pooled, s)
  ↓
Refinement: delta = MLP([pooled, s])  # [B, d_model]
  ↓
Output: h_refined = h + delta.unsqueeze(1), s_next
```

**Key Features**:
- Zero-init output projection (preserves baseline at t=0)
- GRU cell for persistent state update
- Residual connection (additive updates)

### 2. BoundaryHead (`kimi_linear/recursive/boundary_head.py`)

**Purpose**: Predicts when to commit a chunk and its effective length.

**Architecture**:
```python
Input: h_block [B, T, d_model] or [B, d_model]
  ↓
Pool: h_pooled = h_block[:, -1]  # right edge
  ↓
MLP: g = SiLU(Linear(h_pooled)) → SiLU(Linear(g))
  ↓
Heads:
  - commit: sigmoid(Linear(g)) → p_commit [B]
  - length: log_softmax(Linear(g)) → p_len [B, max_len+1]
```

**Outputs**:
- `p_commit`: Probability chunk should be committed
- `p_len`: Categorical distribution over effective length (0..W)

### 3. LatentToken (`kimi_linear/recursive/latent_token.py`)

**Purpose**: Optional learned token `[Z]` for global control signal.

**Usage**:
- Can be a new learned embedding (not in vocab)
- Or reuse existing special token ID
- Flows through all layers via attention
- Enables top→bottom skip connections

### 4. ChunkRefineWrapper (`kimi_linear/recursive/wrapper.py`)

**Purpose**: Main wrapper orchestrating chunked generation.

**Key Methods**:
- `generate_chunks()`: Public API for chunked generation
- `_forward_hidden()`: Run full model forward, return logits + hiddens
- `_seed_window()`: Generate initial chunk proposal
- `_pad_to_width()` / `_strip_trailing_blanks()`: Utility functions

### 5. Loss Functions (`kimi_linear/recursive/losses.py`)

**Components**:
- `L_final`: CE on committed tokens
- `L_masked`: CE on intermediate steps (deep supervision)
- `L_halt`: BCE for boundary prediction
- `L_length`: CE for effective length prediction
- `L_ponder`: Encourage fewer steps (ACT-style)
- `L_stability`: Temporal consistency penalty

**Total Loss**:
```
L = λ_final * L_final
  + λ_masked * L_masked
  + λ_halt * L_halt
  + λ_len * L_length
  + λ_ponder * L_ponder
  + λ_stability * L_stability
```

Default weights: `[1.0, 0.5, 0.5, 0.05, 0.01, 0.01]`

### 6. Data Utilities (`kimi_linear/recursive/data.py`)

**Components**:
- `ChunkCollator`: Collates sequences into fixed-width chunks
- `create_corruption_mask()`: Creates editable position masks
- `create_teacher_boundaries()`: Creates ground-truth chunk boundaries
- `create_noise_schedule()`: Corruption rate schedule across steps

## Training Recipe

### Phase A: Sidecar-Only Training

**Duration**: 50-100k steps

**Frozen**:
- All base model weights
- Optional: LoRA adapters (inactive)

**Trainable**:
- Refine cells
- Boundary head
- Latent token (if used)

**Losses**: All components, focus on boundary + masked CE

**Learning Rate**: 3e-4 (sidecar), 5e-4 (heads)

### Phase B: Light Unfreeze

**Duration**: 20-50k steps

**Changes**:
- Enable LoRA on top 1/3 layers (r=8-16, lr=1e-5)
- Scheduled sampling on boundaries (mix teacher vs model)
- Add chunk perplexity constraint

### Phase C: End-to-End Polish

**Duration**: Short epochs

**Changes**:
- Unfreeze every 4th layer with L2 to pretrained (EWC-lite)
- Monitor n-gram drift
- Short runs only

## Inference Algorithm

```python
z = z0  # Initialize latent
offset = 0  # Position offset for RoPE
output = []

while not EOS and len(output) < max_tokens:
    # Seed window
    chunk = seed_window(prefix, W)
    
    # Inner refinement
    for t in range(K):
        logits, hiddens, cache, z = forward(chunk, cache, z)
        p_commit, p_len = boundary_head(hiddens[-1])
        
        if p_commit > tau:
            break
        
        # Refine
        for layer, cell in zip(layers, refine_cells):
            hiddens[layer], states[layer] = cell(hiddens[layer], states[layer])
        
        # Update tokens (masked only)
        chunk = update_tokens(chunk, logits, mask)
    
    # Commit
    B_eff = predict_length(p_len) or strip_blanks(chunk)
    commit = chunk[:B_eff]
    output.append(commit)
    prefix = concat(prefix, commit)
    offset += len(commit)
```

## Integration Points

### Hugging Face Transformers

The implementation assumes:
- Model loaded with `trust_remote_code=True`
- `output_hidden_states=True` to get per-layer hiddens
- `use_cache=True` for KDA/MLA state persistence
- `past_key_values` support for incremental generation

### KDA State Persistence

Kimi-Linear uses `KimiDynamicCache` which maintains:
- `conv_states`: Convolutional state
- `recurrent_states`: KDA recurrent state (key for inner loop)
- `key_cache`, `value_cache`: Standard KV cache

**Strategy**: Keep recurrent states across inner steps, recompute MLA KV per step.

### Configuration

Add to `KimiLinearConfig`:

```python
trm_enabled: bool = False
trm_block_size: int = 64
trm_inner_steps: int = 4
trm_max_blocks: int = 4096
trm_gate_hidden: int = 1024
trm_commit_threshold: float = 0.85
trm_len_head: bool = True
```

## Evaluation Metrics

1. **Stability@K**: Fraction of blocks with identical tokens across inner steps
2. **Length MSE**: |B' - B*_teacher|
3. **TPOT Uplift**: Time per output token vs baseline
4. **Reasoning Lift**: GSM8K / MATH / ARC-AGI accuracy
5. **Length Generalization**: Train W=128, eval W=256/384

## Ablations

Control flags:
- `REC_NO_Z`: Drop latent token
- `REC_NO_CELL`: Boundary head only (no refinement)
- `REC_GLOB_PASS=1`: One MLA pass per inner step

## Risks & Mitigations

1. **Mode Collapse** (always commit @ t=0):
   - Add masked noise
   - Explicit reward for loss drop
   - Minimum refinement steps during training

2. **Oscillation**:
   - Temporal consistency loss
   - EMA blend on hiddens
   - Cosine annealed alpha

3. **Cache Bugs**:
   - Verify `cache_position` semantics
   - Unit test state persistence

4. **Memory Blowup**:
   - Keep only KDA recurrent state
   - Recompute MLA KV per step
   - Limit K (2-4 steps)

## Default Hyperparameters

- **W (chunk width)**: 128 (code/math), 64 (chat)
- **K (max inner steps)**: 4
- **τ (commit threshold)**: 0.7
- **τ_min (min threshold)**: 0.3
- **d_state**: 512
- **gate_hidden**: 1024
- **corruption_rate**: 0.4 → 0.1 (linear schedule)
- **batch_size**: 8-16 chunks
- **lr**: 3e-4 (sidecar), 5e-4 (heads), 1e-5 (LoRA)

## References

- **Kimi-Linear**: [GitHub](https://github.com/MoonshotAI/Kimi-Linear) | [HF](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- **FLA KDA**: [GitHub](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- **TRM**: [GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | [Paper](https://arxiv.org/abs/2510.04871)

