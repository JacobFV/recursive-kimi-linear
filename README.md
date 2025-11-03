# Recursive Kimi-Linear: Chunked Generation with Latent Refinement

This repository implements **recursive chunked generation** on top of [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct), adding TRM-style latent recursion to enable iterative refinement before committing tokens.

## Overview

**Recursive Kimi-Linear** extends the Kimi-Linear architecture with a chunked, recursive refinement mechanism:

1. **Chunked Generation**: Text is generated in fixed-width chunks (e.g., 64-128 tokens)
2. **Inner Refinement Loop**: Within each chunk, K recursive refinement steps edit latent representations before committing tokens
3. **Dynamic Boundaries**: A learned boundary head predicts when to commit and the effective chunk length
4. **Full-Depth Recursion**: All transformer layers participate in each refinement step (no layer reduction)

### Key Innovation

Unlike typical recursive models that reduce layers, this implementation maintains **full transformer depth** across all refinement steps. The recursion is an *outer* loop that reuses the entire 32-layer stack, with lightweight per-layer refine cells that make residual updates to hidden states.

## Architecture

```
[Prefix Tokens] → [Chunk Draft (W tokens)] 
                    ↓
            Inner Loop (K steps):
                - Full Transformer Forward
                - Per-Layer Refine Cells (latent edits)
                - Boundary Head (halt/commit decision)
                - Token Updates (masked positions only)
                    ↓
            [Commit B' ≤ W tokens] → [Next Chunk]
```

**Components:**

- **RefineCell**: Tiny per-layer MLP that refines hidden states via residual connections
- **BoundaryHead**: Predicts commit probability and effective chunk length
- **LatentToken**: Optional learned `[Z]` token for global control signal
- **ChunkRefineWrapper**: Main wrapper that orchestrates chunked generation

## Installation

```bash
# Clone repository
git clone <repo-url>
cd recursive-kimi-linear

# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.5
- transformers >= 4.45.0
- flash-linear-attention >= 0.4.0 (includes KDA kernels)
- accelerate >= 0.30.0

## Usage

### Inference (Chunked Generation)

```python
from transformers import AutoModelForCausalLM
from kimi_linear.recursive import ChunkRefineWrapper

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Wrap with recursive controller
wrapper = ChunkRefineWrapper(
    base_model=base_model,
    layers_to_refine="all",
    use_latent_token=True,
    max_chunk_len=128,
)

# Generate with recursive refinement
input_ids = tokenizer("Your prompt here", return_tensors="pt")["input_ids"]
output = wrapper.generate_chunks(
    input_ids=input_ids,
    max_new_tokens=512,
    chunk_width=128,
    max_inner_steps=4,
    commit_threshold=0.7,
)
```

### Training

```bash
# Phase A: Train sidecar only (refine cells + boundary head)
python scripts/training/train_recursive.py \
    --model_name moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --chunk_width 128 \
    --max_inner_steps 4 \
    --batch_size 8 \
    --num_steps 50000 \
    --phase a \
    --trust_remote_code
```

**Training Phases:**

- **Phase A**: Freeze base model, train only sidecar components (50-100k steps)
- **Phase B**: Light LoRA unfreeze on attention/MLP, scheduled sampling on boundaries
- **Phase C**: End-to-end polish with subset of layers unfrozen

## Implementation Details

### Refine Cell

Per-layer tiny MLP that takes hidden states `h` and persistent state `s`, returns:
- `h_refined = h + Δh` (residual update)
- `s_next` (updated persistent state)

Output projection is **zero-initialized** to preserve baseline behavior at step 0.

### Boundary Head

Two-head classifier from top-layer pooled hidden states:
- **Commit Head**: Sigmoid → `p(commit)` probability
- **Length Head**: Categorical → effective chunk length (0..W)

### Inner Refinement Loop

```python
for t in range(K):  # K refinement steps
    # Full transformer forward
    logits, hiddens, cache = model.forward(chunk, cache)
    
    # Boundary decision
    p_commit, p_len = boundary_head(hiddens[-1])
    if p_commit > threshold: break
    
    # Refine hidden states
    for layer, refine_cell in zip(layers, refine_cells):
        hiddens[layer] = refine_cell(hiddens[layer], states[layer])
    
    # Update tokens (masked positions only)
    chunk = update_tokens(chunk, logits, mask)
```

### KDA State Persistence

Leverages Kimi-Linear's **KDA recurrent states** which are efficiently maintained across inner steps:
- KDA layers: Keep recurrent state `S` across refinement steps (cheap)
- MLA layers: Recompute KV cache for current chunk (sparse, 3:1 ratio)

This enables multiple refinement passes without ballooning memory.

## Loss Functions

Training uses multiple loss components:

- **Final CE**: Cross-entropy on committed tokens
- **Masked CE**: Deep supervision on intermediate steps (editable positions only)
- **Halt Loss**: Binary cross-entropy for boundary prediction
- **Length Loss**: Cross-entropy for effective length prediction
- **Ponder Loss**: Encourage fewer refinement steps (ACT-style)
- **Stability Loss**: Temporal consistency to prevent flip-flopping

## Performance Considerations

- **Memory**: KDA state persistence keeps memory overhead low (~2-4× inner steps)
- **Latency**: K refinement steps add latency but improve quality
- **Throughput**: Chunked generation can parallelize across batches

**Baseline (Kimi-Linear):**
- Up to 75% KV cache reduction
- Up to 6× faster decoding at 1M tokens
- 1M token context length

**With Recursion:**
- Additional cost: K × (chunk forward pass + refine overhead)
- Benefit: Iterative refinement improves coherence and reasoning

## Evaluation

Proposed evaluation metrics:

1. **Stability@K**: Fraction of blocks with consistent tokens across inner steps
2. **Length MSE**: |B' - B*_teacher| error
3. **TPOT Uplift**: Time per output token vs baseline
4. **Reasoning Lift**: GSM8K / MATH / ARC-AGI accuracy improvements
5. **Length Generalization**: Train on W=128, eval on W=256/384

## References

- **Kimi-Linear**: [GitHub](https://github.com/MoonshotAI/Kimi-Linear) | [HF Model](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) | [Paper](https://arxiv.org/abs/2510.26692)
- **FLA KDA**: [GitHub](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- **TRM**: [GitHub](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | [Paper](https://arxiv.org/abs/2510.04871)

## License

This implementation builds on:
- Kimi-Linear (subject to its license)
- Flash Linear Attention (subject to its license)
- TRM (subject to its license)

Please refer to the original repositories for licensing information.
