# Complete Implementation Specification: Recursive Kimi-Linear with TRM-style Latent Refinement

**Version**: 1.0  
**Date**: 2025-11-03  
**Status**: Ready for Implementation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Foundation](#research-foundation)
3. [Architecture Overview](#architecture-overview)
4. [Configuration System](#configuration-system)
5. [Implementation Details](#implementation-details)
6. [Surgical Operation Plan](#surgical-operation-plan)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation Framework](#evaluation-framework)
9. [Metrics & Monitoring](#metrics--monitoring)
10. [Reproducibility & Idempotency](#reproducibility--idempotency)
11. [Testing Strategy](#testing-strategy)
12. [Deployment & Usage](#deployment--usage)

---

## Executive Summary

### Goal

Add TRM-style (Tiny Recursive Model) latent recursion to Kimi-Linear-48B-A3B-Instruct, enabling iterative refinement of generated text chunks before committing tokens. The key innovation is using **full transformer depth** (all 32 layers) in an outer refinement loop**, not reducing to a tiny 2-layer network like original TRM.

### Core Innovation

- **Chunked Generation**: Generate text in fixed-width chunks (64-128 tokens)
- **Inner Refinement Loop**: Within each chunk, run K recursive refinement steps that:
  - Refine latent representations via per-layer refine cells
  - Predict dynamic chunk boundaries (when to commit)
  - Update persistent state across refinement steps
- **Full Depth Preservation**: All transformer layers participate in each refinement step
- **Zero-Init Safety**: Refine cells use zero-initialized output projections to preserve baseline behavior when untrained

### Key Requirements

1. **Idempotency**: Configuration-driven, reproducible experiments
2. **Backward Compatibility**: Can disable recursion without changing baseline behavior
3. **Minimal Surgery**: Add components without breaking existing functionality
4. **Performance**: Leverage KDA's stateful recurrence for efficient inner loops

### References

- **TRM Paper**: [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- **TRM Repo**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Kimi-Linear**: https://github.com/MoonshotAI/Kimi-Linear
- **HF Model**: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- **FLA KDA**: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda

---

## Research Foundation

### Original TRM Approach

TRM achieves 45% on ARC-AGI-1 with a tiny 7M parameter model by:

1. **Recursive Refinement**: Takes input x, current answer guess y, latent state z
2. **Iterative Updates**: 
   - Recursively updates latent z given (x, y, z)
   - Updates answer y given (y, z)
   - Runs up to K refinement steps
3. **Deep Supervision**: Multiple supervision steps where latent/state carries across iterations
4. **Early Halting**: ACT-style halting mechanism decides when to stop refining

**Key Insight**: Tiny network + many refinement steps + deep supervision → strong generalization on discrete boolean problems despite minimal parameters.

### Adaptation to Text Generation

**Transfer the Pattern, Not the Architecture**:

- Original TRM: 2-layer tiny net for reasoning tasks
- Our Adaptation: Full 32-layer transformer for text generation
- Key Difference: Recursion is **temporal** (outer loop over chunks), not architectural (not a 2-layer controller)

### Design Decisions

1. **Why Full Depth?**
   - User requirement: "the whole token generation process has to go from the bottom all the way to the top"
   - Skip connection from top→bottom requires full stack: output tokens → re-embed → flow through all layers again
   - Cannot use 2-layer controller; must use all 32 layers

2. **Why Chunked?**
   - Enables iterative refinement before committing
   - Allows variable-width chunks (truncate trailing blanks)
   - Maintains coherence via persistent state across chunks

3. **Why Zero-Init?**
   - Critical for preserving baseline behavior when recursion is enabled but untrained
   - Refine cell output projections zero-initialized → delta = 0 at initialization
   - Enables clean A/B testing: enable recursion without training should match baseline

4. **Why KDA State Persistence?**
   - Kimi-Linear uses KDA (Kimi Delta Attention) with recurrent states
   - Can efficiently carry state across inner refinement steps
   - Maintains ~75% KV cache reduction benefits

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATION LOOP                          │
└─────────────────────────────────────────────────────────────┘

1. Prefix Tokens (committed so far)
   ↓
2. Seed Chunk (W tokens, greedy/sampling, one forward pass)
   ↓
3. INNER REFINEMENT LOOP (K steps):
   
   For each step t = 1 to K:
   
   a) Full Transformer Forward
      - All 32 layers
      - Input: prefix + current chunk tokens
      - Output: logits, hidden states per layer
      - State: KDA recurrent states persisted
   
   b) Boundary Decision
      - BoundaryHead(hiddens[-1]) → p_commit, p_len
      - If p_commit > threshold: BREAK (commit now)
   
   c) Refine Hidden States
      - For each layer: RefineCell(hidden[layer], state[layer])
      - Updates: h[layer] += delta, state[layer] = update(state)
   
   d) Update Tokens (masked positions only)
      - logits → token IDs
      - Update only editable positions (corruption mask)
      - Anchor positions unchanged
   
   ↓
4. Commit Chunk
   - Effective length: B' = predict_length(p_len) or strip_blanks(chunk)
   - Commit tokens[:B']
   - Append to output
   
   ↓
5. Advance to Next Chunk
   - Update position offsets
   - Carry forward KDA recurrent states
   - Repeat from step 2
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BASE MODEL                                │
│           (Kimi-Linear-48B-A3B-Instruct)                     │
│                                                              │
│  Input → Embedding → [KDA Layer × 3] → [MLA Layer] → ...   │
│                    ↓           ↓                            │
│              KDA States    MLA KV Cache                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              RECURSIVE COMPONENTS (Optional)                 │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RefineCell (per layer)                           │    │
│  │  Input: h[layer] [B,T,D], s[layer] [B,512]        │    │
│  │  → pooled = mean(h[layer])                        │    │
│  │  → s_next = GRUCell(pooled, s[layer])             │    │
│  │  → delta = MLP([pooled, s[layer]]) [zero-init]   │    │
│  │  Output: h_refined = h + delta, s_next            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  BoundaryHead                                      │    │
│  │  Input: h_top [B,T,D]                              │    │
│  │  → h_pooled = h_top[:, -1] (right edge)           │    │
│  │  → g = MLP(h_pooled)                               │    │
│  │  → p_commit = sigmoid(Linear(g))                   │    │
│  │  → p_len = log_softmax(Linear(g)) [0..W]          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  LatentToken (optional)                            │    │
│  │  Learned [Z] token that flows through all layers  │    │
│  │  Provides global control signal                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Full Depth**: All 32 layers participate in each refinement step
2. **Minimal Surgery**: Additive components only; no modification to base attention
3. **State Persistence**: KDA recurrent states carried across inner steps (cheap)
4. **Zero-Init Residuals**: RefineCell outputs zero at init → preserves baseline

---

## Configuration System

### RecursiveConfig Dataclass

All recursive parameters consolidated into a single configuration class:

```python
@dataclass
class RecursiveConfig:
    """
    Configuration for recursive chunked generation.
    
    When recursive_enabled=False, all components are inactive
    and behavior is identical to baseline model.
    """
    
    # ========== Core Enable/Disable ==========
    recursive_enabled: bool = False
    """Master switch: if False, recursion is completely disabled (idempotent)"""
    
    # ========== Chunking Parameters ==========
    chunk_width: int = 128
    """Fixed chunk width W (64 for chat, 128 for code/math)"""
    
    max_inner_steps: int = 4
    """Maximum inner refinement steps K per chunk"""
    
    commit_threshold: float = 0.7
    """Halt threshold tau: commit when p_commit >= this"""
    
    min_commit_threshold: float = 0.3
    """Minimum threshold to avoid deadlocks (force commit if below)"""
    
    # ========== Architecture Parameters ==========
    layers_to_refine: Union[str, List[int]] = "all"
    """Which layers to apply refinement: "all" or list of indices"""
    
    use_latent_token: bool = True
    """Whether to use learned [Z] token for global control"""
    
    d_state: int = 512
    """Dimension of persistent refine state per layer"""
    
    gate_hidden: int = 1024
    """Hidden dimension for boundary head MLP"""
    
    max_chunk_len: int = 128
    """Maximum chunk length for length prediction head"""
    
    # ========== Training Parameters ==========
    corruption_rate: float = 0.3
    """Initial corruption rate for masked positions"""
    
    corruption_schedule: str = "linear"
    """Corruption schedule: "linear", "cosine", or "none" """
    
    corruption_rate_final: float = 0.1
    """Final corruption rate (for schedules)"""
    
    # ========== Loss Weights ==========
    lambda_final: float = 1.0
    """Weight for final CE loss on committed tokens"""
    
    lambda_masked: float = 0.5
    """Weight for masked CE loss (deep supervision)"""
    
    lambda_halt: float = 0.5
    """Weight for halt/commit prediction loss"""
    
    lambda_len: float = 0.05
    """Weight for length prediction loss"""
    
    lambda_ponder: float = 0.01
    """Weight for ponder cost (encourage fewer steps)"""
    
    lambda_stability: float = 0.01
    """Weight for temporal consistency loss"""
    
    # ========== Advanced Parameters ==========
    cache_reset_policy: str = "keep_kda_recurrent_reset_mla"
    """Policy for cache management across inner steps:
       - keep_kda_recurrent_reset_mla: Keep KDA states, reset MLA KV
       - keep_all: Keep all cache
       - reset_all: Reset all cache each step
    """
    
    enable_length_head: bool = True
    """Whether to predict effective chunk length"""
    
    use_soft_tokens: bool = False
    """Use Gumbel-softmax for differentiable token updates (training only)"""
    
    # ========== Validation ==========
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.chunk_width <= 0:
            errors.append("chunk_width must be positive")
        if self.max_inner_steps <= 0:
            errors.append("max_inner_steps must be positive")
        if not 0.0 <= self.commit_threshold <= 1.0:
            errors.append("commit_threshold must be in [0, 1]")
        # ... more validation
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RecursiveConfig':
        """Load from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> 'RecursiveConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

### Default Configurations

**Baseline (No Recursion)**:
```python
baseline_config = RecursiveConfig(
    recursive_enabled=False,  # Everything disabled
)
```

**After Surgery (Enabled but Untrained)**:
```python
after_surgery_config = RecursiveConfig(
    recursive_enabled=True,
    chunk_width=128,
    max_inner_steps=4,
    # All other defaults
)
# Should produce identical results to baseline (zero-init ensures this)
```

**Training Phase A (Sidecar Only)**:
```python
phase_a_config = RecursiveConfig(
    recursive_enabled=True,
    chunk_width=128,
    max_inner_steps=4,
    lambda_final=1.0,
    lambda_masked=0.5,
    lambda_halt=0.5,
    corruption_rate=0.3,
)
```

### Config File Format

Example: `configs/baseline.json`
```json
{
  "recursive_enabled": false
}
```

Example: `configs/recursive_phase_a.json`
```json
{
  "recursive_enabled": true,
  "chunk_width": 128,
  "max_inner_steps": 4,
  "commit_threshold": 0.7,
  "layers_to_refine": "all",
  "use_latent_token": true,
  "d_state": 512,
  "gate_hidden": 1024,
  "lambda_final": 1.0,
  "lambda_masked": 0.5,
  "lambda_halt": 0.5,
  "lambda_len": 0.05,
  "lambda_ponder": 0.01,
  "lambda_stability": 0.01,
  "corruption_rate": 0.3,
  "corruption_schedule": "linear"
}
```

---

## Implementation Details

### Component Specifications

#### 1. RefineCell (`kimi_linear/recursive/refine_cell.py`)

**Purpose**: Per-layer tiny MLP that refines hidden states via residual updates.

**Architecture**:
```python
class RefineCell(nn.Module):
    def __init__(self, d_model: int, d_state: int = 512):
        self.inp = nn.Linear(d_model + d_state, 4*d_model, bias=False)
        self.act = nn.SiLU()
        self.out = nn.Linear(4*d_model, d_model, bias=False)
        nn.init.zeros_(self.out.weight)  # CRITICAL: zero-init
        self.state_upd = nn.GRUCell(d_model, d_state)
    
    def forward(self, h: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        pooled = h.mean(dim=1)  # [B, d_model]
        s_next = self.state_upd(pooled, s)  # [B, d_state]
        x = torch.cat([pooled, s], dim=-1)  # [B, d_model + d_state]
        delta = self.out(self.act(self.inp(x)))  # [B, d_model]
        h_refined = h + delta.unsqueeze(1)  # Residual connection
        return h_refined, s_next
```

**Key Features**:
- Zero-init output projection ensures `delta = 0` at initialization
- GRU cell for persistent state update
- Residual connection (additive, not replacing)
- Cheap: only MLP operations, no attention

**Idempotency**: When recursion disabled, RefineCell never called.

#### 2. BoundaryHead (`kimi_linear/recursive/boundary_head.py`)

**Purpose**: Predicts when to commit a chunk and its effective length.

**Architecture**:
```python
class BoundaryHead(nn.Module):
    def __init__(self, d_model: int, gate_hidden: int = 1024, max_len: int = 128):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, gate_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(gate_hidden, gate_hidden, bias=False),
            nn.SiLU(),
        )
        self.commit = nn.Linear(gate_hidden, 1, bias=True)
        nn.init.constant_(self.commit.bias, -2.0)  # Bias toward early commit
        self.len_head = nn.Linear(gate_hidden, max_len + 1, bias=True)
    
    def forward(self, h_block: Tensor) -> Tuple[Tensor, Tensor]:
        h_pooled = h_block[:, -1]  # Right edge [B, d_model]
        g = self.mlp(h_pooled)  # [B, gate_hidden]
        p_commit = torch.sigmoid(self.commit(g)).squeeze(-1)  # [B]
        p_len = torch.log_softmax(self.len_head(g), dim=-1)  # [B, max_len+1]
        return p_commit, p_len
```

**Key Features**:
- Uses right edge of chunk (last position) for boundary decision
- Categorical length prediction (0..W tokens)
- Bias initialization encourages early commit initially

#### 3. LatentToken (`kimi_linear/recursive/latent_token.py`)

**Purpose**: Optional learned token `[Z]` for global control signal.

**Architecture**:
```python
class LatentToken(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, token_id: int = None):
        self.token_id = token_id
        if token_id is None:
            # New learned embedding (not in vocab)
            self.embedding = nn.Parameter(torch.randn(1, hidden_size))
        else:
            # Reuse existing token
            self.embedding = None
    
    def forward(self, batch_size: int, device: torch.device) -> Tensor:
        # Returns [B, 1, hidden_size]
        if self.embedding is not None:
            return self.embedding.unsqueeze(0).expand(batch_size, 1, -1)
        # ... handle existing token case
```

**Usage**: Flows through all layers via attention, enables top→bottom skip connections.

#### 4. ChunkRefineWrapper (`kimi_linear/recursive/wrapper.py`)

**Purpose**: Main wrapper orchestrating chunked generation.

**Idempotent Design**:
```python
class ChunkRefineWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        config: RecursiveConfig,  # NEW: config-driven
    ):
        super().__init__()
        self.base = base_model
        self.config = config
        
        # Only create components if recursion enabled
        if config.recursive_enabled:
            self.refine_cells = nn.ModuleList([...])
            self.boundary = BoundaryHead(...)
            if config.use_latent_token:
                self.latent_token = LatentToken(...)
        else:
            # Pass-through mode: no components created
            self.refine_cells = None
            self.boundary = None
            self.latent_token = None
    
    def generate_chunks(self, ...):
        if not self.config.recursive_enabled:
            # IDEMPOTENT: fallback to standard generation
            return self.base.generate(...)
        
        # Full recursive generation
        # ... inner refinement loop ...
```

**Key Methods**:
- `generate_chunks()`: Public API for chunked generation
- `_forward_hidden()`: Run full model forward, return logits + hiddens
- `_seed_window()`: Generate initial chunk proposal
- `_pad_to_width()` / `_strip_trailing_blanks()`: Utilities

**Idempotency Guarantee**:
- When `recursive_enabled=False`: `generate_chunks()` → `base.generate()`
- No components created, zero overhead
- Identical behavior to baseline model

---

## Surgical Operation Plan

### Overview

The "surgery" refers to adding recursive components to the Kimi-Linear model. This is a **minimal, additive** operation that doesn't modify base model weights.

### Pre-Surgery Checklist

- [ ] Model weights present: `./models/kimi-linear-48b/` (92GB, verified)
- [ ] Architecture tests passing: `python test_architecture.py`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Configuration system ready: `RecursiveConfig` implemented
- [ ] Baseline evaluation completed: `eval_results/baseline_results.json`

### Surgery Steps

#### Step 1: Create Configuration System

**File**: `kimi_linear/recursive/config.py`

1. Define `RecursiveConfig` dataclass with all parameters
2. Add validation logic
3. Add save/load methods (JSON)
4. Create default configs for each stage

**Test**: 
```python
from kimi_linear.recursive.config import RecursiveConfig
config = RecursiveConfig(recursive_enabled=False)
assert config.validate() == []
config.save(Path("test_config.json"))
loaded = RecursiveConfig.from_file(Path("test_config.json"))
assert loaded == config
```

#### Step 2: Modify ChunkRefineWrapper

**File**: `kimi_linear/recursive/wrapper.py`

1. Add `RecursiveConfig` parameter to `__init__`
2. Implement conditional component creation:
   - If `recursive_enabled=False`: create no components
   - If `recursive_enabled=True`: create all components
3. Modify `generate_chunks()`:
   - If disabled: call `base.generate()` directly
   - If enabled: run full recursive loop
4. Ensure zero-init on RefineCell outputs (already done, verify)

**Test**:
```python
# Disabled mode (should match baseline)
wrapper_disabled = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=False))
output_disabled = wrapper_disabled.generate_chunks(...)

# Baseline
output_baseline = base_model.generate(...)

# Should be identical (within numerical precision)
assert torch.allclose(output_disabled, output_baseline)
```

#### Step 3: Update Model Loading

**File**: Create `kimi_linear/recursive/loader.py`

```python
def load_model_with_config(
    model_path: str,
    config_path: Optional[str] = None,
    recursive_enabled: Optional[bool] = None,
    **kwargs
) -> Tuple[nn.Module, AutoTokenizer, RecursiveConfig]:
    """
    Load model with recursive configuration.
    
    Args:
        model_path: Path to model (local or HF)
        config_path: Optional path to RecursiveConfig JSON
        recursive_enabled: Override enable flag
        **kwargs: Additional model loading kwargs
    
    Returns:
        (model, tokenizer, config)
    """
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        **kwargs
    )
    
    # Load or create config
    if config_path:
        config = RecursiveConfig.from_file(Path(config_path))
    else:
        config = RecursiveConfig(recursive_enabled=recursive_enabled or False)
    
    # Wrap if enabled
    if config.recursive_enabled:
        model = ChunkRefineWrapper(base_model, config)
    else:
        model = base_model
    
    return model, tokenizer, config
```

#### Step 4: Integration Testing

**Test Sequence**:

1. **Baseline Test**:
   ```python
   model, tokenizer, config = load_model_with_config(
       "./models/kimi-linear-48b",
       recursive_enabled=False
   )
   # Evaluate and save results
   ```

2. **After Surgery Test** (Critical):
   ```python
   model, tokenizer, config = load_model_with_config(
       "./models/kimi-linear-48b",
       recursive_enabled=True  # Enabled but UNTRAINED
   )
   # Evaluate - should match baseline (zero-init ensures this)
   # Compare: results should be nearly identical
   ```

3. **Component Verification**:
   ```python
   # Verify zero-init
   wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=True))
   for cell in wrapper.refine_cells:
       assert torch.allclose(cell.out.weight, torch.zeros_like(cell.out.weight))
   
   # Verify pass-through when disabled
   wrapper_disabled = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=False))
   assert wrapper_disabled.refine_cells is None
   ```

### Post-Surgery Verification

**Must Pass**:
- [ ] Baseline evaluation completes successfully
- [ ] After-surgery evaluation produces identical results to baseline
- [ ] Config save/load works correctly
- [ ] All components created with correct initialization
- [ ] Zero-init verified on all RefineCell output projections

**Success Criteria**:
- Perplexity difference < 0.1% (numerical precision)
- Token accuracy difference < 0.1%
- Generation outputs visually identical

---

## Training Pipeline

### Training Phases

#### Phase A: Sidecar Only (50-100k steps)

**Goal**: Train recursive components while base model frozen.

**Config**:
```python
phase_a_config = RecursiveConfig(
    recursive_enabled=True,
    chunk_width=128,
    max_inner_steps=4,
    lambda_final=1.0,
    lambda_masked=0.5,
    lambda_halt=0.5,
    corruption_rate=0.3,
)
```

**Training Setup**:
- Freeze: All base model weights
- Train: Refine cells, boundary head, latent token (if used)
- Learning rates:
  - Refine cells: 3e-4
  - Boundary head: 5e-4
  - Latent token: 3e-4

**Loss Components**:
- Final CE: On committed tokens
- Masked CE: Deep supervision on intermediate steps (editable positions only)
- Halt loss: BCE for commit prediction
- Length loss: CE for effective length prediction
- Ponder loss: Cost per refinement step (encourage efficiency)
- Stability loss: Temporal consistency (prevent flip-flopping)

#### Phase B: Light Unfreeze (20-50k steps)

**Goal**: Light fine-tuning with LoRA on top layers.

**Changes**:
- Enable LoRA (r=8-16) on top 1/3 layers
- Learning rate: 1e-5 (LoRA), keep sidecar at 3e-4
- Scheduled sampling on boundaries (mix teacher vs model)

#### Phase C: End-to-End Polish (Short)

**Goal**: Final polish with subset of layers.

**Changes**:
- Unfreeze every 4th layer with L2 regularization to pretrained (EWC-lite)
- Monitor n-gram drift
- Short runs only

### Training Script Integration

**File**: `train_recursive.py`

```python
def main():
    parser = argparse.ArgumentParser()
    # ... existing args ...
    
    # NEW: Config support
    parser.add_argument("--recursive-config", type=str, default=None,
                       help="Path to RecursiveConfig JSON file")
    parser.add_argument("--enable-recursive", action="store_true",
                       help="Enable recursion (overrides config file)")
    
    args = parser.parse_args()
    
    # Load config
    if args.recursive_config:
        config = RecursiveConfig.from_file(Path(args.recursive_config))
    else:
        config = RecursiveConfig(
            recursive_enabled=args.enable_recursive,
            chunk_width=args.chunk_width,
            max_inner_steps=args.max_inner_steps,
            # ... map other args
        )
    
    # Load model with config
    model, tokenizer, config = load_model_with_config(
        args.model_name,
        config=config,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Training loop uses config for all parameters
    train_phase_a(model, dataloader, optimizer, scheduler, accelerator, config)
```

### Data Pipeline

**Chunk Collation**:
```python
collator = ChunkCollator(
    chunk_width=config.chunk_width,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

**Corruption Masks**:
```python
# During training
corruption_mask = create_corruption_mask(
    length=chunk_width,
    corruption_rate=config.corruption_rate,
    strategy="random",  # or "span"
)
```

**Post-Training Data**:
- Use `FlywheelDataset` from `kimi_linear/recursive/post_training_data.py`
- Load datasets from HuggingFace Hub
- Collate into chunks for training

---

## Evaluation Framework

### Evaluation Stages

#### Stage 1: Baseline

**Purpose**: Establish baseline before any modifications.

**Config**: `RecursiveConfig(recursive_enabled=False)`

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --recursive-config configs/baseline.json \
    --output_dir ./eval_results
```

**Metrics**:
- Perplexity
- Token accuracy
- Reasoning accuracy (GSM8K-style)
- Generation quality
- Throughput
- Memory usage

#### Stage 2: After Surgery

**Purpose**: Verify zero-init preserves baseline (critical test).

**Config**: `RecursiveConfig(recursive_enabled=True)` (but untrained)

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --recursive-config configs/after_surgery.json \
    --output_dir ./eval_results
```

**Success Criteria**: Results should match baseline within numerical precision.

**Expected**:
- Perplexity: Within 0.1% of baseline
- Token accuracy: Within 0.1% of baseline
- Generation: Visually identical outputs

#### Stage 3: Fine-tuned Vanilla

**Purpose**: Measure improvement from fine-tuning alone.

**Config**: `RecursiveConfig(recursive_enabled=False)` (fine-tuned base model)

**Command**:
```bash
python evaluate_model.py \
    --model_path ./checkpoints/vanilla_finetune \
    --stage finetuned_vanilla \
    --recursive-config configs/baseline.json \
    --output_dir ./eval_results
```

#### Stage 4: Fine-tuned Recursive

**Purpose**: Measure improvement from recursive reasoning.

**Config**: `RecursiveConfig(recursive_enabled=True)` (trained recursive model)

**Command**:
```bash
python evaluate_model.py \
    --model_path ./checkpoints/recursive_finetune \
    --stage finetuned_recursive \
    --recursive-config configs/recursive_phase_a.json \
    --output_dir ./eval_results
```

### Evaluation Metrics

**Core Metrics** (all stages):
1. **Perplexity**: Lower is better, baseline ~10-20 for 48B models
2. **Token Accuracy**: Higher is better, baseline ~50-60%
3. **Reasoning Accuracy**: Higher is better, tests generalization
4. **Generation Quality**: Qualitative (coherence, length, relevance)

**Recursive-Specific Metrics** (when enabled):
1. **Average Refinement Steps**: Target 1-2 steps (efficient)
2. **Commit Rate**: Fraction of chunks committed per refinement
3. **Chunk Length Distribution**: Should match natural boundaries
4. **Stability@K**: Consistency across refinement steps

**Performance Metrics**:
1. **Throughput**: Tokens/second (compare baseline vs recursive)
2. **Memory**: Peak memory usage
3. **Training Speed**: Steps/second during training

### Comparison Script

**File**: `compare_results.py`

Usage:
```bash
python compare_results.py baseline after_surgery finetuned_vanilla finetuned_recursive
```

Output:
- Comparison table of all metrics
- Improvement percentages over baseline
- Stage-by-stage analysis

---

## Metrics & Monitoring

### TensorBoard Setup

**Start TensorBoard**:
```bash
# On Lambda instance (192.222.58.183)
tmux new -s tensorboard
cd recursive-kimi-linear
./start_tensorboard.sh
# Ctrl+B, D to detach
```

**Access via SSH Port Forwarding**:
```bash
# On your local machine
ssh -L 6006:localhost:6006 ubuntu@192.222.58.183 -N
# Then open: http://localhost:6006
```

### MetricsTracker Integration

**During Training**:
```python
metrics_tracker = MetricsTracker(log_dir="./logs/phase_a", use_tensorboard=True)

# Log losses
metrics_tracker.log_losses(loss_dict, step, prefix="train")

# Log generation metrics
metrics_tracker.log_generation_metrics(
    avg_steps=avg_refine_steps,
    commit_rate=commit_rate,
    avg_chunk_len=avg_chunk_len,
    step=step,
)

# Periodic evaluation
if step % eval_interval == 0:
    eval_results = evaluate_model(
        model, tokenizer, eval_data,
        metrics_tracker=metrics_tracker,
        step=step,
    )
```

**TensorBoard Dashboards**:
- **SCALARS**: Loss curves, metrics over time
- **HISTOGRAMS**: Parameter distributions
- **IMAGES**: Sample generations (if logged)

**Key Metrics to Monitor**:
- `train/loss/total`: Should decrease
- `train/generation/avg_refine_steps`: Should converge to 1-2
- `train/generation/commit_rate`: Should increase
- `evaluation/*/perplexity`: Should improve over baseline

---

## Reproducibility & Idempotency

### Idempotency Guarantees

1. **Config-Driven**: Same config file = same behavior
2. **Zero-Init Safety**: Enabled but untrained = baseline
3. **Pass-Through Mode**: Disabled = identical to no wrapper
4. **Deterministic**: Fixed seeds produce identical results

### Configuration Management

**Save Config with Checkpoints**:
```python
# During training
checkpoint_dir = Path(args.output_dir) / f"step_{step}"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save model
accelerator.save_model(model, checkpoint_dir / "model")

# Save config (NEW)
config.save(checkpoint_dir / "recursive_config.json")

# Save metrics
metrics_tracker.save_checkpoint_metrics(checkpoint_dir)
```

**Load Config from Checkpoint**:
```python
def load_checkpoint(checkpoint_dir: Path):
    config = RecursiveConfig.from_file(checkpoint_dir / "recursive_config.json")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir / "model",
        trust_remote_code=True,
    )
    if config.recursive_enabled:
        model = ChunkRefineWrapper(model, config)
    return model, config
```

### Reproducibility Checklist

- [ ] Config files saved with all checkpoints
- [ ] Random seeds documented in config
- [ ] Evaluation uses same dataset across stages
- [ ] Hardware requirements documented
- [ ] Dependencies pinned (requirements.txt)

---

## Testing Strategy

### Unit Tests

**Test RefineCell**:
```python
def test_refine_cell_zero_init():
    cell = RefineCell(d_model=768, d_state=512)
    h = torch.randn(2, 32, 768)
    s = torch.randn(2, 512)
    
    h_refined, s_next = cell(h, s)
    
    # At init, delta should be near zero (output projection zero-init)
    # So h_refined ≈ h (within numerical precision)
    assert torch.allclose(h_refined, h, atol=1e-5)
```

**Test Wrapper Pass-Through**:
```python
def test_wrapper_pass_through():
    base_model = DummyBaseModel()
    config_disabled = RecursiveConfig(recursive_enabled=False)
    wrapper = ChunkRefineWrapper(base_model, config_disabled)
    
    # Should have no components
    assert wrapper.refine_cells is None
    assert wrapper.boundary is None
    
    # generate_chunks should call base.generate
    input_ids = torch.randint(1, 1000, (1, 8))
    output = wrapper.generate_chunks(input_ids, max_new_tokens=16)
    
    # Should match base model output
    base_output = base_model.generate(input_ids, max_length=24)
    assert torch.equal(output, base_output)
```

### Integration Tests

**Test After-Surgery Equals Baseline**:
```python
def test_after_surgery_equals_baseline():
    # Load baseline
    baseline_model, tokenizer, _ = load_model_with_config(
        "./models/kimi-linear-48b",
        recursive_enabled=False,
    )
    
    # Load after surgery
    surgery_model, _, config = load_model_with_config(
        "./models/kimi-linear-48b",
        recursive_enabled=True,
    )
    
    # Same input
    input_ids = tokenizer("Test prompt", return_tensors="pt")["input_ids"]
    
    # Generate
    baseline_output = baseline_model.generate(input_ids, max_new_tokens=64)
    surgery_output = surgery_model.generate_chunks(
        input_ids,
        max_new_tokens=64,
        chunk_width=64,
    )
    
    # Should be identical (within numerical precision)
    assert torch.allclose(
        baseline_output,
        surgery_output,
        atol=1e-3,  # Allow small numerical differences
    )
```

### Validation Tests

**Config Validation**:
```python
def test_config_validation():
    # Invalid configs should raise errors
    invalid_config = RecursiveConfig(
        chunk_width=-10,  # Invalid
        commit_threshold=1.5,  # Invalid (> 1.0)
    )
    errors = invalid_config.validate()
    assert len(errors) > 0
    assert any("chunk_width" in err for err in errors)
    assert any("commit_threshold" in err for err in errors)
```

---

## Deployment & Usage

### Quick Start

**1. Setup**:
```bash
cd recursive-kimi-linear
python setup_metrics.py
```

**2. Baseline Evaluation**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --output_dir ./eval_results
```

**3. Enable Recursion**:
```bash
# Create config
python -c "
from kimi_linear.recursive.config import RecursiveConfig
config = RecursiveConfig(recursive_enabled=True, chunk_width=128)
config.save('configs/my_config.json')
"

# Evaluate after surgery
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --recursive-config configs/my_config.json \
    --output_dir ./eval_results
```

**4. Training**:
```bash
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --recursive-config configs/recursive_phase_a.json \
    --batch_size 4 \
    --num_steps 50000 \
    --phase a \
    --trust_remote_code \
    --log_dir ./logs/phase_a \
    --output_dir ./checkpoints/phase_a
```

**5. Monitor**:
```bash
# Start TensorBoard
./start_tensorboard.sh

# On local machine, SSH forward:
ssh -L 6006:localhost:6006 ubuntu@192.222.58.183 -N

# Open http://localhost:6006
```

### Config Files Structure

```
configs/
├── baseline.json                    # No recursion
├── after_surgery.json               # Enabled but untrained
├── recursive_phase_a.json          # Training Phase A
├── recursive_phase_b.json          # Training Phase B
└── recursive_phase_c.json          # Training Phase C
```

### Evaluation Workflow

**Complete Evaluation Sequence**:
```bash
# Stage 1: Baseline
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --recursive-config configs/baseline.json

# Stage 2: After Surgery (VERIFY ZERO-INIT)
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --recursive-config configs/after_surgery.json

# Stage 3: After Vanilla Fine-tuning
python evaluate_model.py \
    --model_path ./checkpoints/vanilla_finetune \
    --stage finetuned_vanilla \
    --recursive-config configs/baseline.json

# Stage 4: After Recursive Fine-tuning
python evaluate_model.py \
    --model_path ./checkpoints/recursive_finetune \
    --stage finetuned_recursive \
    --recursive-config configs/recursive_phase_a.json

# Compare all stages
python compare_results.py baseline after_surgery finetuned_vanilla finetuned_recursive
```

---

## File Structure Reference

```
recursive-kimi-linear/
├── kimi_linear/
│   ├── __init__.py
│   └── recursive/
│       ├── __init__.py
│       ├── config.py              # NEW: RecursiveConfig
│       ├── refine_cell.py          # Per-layer refinement
│       ├── boundary_head.py        # Commit/length prediction
│       ├── latent_token.py        # Optional [Z] token
│       ├── wrapper.py             # Main wrapper (MODIFIED: config-driven)
│       ├── losses.py              # Loss functions
│       ├── data.py                # Data collation
│       ├── metrics.py             # Metrics tracking
│       ├── loader.py              # NEW: Model loading with config
│       └── post_training_data.py  # Flywheel dataset integration
│
├── configs/                        # NEW: Config files directory
│   ├── baseline.json
│   ├── after_surgery.json
│   ├── recursive_phase_a.json
│   ├── recursive_phase_b.json
│   └── recursive_phase_c.json
│
├── train_recursive.py              # MODIFIED: Config support
├── evaluate_model.py               # MODIFIED: Config support
├── compare_results.py              # Comparison tool
├── setup_metrics.py                # Metrics setup
├── start_tensorboard.sh            # TensorBoard launcher
├── test_architecture.py            # Architecture tests
│
├── docs/
│   └── archive/                    # Old setup docs
│
├── logs/                           # TensorBoard logs
├── checkpoints/                    # Model checkpoints
│   ├── vanilla_finetune/
│   │   ├── model/
│   │   └── recursive_config.json  # Config saved with checkpoint
│   └── recursive_finetune/
│       └── ...
└── eval_results/                   # Evaluation results
    ├── baseline_results.json
    ├── after_surgery_results.json
    └── ...
```

---

## Implementation Checklist

### Phase 1: Configuration System

- [ ] Create `kimi_linear/recursive/config.py`
  - [ ] Define `RecursiveConfig` dataclass
  - [ ] Add validation method
  - [ ] Add save/load methods (JSON)
  - [ ] Add from_dict/to_dict
  - [ ] Test config save/load

### Phase 2: Idempotent Wrapper

- [ ] Modify `ChunkRefineWrapper.__init__` to accept `RecursiveConfig`
- [ ] Add conditional component creation (if enabled)
- [ ] Modify `generate_chunks()` to pass-through when disabled
- [ ] Verify zero-init on RefineCell outputs
- [ ] Test pass-through mode equals baseline

### Phase 3: Model Loading

- [ ] Create `kimi_linear/recursive/loader.py`
- [ ] Implement `load_model_with_config()`
- [ ] Support config file or direct config object
- [ ] Test loading with various configs

### Phase 4: Script Updates

- [ ] Update `train_recursive.py`:
  - [ ] Add `--recursive-config` argument
  - [ ] Add `--enable-recursive` flag
  - [ ] Use config throughout training loop
  - [ ] Save config with checkpoints
- [ ] Update `evaluate_model.py`:
  - [ ] Add config support
  - [ ] Use config when loading model
- [ ] Update `compare_results.py`:
  - [ ] Load configs from checkpoints
  - [ ] Include config info in comparison

### Phase 5: Documentation

- [ ] Update `EVALUATION_PLAN.md`:
  - [ ] Config-based examples
  - [ ] Idempotency guarantees
  - [ ] Testing workflow
- [ ] Update `METRICS_GUIDE.md`:
  - [ ] Config-driven workflow
  - [ ] TensorBoard integration
- [ ] Update `README.md`:
  - [ ] Config usage examples
  - [ ] Quick start with configs
- [ ] Create `SURGICAL_IMPLEMENTATION.md`:
  - [ ] Step-by-step surgery guide
  - [ ] Verification procedures
  - [ ] Rollback instructions

### Phase 6: Testing

- [ ] Test config save/load
- [ ] Test wrapper pass-through (disabled = baseline)
- [ ] Test after-surgery equals baseline (zero-init)
- [ ] Test config validation
- [ ] Test all scripts with config files

### Phase 7: Example Configs

- [ ] Create `configs/baseline.json`
- [ ] Create `configs/after_surgery.json`
- [ ] Create `configs/recursive_phase_a.json`
- [ ] Create `configs/recursive_phase_b.json`
- [ ] Create `configs/recursive_phase_c.json`

---

## Critical Implementation Details

### Zero-Init Verification

**Must Verify**:
```python
# When recursion enabled but untrained
wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=True))

# All RefineCell output projections should be zero
for cell in wrapper.refine_cells:
    assert torch.allclose(
        cell.out.weight,
        torch.zeros_like(cell.out.weight),
        atol=1e-6
    ), "RefineCell not zero-initialized!"

# This ensures h_refined = h + 0 = h at initialization
```

### Pass-Through Implementation

**When `recursive_enabled=False`**:
```python
def generate_chunks(self, input_ids, ...):
    if not self.config.recursive_enabled:
        # IDEMPOTENT: Direct pass-through to base model
        return self.base.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
    
    # Full recursive generation
    # ... inner refinement loop ...
```

### Config File Format

**Minimum Valid Config** (baseline):
```json
{
  "recursive_enabled": false
}
```

**Full Config Example**:
```json
{
  "recursive_enabled": true,
  "chunk_width": 128,
  "max_inner_steps": 4,
  "commit_threshold": 0.7,
  "min_commit_threshold": 0.3,
  "layers_to_refine": "all",
  "use_latent_token": true,
  "d_state": 512,
  "gate_hidden": 1024,
  "max_chunk_len": 128,
  "corruption_rate": 0.3,
  "corruption_schedule": "linear",
  "corruption_rate_final": 0.1,
  "lambda_final": 1.0,
  "lambda_masked": 0.5,
  "lambda_halt": 0.5,
  "lambda_len": 0.05,
  "lambda_ponder": 0.01,
  "lambda_stability": 0.01,
  "cache_reset_policy": "keep_kda_recurrent_reset_mla",
  "enable_length_head": true,
  "use_soft_tokens": false
}
```

---

## Key Research Insights

### Why This Should Work

1. **Semantic Prior**: Pretrained LLMs have token-local manifolds where small iterative updates converge fast
2. **KDA Recurrent**: Efficient stateful kernel allows 2-4 refinement steps without major overhead
3. **TRM Pattern**: Tiny recursive nets generalize via iterative refinement; we transfer the *procedure* not the architecture
4. **Zero-Init Safety**: Allows clean A/B testing without breaking baseline

### Expected Behavior

**After Surgery (Untrained)**:
- Should match baseline exactly (zero-init ensures this)
- All refine cells produce zero deltas
- Boundary head may predict randomly, but shouldn't affect logits (if not used in forward)

**During Training**:
- Loss should decrease as model learns to refine
- Average refinement steps should converge to 1-2 (efficient)
- Commit rate should increase (better boundary prediction)

**After Training**:
- Perplexity should improve over baseline
- Reasoning accuracy should improve (if recursion helps)
- Generation quality should improve (more coherent, longer outputs)

---

## Troubleshooting Guide

### Issue: After-surgery doesn't match baseline

**Check**:
1. Zero-init verified on all RefineCell outputs
2. Pass-through mode works when disabled
3. No components active in forward pass when untrained

**Fix**:
- Ensure RefineCell output projections are zero-initialized
- Verify wrapper doesn't use boundary head in forward when untrained

### Issue: Config not loading

**Check**:
1. JSON file is valid
2. Config path is correct
3. All required fields present

**Fix**:
- Use `RecursiveConfig.validate()` to check errors
- Provide default values for missing fields

### Issue: Training loss not decreasing

**Check**:
1. Learning rates appropriate
2. Data pipeline working
3. Corruption masks correct
4. Components actually trainable (not frozen)

**Fix**:
- Verify which parameters require grad
- Check learning rate schedules
- Validate loss computation

---

## Next Steps for Implementation

1. **Start with Configuration System**: Create `config.py` first, test thoroughly
2. **Modify Wrapper**: Add config support, implement pass-through
3. **Update Scripts**: Add config arguments, save configs with checkpoints
4. **Test Continuously**: Verify idempotency at each step
5. **Document as You Go**: Update docs as features are added

---

## Appendix: Full Code Snippets

### RecursiveConfig Complete Implementation

```python
from dataclasses import dataclass, asdict
from typing import List, Union, Optional
from pathlib import Path
import json

@dataclass
class RecursiveConfig:
    """Configuration for recursive chunked generation."""
    
    # Core
    recursive_enabled: bool = False
    chunk_width: int = 128
    max_inner_steps: int = 4
    commit_threshold: float = 0.7
    min_commit_threshold: float = 0.3
    
    # Architecture
    layers_to_refine: Union[str, List[int]] = "all"
    use_latent_token: bool = True
    d_state: int = 512
    gate_hidden: int = 1024
    max_chunk_len: int = 128
    
    # Training
    corruption_rate: float = 0.3
    corruption_schedule: str = "linear"
    corruption_rate_final: float = 0.1
    
    # Loss weights
    lambda_final: float = 1.0
    lambda_masked: float = 0.5
    lambda_halt: float = 0.5
    lambda_len: float = 0.05
    lambda_ponder: float = 0.01
    lambda_stability: float = 0.01
    
    # Advanced
    cache_reset_policy: str = "keep_kda_recurrent_reset_mla"
    enable_length_head: bool = True
    use_soft_tokens: bool = False
    
    def validate(self) -> List[str]:
        errors = []
        if self.chunk_width <= 0:
            errors.append("chunk_width must be positive")
        if self.max_inner_steps <= 0:
            errors.append("max_inner_steps must be positive")
        if not 0.0 <= self.commit_threshold <= 1.0:
            errors.append("commit_threshold must be in [0, 1]")
        if not 0.0 <= self.min_commit_threshold <= 1.0:
            errors.append("min_commit_threshold must be in [0, 1]")
        if self.commit_threshold < self.min_commit_threshold:
            errors.append("commit_threshold must be >= min_commit_threshold")
        if self.corruption_schedule not in ["linear", "cosine", "none"]:
            errors.append("corruption_schedule must be linear, cosine, or none")
        return errors
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RecursiveConfig':
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> 'RecursiveConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
```

### Loader Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kimi_linear.recursive.config import RecursiveConfig
from kimi_linear.recursive.wrapper import ChunkRefineWrapper
import torch

def load_model_with_config(
    model_path: str,
    config: Optional[RecursiveConfig] = None,
    config_path: Optional[str] = None,
    recursive_enabled: Optional[bool] = None,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    **kwargs
) -> tuple:
    """
    Load model with recursive configuration.
    
    Returns:
        (model, tokenizer, config)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **kwargs
    )
    
    # Resolve config
    if config is not None:
        # Use provided config
        final_config = config
    elif config_path:
        # Load from file
        final_config = RecursiveConfig.from_file(Path(config_path))
    elif recursive_enabled is not None:
        # Create from flag
        final_config = RecursiveConfig(recursive_enabled=recursive_enabled)
    else:
        # Default: disabled
        final_config = RecursiveConfig(recursive_enabled=False)
    
    # Validate
    errors = final_config.validate()
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    
    # Wrap if enabled
    if final_config.recursive_enabled:
        model = ChunkRefineWrapper(base_model, final_config)
    else:
        model = base_model
    
    return model, tokenizer, final_config
```

---

## Final Notes

This document provides complete context for implementing the recursive configuration system. Key principles:

1. **Idempotency First**: Every feature must support being disabled cleanly
2. **Config-Driven**: All parameters in RecursiveConfig
3. **Zero-Init Critical**: Must preserve baseline when enabled but untrained
4. **Reproducible**: Configs saved with checkpoints
5. **Minimal Surgery**: Additive components only

The implementation should proceed incrementally, testing idempotency at each step.

