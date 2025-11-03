# Research Agent Directions: Config-Driven Recursive System

**Date**: 2025-01-03  
**Status**: Implementation Complete - Ready for Testing  
**Commit**: e5f5217

---

## Executive Summary

A configuration-driven recursive system has been implemented for the Kimi-Linear recursive architecture. The system ensures **idempotency** (baseline behavior when recursion is disabled) and provides a **reproducible** configuration framework for all experiments.

### Key Achievements

✅ **RecursiveConfig System**: Centralized configuration with validation  
✅ **Idempotent Wrapper**: Pass-through behavior when recursion disabled  
✅ **Model Loader**: Unified loading with config support  
✅ **Backward Compatibility**: Legacy code still works  
✅ **Example Configs**: Pre-configured for all testing stages  

---

## What Was Implemented

### 1. Configuration System (`kimi_linear/recursive/config.py`)

**Purpose**: Centralized configuration for all recursive parameters.

**Key Features**:
- `RecursiveConfig` dataclass with 25+ parameters
- Validation logic (returns list of errors)
- JSON save/load functionality
- Default config factories for common use cases

**Usage**:
```python
from kimi_linear.recursive.config import RecursiveConfig

# Create config
config = RecursiveConfig(
    recursive_enabled=True,
    chunk_width=128,
    max_inner_steps=4,
)

# Validate
errors = config.validate()
if errors:
    raise ValueError(f"Invalid config: {errors}")

# Save/load
config.save("my_config.json")
loaded = RecursiveConfig.from_file("my_config.json")
```

**Default Configs Available**:
- `get_baseline_config()` - Recursion disabled (baseline)
- `get_after_surgery_config()` - Enabled but untrained
- `get_phase_a_config()` - Training Phase A (sidecar only)
- `get_phase_b_config()` - Training Phase B (light unfreeze)
- `get_phase_c_config()` - Training Phase C (end-to-end)

### 2. Idempotent Wrapper (`kimi_linear/recursive/wrapper.py`)

**Purpose**: Chunked recursive generation that preserves baseline behavior when disabled.

**Key Changes**:
- Accepts `RecursiveConfig` object (preferred) or legacy parameters
- **Conditional component creation**: When `recursive_enabled=False`, no components are created
- **Pass-through mode**: `generate_chunks()` calls `base.generate()` when disabled
- Zero-init on RefineCell outputs preserved (critical for baseline preservation)

**Idempotency Guarantee**:
```python
# When recursive_enabled=False:
wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=False))
output = wrapper.generate_chunks(input_ids, max_new_tokens=64)

# This is IDENTICAL to:
baseline_output = base_model.generate(input_ids, max_new_tokens=64)
# (within numerical precision)
```

**Legacy Support**:
Still supports old-style initialization:
```python
wrapper = ChunkRefineWrapper(
    base_model,
    layers_to_refine="all",
    use_latent_token=True,
    d_state=512,
)
# This creates a config internally with recursive_enabled=True
```

### 3. Model Loader (`kimi_linear/recursive/loader.py`)

**Purpose**: Unified model loading with config support.

**Function**: `load_model_with_config()`

**Usage**:
```python
from kimi_linear.recursive import load_model_with_config

# Option 1: From config file
model, tokenizer, config = load_model_with_config(
    "./models/kimi-linear-48b",
    config_path="configs/after_surgery.json"
)

# Option 2: Direct config object
from kimi_linear.recursive.config import get_baseline_config
config = get_baseline_config()
model, tokenizer, config = load_model_with_config(
    "./models/kimi-linear-48b",
    config=config
)

# Option 3: Boolean flag (creates default config)
model, tokenizer, config = load_model_with_config(
    "./models/kimi-linear-48b",
    recursive_enabled=True
)
```

**Returns**: `(model, tokenizer, config)` tuple

### 4. Updated Evaluation Script (`evaluate_model.py`)

**New Features**:
- `--recursive-config` argument for config file path
- Auto-selects config based on `--stage` argument
- Maintains backward compatibility with `--use_recursive` flag

**Usage**:
```bash
# New way (config-driven)
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --recursive-config configs/after_surgery.json

# Old way (still works)
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --use_recursive
```

**Auto-Config Selection**:
If `--recursive-config` not provided, script auto-selects based on stage:
- `baseline` → `configs/baseline.json`
- `after_surgery` → `configs/after_surgery.json`
- `finetuned_vanilla` → `configs/baseline.json`
- `finetuned_recursive` → `configs/recursive_phase_a.json`

### 5. Configuration Files (`configs/`)

**Created Files**:
- `baseline.json` - Recursion disabled (baseline behavior)
- `after_surgery.json` - Enabled but untrained (should match baseline due to zero-init)
- `recursive_phase_a.json` - Training Phase A (sidecar only)
- `recursive_phase_b.json` - Training Phase B (light unfreeze)
- `recursive_phase_c.json` - Training Phase C (end-to-end polish)

**Location**: `/home/ubuntu/recursive-kimi-linear/configs/`

---

## Critical Concepts

### 1. Idempotency

**Definition**: When `recursive_enabled=False`, the wrapper behaves **identically** to the base model (no wrapper overhead, same outputs).

**Why Critical**:
- Enables clean A/B testing: enable recursion without training should match baseline
- Verifies zero-init works: after surgery (enabled but untrained) should equal baseline
- Allows disabling recursion without code changes

**Verification**:
```python
# Test idempotency
base_model = load_base_model()
config_disabled = RecursiveConfig(recursive_enabled=False)
wrapper = ChunkRefineWrapper(base_model, config=config_disabled)

input_ids = tokenizer("Test", return_tensors="pt")["input_ids"]

# Should be identical
output_wrapper = wrapper.generate_chunks(input_ids, max_new_tokens=64)
output_baseline = base_model.generate(input_ids, max_new_tokens=64)

assert torch.allclose(output_wrapper, output_baseline, atol=1e-3)
```

### 2. Zero-Init Safety

**Definition**: RefineCell output projections are zero-initialized, so `delta = 0` at initialization.

**Implementation**:
```python
# In RefineCell.__init__()
self.out = nn.Linear(4 * d_model, d_model, bias=False)
nn.init.zeros_(self.out.weight)  # CRITICAL
```

**Effect**:
- When recursion enabled but untrained: `h_refined = h + 0 = h` (no change)
- After-surgery model should match baseline exactly

**Verification**:
```python
wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=True))
# Verify zero-init
for cell in wrapper.refine_cells:
    assert torch.allclose(cell.out.weight, torch.zeros_like(cell.out.weight))
```

### 3. Pass-Through Mode

**When**: `recursive_enabled=False`

**Behavior**:
- No components created (`refine_cells = None`, `boundary = None`)
- `generate_chunks()` calls `base.generate()` directly
- Zero overhead, identical behavior

**Code**:
```python
def generate_chunks(self, input_ids, ...):
    if not self.config.recursive_enabled:
        # IDEMPOTENT: Direct pass-through
        return self.base.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            ...
        )
    # Full recursive generation...
```

---

## Testing Protocol

### Stage 1: Baseline Evaluation

**Purpose**: Establish baseline performance before any modifications.

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --output_dir ./eval_results
```

**Expected**:
- Uses `configs/baseline.json` (auto-selected)
- `recursive_enabled=False`
- Normal model generation

**Save Results**: `eval_results/baseline_results.json`

### Stage 2: After-Surgery Verification (CRITICAL)

**Purpose**: Verify zero-init preserves baseline (must match Stage 1).

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --output_dir ./eval_results
```

**Expected**:
- Uses `configs/after_surgery.json` (auto-selected)
- `recursive_enabled=True` but components untrained
- Results should match baseline within numerical precision (< 0.1% difference)

**Success Criteria**:
- Perplexity difference < 0.1%
- Token accuracy difference < 0.1%
- Generation outputs visually identical

**If This Fails**:
- Check zero-init on RefineCell outputs
- Verify components not affecting forward pass when untrained
- Check pass-through mode works correctly

**Save Results**: `eval_results/after_surgery_results.json`

### Stage 3: Comparison Test

**Purpose**: Verify baseline = after_surgery (critical test).

**Script**:
```python
import json
from pathlib import Path

baseline = json.load(open("eval_results/baseline_results.json"))
after_surgery = json.load(open("eval_results/after_surgery_results.json"))

baseline_ppl = baseline["perplexity"]["perplexity"]
surgery_ppl = after_surgery["perplexity"]["perplexity"]

diff_pct = abs(baseline_ppl - surgery_ppl) / baseline_ppl * 100

print(f"Baseline PPL: {baseline_ppl:.4f}")
print(f"After Surgery PPL: {surgery_ppl:.4f}")
print(f"Difference: {diff_pct:.2f}%")

if diff_pct > 0.1:
    print("⚠ WARNING: Difference > 0.1% - zero-init may not be working!")
else:
    print("✓ SUCCESS: Zero-init verified!")
```

---

## Next Steps for Research Agent

### Immediate Actions (Priority 1)

1. **Verify After-Surgery Equals Baseline**
   ```bash
   # Run both evaluations
   python evaluate_model.py --model_path ./models/kimi-linear-48b --stage baseline
   python evaluate_model.py --model_path ./models/kimi-linear-48b --stage after_surgery
   
   # Compare results
   python -c "
   import json
   b = json.load(open('eval_results/baseline_results.json'))
   a = json.load(open('eval_results/after_surgery_results.json'))
   print(f'PPL diff: {abs(b[\"perplexity\"][\"perplexity\"] - a[\"perplexity\"][\"perplexity\"]) / b[\"perplexity\"][\"perplexity\"] * 100:.2f}%')
   "
   ```
   
   **If difference > 0.1%**: Investigate zero-init and pass-through implementation.

2. **Test Config System**
   ```bash
   # Test config load/save
   python -c "
   from kimi_linear.recursive.config import RecursiveConfig, get_baseline_config
   from pathlib import Path
   
   # Test validation
   config = RecursiveConfig(chunk_width=-1)  # Invalid
   errors = config.validate()
   assert len(errors) > 0, 'Should have validation errors'
   print('✓ Validation works')
   
   # Test save/load
   config = get_baseline_config()
   config.save('test_config.json')
   loaded = RecursiveConfig.from_file('test_config.json')
   assert loaded == config, 'Save/load failed'
   print('✓ Save/load works')
   "
   ```

3. **Test Model Loading**
   ```python
   # Test loader with different configs
   from kimi_linear.recursive import load_model_with_config
   
   # Test disabled mode
   model, tokenizer, config = load_model_with_config(
       "./models/kimi-linear-48b",
       recursive_enabled=False
   )
   assert not config.recursive_enabled
   assert not hasattr(model, 'refine_cells') or model.refine_cells is None
   print("✓ Disabled mode works")
   
   # Test enabled mode
   model, tokenizer, config = load_model_with_config(
       "./models/kimi-linear-48b",
       config_path="configs/after_surgery.json"
   )
   assert config.recursive_enabled
   assert model.refine_cells is not None
   print("✓ Enabled mode works")
   ```

### Medium Priority (Priority 2)

4. **Update Training Script** (`train_recursive.py`)
   - Add `--recursive-config` argument
   - Use config for all parameters
   - Save config with checkpoints
   - Load config when resuming

   **Example**:
   ```python
   parser.add_argument("--recursive-config", type=str, default=None)
   
   # In main():
   if args.recursive_config:
       config = RecursiveConfig.from_file(Path(args.recursive_config))
   else:
       config = RecursiveConfig(recursive_enabled=args.use_recursive)
   
   model, tokenizer, config = load_model_with_config(
       args.model_name,
       config=config,
   )
   
   # Save config with checkpoint
   checkpoint_dir / "recursive_config.json"
   config.save(checkpoint_dir / "recursive_config.json")
   ```

5. **Create Unit Tests**
   - Test config validation
   - Test wrapper pass-through
   - Test zero-init preservation
   - Test config save/load

   **File**: `tests/test_config_system.py`

### Lower Priority (Priority 3)

6. **Documentation Updates**
   - Update `EVALUATION_PLAN.md` with config examples
   - Update `METRICS_GUIDE.md` with config workflow
   - Add config examples to `README.md`

7. **Compare Results Script**
   - Update `compare_results.py` to load configs from checkpoints
   - Include config info in comparison tables

---

## Configuration Parameters Reference

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recursive_enabled` | `False` | Master switch (must be True to enable) |
| `chunk_width` | `128` | Fixed chunk width W |
| `max_inner_steps` | `4` | Maximum refinement steps K |
| `commit_threshold` | `0.7` | Halt threshold tau |
| `min_commit_threshold` | `0.3` | Minimum threshold (deadlock prevention) |

### Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers_to_refine` | `"all"` | Which layers to refine ("all" or list) |
| `use_latent_token` | `True` | Use learned [Z] token |
| `d_state` | `512` | Persistent state dimension |
| `gate_hidden` | `1024` | Boundary head MLP hidden dim |
| `max_chunk_len` | `128` | Max chunk length for prediction |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corruption_rate` | `0.3` | Initial corruption rate |
| `corruption_schedule` | `"linear"` | Schedule type: linear/cosine/none |
| `corruption_rate_final` | `0.1` | Final corruption rate |

### Loss Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_final` | `1.0` | Final CE loss weight |
| `lambda_masked` | `0.5` | Masked CE loss (deep supervision) |
| `lambda_halt` | `0.5` | Halt/commit prediction loss |
| `lambda_len` | `0.05` | Length prediction loss |
| `lambda_ponder` | `0.01` | Ponder cost (step penalty) |
| `lambda_stability` | `0.01` | Temporal consistency loss |

---

## Troubleshooting Guide

### Issue: After-surgery doesn't match baseline

**Symptoms**:
- Perplexity difference > 0.1%
- Different generation outputs

**Diagnosis**:
1. Check zero-init:
   ```python
   wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=True))
   for cell in wrapper.refine_cells:
       print(f"Cell output weight norm: {cell.out.weight.norm()}")
       # Should be near zero (< 1e-5)
   ```

2. Verify pass-through:
   ```python
   wrapper_disabled = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=False))
   assert wrapper_disabled.refine_cells is None
   ```

3. Check if boundary head affects forward:
   - Boundary head should only be used in `generate_chunks()`
   - Not used in base model forward pass

**Fix**:
- Ensure RefineCell output projection is zero-initialized
- Verify wrapper doesn't modify base model forward when disabled

### Issue: Config not loading

**Symptoms**:
- `FileNotFoundError` or invalid config

**Diagnosis**:
```python
from kimi_linear.recursive.config import RecursiveConfig

try:
    config = RecursiveConfig.from_file("path/to/config.json")
    errors = config.validate()
    if errors:
        print(f"Validation errors: {errors}")
except Exception as e:
    print(f"Load error: {e}")
```

**Fix**:
- Check file path exists
- Validate JSON syntax
- Use `config.validate()` to check parameter ranges

### Issue: Wrapper not idempotent

**Symptoms**:
- Disabled mode produces different outputs than base model

**Diagnosis**:
```python
# Test pass-through
base_output = base_model.generate(input_ids, max_new_tokens=64)
wrapper = ChunkRefineWrapper(base_model, RecursiveConfig(recursive_enabled=False))
wrapper_output = wrapper.generate_chunks(input_ids, max_new_tokens=64)

diff = (base_output != wrapper_output).sum()
print(f"Token differences: {diff}")
```

**Fix**:
- Verify `generate_chunks()` pass-through calls `base.generate()` with same args
- Check no components are created when disabled

---

## Code Examples

### Example 1: Load Model with Config

```python
from kimi_linear.recursive import load_model_with_config

# Load baseline (no recursion)
model, tokenizer, config = load_model_with_config(
    "./models/kimi-linear-48b",
    config_path="configs/baseline.json"
)
print(f"Recursive enabled: {config.recursive_enabled}")  # False

# Generate
input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
output = model.generate(input_ids, max_new_tokens=64)

# Load with recursion enabled
model, tokenizer, config = load_model_with_config(
    "./models/kimi-linear-48b",
    config_path="configs/after_surgery.json"
)
print(f"Recursive enabled: {config.recursive_enabled}")  # True

# Generate with recursion (should match baseline initially)
output = model.generate_chunks(input_ids, max_new_tokens=64)
```

### Example 2: Create Custom Config

```python
from kimi_linear.recursive.config import RecursiveConfig

# Create custom config for testing
custom_config = RecursiveConfig(
    recursive_enabled=True,
    chunk_width=64,  # Smaller chunks
    max_inner_steps=2,  # Fewer refinement steps
    commit_threshold=0.8,  # Higher threshold
    lambda_halt=0.3,  # Lower halt loss weight
)

# Validate
errors = custom_config.validate()
if errors:
    print(f"Errors: {errors}")
else:
    custom_config.save("configs/custom_test.json")
    print("Config saved!")
```

### Example 3: Compare Configurations

```python
from kimi_linear.recursive.config import get_baseline_config, get_after_surgery_config

baseline = get_baseline_config()
after_surgery = get_after_surgery_config()

print("Baseline:", baseline.recursive_enabled)  # False
print("After Surgery:", after_surgery.recursive_enabled)  # True
print("Chunk Width:", after_surgery.chunk_width)  # 128
print("Max Inner Steps:", after_surgery.max_inner_steps)  # 4
```

---

## File Structure

```
recursive-kimi-linear/
├── kimi_linear/
│   └── recursive/
│       ├── config.py          # NEW: RecursiveConfig
│       ├── loader.py           # NEW: load_model_with_config()
│       ├── wrapper.py          # MODIFIED: Config-driven, idempotent
│       ├── refine_cell.py      # (unchanged, zero-init preserved)
│       ├── boundary_head.py    # (unchanged)
│       ├── latent_token.py     # (unchanged)
│       └── __init__.py         # MODIFIED: Exports config system
│
├── configs/                    # NEW: Configuration files
│   ├── baseline.json
│   ├── after_surgery.json
│   ├── recursive_phase_a.json
│   ├── recursive_phase_b.json
│   └── recursive_phase_c.json
│
├── evaluate_model.py           # MODIFIED: Config support
├── train_recursive.py          # TODO: Update to use configs
└── RESEARCH_AGENT_DIRECTIONS.md # NEW: This file
```

---

## Research Questions to Answer

1. **Does zero-init preserve baseline?**
   - Compare baseline vs after-surgery results
   - Should be < 0.1% difference

2. **Does recursion help after training?**
   - Compare finetuned_vanilla vs finetuned_recursive
   - Should show improvement if recursion helps

3. **Is recursion efficient?**
   - Check average refinement steps (target: 1-2)
   - Monitor throughput vs baseline

4. **Do configs enable reproducibility?**
   - Load same config multiple times
   - Should produce identical behavior

---

## Contact & Support

If you encounter issues:

1. **Check this document first** - Troubleshooting section
2. **Verify zero-init** - Most common issue
3. **Test config load** - Validate JSON syntax
4. **Check pass-through** - Verify idempotency

**Key Files to Review**:
- `kimi_linear/recursive/config.py` - Config system
- `kimi_linear/recursive/wrapper.py` - Idempotent wrapper
- `kimi_linear/recursive/loader.py` - Model loading

---

## Summary Checklist

Before starting research, verify:

- [ ] Config system loads/saves correctly
- [ ] Baseline evaluation runs successfully
- [ ] After-surgery evaluation matches baseline (< 0.1% diff)
- [ ] Wrapper pass-through works (disabled = baseline)
- [ ] Model loader works with all config types
- [ ] Config files exist in `configs/` directory

If all checked, you're ready to proceed with recursive training and evaluation!

---

**Good luck with your research! 🚀**

