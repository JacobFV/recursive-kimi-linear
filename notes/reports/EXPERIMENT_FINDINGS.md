# Experiment Findings Log

**Date**: 2025-01-03  
**Status**: Active Research

---

## Issue Fixed: Learning Rate Recording

### Problem
Learning rate was showing as `0.0` in all metrics JSON files.

### Root Cause
1. `scheduler.get_last_lr()` returns `[0.0]` before any `scheduler.step()` calls
2. For baseline experiments (no recursion), we never call `scheduler.step()`, so LR stayed at 0.0
3. For recursive training, we need to get LR **after** `scheduler.step()`, not before

### Solution Applied
1. **For recursive training**: Get LR after `scheduler.step()` call (correct sequence)
2. **For baseline**: Use `initial_lr` parameter to show what LR would be (informational)
3. **Proper float conversion**: All LR values explicitly converted to `float()` for JSON serialization
4. **Enhanced error handling**: Fallback to optimizer `param_groups[0]['lr']` if scheduler fails

### Code Changes
- Modified `train_with_tracking()` to get LR after scheduler.step()
- Added `initial_lr` parameter for baseline mode logging
- Enhanced `save_metrics()` to ensure proper float conversion for all values
- Added fallback to optimizer LR if scheduler fails

### Verification
```python
# Test confirmed:
# - get_last_lr() returns [0.0] before first step
# - After scheduler.step(), returns correct scheduled value
# - Float values serialize correctly to JSON
```

---

## Experiments Conducted

### Experiment 1: Baseline Validation (baseline_001_validation)

**Purpose**: Validate experiment infrastructure with baseline (no recursion)

**Status**: ✅ Completed

**Results**:
- Loss: ~12.5 (stable, as expected for dummy data)
- Learning Rate: Now shows 0.0003 (initial LR) for reference
- Checkpoints: Saved successfully at steps 25, 50
- Metadata: Complete tracking working
- Notes: 15 observations recorded
- Issues: 0

**Key Observations**:
- Experiment infrastructure works correctly
- Model requires eval mode (MoE gate constraint)
- Metrics properly logged to JSON and TensorBoard
- Checkpointing with configs working

### Experiment 2: Phase A Recursive Training (phase_a_001_recursive)

**Purpose**: Train recursive sidecar components

**Status**: ⏳ In Progress / Needs Fix

**Issues Encountered**:
1. **MoE Gate Constraint**: Base model must be in eval mode
   - **Solution**: Set `model.base.eval()` before forward pass
   - Use `torch.enable_grad()` context to allow gradients through

2. **Gradient Flow**: Initial attempts had gradient issues
   - **Solution**: Properly ensure gradients enabled even with base in eval

**Current Status**: Fixing gradient flow issues

---

## Technical Notes

### Kimi Model Constraints

The Kimi-Linear model has a critical constraint:
- **MoE Gate Assertion**: `assert not self.training` in the MoE gate forward
- **Implication**: Base model **must** be in eval mode during forward pass
- **Solution**: Keep base in eval, enable gradients via `torch.enable_grad()`

### Learning Rate Schedule

**Warmup Schedule** (default):
- Warmup steps: 20 (for 200 step run)
- Initial LR: 3e-4
- Final LR: 3e-4 (no decay for short runs)
- LR at step 0: ~0 (during warmup)
- LR at step 20+: ~3e-4

**For Baseline**:
- LR logged as initial LR (3e-4) for reference
- Not actually used (no training)

---

## Next Steps

1. ✅ Fix LR recording (completed)
2. ⏳ Fix gradient flow for recursive training
3. ⏳ Complete Phase A experiment
4. ⏳ Compare baseline vs Phase A results
5. ⏳ Document full findings

---

**Updated**: 2025-01-03 08:45 UTC

