# Evaluation Plan: Surgery & Fine-tuning Tracking

## Overview

This document outlines the comprehensive evaluation plan for testing the recursive surgery on Kimi-Linear at different stages.

## Testing Stages

### 1. Baseline: Regular Kimi-Linear Model
**Purpose**: Establish baseline performance before any modifications.

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --output_dir ./eval_results
```

**Metrics to Track**:
- Perplexity on evaluation dataset
- Token-level accuracy
- Reasoning accuracy (GSM8K-style problems)
- Generation quality (coherence, length)
- Throughput (tokens/second)
- Memory usage

**Expected Output**: `eval_results/baseline_results.json`

---

### 2. Fine-tune Vanilla (Unmodified)
**Purpose**: Fine-tune the original model to establish improvement from fine-tuning alone.

**Training Command**:
```bash
# Fine-tune without recursive wrapper
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --chunk_width 128 \
    --batch_size 4 \
    --num_steps 10000 \
    --phase a \
    --trust_remote_code \
    --log_dir ./logs/vanilla_finetune
```

**Evaluation Command**:
```bash
# Evaluate after vanilla fine-tuning
python evaluate_model.py \
    --model_path ./checkpoints/vanilla_finetune \
    --stage finetuned_vanilla \
    --output_dir ./eval_results
```

**Metrics to Track**:
- All baseline metrics
- Training loss curves (TensorBoard)
- Improvement over baseline
- Generation quality comparison

---

### 3. After Surgery (Before Fine-tuning)
**Purpose**: Test the architecture immediately after adding recursive components (zero-init should preserve baseline).

**Command**:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --use_recursive \
    --output_dir ./eval_results
```

**Metrics to Track**:
- All baseline metrics
- **Critical**: Should match baseline (zero-init ensures no change)
- Generation metrics with recursive wrapper
- Initial commit rates, refine steps
- Any performance overhead

**Expected**: Metrics should be nearly identical to baseline (within numerical precision).

---

### 4. Fine-tune Recursive (Modified Version)
**Purpose**: Fine-tune the modified model with recursive components to see improvement from recursive reasoning.

**Training Command**:
```bash
# Fine-tune with recursive wrapper
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --chunk_width 128 \
    --max_inner_steps 4 \
    --batch_size 4 \
    --num_steps 50000 \
    --phase a \
    --use_recursive \
    --trust_remote_code \
    --log_dir ./logs/recursive_finetune
```

**Evaluation Command**:
```bash
# Evaluate after recursive fine-tuning
python evaluate_model.py \
    --model_path ./checkpoints/recursive_finetune \
    --stage finetuned_recursive \
    --use_recursive \
    --output_dir ./eval_results
```

**Metrics to Track**:
- All previous metrics
- Recursive-specific metrics:
  - Average refinement steps per chunk
  - Commit rate
  - Effective chunk lengths
  - Stability across refinement steps
- Comparison to:
  - Baseline
  - Vanilla fine-tuned
  - After surgery (before training)

---

## Comprehensive Metrics

### Core Metrics (All Stages)

1. **Perplexity**
   - Measures language modeling quality
   - Lower is better
   - Baseline: ~10-20 (typical for 48B models)

2. **Token Accuracy**
   - Percentage of correctly predicted tokens
   - Higher is better
   - Baseline: ~50-60%

3. **Reasoning Accuracy**
   - Accuracy on simple math/reasoning problems
   - Higher is better
   - Baseline: Variable (0-100% depending on training)

4. **Generation Quality**
   - Coherence, length, relevance
   - Qualitative assessment
   - Compare across stages

### Recursive-Specific Metrics

1. **Average Refinement Steps**
   - How many inner refinement steps used per chunk
   - Target: 1-2 steps (efficient)
   - Too high = inefficient
   - Too low = not using recursion

2. **Commit Rate**
   - Fraction of chunks committed per refinement
   - Should increase during training

3. **Chunk Length Distribution**
   - Distribution of effective chunk lengths
   - Should match natural boundaries

4. **Stability@K**
   - Fraction of chunks with consistent outputs across steps
   - Higher = more stable refinement

### Performance Metrics

1. **Throughput (Tokens/Second)**
   - Generation speed
   - Compare: baseline vs recursive
   - Expected: Some overhead from recursion

2. **Memory Usage**
   - Peak memory during generation
   - Compare across stages

3. **Training Speed**
   - Steps per second during training
   - Compare vanilla vs recursive

---

## TensorBoard Setup

### Start TensorBoard
```bash
# In a separate terminal/tmux session
tensorboard --logdir=./logs --port=6006
```

### Access
- Local: http://localhost:6006
- Remote (SSH tunnel): `ssh -L 6006:localhost:6006 user@host`

### Monitoring During Training

**Losses**:
- `train/loss/total`
- `train/loss/final_ce`
- `train/loss/masked_ce`
- `train/loss/halt`
- `train/loss/ponder`

**Generation Metrics**:
- `train/generation/avg_refine_steps`
- `train/generation/commit_rate`
- `train/generation/avg_chunk_length`

**Performance**:
- `train/performance/tokens_per_sec`
- `train/performance/memory_mb`

**Evaluation**:
- `evaluation/{stage}/perplexity`
- `evaluation/{stage}/reasoning_accuracy`

---

## Evaluation Workflow

### Step 1: Baseline (Day 0)
```bash
# 1. Setup metrics
python setup_metrics.py

# 2. Evaluate baseline
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --output_dir ./eval_results

# 3. Start TensorBoard (in background)
tensorboard --logdir=./logs --port=6006 &
```

### Step 2: Fine-tune Vanilla (Day 1-2)
```bash
# 1. Train vanilla
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --chunk_width 128 \
    --batch_size 4 \
    --num_steps 10000 \
    --phase a \
    --output_dir ./checkpoints/vanilla_finetune \
    --log_dir ./logs/vanilla_finetune \
    --trust_remote_code

# 2. Evaluate
python evaluate_model.py \
    --model_path ./checkpoints/vanilla_finetune \
    --stage finetuned_vanilla \
    --output_dir ./eval_results
```

### Step 3: Surgery & Immediate Test (Day 3)
```bash
# 1. Test immediately after surgery (should match baseline)
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --use_recursive \
    --output_dir ./eval_results

# 2. Compare to baseline
python compare_results.py baseline after_surgery
```

### Step 4: Fine-tune Recursive (Day 4-10)
```bash
# 1. Train recursive
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --chunk_width 128 \
    --max_inner_steps 4 \
    --batch_size 4 \
    --num_steps 50000 \
    --phase a \
    --use_recursive \
    --output_dir ./checkpoints/recursive_finetune \
    --log_dir ./logs/recursive_finetune \
    --trust_remote_code

# 2. Periodic evaluations (every 5k steps)
for step in 5000 10000 20000 30000 40000 50000; do
    python evaluate_model.py \
        --model_path ./checkpoints/recursive_finetune/step_${step} \
        --stage finetuned_recursive \
        --use_recursive \
        --output_dir ./eval_results
done
```

---

## Comparison Script

Create `compare_results.py` to compare stages:

```bash
python compare_results.py baseline after_surgery finetuned_vanilla finetuned_recursive
```

This will:
- Load all evaluation JSON files
- Create comparison tables
- Plot metrics over stages
- Highlight improvements/regressions

---

## Key Questions to Answer

1. **Does surgery preserve baseline?**
   - Compare: baseline vs after_surgery
   - Should be nearly identical (zero-init)

2. **Does fine-tuning help?**
   - Compare: baseline vs finetuned_vanilla
   - Should show improvement from data

3. **Does recursion help?**
   - Compare: finetuned_vanilla vs finetuned_recursive
   - Should show additional improvement

4. **Is recursion efficient?**
   - Check: avg_refine_steps, throughput
   - Should converge to 1-2 steps efficiently

---

## Files Generated

```
eval_results/
├── baseline_results.json
├── finetuned_vanilla_results.json
├── after_surgery_results.json
└── finetuned_recursive_results.json

logs/
├── vanilla_finetune/    # TensorBoard logs
└── recursive_finetune/   # TensorBoard logs

checkpoints/
├── vanilla_finetune/
└── recursive_finetune/
```

---

## Notes

- Run evaluations on same hardware for fair comparison
- Use same evaluation dataset across all stages
- Save random seeds for reproducibility
- Document any anomalies or unexpected results
- Keep TensorBoard running throughout training for real-time monitoring

