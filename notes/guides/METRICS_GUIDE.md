# Metrics & Evaluation Guide

## Quick Start

### 1. Setup Metrics Infrastructure
```bash
./setup_metrics.py
```

This will:
- Install TensorBoard (if needed)
- Create `logs/`, `checkpoints/`, `eval_results/` directories
- Verify TensorBoard is working

### 2. Run Evaluation (Any Stage)
```bash
# Baseline (original model)
./quick_start_evaluation.sh ./models/kimi-linear-48b baseline

# After surgery (with recursive wrapper, before training)
./quick_start_evaluation.sh ./models/kimi-linear-48b after_surgery --use_recursive

# After fine-tuning
./quick_start_evaluation.sh ./checkpoints/my_checkpoint finetuned_recursive --use_recursive
```

Or use the Python script directly:
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline \
    --output_dir ./eval_results
```

### 3. Start TensorBoard
```bash
# In a separate terminal or tmux session
tensorboard --logdir=./logs --port=6006

# Access via browser:
# - Local: http://localhost:6006
# - Remote: ssh -L 6006:localhost:6006 user@host (then open localhost:6006)
```

### 4. Compare Results
```bash
python compare_results.py baseline after_surgery finetuned_vanilla finetuned_recursive
```

## Metrics Overview

### Training Metrics (TensorBoard)

**Losses** (`train/loss/*`):
- `total`: Total training loss
- `final_ce`: Cross-entropy on committed tokens
- `masked_ce`: Deep supervision on intermediate steps
- `halt`: Boundary prediction loss
- `ponder`: Cost for refinement steps
- `stability`: Temporal consistency loss

**Generation** (`train/generation/*`):
- `avg_refine_steps`: Average refinement steps per chunk
- `commit_rate`: Fraction of chunks committed
- `avg_chunk_length`: Average effective chunk length

**Performance** (`train/performance/*`):
- `tokens_per_sec`: Generation throughput
- `memory_mb`: Peak memory usage

### Evaluation Metrics

**Perplexity**:
- Lower is better
- Measures language modeling quality
- Baseline: ~10-20 for 48B models

**Token Accuracy**:
- Percentage of correctly predicted tokens
- Higher is better
- Baseline: ~50-60%

**Reasoning Accuracy**:
- Accuracy on math/reasoning problems
- Higher is better
- Tests generalization capability

**Generation Quality**:
- Coherence, length, relevance
- Qualitative assessment
- Compare outputs across stages

## Evaluation Stages

### Stage 1: Baseline
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage baseline
```
**Purpose**: Establish baseline before any changes

### Stage 2: After Surgery
```bash
python evaluate_model.py \
    --model_path ./models/kimi-linear-48b \
    --stage after_surgery \
    --use_recursive
```
**Purpose**: Verify zero-init preserves baseline (should match Stage 1)

### Stage 3: Fine-tuned Vanilla
```bash
python evaluate_model.py \
    --model_path ./checkpoints/vanilla_finetune \
    --stage finetuned_vanilla
```
**Purpose**: Measure improvement from fine-tuning alone

### Stage 4: Fine-tuned Recursive
```bash
python evaluate_model.py \
    --model_path ./checkpoints/recursive_finetune \
    --stage finetuned_recursive \
    --use_recursive
```
**Purpose**: Measure improvement from recursive reasoning

## What to Monitor

### During Training

**Watch for**:
1. **Loss decreasing**: Should trend downward
2. **Stability**: Metrics shouldn't oscillate wildly
3. **Refinement steps**: Should converge to 1-2 steps
4. **Commit rate**: Should increase as model learns boundaries
5. **Memory**: Should stay within GPU limits

**Red flags**:
- Loss not decreasing → learning rate too low or data issue
- Loss exploding → learning rate too high
- Refinement steps always max → not learning to halt
- Memory OOM → reduce batch size or chunk width

### After Training

**Compare metrics**:
- Perplexity: Should improve over baseline
- Reasoning: Should show gains if recursion helps
- Generation: Should be more coherent/longer if recursion helps

**Check improvements**:
```bash
python compare_results.py baseline after_surgery finetuned_vanilla finetuned_recursive
```

## TensorBoard Tips

### Key Dashboards

1. **SCALARS**: Loss curves, metrics over time
2. **HISTOGRAMS**: Parameter distributions (if logged)
3. **IMAGES**: Sample generations (if logged)

### Useful Views

- **Loss Comparison**: Overlay `train/loss/total` across experiments
- **Generation Metrics**: Track `train/generation/avg_refine_steps`
- **Performance**: Monitor `train/performance/tokens_per_sec`

### Filtering

- Use regex in TensorBoard: `train/loss.*`
- Group by experiment: Organize logs in subdirectories
- Compare runs: Use TensorBoard's compare feature

## Integration with Training

The `train_recursive.py` script automatically logs to TensorBoard:

```bash
python train_recursive.py \
    --model_name ./models/kimi-linear-48b \
    --log_dir ./logs/my_experiment \
    --eval_interval 1000 \
    --save_interval 5000 \
    ...
```

Metrics are logged at each step, evaluations run at intervals.

## Troubleshooting

### TensorBoard not showing data
- Check `logs/` directory has subdirectories
- Verify TensorBoard logdir path is correct
- Restart TensorBoard: `tensorboard --logdir=./logs --reload_interval=5`

### Evaluation fails
- Check model path exists
- Verify model is compatible (trust_remote_code if needed)
- Check GPU memory availability

### Metrics seem wrong
- Verify zero-init: After surgery should match baseline
- Check evaluation data: Should use same dataset
- Compare across same hardware: Different GPUs may vary

## Advanced Usage

### Custom Metrics

Add custom metrics in training loop:
```python
metrics_tracker.log_scalar("custom/my_metric", value, step)
```

### Periodic Evaluations

Add to training script:
```python
if step % eval_interval == 0:
    eval_results = evaluate_model(
        model, tokenizer, eval_data,
        metrics_tracker=metrics_tracker,
        step=step,
    )
```

### Export Results

```python
# Save checkpoint with metrics
metrics_tracker.save_checkpoint_metrics(checkpoint_path)
```

## File Structure

```
logs/
├── phase_a/
│   └── events.out.tfevents.*  # TensorBoard logs
└── phase_b/
    └── events.out.tfevents.*

eval_results/
├── baseline_results.json
├── after_surgery_results.json
├── finetuned_vanilla_results.json
└── finetuned_recursive_results.json

checkpoints/
├── phase_a_checkpoint/
│   ├── model.safetensors
│   └── metrics.json
└── phase_b_checkpoint/
    └── ...
```

