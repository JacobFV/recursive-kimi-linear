# Experiment Guide: Conducting Science with the Recursive System

This guide explains how to run experiments with proper scientific rigor, logging, and checkpointing.

## Quick Start

```bash
# Run a baseline experiment (Phase A)
python run_experiment.py \
    --experiment-name "baseline_phase_a_001" \
    --description "Baseline Phase A training with default config" \
    --model-path ./models/kimi-linear-48b \
    --recursive-config configs/recursive_phase_a.json \
    --phase a \
    --num-steps 10000 \
    --batch-size 8 \
    --lr 3e-4 \
    --save-interval 2000 \
    --log-interval 100
```

## Experiment Infrastructure

### Directory Structure

Each experiment creates a structured directory:

```
experiments/
└── experiment_name/
    ├── experiment_metadata.json    # Full experiment metadata
    ├── recursive_config.json        # Config used for this experiment
    ├── checkpoints/                 # All checkpoints
    │   └── step_XXXXXX/
    │       ├── model/               # Model weights
    │       ├── optimizer.pt         # Optimizer state
    │       ├── scheduler.pt         # Scheduler state
    │       ├── recursive_config.json # Config at this checkpoint
    │       └── checkpoint_metadata.json
    ├── logs/                        # TensorBoard logs
    ├── metrics/                     # JSON metrics files
    │   └── step_XXXXXX.json
    └── notes/                       # Human notes and observations
        ├── notes.txt                # General notes
        ├── issues.txt               # Problems encountered
        └── generation_samples.jsonl  # Sample generations
```

## Key Features

### 1. Automatic Metadata Tracking

Every experiment automatically tracks:
- Experiment ID and name
- Creation timestamp
- Python/Torch/CUDA versions
- Training parameters (steps, batch size, LR, etc.)
- Model and config paths
- Final results

### 2. Checkpointing

Checkpoints are saved with:
- Full model state
- Optimizer state
- Scheduler state
- Config used at that step
- Metrics at that step
- Best checkpoint marker

### 3. Observational Logging

Use the tracker to record observations:

```python
from kimi_linear.recursive.experiment import ExperimentTracker

tracker = ExperimentTracker(...)

# Record observations
tracker.add_note("Loss plateaued around step 5000, trying LR decay")
tracker.add_note("Generation quality improved after step 8000")

# Record issues
tracker.add_issue("Out of memory at step 12000, reduced batch size")
tracker.add_issue("Generation dtype error fixed at step 5000")
```

### 4. Metrics Tracking

All metrics are automatically:
- Logged to TensorBoard
- Saved as JSON files
- Included in checkpoint metadata

## Running Experiments

### Phase A: Sidecar Only

```bash
python run_experiment.py \
    --experiment-name "phase_a_sidecar_only" \
    --description "Training sidecar components only (refine cells + boundary)" \
    --model-path ./models/kimi-linear-48b \
    --recursive-config configs/recursive_phase_a.json \
    --phase a \
    --num-steps 50000 \
    --batch-size 8 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --save-interval 5000 \
    --eval-interval 1000 \
    --log-interval 100
```

### Phase B: Light Unfreeze

```bash
# TODO: Implement Phase B (LoRA on top layers)
```

### Phase C: End-to-End Polish

```bash
# TODO: Implement Phase C (subset layer unfreeze)
```

## HuggingFace Integration

### Upload Experiment to HF

```bash
# After experiment completes
python scripts/upload_to_hf.py \
    --experiment-dir ./experiments/experiment_name \
    --hf-repo username/model-name \
    --hf-token YOUR_TOKEN
```

Or set `HF_TOKEN` environment variable:
```bash
export HF_TOKEN=your_token_here
python scripts/upload_to_hf.py \
    --experiment-dir ./experiments/experiment_name \
    --hf-repo username/model-name
```

### What Gets Uploaded

- Best checkpoint (from `best_checkpoint_step`)
- Config file
- Experiment metadata
- Notes and observations
- Model card (README.md)

## Best Practices for Science

### 1. Be a Good Observer

**Record everything:**
- When you notice something interesting
- When something unexpected happens
- When you make changes
- When you encounter problems

**Example:**
```python
tracker.add_note("Loss decreased rapidly in first 5000 steps, then plateaued")
tracker.add_note("Generation samples show improved coherence after step 15000")
tracker.add_issue("GPU memory spike at step 8000, investigating")
```

### 2. Document Your Config

Each experiment should have a clear description:
```bash
--description "Phase A training with chunk_width=128, corruption_rate=0.3, testing if higher corruption helps"
```

### 3. Save Frequent Checkpoints

Don't lose work! Use reasonable intervals:
- `--save-interval 5000` for long runs
- `--save-interval 1000` for short experiments
- Always save final checkpoint

### 4. Track Metrics Carefully

Monitor:
- Training loss
- Evaluation perplexity
- Generation quality
- Computational efficiency (steps per second)

### 5. Compare Baselines

Always run baseline first:
```bash
# Baseline (no recursion)
python run_experiment.py \
    --experiment-name "baseline_001" \
    --recursive-config configs/baseline.json \
    ...

# Then compare with recursive
python run_experiment.py \
    --experiment-name "recursive_phase_a_001" \
    --recursive-config configs/recursive_phase_a.json \
    ...
```

## Loading Checkpoints

```python
from pathlib import Path
from kimi_linear.recursive.config import RecursiveConfig
from kimi_linear.recursive import load_model_with_config

# Load checkpoint
checkpoint_dir = Path("./experiments/experiment_name/checkpoints/step_50000")

# Load config
config = RecursiveConfig.from_file(checkpoint_dir / "recursive_config.json")

# Load model
model, tokenizer, config = load_model_with_config(
    str(checkpoint_dir / "model"),
    config=config,
)
```

## Experiment Notes Template

For each experiment, document:

1. **Hypothesis**: What are you testing?
2. **Setup**: Config, data, parameters
3. **Observations**: What happened?
4. **Issues**: Problems encountered
5. **Results**: Final metrics
6. **Conclusions**: What did you learn?

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce chunk width
- Use gradient checkpointing

### Slow Training
- Check data loading
- Monitor GPU utilization
- Consider mixed precision

### Generation Errors
- Check dtype matching
- Verify tokenizer is correct
- Check model.device vs input.device

## Next Steps

1. Run baseline experiment
2. Run Phase A with recursive config
3. Compare results
4. Iterate and improve
5. Upload to HuggingFace for sharing

---

**Remember**: Good science requires good observations. Document everything!

