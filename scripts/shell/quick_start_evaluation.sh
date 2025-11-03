#!/bin/bash
# Quick start script for evaluation workflow

set -e

echo "============================================================"
echo "Recursive Kimi-Linear Evaluation Workflow"
echo "============================================================"
echo ""

# Check if model path is provided
MODEL_PATH=${1:-"./models/kimi-linear-48b"}
STAGE=${2:-"baseline"}

echo "Model path: $MODEL_PATH"
echo "Stage: $STAGE"
echo ""

# Step 1: Setup metrics (if needed)
echo "[1/3] Setting up metrics..."
python scripts/setup/setup_metrics.py

# Step 2: Run evaluation
echo ""
echo "[2/3] Running evaluation..."
python scripts/evaluation/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --stage "$STAGE" \
    --output_dir ./results/eval \
    ${3:+--use_recursive}  # Pass additional args

# Step 3: Display results
echo ""
echo "[3/3] Evaluation complete!"
echo ""
echo "Results saved to: ./results/eval/${STAGE}_results.json"
echo ""
echo "To view results:"
echo "  python scripts/evaluation/compare_results.py $STAGE"
echo ""
echo "To start TensorBoard:"
echo "  tensorboard --logdir=./results/logs --port=6006"

