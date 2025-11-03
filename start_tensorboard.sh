#!/bin/bash
# Start TensorBoard on Lambda instance
# Run this on the Lambda instance (192.222.58.183)

LOG_DIR=${1:-"./logs"}
PORT=${2:-"6006"}

echo "============================================================"
echo "Starting TensorBoard"
echo "============================================================"
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "To access from your local machine:"
echo "  ssh -L ${PORT}:localhost:${PORT} ubuntu@192.222.58.183"
echo ""
echo "Then open in browser:"
echo "  http://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "============================================================"
echo ""

# Check if TensorBoard is already running
if pgrep -f "tensorboard.*--logdir" > /dev/null; then
    echo "⚠ TensorBoard is already running!"
    echo "   Kill it with: pkill -f tensorboard"
    echo ""
    read -p "Kill existing TensorBoard and start new? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f tensorboard
        sleep 2
    else
        exit 0
    fi
fi

# Start TensorBoard in background (or foreground)
tensorboard --logdir="$LOG_DIR" --port="$PORT" --host=0.0.0.0 --reload_interval=5

