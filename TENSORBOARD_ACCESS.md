# TensorBoard Access Guide

## Your Lambda Instance
- **IP**: 192.222.58.183
- **User**: ubuntu (default)

## Quick Start

### Option 1: SSH Port Forwarding (Recommended)

**On your local machine** (Windows/Mac/Linux):

1. **Start TensorBoard on Lambda** (SSH into the instance):
```bash
ssh ubuntu@192.222.58.183
cd recursive-kimi-linear
./start_tensorboard.sh
```

Or manually:
```bash
tensorboard --logdir=./logs --port=6006 --host=0.0.0.0
```

2. **Keep TensorBoard running** (in tmux/screen or let it run):
```bash
# On Lambda instance
tmux new -s tensorboard
./start_tensorboard.sh
# Press Ctrl+B then D to detach
```

3. **Set up SSH port forwarding** (on your local machine):
```bash
ssh -L 6006:localhost:6006 ubuntu@192.222.58.183 -N
```

Or keep it in background:
```bash
# Windows PowerShell
Start-Process ssh -ArgumentList "-L 6006:localhost:6006 ubuntu@192.222.58.183 -N"

# Linux/Mac
ssh -L 6006:localhost:6006 ubuntu@192.222.58.183 -N &
```

4. **Open in browser** (on your local machine):
```
http://localhost:6006
```

### Option 2: Expose TensorBoard Directly (Less Secure)

**On Lambda instance**, start with:
```bash
tensorboard --logdir=./logs --port=6006 --host=0.0.0.0
```

**Then access directly** (if firewall allows):
```
http://192.222.58.183:6006
```

⚠️ **Warning**: This exposes TensorBoard to the internet. Only use if Lambda firewall is configured properly.

## Using tmux (Recommended for SSH Sessions)

Since you're using SSH, use tmux to keep TensorBoard running:

```bash
# On Lambda instance
tmux new -s tensorboard

# Inside tmux, run:
./start_tensorboard.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t tensorboard
# Kill: tmux kill-session -t tensorboard
```

## Verify TensorBoard is Running

On Lambda instance:
```bash
# Check if TensorBoard is running
pgrep -f tensorboard

# Check listening ports
netstat -tlnp | grep 6006
# or
ss -tlnp | grep 6006
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 6006
lsof -i :6006
# or
fuser 6006/tcp

# Kill it
kill -9 <PID>
```

### Can't Access via Localhost
- Make sure SSH port forwarding is active
- Check that TensorBoard is running: `pgrep -f tensorboard`
- Verify TensorBoard is listening: `ss -tlnp | grep 6006`
- Try different port: `--port=6007`

### Firewall Issues
Lambda instances usually have security groups. TensorBoard with `--host=0.0.0.0` should work if:
- Security group allows port 6006 inbound
- Or use SSH port forwarding (recommended)

## Auto-Start Script

Create `~/.bashrc` alias on Lambda:
```bash
alias tb='cd ~/recursive-kimi-linear && ./start_tensorboard.sh'
```

Then just run: `tb`

## Check Current TensorBoard Status

```bash
# See what TensorBoard is logging
ps aux | grep tensorboard

# See recent TensorBoard logs
tail -f ~/.tensorboard.log  # if you redirected output
```

## Multiple Experiments

If you have multiple experiments in different log directories:

```bash
# Start TensorBoard for specific experiment
./start_tensorboard.sh ./logs/phase_a 6006

# Or start multiple on different ports
tensorboard --logdir=./logs/phase_a --port=6006 --host=0.0.0.0 &
tensorboard --logdir=./logs/phase_b --port=6007 --host=0.0.0.0 &
```

Then forward multiple ports:
```bash
ssh -L 6006:localhost:6006 -L 6007:localhost:6007 ubuntu@192.222.58.183 -N
```

