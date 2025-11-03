# Lambda Cloud Setup Instructions

## Step 1: Get API Key

1. Go to https://cloud.lambdalabs.com
2. Sign up or log in
3. Go to https://cloud.lambdalabs.com/api-keys
4. Create a new API key
5. Copy the key (you'll only see it once!)

## Step 2: Run the Launch Script

### Option A: Using Python directly

```bash
python launch_lambda_instance.py YOUR_API_KEY
```

### Option B: Using environment variable

```bash
# Windows PowerShell
$env:LAMBDA_API_KEY="YOUR_API_KEY"
python launch_lambda_instance.py

# Or add to script
python launch_lambda_instance.py
```

### Option C: Using uv (if Python not available)

```bash
# Install requests first
uv pip install requests

# Run script
uv run python launch_lambda_instance.py YOUR_API_KEY
```

## Step 3: Add SSH Key to Lambda Cloud

If you don't have SSH keys set up:

1. Generate SSH key (if needed):
   ```bash
   ssh-keygen -t ed25519 -C "lambda-cloud" -f ~/.ssh/lambda_key
   ```

2. Add public key to Lambda Cloud:
   - Go to https://cloud.lambdalabs.com/ssh-keys
   - Click "Add SSH Key"
   - Paste contents of `~/.ssh/lambda_key.pub` (or `~/.ssh/id_rsa.pub`)

## Step 4: Connect via SSH

After instance is launched, you'll get connection info. Add to your SSH config:

```
Host lambda-gh200
    HostName <IP_FROM_LAMBDA>
    User ubuntu
    IdentityFile ~/.ssh/lambda_key  # or your key path
```

Then SSH:
```bash
ssh lambda-gh200
```

## Step 5: Follow Setup Guide

Once SSH'd into the instance, follow [SETUP_GCP.md](./SETUP_GCP.md] for:
- Cloning the repository
- Setting up Python environment  
- Downloading Hugging Face weights
- Converting weights

**Note**: Lambda instances come with CUDA/GPU drivers pre-installed, so you can install GPU PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Managing Instances

### List instances:
```bash
curl -X GET "https://cloud.lambdalabs.com/api/v1/instances" \
  -H "Authorization: Bearer $LAMBDA_API_KEY"
```

### Terminate instance:
```bash
curl -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"instance_ids": ["INSTANCE_ID"]}'
```

## Costs

GH200 instances are expensive. Monitor usage at https://cloud.lambdalabs.com/billing
- **Always terminate instances when not in use**
- Check pricing: https://cloud.lambdalabs.com/pricing

