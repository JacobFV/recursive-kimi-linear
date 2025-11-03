# Lambda Cloud Setup - GH200 96GB GPU

## Prerequisites

You'll need:
1. Lambda Cloud account (sign up at https://cloud.lambdalabs.com)
2. API key from Lambda Cloud dashboard

## Option 1: Using Lambda Cloud API (Python/curl)

### Step 1: Install Python requests (if needed)

```bash
# Using uv
uv pip install requests

# Or using pip
pip install requests
```

### Step 2: Get API Key

1. Go to https://cloud.lambdalabs.com/api-keys
2. Create a new API key
3. Save it securely

### Step 3: Create Instance Script

Create a script to launch the instance:

```python
#!/usr/bin/env python3
import requests
import json
import sys

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
API_URL = "https://cloud.lambdalabs.com/api/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Check available instance types
print("Checking available GH200 instances...")
response = requests.get(f"{API_URL}/instance-types", headers=headers)
instance_types = response.json()

# Find GH200 96GB
gh200_types = [t for t in instance_types.get("data", []) if "gh200" in t.get("name", "").lower() and "96" in t.get("name", "")]
if not gh200_types:
    print("GH200 96GB not found. Available types:")
    for it in instance_types.get("data", []):
        print(f"  - {it.get('name')}: {it.get('description', 'N/A')}")
    sys.exit(1)

gh200 = gh200_types[0]
print(f"Found: {gh200['name']} - {gh200.get('description', 'N/A')}")

# Launch instance
instance_config = {
    "region_name": "us-east-1",  # or check available regions
    "instance_type_name": gh200["name"],
    "ssh_key_names": [],  # You'll need to add SSH keys first via API
    "quantity": 1
}

print(f"\nLaunching instance: {gh200['name']}...")
response = requests.post(f"{API_URL}/instance-operations/launch", 
                        headers=headers, 
                        json=instance_config)

if response.status_code == 201:
    result = response.json()
    print("✓ Instance launched successfully!")
    print(json.dumps(result, indent=2))
else:
    print(f"✗ Error: {response.status_code}")
    print(response.text)
```

## Option 2: Using curl/HTTP requests

```bash
# Set your API key
export LAMBDA_API_KEY="YOUR_API_KEY_HERE"

# List instance types
curl -X GET "https://cloud.lambdalabs.com/api/v1/instance-types" \
  -H "Authorization: Bearer $LAMBDA_API_KEY"

# Launch GH200 instance (adjust instance_type_name based on available types)
curl -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
  -H "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "region_name": "us-east-1",
    "instance_type_name": "gpu_24x_gh200",
    "quantity": 1
  }'
```

## Option 3: Manual Setup via Web Dashboard

1. Go to https://cloud.lambdalabs.com
2. Click "Launch Instance"
3. Select GH200 96GB instance type
4. Choose region and launch
5. Note the SSH command shown

## Step 4: Configure SSH

**Instance Created**: IP 192.222.58.183

SSH config has been added to `C:\Users\Jacob\.ssh\config`:

```
Host lambda-gh200
    HostName 192.222.58.183
    User ubuntu
    IdentityFile C:\Users\Jacob\.ssh\google_compute_engine
```

**SSH Command**: `ssh lambda-gh200`

**Instance Details**:
- GPU: NVIDIA GH200 480GB (97GB memory available)
- CUDA: 12.8
- Architecture: ARM64 (aarch64)
- OS: Ubuntu 22.04

## Important Notes

- **SSH Keys**: You need to add SSH keys to Lambda Cloud first
  - Go to https://cloud.lambdalabs.com/ssh-keys
  - Add your public key
  - Use the key name in the launch request

- **Instance Names**: Lambda Cloud uses specific naming. Common formats:
  - `gpu_1x_gh200_96gb`
  - `gh200_96gb` 
  - Check available types via API first

- **Cost**: GH200 instances are expensive. Monitor usage and shut down when not needed.

## Next Steps

**Repository already cloned on the instance at**: `/home/ubuntu/recursive-kimi-linear`

Follow [SETUP_LAMBDA.md](./SETUP_LAMBDA.md) for complete setup instructions:

1. ✅ Repository cloned
2. Set up Python environment
3. Download Hugging Face weights
4. Convert weights to custom implementation
5. Verify GPU access and test KDA layer

**Important Notes**:

- **Architecture**: ARM64 (aarch64) - verify PyTorch compatibility
- **GPU Memory**: 97GB available - sufficient for the 48B model
- **Cost**: GH200 instances are expensive - terminate when not in use!

