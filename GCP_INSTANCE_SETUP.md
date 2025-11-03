# GCP Instance Setup - Manual Instructions

Since GPU availability varies by zone, follow these instructions to create your GCP instance manually.

## Step 1: Create GCP Instance

Run this command (adjust zone and GPU type as needed based on availability):

```bash
gcloud compute instances create kimi-linear-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --metadata="install-nvidia-driver=True" \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Alternative GPU Options

If T4 is unavailable, try:
- **L4**: Requires `g2-standard-4` machine type
  ```bash
  --machine-type=g2-standard-4 --accelerator="type=nvidia-l4,count=1"
  ```
- **P4**: Same machine type as T4
  ```bash
  --accelerator="type=nvidia-tesla-p4,count=1"
  ```
- **A100** (more expensive): Requires `a2-highgpu-1g` machine type
  ```bash
  --machine-type=a2-highgpu-1g --accelerator="type=nvidia-tesla-a100,count=1"
  ```

### Check GPU Availability

```bash
# List available GPU types in a zone
gcloud compute accelerator-types list --filter="zone:us-central1-a"

# Check quota
gcloud compute project-info describe --project=YOUR_PROJECT_ID
```

## Step 2: Get Instance IP

After creation, get the external IP:

```bash
gcloud compute instances describe kimi-linear-gpu --zone=YOUR_ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

Or list all instances:
```bash
gcloud compute instances list
```

## Step 3: Update SSH Config

Add this entry to `C:\Users\Jacob\.ssh\config` (replace EXTERNAL_IP with your instance's IP):

```
Host kimi-gcp
    HostName EXTERNAL_IP
    User YOUR_USERNAME
    IdentityFile C:\Users\Jacob\.ssh\google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile C:\Users\Jacob\.ssh\known_hosts
```

Or use gcloud's built-in SSH config:
```bash
gcloud compute config-ssh
```

This will automatically add entries for all your GCP instances.

## Step 4: SSH into Instance

Wait 2-3 minutes for the instance to initialize, then:

```bash
ssh kimi-gcp
```

Or use gcloud directly:
```bash
gcloud compute ssh kimi-linear-gpu --zone=YOUR_ZONE
```

## Step 5: Follow Setup Instructions

Once connected, follow the instructions in [SETUP_GCP.md](./SETUP_GCP.md) to:
1. Clone the repository
2. Set up Python environment
3. Download Hugging Face weights
4. Convert weights to custom implementation

## Troubleshooting

### Zone Resource Exhaustion
If you get "ZONE_RESOURCE_POOL_EXHAUSTED":
- Try different zones (us-central1-b, us-west1-a, etc.)
- Try different GPU types (L4, P4)
- Wait and retry later
- Check quota limits

### No GPU Available
- Verify GPU quota: `gcloud compute project-info describe`
- Request quota increase if needed
- Try preemptible instances (cheaper but can be terminated)

### SSH Connection Issues
- Ensure firewall rules allow SSH (port 22)
- Check that the instance is running: `gcloud compute instances list`
- Try using gcloud's built-in SSH: `gcloud compute ssh INSTANCE_NAME`

