# GCP Instance Setup - Manual Instructions

## Step 1: Create GCP Instance

Run this command to create a CPU-only instance (4 vCPUs, 16GB RAM):

```bash
gcloud compute instances create kimi-linear-cpu \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Machine Type Options:**
- `e2-standard-4`: 4 vCPUs, 16GB RAM (cost-effective)
- `n2-standard-4`: 4 vCPUs, 16GB RAM (better performance)
- `n1-standard-4`: 4 vCPUs, 15GB RAM (legacy, still works)

### Alternative Zones

If the zone is unavailable, try:
- `us-central1-b`
- `us-central1-c`
- `us-west1-a`
- `us-east1-b`

Simply change the `--zone` parameter in the command above.

## Step 2: Get Instance IP

After creation, get the external IP:

```bash
gcloud compute instances describe kimi-linear-cpu --zone=YOUR_ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
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
gcloud compute ssh kimi-linear-cpu --zone=YOUR_ZONE
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
- Try different machine types (n2-standard-4, n1-standard-4)
- Wait and retry later
- Check quota limits

### SSH Connection Issues
- Ensure firewall rules allow SSH (port 22)
- Check that the instance is running: `gcloud compute instances list`
- Try using gcloud's built-in SSH: `gcloud compute ssh INSTANCE_NAME`

