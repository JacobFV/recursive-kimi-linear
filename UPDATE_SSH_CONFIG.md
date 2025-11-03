# Updating SSH Config

After creating your GCP instance manually, follow these steps to update your SSH config.

## Step 1: Get Instance IP

```bash
gcloud compute instances describe kimi-linear-cpu --zone=YOUR_ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

Or simply:
```bash
gcloud compute instances list
```

## Step 2: Add to SSH Config

Add this entry to `C:\Users\Jacob\.ssh\config`:

Replace `EXTERNAL_IP` with the IP from Step 1, and `YOUR_USERNAME` with your actual username (usually `jacob` or `ubuntu`).

```
Host kimi-gcp
    HostName EXTERNAL_IP
    User YOUR_USERNAME
    IdentityFile C:\Users\Jacob\.ssh\google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile C:\Users\Jacob\.ssh\known_hosts
```

## Alternative: Use gcloud config-ssh

The easiest way is to use gcloud's built-in SSH config management:

```bash
gcloud compute config-ssh
```

This automatically adds entries for all your GCP instances. The hostname will be something like:
`kimi-linear-gpu.us-central1-a.jacobfv123-main-project`

You can then SSH with:
```bash
ssh kimi-linear-gpu.us-central1-a.jacobfv123-main-project
```

Or add an alias in your SSH config:
```
Host kimi-gcp
    HostName kimi-linear-cpu.us-central1-a.jacobfv123-main-project
    User YOUR_USERNAME
    IdentityFile C:\Users\Jacob\.ssh\google_compute_engine
```

