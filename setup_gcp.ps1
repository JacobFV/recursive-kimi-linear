# PowerShell script to set up GCP instance with GPU
# Make sure you're authenticated: gcloud auth login

$INSTANCE_NAME = "kimi-linear-gpu"
$ZONE = "us-central1-a"
$MACHINE_TYPE = "n1-standard-4"
$GPU_TYPE = "nvidia-tesla-t4"
$GPU_COUNT = 1
$IMAGE_FAMILY = "ubuntu-2204-jammy-v20250109"
$IMAGE_PROJECT = "ubuntu-os-cloud"
$DISK_SIZE = "200GB"

Write-Host "Creating GCP instance: $INSTANCE_NAME" -ForegroundColor Green

# Create instance with GPU
gcloud compute instances create $INSTANCE_NAME `
    --zone=$ZONE `
    --machine-type=$MACHINE_TYPE `
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" `
    --maintenance-policy=TERMINATE `
    --image-family=$IMAGE_FAMILY `
    --image-project=$IMAGE_PROJECT `
    --boot-disk-size=$DISK_SIZE `
    --boot-disk-type=pd-standard `
    --metadata="install-nvidia-driver=True" `
    --scopes=https://www.googleapis.com/auth/cloud-platform

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Instance created successfully!" -ForegroundColor Green
    
    # Get external IP
    $EXTERNAL_IP = (gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    
    Write-Host ""
    Write-Host "Instance Details:" -ForegroundColor Cyan
    Write-Host "  Name: $INSTANCE_NAME"
    Write-Host "  Zone: $ZONE"
    Write-Host "  External IP: $EXTERNAL_IP"
    Write-Host ""
    Write-Host "SSH Command:" -ForegroundColor Yellow
    Write-Host "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    Write-Host ""
    Write-Host "Or with regular SSH (after updating ~/.ssh/config):" -ForegroundColor Yellow
    Write-Host "  ssh kimi-gcp"
    Write-Host ""
    
    # Update SSH config
    Write-Host "Updating SSH config..." -ForegroundColor Green
    $SSH_CONFIG = "$env:USERPROFILE\.ssh\config"
    
    $SSH_ENTRY = @"
`nHost kimi-gcp
    HostName $EXTERNAL_IP
    User $env:USERNAME
    IdentityFile $env:USERPROFILE\.ssh\google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile $env:USERPROFILE\.ssh\known_hosts

"@
    
    if (Test-Path $SSH_CONFIG) {
        $currentConfig = Get-Content $SSH_CONFIG -Raw
        if ($currentConfig -notmatch 'Host kimi-gcp') {
            Add-Content $SSH_CONFIG $SSH_ENTRY
            Write-Host "[SUCCESS] SSH config updated" -ForegroundColor Green
        } else {
            Write-Host "SSH config already contains kimi-gcp entry" -ForegroundColor Yellow
        }
    } else {
        New-Item -Path $SSH_CONFIG -ItemType File -Force | Out-Null
        Set-Content $SSH_CONFIG $SSH_ENTRY
        Write-Host "[SUCCESS] SSH config created" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Wait 2-3 minutes for the instance to fully initialize"
    Write-Host "2. SSH into the instance: ssh kimi-gcp"
    Write-Host "3. Follow instructions in SETUP_GCP.md"
    
} else {
    Write-Host "[ERROR] Failed to create instance. Check your gcloud configuration." -ForegroundColor Red
    exit 1
}
