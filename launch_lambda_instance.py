#!/usr/bin/env python3
"""
Launch Lambda Cloud instance with GH200 96GB GPU
Usage: python launch_lambda_instance.py <API_KEY>
"""
import requests
import json
import sys
import time
import os

API_URL = "https://cloud.lambdalabs.com/api/v1"

def get_api_key():
    """Get API key from argument or environment variable"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        print("Error: API key required")
        print("Usage: python launch_lambda_instance.py <API_KEY>")
        print("Or set LAMBDA_API_KEY environment variable")
        sys.exit(1)
    return api_key

def get_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def list_instance_types(headers):
    """List available instance types"""
    print("Fetching available instance types...")
    response = requests.get(f"{API_URL}/instance-types", headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    
    return response.json()

def find_gh200_instance(instance_types):
    """Find GH200 96GB instance type"""
    data = instance_types.get("data", [])
    
    # Look for GH200 instances
    gh200_candidates = []
    for it in data:
        name = it.get("name", "").lower()
        desc = it.get("description", "").lower()
        
        if "gh200" in name or "gh200" in desc:
            if "96" in name or "96" in desc:
                gh200_candidates.append(it)
    
    if gh200_candidates:
        print(f"\nFound {len(gh200_candidates)} GH200 96GB instance type(s):")
        for inst in gh200_candidates:
            print(f"  - {inst['name']}: {inst.get('description', 'N/A')}")
        return gh200_candidates[0]
    
    # Fallback: list all GPU instances
    print("\nGH200 96GB not found. Available GPU instances:")
    for it in data:
        if "gpu" in it.get("name", "").lower():
            print(f"  - {it['name']}: {it.get('description', 'N/A')}")
    
    return None

def list_ssh_keys(headers):
    """List SSH keys"""
    response = requests.get(f"{API_URL}/ssh-keys", headers=headers)
    if response.status_code == 200:
        keys = response.json().get("data", [])
        if keys:
            print("\nAvailable SSH keys:")
            for key in keys:
                print(f"  - {key['name']} ({key.get('public_key', '')[:50]}...)")
            return [key["name"] for key in keys]
        else:
            print("\nNo SSH keys found. You need to add one at https://cloud.lambdalabs.com/ssh-keys")
    return []

def launch_instance(headers, instance_type_name, ssh_key_names, region="us-east-1"):
    """Launch instance"""
    config = {
        "region_name": region,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
        "quantity": 1
    }
    
    print(f"\nLaunching instance: {instance_type_name}...")
    print(f"Region: {region}")
    print(f"SSH Keys: {ssh_key_names if ssh_key_names else 'None (will need to add)'}")
    
    response = requests.post(
        f"{API_URL}/instance-operations/launch",
        headers=headers,
        json=config
    )
    
    if response.status_code == 201:
        result = response.json()
        print("\n✓ Instance launched successfully!")
        return result
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)
        return None

def get_instance_info(headers, instance_id):
    """Get instance details"""
    response = requests.get(f"{API_URL}/instances/{instance_id}", headers=headers)
    if response.status_code == 200:
        return response.json()["data"]
    return None

def main():
    api_key = get_api_key()
    headers = get_headers(api_key)
    
    # List instance types
    instance_types = list_instance_types(headers)
    
    # Find GH200
    gh200 = find_gh200_instance(instance_types)
    if not gh200:
        print("\nPlease check available instance types and update the script")
        sys.exit(1)
    
    instance_type_name = gh200["name"]
    
    # List SSH keys
    ssh_keys = list_ssh_keys(headers)
    if not ssh_keys:
        print("\n⚠️  Warning: No SSH keys found. Add one at https://cloud.lambdalabs.com/ssh-keys")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Launch instance
    result = launch_instance(headers, instance_type_name, ssh_keys)
    
    if result:
        instance_id = result["data"]["instance_ids"][0]
        print(f"\nInstance ID: {instance_id}")
        
        # Wait a moment for instance to start
        print("\nWaiting for instance to initialize...")
        time.sleep(5)
        
        # Get instance details
        instance_info = get_instance_info(headers, instance_id)
        if instance_info:
            ip = instance_info.get("ip")
            hostname = instance_info.get("hostname")
            
            print(f"\nInstance Details:")
            print(f"  IP: {ip}")
            print(f"  Hostname: {hostname}")
            print(f"  Status: {instance_info.get('status')}")
            
            print(f"\nSSH Command:")
            if ip:
                print(f"  ssh ubuntu@{ip}")
            if hostname:
                print(f"  ssh ubuntu@{hostname}")
            
            # Update SSH config instruction
            print(f"\nAdd to ~/.ssh/config:")
            print(f"""
Host lambda-gh200
    HostName {ip or hostname}
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
""")

if __name__ == "__main__":
    main()

