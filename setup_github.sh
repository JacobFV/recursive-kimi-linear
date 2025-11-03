#!/bin/bash
# Script to create GitHub repo and push code
# Usage: ./setup_github.sh <your-github-username>

if [ -z "$1" ]; then
    echo "Usage: ./setup_github.sh <your-github-username>"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME="recursive-kimi-linear"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed."
    echo "Please create the repository manually at https://github.com/new"
    echo "Repository name: $REPO_NAME"
    echo "Make it public"
    echo ""
    echo "Then run:"
    echo "  git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "  git push -u origin main"
    exit 1
fi

# Create repository
echo "Creating GitHub repository..."
gh repo create $REPO_NAME --public --source=. --remote=origin --push

if [ $? -eq 0 ]; then
    echo "✓ Repository created and code pushed successfully!"
    echo "Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
else
    echo "Failed to create repository. Please create it manually."
fi

