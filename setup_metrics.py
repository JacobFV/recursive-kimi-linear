#!/usr/bin/env python3
"""Setup script for metrics tracking and TensorBoard."""

import subprocess
import sys
from pathlib import Path

def check_and_install(package, import_name=None):
    """Check if package is installed, install if not."""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        print(f"✓ {package} already installed")
        return True
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--quiet"],
                check=True,
                capture_output=True,
            )
            print(f"✓ Installed {package}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

def setup_directories():
    """Create necessary directories."""
    dirs = ["logs", "checkpoints", "eval_results"]
    
    base = Path(__file__).parent
    for dir_name in dirs:
        dir_path = base / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified {dir_path}")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Setting up Metrics Tracking & TensorBoard")
    print("=" * 60)
    print()
    
    # Install dependencies
    print("Checking dependencies...")
    check_and_install("tensorboard", "torch.utils.tensorboard")
    check_and_install("numpy")
    
    # Create directories
    print("\nSetting up directories...")
    setup_directories()
    
    # Verify TensorBoard
    print("\nVerifying TensorBoard...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        test_dir = Path("./logs/test")
        test_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(test_dir))
        writer.add_scalar("test/scalar", 1.0, 0)
        writer.close()
        print("✓ TensorBoard working correctly")
    except Exception as e:
        print(f"⚠ TensorBoard verification failed: {e}")
        print("  You can still use file-based logging")
    
    print()
    print("=" * 60)
    print("✓ Metrics setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start TensorBoard: tensorboard --logdir=./logs")
    print("2. Run evaluations: python evaluate_model.py")
    print("3. Check logs in: ./logs/")

if __name__ == "__main__":
    main()

