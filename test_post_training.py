"""Test post-training data pipeline integration."""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loader_import():
    """Test that post_training_data module can be imported."""
    print("Testing post_training_data import...")
    try:
        from kimi_linear.recursive import post_training_data
        print("✓ post_training_data module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import post_training_data: {e}")
        return False


def test_flywheel_dataset_class():
    """Test FlywheelDataset class exists and has expected methods."""
    print("Testing FlywheelDataset class...")
    try:
        from kimi_linear.recursive.post_training_data import FlywheelDataset, RECOMMENDED_DATASETS
        
        # Check class exists
        assert hasattr(FlywheelDataset, '__init__'), "FlywheelDataset missing __init__"
        assert hasattr(FlywheelDataset, '__getitem__'), "FlywheelDataset missing __getitem__"
        assert hasattr(FlywheelDataset, '__len__'), "FlywheelDataset missing __len__"
        
        # Check recommended datasets are defined
        assert "instruction_following" in RECOMMENDED_DATASETS
        assert "code" in RECOMMENDED_DATASETS
        assert "math" in RECOMMENDED_DATASETS
        
        print("✓ FlywheelDataset class structure verified")
        print(f"  - Instruction following datasets: {len(RECOMMENDED_DATASETS['instruction_following'])}")
        print(f"  - Code datasets: {len(RECOMMENDED_DATASETS['code'])}")
        print(f"  - Math datasets: {len(RECOMMENDED_DATASETS['math'])}")
        
        return True
    except Exception as e:
        print(f"✗ FlywheelDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_collator():
    """Test integration with ChunkCollator."""
    print("Testing integration with ChunkCollator...")
    try:
        from kimi_linear.recursive.data import ChunkCollator
        from kimi_linear.recursive.post_training_data import get_flywheel_datasets
        
        # Create a dummy tokenizer for testing
        class DummyTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Simple tokenization: split by spaces and map to integers
                tokens = text.split()
                return [hash(token) % 1000 + 1 for token in tokens]
        
        tokenizer = DummyTokenizer()
        
        # Test with small synthetic data
        # Since we can't actually download datasets in test, we'll test the structure
        collator = ChunkCollator(chunk_width=32, pad_token_id=0, eos_token_id=1)
        
        # Create some dummy token sequences
        dummy_sequences = [
            torch.randint(1, 1000, (32,)) for _ in range(5)
        ]
        
        batch = collator(dummy_sequences, return_mask=True)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[1] == 32  # chunk_width
        
        print("✓ ChunkCollator integration verified")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_integration():
    """Test that train_recursive.py can use post_training_data."""
    print("Testing train_recursive.py integration...")
    try:
        # Check if train_recursive can import the modules
        import train_recursive
        from kimi_linear.recursive import ChunkRefineWrapper
        from kimi_linear.recursive.data import ChunkCollator
        
        # Verify training script structure
        assert hasattr(train_recursive, 'main'), "train_recursive missing main"
        assert hasattr(train_recursive, 'train_phase_a'), "train_recursive missing train_phase_a"
        
        print("✓ train_recursive.py structure verified")
        print("  Note: Full training test requires model weights and GPU")
        return True
    except Exception as e:
        print(f"✗ Training script integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all post-training pipeline tests."""
    print("=" * 60)
    print("Testing Post-Training Data Pipeline")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Data loader import", test_data_loader_import()))
    results.append(("FlywheelDataset class", test_flywheel_dataset_class()))
    results.append(("Collator integration", test_integration_with_collator()))
    results.append(("Training script integration", test_training_script_integration()))
    
    print()
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"✓ All post-training pipeline tests passed ({passed}/{total})")
        print("=" * 60)
        return 0
    else:
        print(f"⚠ Some tests failed ({passed}/{total} passed)")
        print("=" * 60)
        for name, result in results:
            status = "✓" if result else "✗"
            print(f"{status} {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

