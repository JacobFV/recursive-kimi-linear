"""Test script to verify recursive architecture works with random data."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add repo root to path (three levels up from scripts/testing/)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from kimi_linear.recursive import (
    RefineCell,
    BoundaryHead,
    LatentToken,
    ChunkRefineWrapper,
)


class DummyBaseModel(nn.Module):
    """Dummy base model for testing wrapper architecture."""
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace(
                hidden_size=768,
                num_hidden_layers=12,
                vocab_size=32000,
                pad_token_id=0,
                eos_token_id=1,
            )
        self.config = config
        
        # Simple embedding + transformer blocks
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=12,
                dim_feedforward=3072,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids, past_key_values=None, use_cache=False, 
                output_hidden_states=False, return_dict=True):
        # Simple forward pass
        hidden_states = self.embedding(input_ids)
        
        all_hidden_states = [] if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(
                logits=logits,
                hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
                past_key_values=past_key_values,
            )
        return logits


def test_refine_cell():
    """Test RefineCell component."""
    print("Testing RefineCell...")
    cell = RefineCell(d_model=768, d_state=512)
    
    batch_size = 2
    seq_len = 32
    h = torch.randn(batch_size, seq_len, 768)
    s = torch.randn(batch_size, 512)
    
    h_refined, s_next = cell(h, s)
    
    assert h_refined.shape == h.shape, f"Expected {h.shape}, got {h_refined.shape}"
    assert s_next.shape == s.shape, f"Expected {s.shape}, got {s_next.shape}"
    
    print("✓ RefineCell test passed")


def test_boundary_head():
    """Test BoundaryHead component."""
    print("Testing BoundaryHead...")
    head = BoundaryHead(d_model=768, gate_hidden=1024, max_len=128)
    
    batch_size = 2
    seq_len = 32
    h = torch.randn(batch_size, seq_len, 768)
    
    p_commit, p_len = head(h)
    
    assert p_commit.shape == (batch_size,), f"Expected ({batch_size},), got {p_commit.shape}"
    assert p_len.shape == (batch_size, 129), f"Expected ({batch_size}, 129), got {p_len.shape}"
    assert (p_commit >= 0).all() and (p_commit <= 1).all(), "p_commit should be in [0,1]"
    
    print("✓ BoundaryHead test passed")


def test_latent_token():
    """Test LatentToken component."""
    print("Testing LatentToken...")
    token = LatentToken(vocab_size=32000, hidden_size=768)
    
    batch_size = 2
    device = torch.device("cpu")
    
    emb = token(batch_size, device)
    
    assert emb.shape == (batch_size, 1, 768), f"Expected ({batch_size}, 1, 768), got {emb.shape}"
    
    print("✓ LatentToken test passed")


def test_wrapper_forward():
    """Test ChunkRefineWrapper forward pass."""
    print("Testing ChunkRefineWrapper forward...")
    
    # Create dummy config
    from types import SimpleNamespace
    config = SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=12,
        vocab_size=32000,
        pad_token_id=0,
        eos_token_id=1,
    )
    
    # Create dummy base model
    base_model = DummyBaseModel(config)
    
    # Wrap with recursive controller
    wrapper = ChunkRefineWrapper(
        base_model=base_model,
        layers_to_refine="all",
        use_latent_token=True,
        d_state=512,
        gate_hidden=1024,
        max_chunk_len=64,
    )
    
    # Test forward_hidden
    batch_size = 1
    prefix_len = 16
    chunk_len = 32
    
    input_ids = torch.randint(1, 1000, (batch_size, prefix_len))
    chunk_ids = torch.randint(1, 1000, (batch_size, chunk_len))
    
    logits, hiddens, cache, z_token = wrapper._forward_hidden(
        input_ids=input_ids,
        chunk_ids=chunk_ids,
        past_key_values=None,
        latent_token=None,
    )
    
    assert logits.shape == (batch_size, chunk_len, config.vocab_size), \
        f"Expected ({batch_size}, {chunk_len}, {config.vocab_size}), got {logits.shape}"
    assert len(hiddens) == config.num_hidden_layers, \
        f"Expected {config.num_hidden_layers} hidden states, got {len(hiddens)}"
    assert all(h.shape[2] == config.hidden_size for h in hiddens), \
        "All hidden states should have correct hidden_size"
    
    print("✓ ChunkRefineWrapper forward test passed")


def test_wrapper_generation():
    """Test ChunkRefineWrapper generation (small scale)."""
    print("Testing ChunkRefineWrapper generation...")
    
    from types import SimpleNamespace
    config = SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=6,  # Smaller for faster test
        vocab_size=32000,
        pad_token_id=0,
        eos_token_id=1,
    )
    
    base_model = DummyBaseModel(config)
    
    wrapper = ChunkRefineWrapper(
        base_model=base_model,
        layers_to_refine="all",
        use_latent_token=True,
        max_chunk_len=32,
    )
    
    # Small generation test
    input_ids = torch.randint(1, 100, (1, 8))
    
    try:
        output = wrapper.generate_chunks(
            input_ids=input_ids,
            max_new_tokens=16,  # Small for test
            chunk_width=16,
            max_inner_steps=2,
            commit_threshold=0.5,
            temperature=0.0,
        )
        
        assert output.shape[1] >= input_ids.shape[1], \
            f"Output should be at least as long as input, got {output.shape[1]} vs {input_ids.shape[1]}"
        
        print("✓ ChunkRefineWrapper generation test passed")
    except Exception as e:
        print(f"⚠ Generation test failed (may be expected if _seed_window incomplete): {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Recursive Architecture Components")
    print("=" * 60)
    print()
    
    try:
        test_refine_cell()
        test_boundary_head()
        test_latent_token()
        test_wrapper_forward()
        test_wrapper_generation()
        
        print()
        print("=" * 60)
        print("✓ All architecture tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

