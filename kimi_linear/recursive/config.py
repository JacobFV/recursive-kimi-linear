"""Configuration system for recursive chunked generation."""

from dataclasses import dataclass, asdict
from typing import List, Union, Optional
from pathlib import Path
import json


@dataclass
class RecursiveConfig:
    """
    Configuration for recursive chunked generation.
    
    When recursive_enabled=False, all components are inactive
    and behavior is identical to baseline model.
    """
    
    # ========== Core Enable/Disable ==========
    recursive_enabled: bool = False
    """Master switch: if False, recursion is completely disabled (idempotent)"""
    
    # ========== Chunking Parameters ==========
    chunk_width: int = 128
    """Fixed chunk width W (64 for chat, 128 for code/math)"""
    
    max_inner_steps: int = 4
    """Maximum inner refinement steps K per chunk"""
    
    commit_threshold: float = 0.7
    """Halt threshold tau: commit when p_commit >= this"""
    
    min_commit_threshold: float = 0.3
    """Minimum threshold to avoid deadlocks (force commit if below)"""
    
    # ========== Architecture Parameters ==========
    layers_to_refine: Union[str, List[int]] = "all"
    """Which layers to apply refinement: "all" or list of indices"""
    
    use_latent_token: bool = True
    """Whether to use learned [Z] token for global control"""
    
    d_state: int = 512
    """Dimension of persistent refine state per layer"""
    
    gate_hidden: int = 1024
    """Hidden dimension for boundary head MLP"""
    
    max_chunk_len: int = 128
    """Maximum chunk length for length prediction head"""
    
    # ========== Training Parameters ==========
    corruption_rate: float = 0.3
    """Initial corruption rate for masked positions"""
    
    corruption_schedule: str = "linear"
    """Corruption schedule: "linear", "cosine", or "none" """
    
    corruption_rate_final: float = 0.1
    """Final corruption rate (for schedules)"""
    
    # ========== Loss Weights ==========
    lambda_final: float = 1.0
    """Weight for final CE loss on committed tokens"""
    
    lambda_masked: float = 0.5
    """Weight for masked CE loss (deep supervision)"""
    
    lambda_halt: float = 0.5
    """Weight for halt/commit prediction loss"""
    
    lambda_len: float = 0.05
    """Weight for length prediction loss"""
    
    lambda_ponder: float = 0.01
    """Weight for ponder cost (encourage fewer steps)"""
    
    lambda_stability: float = 0.01
    """Weight for temporal consistency loss"""
    
    # ========== Advanced Parameters ==========
    cache_reset_policy: str = "keep_kda_recurrent_reset_mla"
    """Policy for cache management across inner steps:
       - keep_kda_recurrent_reset_mla: Keep KDA states, reset MLA KV
       - keep_all: Keep all cache
       - reset_all: Reset all cache each step
    """
    
    enable_length_head: bool = True
    """Whether to predict effective chunk length"""
    
    use_soft_tokens: bool = False
    """Use Gumbel-softmax for differentiable token updates (training only)"""
    
    # ========== Validation ==========
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if self.chunk_width <= 0:
            errors.append("chunk_width must be positive")
        
        if self.max_inner_steps <= 0:
            errors.append("max_inner_steps must be positive")
        
        if not 0.0 <= self.commit_threshold <= 1.0:
            errors.append("commit_threshold must be in [0, 1]")
        
        if not 0.0 <= self.min_commit_threshold <= 1.0:
            errors.append("min_commit_threshold must be in [0, 1]")
        
        if self.commit_threshold < self.min_commit_threshold:
            errors.append("commit_threshold must be >= min_commit_threshold")
        
        if self.corruption_schedule not in ["linear", "cosine", "none"]:
            errors.append("corruption_schedule must be 'linear', 'cosine', or 'none'")
        
        if not 0.0 <= self.corruption_rate <= 1.0:
            errors.append("corruption_rate must be in [0, 1]")
        
        if not 0.0 <= self.corruption_rate_final <= 1.0:
            errors.append("corruption_rate_final must be in [0, 1]")
        
        if self.layers_to_refine != "all" and not isinstance(self.layers_to_refine, list):
            errors.append("layers_to_refine must be 'all' or a list of integers")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert layers_to_refine list to string representation if needed
        if isinstance(data["layers_to_refine"], list):
            # Keep as list in dict (JSON can handle it)
            pass
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RecursiveConfig':
        """Load from dictionary."""
        # Handle layers_to_refine: convert string "all" or keep list
        if "layers_to_refine" in data and isinstance(data["layers_to_refine"], str):
            if data["layers_to_refine"] != "all":
                raise ValueError(f"Invalid layers_to_refine string: {data['layers_to_refine']}")
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'RecursiveConfig':
        """Load from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Union[str, Path]):
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        errors = self.validate()
        if errors:
            raise ValueError(f"Cannot save invalid config: {errors}")
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __eq__(self, other) -> bool:
        """Check equality (for testing)."""
        if not isinstance(other, RecursiveConfig):
            return False
        return self.to_dict() == other.to_dict()


# Default configurations for common use cases

def get_baseline_config() -> RecursiveConfig:
    """Baseline config: recursion disabled."""
    return RecursiveConfig(recursive_enabled=False)


def get_after_surgery_config() -> RecursiveConfig:
    """After surgery config: enabled but untrained (should match baseline due to zero-init)."""
    return RecursiveConfig(
        recursive_enabled=True,
        chunk_width=128,
        max_inner_steps=4,
        commit_threshold=0.7,
        min_commit_threshold=0.3,
    )


def get_phase_a_config() -> RecursiveConfig:
    """Training Phase A: sidecar only."""
    return RecursiveConfig(
        recursive_enabled=True,
        chunk_width=128,
        max_inner_steps=4,
        commit_threshold=0.7,
        min_commit_threshold=0.3,
        lambda_final=1.0,
        lambda_masked=0.5,
        lambda_halt=0.5,
        lambda_len=0.05,
        lambda_ponder=0.01,
        lambda_stability=0.01,
        corruption_rate=0.3,
        corruption_schedule="linear",
        corruption_rate_final=0.1,
    )


def get_phase_b_config() -> RecursiveConfig:
    """Training Phase B: light unfreeze."""
    config = get_phase_a_config()
    # Phase B uses same config but training will enable LoRA
    return config


def get_phase_c_config() -> RecursiveConfig:
    """Training Phase C: end-to-end polish."""
    config = get_phase_a_config()
    # Phase C uses same config but training will unfreeze subset of layers
    return config

