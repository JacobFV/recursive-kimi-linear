"""Model loading utilities with recursive configuration support."""

from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RecursiveConfig
from .wrapper import ChunkRefineWrapper


def load_model_with_config(
    model_path: str,
    config: Optional[RecursiveConfig] = None,
    config_path: Optional[str] = None,
    recursive_enabled: Optional[bool] = None,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[torch.nn.Module, AutoTokenizer, RecursiveConfig]:
    """
    Load model with recursive configuration.
    
    Args:
        model_path: Path to model (local or HuggingFace Hub ID)
        config: RecursiveConfig object (preferred)
        config_path: Path to RecursiveConfig JSON file
        recursive_enabled: Override enable flag (used if config/config_path not provided)
        torch_dtype: Model dtype
        device_map: Device placement strategy
        trust_remote_code: Whether to trust custom model code
        **kwargs: Additional model loading kwargs
    
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        **kwargs
    )
    
    # Resolve config: precedence is config > config_path > recursive_enabled > default
    if config is not None:
        # Use provided config
        final_config = config
    elif config_path:
        # Load from file
        final_config = RecursiveConfig.from_file(Path(config_path))
    elif recursive_enabled is not None:
        # Create from flag
        final_config = RecursiveConfig(recursive_enabled=recursive_enabled)
    else:
        # Default: disabled (baseline)
        final_config = RecursiveConfig(recursive_enabled=False)
    
    # Validate config
    errors = final_config.validate()
    if errors:
        raise ValueError(f"Invalid RecursiveConfig: {errors}")
    
    # Wrap if enabled
    if final_config.recursive_enabled:
        model = ChunkRefineWrapper(base_model, config=final_config)
    else:
        model = base_model
    
    return model, tokenizer, final_config

