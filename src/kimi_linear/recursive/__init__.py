"""Recursive chunked generation module for Kimi-Linear."""

from .refine_cell import RefineCell
from .boundary_head import BoundaryHead
from .latent_token import LatentToken
from .wrapper import ChunkRefineWrapper
from .config import (
    RecursiveConfig,
    get_baseline_config,
    get_after_surgery_config,
    get_phase_a_config,
    get_phase_b_config,
    get_phase_c_config,
)
from .loader import load_model_with_config
from .losses import (
    compute_final_ce_loss,
    compute_masked_ce_loss,
    compute_halt_loss,
    compute_ponder_loss,
    compute_stability_loss,
    compute_total_loss,
)
from .data import ChunkCollator, create_corruption_mask, create_dummy_data, ChunkDataset
from .experiment import ExperimentTracker, ExperimentMetadata

__all__ = [
    "RefineCell",
    "BoundaryHead",
    "LatentToken",
    "ChunkRefineWrapper",
    "RecursiveConfig",
    "get_baseline_config",
    "get_after_surgery_config",
    "get_phase_a_config",
    "get_phase_b_config",
    "get_phase_c_config",
    "load_model_with_config",
    "compute_final_ce_loss",
    "compute_masked_ce_loss",
    "compute_halt_loss",
    "compute_ponder_loss",
    "compute_stability_loss",
    "compute_total_loss",
    "ChunkCollator",
    "create_corruption_mask",
    "create_dummy_data",
    "ChunkDataset",
    "ExperimentTracker",
    "ExperimentMetadata",
]

