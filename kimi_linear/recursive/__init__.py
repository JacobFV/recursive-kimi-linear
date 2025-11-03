"""Recursive chunked generation module for Kimi-Linear."""

from .refine_cell import RefineCell
from .boundary_head import BoundaryHead
from .latent_token import LatentToken
from .wrapper import ChunkRefineWrapper
from .losses import (
    compute_final_ce_loss,
    compute_masked_ce_loss,
    compute_halt_loss,
    compute_ponder_loss,
    compute_stability_loss,
    compute_total_loss,
)
from .data import ChunkCollator, create_corruption_mask

__all__ = [
    "RefineCell",
    "BoundaryHead",
    "LatentToken",
    "ChunkRefineWrapper",
    "compute_final_ce_loss",
    "compute_masked_ce_loss",
    "compute_halt_loss",
    "compute_ponder_loss",
    "compute_stability_loss",
    "compute_total_loss",
    "ChunkCollator",
    "create_corruption_mask",
]

