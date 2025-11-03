"""Loss functions for recursive chunked generation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_final_ce_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Standard cross-entropy loss on final committed tokens.
    
    Args:
        logits: [B, T, V] model logits
        target_ids: [B, T] target token IDs
        ignore_index: Token index to ignore in loss
        
    Returns:
        Scalar loss
    """
    # Flatten for CE
    logits_flat = logits.view(-1, logits.size(-1))  # [B*T, V]
    targets_flat = target_ids.view(-1)  # [B*T]
    
    return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


def compute_masked_ce_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -100,
    step_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Masked cross-entropy loss for intermediate refinement steps (deep supervision).
    
    Only computes loss on positions marked as editable (masked).
    
    Args:
        logits: [B, T, V] model logits at intermediate step
        target_ids: [B, T] target token IDs
        mask: [B, T] boolean mask (True = editable/masked position)
        ignore_index: Token index to ignore
        step_weight: Optional decay weight for this refinement step
        
    Returns:
        Scalar loss
    """
    # Apply mask
    mask_expanded = mask.unsqueeze(-1).expand_as(logits)  # [B, T, V]
    logits_masked = logits.masked_fill(~mask_expanded[..., 0:1], float('-inf'))
    
    # Flatten
    logits_flat = logits_masked.view(-1, logits.size(-1))  # [B*T, V]
    targets_flat = target_ids.view(-1)  # [B*T]
    
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction='none')
    
    # Only include masked positions
    mask_flat = mask.view(-1)  # [B*T]
    loss = (loss * mask_flat.float()).sum() / (mask_flat.float().sum() + 1e-8)
    
    # Apply step weight if provided
    if step_weight is not None:
        loss = loss * step_weight
    
    return loss


def compute_halt_loss(
    p_commit: torch.Tensor,
    target_commit: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for boundary (halt) prediction.
    
    Args:
        p_commit: [B] predicted commit probabilities
        target_commit: [B] target commit labels (0 or 1)
        
    Returns:
        Scalar BCE loss
    """
    return F.binary_cross_entropy(p_commit, target_commit.float())


def compute_length_loss(
    p_len: torch.Tensor,
    target_len: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for effective chunk length prediction.
    
    Args:
        p_len: [B, max_len+1] log-probabilities over lengths
        target_len: [B] target length indices (0..max_len)
        
    Returns:
        Scalar CE loss
    """
    return F.nll_loss(p_len, target_len.long())


def compute_ponder_loss(
    num_steps: int,
    lambda_ponder: float = 0.01,
) -> torch.Tensor:
    """
    Ponder cost (ACT-style) to encourage fewer refinement steps.
    
    Args:
        num_steps: Number of inner refinement steps used
        lambda_ponder: Weight for ponder cost
        
    Returns:
        Scalar loss
    """
    return torch.tensor(float(num_steps) * lambda_ponder, requires_grad=True)


def compute_stability_loss(
    logits_prev: torch.Tensor,
    logits_curr: torch.Tensor,
    mask_stable: Optional[torch.Tensor] = None,
    mu: float = 0.01,
) -> torch.Tensor:
    """
    Temporal consistency loss to prevent flip-flopping.
    
    Penalizes large changes in logits for positions that should be stable.
    
    Args:
        logits_prev: [B, T, V] logits from previous step
        logits_curr: [B, T, V] logits from current step
        mask_stable: [B, T] boolean mask (True = should be stable)
        mu: Weight for stability loss
        
    Returns:
        Scalar loss
    """
    # L1 distance between logits (softmax space)
    prob_prev = F.softmax(logits_prev, dim=-1)
    prob_curr = F.softmax(logits_curr, dim=-1)
    
    diff = torch.abs(prob_curr - prob_prev)  # [B, T, V]
    diff = diff.mean(dim=-1)  # [B, T]
    
    if mask_stable is not None:
        # Only penalize changes on stable positions
        diff = diff * mask_stable.float()
        loss = diff.sum() / (mask_stable.float().sum() + 1e-8)
    else:
        loss = diff.mean()
    
    return mu * loss


def compute_total_loss(
    logits_final: torch.Tensor,
    logits_intermediate: list[torch.Tensor],
    target_ids: torch.Tensor,
    p_commit: torch.Tensor,
    target_commit: torch.Tensor,
    p_len: Optional[torch.Tensor] = None,
    target_len: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    num_steps: int = 0,
    lambda_final: float = 1.0,
    lambda_masked: float = 0.5,
    lambda_halt: float = 0.5,
    lambda_len: float = 0.05,
    lambda_ponder: float = 0.01,
    lambda_stability: float = 0.01,
    step_weights: Optional[list[float]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total training loss with all components.
    
    Args:
        logits_final: [B, T, V] final step logits
        logits_intermediate: List of [B, T, V] intermediate step logits
        target_ids: [B, T] target tokens
        p_commit: [B] predicted commit probabilities
        target_commit: [B] target commit labels
        p_len: Optional [B, max_len+1] length log-probs
        target_len: Optional [B] target lengths
        mask: [B, T] editable position mask
        num_steps: Number of refinement steps used
        lambda_*: Loss weights
        step_weights: Optional decay weights per intermediate step
        
    Returns:
        total_loss: Scalar total loss
        loss_dict: Dictionary of component losses
    """
    losses = {}
    
    # Final CE loss
    L_final = compute_final_ce_loss(logits_final, target_ids)
    losses['final_ce'] = L_final
    
    # Intermediate masked CE losses (deep supervision)
    if logits_intermediate:
        L_masked_list = []
        if step_weights is None:
            step_weights = [1.0 / (i + 1) for i in range(len(logits_intermediate))]
        
        for i, logits_i in enumerate(logits_intermediate):
            L_i = compute_masked_ce_loss(
                logits_i, target_ids, mask,
                step_weight=step_weights[i] if step_weights else None
            )
            L_masked_list.append(L_i)
        
        L_masked = sum(L_masked_list) / len(L_masked_list)
        losses['masked_ce'] = L_masked
    else:
        L_masked = torch.tensor(0.0, device=logits_final.device)
    
    # Halt loss
    L_halt = compute_halt_loss(p_commit, target_commit)
    losses['halt'] = L_halt
    
    # Length loss (optional)
    if p_len is not None and target_len is not None:
        L_len = compute_length_loss(p_len, target_len)
        losses['length'] = L_len
    else:
        L_len = torch.tensor(0.0, device=logits_final.device)
    
    # Ponder loss
    L_ponder = compute_ponder_loss(num_steps, lambda_ponder)
    losses['ponder'] = L_ponder
    
    # Stability loss (if intermediate steps available)
    L_stability = torch.tensor(0.0, device=logits_final.device)
    if len(logits_intermediate) >= 2:
        # Compare consecutive steps
        for i in range(len(logits_intermediate) - 1):
            L_stab_i = compute_stability_loss(
                logits_intermediate[i],
                logits_intermediate[i + 1],
                mask_stable=~mask if mask is not None else None,
            )
            L_stability = L_stability + L_stab_i
        L_stability = L_stability / (len(logits_intermediate) - 1)
    losses['stability'] = L_stability
    
    # Total loss
    total_loss = (
        lambda_final * L_final
        + lambda_masked * L_masked
        + lambda_halt * L_halt
        + lambda_len * L_len
        + lambda_ponder * L_ponder
        + lambda_stability * L_stability
    )
    
    return total_loss, losses

