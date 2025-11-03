"""Utility functions for FLA layers."""

import torch


def get_unpad_data(attention_mask: torch.Tensor):
    """
    Get unpadding data from attention mask.
    
    Args:
        attention_mask: [B, S] mask where 1 = valid token, 0 = padding
    
    Returns:
        indices: Flattened indices of valid tokens
        cu_seqlens: Cumulative sequence lengths [0, s1, s1+s2, ...]
        max_seqlen_in_batch: Maximum sequence length
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.zeros(
        (len(seqlens_in_batch) + 1,),
        device=seqlens_in_batch.device,
        dtype=torch.int32
    )
    cu_seqlens[1:] = seqlens_in_batch.cumsum(dim=0)
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Index the first axis of a tensor.
    
    Args:
        x: Tensor of shape [N, ...]
        indices: Indices to select [M]
    
    Returns:
        Tensor of shape [M, ...]
    """
    return x[indices]


def pad_input(
    x: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Pad input tensor back to original shape.
    
    Args:
        x: Unpadded tensor of shape [M, ...]
        indices: Original indices [M]
        batch_size: Original batch size
        seq_len: Original sequence length
    
    Returns:
        Padded tensor of shape [batch_size, seq_len, ...]
    """
    # Create output tensor filled with zeros
    output_shape = (batch_size, seq_len) + x.shape[1:]
    output = torch.zeros(
        output_shape,
        dtype=x.dtype,
        device=x.device
    )
    
    # Flatten for indexing
    output_flat = output.view(-1, *x.shape[1:])
    output_flat[indices] = x
    
    return output

