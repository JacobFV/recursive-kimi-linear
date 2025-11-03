"""Data collation and corruption utilities for chunked training."""

import torch
from typing import List, Dict, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence


def create_dummy_data(num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 32000) -> List[torch.Tensor]:
    """
    Create dummy data for testing.
    
    Args:
        num_samples: Number of sequences
        seq_len: Length of each sequence
        vocab_size: Vocabulary size
        
    Returns:
        List of token ID tensors
    """
    return [torch.randint(1, vocab_size, (seq_len,)) for _ in range(num_samples)]


class ChunkDataset(torch.utils.data.Dataset):
    """Simple dataset that yields sequences for chunked training."""
    
    def __init__(
        self,
        sequences: List[torch.Tensor],
        chunk_width: int = 128,
    ):
        """
        Args:
            sequences: List of token ID sequences
            chunk_width: Target chunk width
        """
        self.sequences = sequences
        self.chunk_width = chunk_width
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate or pad to chunk width
        if len(seq) > self.chunk_width:
            seq = seq[:self.chunk_width]
        elif len(seq) < self.chunk_width:
            padding = torch.full((self.chunk_width - len(seq),), 0, dtype=seq.dtype)
            seq = torch.cat([seq, padding])
        return seq


class ChunkCollator:
    """
    Collates sequences into fixed-width chunks with right-padding.
    
    Also handles corruption masking for denoising training.
    """
    
    def __init__(
        self,
        chunk_width: int = 128,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
    ):
        """
        Args:
            chunk_width: Fixed chunk width W
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        """
        self.chunk_width = chunk_width
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    def __call__(
        self,
        sequences: List[torch.Tensor],
        return_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Collate sequences into chunks.
        
        Args:
            sequences: List of token ID tensors [T_i]
            return_mask: Whether to return padding mask
            
        Returns:
            Dictionary with:
                - input_ids: [B, W] padded chunk tokens
                - attention_mask: [B, W] (if return_mask)
                - lengths: [B] actual sequence lengths
        """
        batch_size = len(sequences)
        
        # Pad sequences to chunk width
        padded = []
        lengths = []
        
        for seq in sequences:
            # Truncate if too long
            if len(seq) > self.chunk_width:
                seq = seq[:self.chunk_width]
            
            lengths.append(len(seq))
            
            # Right-pad to chunk width
            if len(seq) < self.chunk_width:
                padding = torch.full(
                    (self.chunk_width - len(seq),),
                    self.pad_token_id,
                    dtype=seq.dtype,
                    device=seq.device
                )
                seq = torch.cat([seq, padding])
            
            padded.append(seq)
        
        # Stack into batch
        input_ids = torch.stack(padded)  # [B, W]
        
        result = {
            'input_ids': input_ids,
            'lengths': torch.tensor(lengths, dtype=torch.long),
        }
        
        if return_mask:
            # Create attention mask (1 = real token, 0 = padding)
            attention_mask = (input_ids != self.pad_token_id).long()
            result['attention_mask'] = attention_mask
        
        return result


def create_corruption_mask(
    length: int,
    corruption_rate: float = 0.3,
    min_corrupt: int = 1,
    max_corrupt: Optional[int] = None,
    strategy: str = "random",
) -> torch.Tensor:
    """
    Create a boolean mask indicating which positions should be corrupted/editable.
    
    Args:
        length: Sequence length
        corruption_rate: Fraction of positions to corrupt
        min_corrupt: Minimum number of corrupted positions
        max_corrupt: Maximum number of corrupted positions
        strategy: "random" or "span" (span masking)
        
    Returns:
        Boolean mask [length] (True = corrupted/editable)
    """
    num_corrupt = max(
        min_corrupt,
        min(
            int(length * corruption_rate),
            max_corrupt or length
        )
    )
    
    if strategy == "random":
        # Random positions
        corrupt_indices = torch.randperm(length)[:num_corrupt]
        mask = torch.zeros(length, dtype=torch.bool)
        mask[corrupt_indices] = True
    
    elif strategy == "span":
        # Contiguous span
        start = torch.randint(0, max(1, length - num_corrupt + 1), (1,)).item()
        mask = torch.zeros(length, dtype=torch.bool)
        mask[start:start + num_corrupt] = True
    
    else:
        raise ValueError(f"Unknown corruption strategy: {strategy}")
    
    return mask


def create_teacher_boundaries(
    sequence: torch.Tensor,
    chunk_width: int,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    strategy: str = "fixed",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create teacher boundaries for chunk segmentation.
    
    Args:
        sequence: Token sequence [T]
        chunk_width: Chunk width
        strategy: "fixed" (every W tokens) or "sentence" (at EOS/punctuation)
        
    Returns:
        boundaries: Boolean tensor [num_chunks] (True = commit here)
        lengths: Effective lengths [num_chunks]
    """
    T = len(sequence)
    
    if strategy == "fixed":
        num_chunks = (T + chunk_width - 1) // chunk_width
        boundaries = torch.ones(num_chunks, dtype=torch.bool)
        lengths = []
        
        for i in range(num_chunks):
            start = i * chunk_width
            end = min(start + chunk_width, T)
            chunk = sequence[start:end]
            
            # Find effective length (strip trailing pads)
            if pad_token_id is not None:
                valid = (chunk != pad_token_id).nonzero(as_tuple=True)[0]
                if len(valid) > 0:
                    length = valid[-1].item() + 1
                else:
                    length = len(chunk)
            else:
                length = len(chunk)
            
            lengths.append(length)
        
        return boundaries, torch.tensor(lengths, dtype=torch.long)
    
    elif strategy == "sentence":
        # Find sentence boundaries (EOS or punctuation followed by space)
        # Simplified: just use EOS
        eos_positions = (sequence == eos_token_id).nonzero(as_tuple=True)[0]
        
        boundaries = []
        lengths = []
        start = 0
        
        for eos_pos in eos_positions:
            if eos_pos >= start:
                chunk_len = eos_pos.item() - start + 1
                if chunk_len <= chunk_width:
                    boundaries.append(True)
                    lengths.append(chunk_len)
                    start = eos_pos.item() + 1
        
        # Handle remainder
        if start < T:
            remainder = T - start
            if remainder > 0:
                boundaries.append(True)
                lengths.append(min(remainder, chunk_width))
        
        return torch.tensor(boundaries, dtype=torch.bool), torch.tensor(lengths, dtype=torch.long)
    
    else:
        raise ValueError(f"Unknown boundary strategy: {strategy}")


def create_noise_schedule(
    num_steps: int,
    initial_rate: float = 0.4,
    final_rate: float = 0.1,
    schedule: str = "linear",
) -> List[float]:
    """
    Create corruption rate schedule across refinement steps.
    
    Args:
        num_steps: Number of refinement steps
        initial_rate: Initial corruption rate
        final_rate: Final corruption rate
        schedule: "linear" or "cosine"
        
    Returns:
        List of corruption rates per step
    """
    if num_steps == 1:
        return [final_rate]
    
    rates = []
    for i in range(num_steps):
        if schedule == "linear":
            alpha = i / (num_steps - 1) if num_steps > 1 else 0.0
        elif schedule == "cosine":
            alpha = 0.5 * (1 - torch.cos(torch.tensor(i / (num_steps - 1)) * torch.pi)).item()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        rate = initial_rate * (1 - alpha) + final_rate * alpha
        rates.append(rate)
    
    return rates

