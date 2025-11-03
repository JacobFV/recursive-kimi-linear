"""Boundary (halt/commit) head for chunked generation."""

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """
    Predicts when to commit a chunk and its effective length.
    
    Takes pooled hidden states from the top layer and outputs:
    - p_commit: Probability that the chunk should be committed
    - p_len: Distribution over effective chunk length (0..max_len)
    """
    
    def __init__(self, d_model: int, gate_hidden: int = 1024, max_len: int = 64):
        """
        Args:
            d_model: Hidden dimension
            gate_hidden: MLP hidden dimension
            max_len: Maximum chunk length (predicts 0..max_len)
        """
        super().__init__()
        self.max_len = max_len
        
        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(d_model, gate_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(gate_hidden, gate_hidden, bias=False),
            nn.SiLU(),
        )
        
        # Commit probability head
        self.commit = nn.Linear(gate_hidden, 1, bias=True)
        nn.init.constant_(self.commit.bias, -2.0)  # Bias toward early commit initially
        
        # Length prediction head (categorical over 0..max_len)
        self.len_head = nn.Linear(gate_hidden, max_len + 1, bias=True)
    
    def forward(self, h_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict commit probability and effective length.
        
        Args:
            h_block: Block hidden states [B, T, d_model] or [B, d_model] (pooled)
            
        Returns:
            p_commit: Commit probability [B]
            p_len: Log-probabilities over lengths [B, max_len+1]
        """
        # If sequence dimension present, pool it
        if h_block.dim() == 3:
            # Use right edge (last position) for boundary decision
            h_pooled = h_block[:, -1]  # [B, d_model]
        else:
            h_pooled = h_block  # [B, d_model]
        
        # Shared features
        g = self.mlp(h_pooled)  # [B, gate_hidden]
        
        # Commit probability
        p_commit = torch.sigmoid(self.commit(g)).squeeze(-1)  # [B]
        
        # Length distribution
        p_len = torch.log_softmax(self.len_head(g), dim=-1)  # [B, max_len+1]
        
        return p_commit, p_len

