"""Latent token utilities for top-to-bottom skip connections."""

import torch
import torch.nn as nn


class LatentToken(nn.Module):
    """
    Optional learned latent token that persists across inner refinement steps.
    
    Provides a global control signal that flows through all layers via attention,
    enabling top→bottom skip connections without FiLM/modulation.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, token_id: int = None):
        """
        Args:
            vocab_size: Vocabulary size (for embedding lookup)
            hidden_size: Hidden dimension
            token_id: Special token ID to reserve (if None, create new embedding)
        """
        super().__init__()
        self.token_id = token_id
        
        if token_id is None:
            # Create a new learned embedding (not in vocab)
            self.embedding = nn.Parameter(torch.randn(1, hidden_size))
        else:
            # Use existing token from vocab (requires access to token embeddings)
            self.embedding = None
            self.token_id = token_id
    
    def get_embedding(self, token_embeddings: nn.Embedding = None) -> torch.Tensor:
        """
        Get the latent token embedding.
        
        Args:
            token_embeddings: If using existing token, provide embedding table
            
        Returns:
            Embedding vector [1, hidden_size]
        """
        if self.embedding is not None:
            return self.embedding
        elif token_embeddings is not None and self.token_id is not None:
            return token_embeddings(torch.tensor([self.token_id], 
                                                 device=token_embeddings.weight.device))
        else:
            raise ValueError("Must provide token_embeddings if using existing token_id")
    
    def forward(self, batch_size: int, device: torch.device, 
                token_embeddings: nn.Embedding = None) -> torch.Tensor:
        """
        Create latent token embeddings for a batch.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            token_embeddings: Embedding table if using existing token
            
        Returns:
            Latent token embeddings [batch_size, 1, hidden_size]
        """
        emb = self.get_embedding(token_embeddings)  # [1, hidden_size]
        # Broadcast to batch
        return emb.unsqueeze(0).expand(batch_size, 1, -1)  # [B, 1, hidden_size]


def append_latent_token(input_ids: torch.Tensor, latent_emb: torch.Tensor) -> torch.Tensor:
    """
    Append latent token embedding to input sequence.
    
    Args:
        input_ids: Input token IDs [B, T]
        latent_emb: Latent token embedding [B, 1, hidden_size]
        
    Returns:
        Concatenated embeddings [B, T+1, hidden_size]
    """
    # In practice, this is handled by modifying the sequence in forward pass
    # This is a utility for clarity
    return torch.cat([input_ids, latent_emb], dim=1)

