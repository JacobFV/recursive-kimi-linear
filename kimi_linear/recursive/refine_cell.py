"""Per-layer refine cell for latent representation editing."""

import torch
import torch.nn as nn


class RefineCell(nn.Module):
    """
    Tiny MLP that refines hidden states via residual updates.
    
    Per-layer cell that takes current hidden states and a persistent state,
    returns a delta to add to hidden states and updates the persistent state.
    
    Zero-init on output projection ensures baseline behavior at initialization.
    """
    
    def __init__(self, d_model: int, d_state: int = 512):
        """
        Args:
            d_model: Hidden dimension of the transformer
            d_state: Dimension of persistent state vector
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projection: concat pooled hidden + state
        self.inp = nn.Linear(d_model + d_state, 4 * d_model, bias=False)
        self.act = nn.SiLU()
        
        # Output projection (zero-init for residual on-ramp)
        self.out = nn.Linear(4 * d_model, d_model, bias=False)
        nn.init.zeros_(self.out.weight)  # Critical: preserves baseline at step 0
        
        # State update via GRU cell
        self.state_upd = nn.GRUCell(d_model, d_state)
    
    def forward(self, h: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Refine hidden states using persistent state.
        
        Args:
            h: Hidden states [B, T, d_model]
            s: Persistent state [B, d_state]
            
        Returns:
            h_refined: Refined hidden states [B, T, d_model]
            s_next: Updated state [B, d_state]
        """
        # Pool over sequence dimension (cheap window summary)
        pooled = h.mean(dim=1)  # [B, d_model]
        
        # Update persistent state
        s_next = self.state_upd(pooled, s)
        
        # Compute refinement delta
        x = torch.cat([pooled, s], dim=-1)  # [B, d_model + d_state]
        g = self.act(self.inp(x))  # [B, 4*d_model]
        delta = self.out(g)  # [B, d_model]
        
        # Residual connection: h + delta (broadcast over T)
        h_refined = h + delta.unsqueeze(1)  # [B, T, d_model]
        
        return h_refined, s_next

