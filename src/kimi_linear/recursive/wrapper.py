"""Chunked recursive generation wrapper for Kimi-Linear models."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from .refine_cell import RefineCell
from .boundary_head import BoundaryHead
from .latent_token import LatentToken
from .config import RecursiveConfig


class ChunkRefineWrapper(nn.Module):
    """
    Wrapper that adds chunked recursive refinement to a base model.
    
    Generates text in fixed-width chunks, applying K inner refinement steps
    before committing tokens. Uses persistent state across inner steps.
    
    Idempotent: When recursive_enabled=False, behaves identically to base model.
    """
    
    def __init__(
        self,
        base_model,
        config: Optional[RecursiveConfig] = None,
        # Legacy parameters (for backward compatibility)
        layers_to_refine: Optional[Union[str, List[int]]] = None,
        use_latent_token: Optional[bool] = None,
        d_state: Optional[int] = None,
        gate_hidden: Optional[int] = None,
        max_chunk_len: Optional[int] = None,
    ):
        """
        Args:
            base_model: Base transformer model (Kimi-Linear)
            config: RecursiveConfig object (preferred) or None to use defaults/legacy
            layers_to_refine: (Legacy) Which layers to apply refinement
            use_latent_token: (Legacy) Whether to use explicit [Z] token
            d_state: (Legacy) Dimension of persistent refine state
            gate_hidden: (Legacy) Hidden dim for boundary head
            max_chunk_len: (Legacy) Maximum chunk length for length prediction
        """
        super().__init__()
        self.base = base_model
        
        # Resolve config: prefer config parameter, fall back to legacy params
        if config is not None:
            self.config = config
            # Validate config
            errors = config.validate()
            if errors:
                raise ValueError(f"Invalid RecursiveConfig: {errors}")
        else:
            # Legacy mode: create config from individual parameters
            self.config = RecursiveConfig(
                recursive_enabled=True,  # Legacy mode assumes enabled
                layers_to_refine=layers_to_refine if layers_to_refine is not None else "all",
                use_latent_token=use_latent_token if use_latent_token is not None else True,
                d_state=d_state if d_state is not None else 512,
                gate_hidden=gate_hidden if gate_hidden is not None else 1024,
                max_chunk_len=max_chunk_len if max_chunk_len is not None else 128,
            )
        
        # Get model config for dimensions
        model_config = getattr(base_model, 'config', None)
        if model_config is None:
            raise ValueError("Base model must have a config attribute")
        
        self.hidden_size = getattr(model_config, 'hidden_size', getattr(model_config, 'd_model', None))
        if self.hidden_size is None:
            raise ValueError("Could not determine hidden_size from config")
        
        self.num_layers = getattr(model_config, 'num_hidden_layers', getattr(model_config, 'num_layers', None))
        if self.num_layers is None:
            raise ValueError("Could not determine num_layers from config")
        
        # IDEMPOTENT: Only create components if recursion enabled
        if self.config.recursive_enabled:
            # Determine which layers to refine
            if self.config.layers_to_refine == "all":
                self.layer_indices = list(range(self.num_layers))
            else:
                self.layer_indices = self.config.layers_to_refine
            
            # Get model dtype for component initialization
            model_dtype = next(base_model.parameters()).dtype
            
            # Create refine cells (one per layer)
            self.refine_cells = nn.ModuleList([
                RefineCell(self.hidden_size, self.config.d_state).to(dtype=model_dtype)
                for _ in range(len(self.layer_indices))
            ])
            
            # Boundary head
            self.boundary = BoundaryHead(
                self.hidden_size,
                self.config.gate_hidden,
                self.config.max_chunk_len
            ).to(dtype=model_dtype)
            
            # Latent token (optional)
            if self.config.use_latent_token:
                self.latent_token = LatentToken(
                    vocab_size=getattr(model_config, 'vocab_size', 32000),
                    hidden_size=self.hidden_size
                ).to(dtype=model_dtype)
            else:
                self.latent_token = None
        else:
            # Pass-through mode: no components created
            self.refine_cells = None
            self.boundary = None
            self.latent_token = None
            self.layer_indices = []
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through wrapper.
        
        When recursion is disabled, passes through to base model.
        When recursion is enabled, also passes through to base model for now
        (recursive generation uses generate_chunks, not forward).
        """
        return self.base(*args, **kwargs)
    
    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        chunk_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        latent_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Any, Optional[torch.Tensor]]:
        """
        Run full model forward and return logits + hidden states.
        
        Args:
            input_ids: Past/prefix tokens [B, S]
            chunk_ids: Current chunk tokens [B, W]
            past_key_values: KDA/MLA cache from previous steps
            latent_token: Optional [Z] token embedding [B, 1, D]
            
        Returns:
            logits: [B, W, V]
            hiddens: List of [B, W, D] per layer
            new_cache: Updated cache
            z_token: Updated latent token state (if used)
        """
        # Concatenate prefix + chunk for forward pass
        # Note: In practice, we pass chunk_ids and use past_key_values for prefix
        # This is a simplification - actual implementation should handle kv cache properly
        
        # For now, concatenate if no cache (training) or use chunk only with cache (inference)
        if past_key_values is None:
            # Training: concatenate prefix + chunk
            full_input = torch.cat([input_ids, chunk_ids], dim=1)  # [B, S+W]
        else:
            # Inference: use chunk only, prefix is in cache
            full_input = chunk_ids  # [B, W]
        
        # If using latent token, we'd need to modify embeddings here
        # For now, we'll handle it via cache position tracking
        # TODO: Implement proper latent token injection via embeddings
        
        # Run model forward
        outputs = self.base(
            input_ids=full_input if past_key_values is None else chunk_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        
        logits = outputs.logits
        hiddens = outputs.hidden_states  # List of [B, T, D]
        new_cache = outputs.past_key_values
        
        # Extract logits/hiddens for chunk only (exclude prefix if concatenated)
        if past_key_values is None and input_ids.size(1) > 0:
            # Training: extract chunk portion
            prefix_len = input_ids.size(1)
            logits = logits[:, prefix_len:, :]  # [B, W, V]
            hiddens = [h[:, prefix_len:, :] for h in hiddens]  # List of [B, W, D]
        
        # Extract latent token state if used
        z_token = None
        if self.config.use_latent_token and latent_token is not None:
            # Use last position's hidden as latent state
            z_token = hiddens[-1][:, -1:, :]  # [B, 1, D]
        elif self.config.use_latent_token and self.latent_token is not None:
            # Initialize from learned embedding
            batch_size = chunk_ids.size(0)
            device = chunk_ids.device
            z_token = self.latent_token(batch_size, device)
        
        return logits, hiddens, new_cache, z_token
    
    def _seed_window(
        self,
        prefix_ids: torch.Tensor,
        chunk_width: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate initial chunk proposal via greedy/sampling single pass.
        
        Args:
            prefix_ids: Current prefix [S] or [B, S]
            chunk_width: Desired chunk width
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Proposed chunk token IDs [chunk_width] or [B, chunk_width]
        """
        # Normalize input
        if prefix_ids.dim() == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        batch_size = prefix_ids.size(0)
        
        # Initialize cache with prefix
        cache = None
        chunk_list = []
        
        for b in range(batch_size):
            prefix = prefix_ids[b]  # [S]
            current_ids = prefix.clone()
            cache_b = None
            
            # Prime cache with prefix (if needed)
            # For KDA, we may need to run forward on prefix first
            # Simplified: assume we can autoregressively extend
            
            chunk_b = []
            for _ in range(chunk_width):
                # Forward on last token (with cache)
                if len(current_ids) == len(prefix):
                    # First step: forward on entire prefix
                    inputs = current_ids.unsqueeze(0)
                else:
                    # Subsequent: forward on last token only
                    inputs = current_ids[-1:].unsqueeze(0)
                
                outputs = self.base(
                    input_ids=inputs,
                    past_key_values=cache_b,
                    use_cache=True,
                )
                logits = outputs.logits[0, -1]  # [V]
                cache_b = outputs.past_key_values
                
                # Sample next token
                if temperature == 0.0:
                    next_id = logits.argmax(-1).unsqueeze(0)
                else:
                    # Sample
                    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                    if top_p < 1.0:
                        # Nucleus sampling
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum = torch.cumsum(sorted_probs, dim=-1)
                        mask = cumsum <= top_p
                        sorted_probs = sorted_probs * mask
                        sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-8)
                        next_id_idx = torch.multinomial(sorted_probs, 1)
                        next_id = sorted_indices[next_id_idx].unsqueeze(0)
                    else:
                        next_id = torch.multinomial(probs, 1)
                
                chunk_b.append(next_id.item())
                current_ids = torch.cat([current_ids, next_id])
            
            chunk_list.append(torch.tensor(chunk_b, device=prefix_ids.device))
        
        if batch_size == 1:
            return chunk_list[0]
        return torch.stack(chunk_list)
    
    def _pad_to_width(self, chunk: torch.Tensor, width: int, pad_id: int = 0) -> torch.Tensor:
        """Pad chunk to fixed width."""
        current_len = chunk.size(0)
        if current_len >= width:
            return chunk[:width]
        padding = torch.full((width - current_len,), pad_id, dtype=chunk.dtype, device=chunk.device)
        return torch.cat([chunk, padding])
    
    def _strip_trailing_blanks(self, chunk: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
        """Remove trailing padding/blank tokens."""
        # Find last non-blank
        mask = chunk != blank_id
        if mask.any():
            last_valid = mask.nonzero()[-1].item() + 1
            return chunk[:last_valid]
        return chunk
    
    @torch.no_grad()
    def generate_chunks(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        chunk_width: Optional[int] = None,
        max_inner_steps: Optional[int] = None,
        commit_threshold: Optional[float] = None,
        min_commit_threshold: Optional[float] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **gen_kwargs
    ) -> torch.Tensor:
        """
        Generate text using chunked recursive refinement.
        
        Args:
            input_ids: Initial prompt [B, S] or [S]
            max_new_tokens: Maximum new tokens to generate
            chunk_width: Fixed chunk width W (uses config if None)
            max_inner_steps: Maximum inner refinement steps K (uses config if None)
            commit_threshold: Halt threshold tau (uses config if None)
            min_commit_threshold: Minimum threshold (uses config if None)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            **gen_kwargs: Additional generation kwargs
            
        Returns:
            Generated token IDs [B, S+new_tokens]
        """
        # IDEMPOTENT: Pass through to base model if recursion disabled
        if not self.config.recursive_enabled:
            return self.base.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if top_p < 1.0 else None,
                **gen_kwargs
            )
        
        # Use config values if parameters not provided
        chunk_width = chunk_width if chunk_width is not None else self.config.chunk_width
        max_inner_steps = max_inner_steps if max_inner_steps is not None else self.config.max_inner_steps
        commit_threshold = commit_threshold if commit_threshold is not None else self.config.commit_threshold
        min_commit_threshold = min_commit_threshold if min_commit_threshold is not None else self.config.min_commit_threshold
        
        # Normalize input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size = input_ids.size(0)
        
        out_ids = input_ids.clone()
        total_new = 0
        cache = None
        z_token = None
        
        # Get pad/blank token ID
        pad_id = getattr(self.base.config, 'pad_token_id', 0)
        
        while total_new < max_new_tokens:
            # Seed window with initial proposal
            prefix = out_ids[0] if batch_size == 1 else out_ids
            proposal = self._seed_window(prefix, chunk_width, temperature, top_p)
            
            # Pad to fixed width
            chunk = self._pad_to_width(proposal, chunk_width, pad_id)
            
            # Initialize per-layer refine states
            states = [
                torch.zeros(batch_size, self.config.d_state, device=input_ids.device)
                for _ in self.refine_cells
            ]
            
            # Inner recursion loop
            last_logits = None
            committed = False
            
            for t in range(max_inner_steps):
                # Forward pass
                logits, hiddens, cache, z_token = self._forward_hidden(
                    out_ids,
                    chunk.unsqueeze(0) if chunk.dim() == 1 else chunk,
                    past_key_values=cache,
                    latent_token=z_token,
                )
                
                # Boundary decision
                h_top = hiddens[-1]  # [B, W, D]
                p_commit, p_len = self.boundary(h_top)
                
                # Check halt condition
                if (p_commit >= commit_threshold).all():
                    committed = True
                    break
                
                # Deadlock prevention: if stuck for all steps, force commit
                if t == max_inner_steps - 1 and (p_commit < min_commit_threshold).all():
                    committed = True
                    break
                
                # Refine hidden states
                for i, (layer_idx, cell) in enumerate(zip(self.layer_indices, self.refine_cells)):
                    hiddens[layer_idx], states[i] = cell(hiddens[layer_idx], states[i])
                
                # Update tokens (only on masked/blank positions)
                # For now, simple: update blank positions
                new_ids = logits[0].argmax(-1)  # [W]
                blank_mask = (chunk == pad_id)
                chunk = torch.where(blank_mask, new_ids, chunk)
                
                last_logits = logits
            
            # Determine effective chunk length
            if not committed:
                # Use length head prediction
                _, p_len = self.boundary(hiddens[-1])
                B_hat = p_len.argmax(-1).item()
                B_eff = min(max(B_hat, 1), len(chunk))
            else:
                # Use actual generated length (strip blanks)
                B_eff = len(self._strip_trailing_blanks(chunk, pad_id))
            
            # Commit chunk
            commit_ids = chunk[:B_eff].unsqueeze(0) if batch_size == 1 else chunk[:B_eff]
            out_ids = torch.cat([out_ids, commit_ids], dim=1)
            total_new += B_eff
            
            # Stop on EOS
            if (commit_ids == getattr(self.base.config, 'eos_token_id', None)).any():
                break
        
        return out_ids

