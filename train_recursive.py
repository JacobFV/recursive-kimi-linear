"""Training script for recursive chunked generation on Kimi-Linear."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from pathlib import Path
import argparse
from typing import Optional

from kimi_linear.recursive import (
    ChunkRefineWrapper,
    ChunkCollator,
    create_corruption_mask,
    compute_total_loss,
)
from kimi_linear.recursive.metrics import MetricsTracker, evaluate_model


class ChunkDataset(Dataset):
    """Simple dataset that yields sequences for chunked training."""
    
    def __init__(
        self,
        sequences: list[torch.Tensor],
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


def create_dummy_data(num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 32000):
    """Create dummy data for testing."""
    return [torch.randint(1, vocab_size, (seq_len,)) for _ in range(num_samples)]


def train_phase_a(
    model: ChunkRefineWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    accelerator: Accelerator,
    num_steps: int = 50000,
    log_interval: int = 100,
):
    """
    Phase A: Train sidecar only (refine cells + boundary head).
    Base model weights are frozen.
    """
    model.train()
    
    # Freeze base model
    for param in model.base.parameters():
        param.requires_grad = False
    
    # Unfreeze sidecar components
    for param in model.refine_cells.parameters():
        param.requires_grad = True
    for param in model.boundary.parameters():
        param.requires_grad = True
    if model.latent_token is not None:
        for param in model.latent_token.parameters():
            param.requires_grad = True
    
    step = 0
    for epoch in range(1000):  # Large epoch count, break on steps
        for batch in dataloader:
            if step >= num_steps:
                return
            
            # Get input chunks
            input_ids = batch['input_ids']  # [B, W]
            target_ids = batch['input_ids'].clone()  # Same for now (can use teacher later)
            
            # Create corruption mask
            batch_size, chunk_width = input_ids.shape
            corruption_mask = torch.stack([
                create_corruption_mask(chunk_width, corruption_rate=0.3)
                for _ in range(batch_size)
            ]).to(input_ids.device)
            
            # Run inner refinement loop
            cache = None
            logits_intermediate = []
            z_token = None
            
            # Initialize refine states
            states = [
                torch.zeros(batch_size, 512, device=input_ids.device)
                for _ in model.refine_cells
            ]
            
            for t in range(4):  # K=4 inner steps
                # Forward pass
                logits, hiddens, cache, z_token = model._forward_hidden(
                    input_ids[:, :0],  # Empty prefix for now
                    input_ids,
                    past_key_values=cache,
                    latent_token=z_token,
                )
                
                logits_intermediate.append(logits)
                
                # Boundary prediction
                h_top = hiddens[-1]
                p_commit, p_len = model.boundary(h_top)
                
                # Refine hidden states
                for i, cell in enumerate(model.refine_cells):
                    hiddens[i], states[i] = cell(hiddens[i], states[i])
            
            # Final step
            logits_final = logits_intermediate[-1]
            
            # Compute losses
            # Target commit: always 1 at final step
            target_commit = torch.ones(batch_size, device=input_ids.device)
            
            # Target length: actual length (from collator)
            target_len = batch.get('lengths', torch.full((batch_size,), chunk_width))
            
            loss, loss_dict = compute_total_loss(
                logits_final=logits_final,
                logits_intermediate=logits_intermediate[:-1],
                target_ids=target_ids,
                p_commit=p_commit,
                target_commit=target_commit,
                p_len=p_len,
                target_len=target_len,
                mask=corruption_mask,
                num_steps=len(logits_intermediate),
            )
            
            # Backward
            accelerator.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            # Log to metrics tracker
            metrics_tracker.update_step(step)
            metrics_tracker.log_losses(loss_dict, step, prefix="train")
            if "total" not in loss_dict:
                metrics_tracker.log_scalar("train/loss/total", loss.item(), step)
            
            if step % log_interval == 0:
                accelerator.print(
                    f"Step {step}: Loss={loss.item():.4f}, "
                    f"FinalCE={loss_dict['final_ce'].item():.4f}, "
                    f"Halt={loss_dict['halt'].item():.4f}"
                )
            
            step += 1
        
        metrics_tracker.close()


def main():
    parser = argparse.ArgumentParser(description="Train recursive chunked generation")
    parser.add_argument("--model_name", type=str, 
                       default="moonshotai/Kimi-Linear-48B-A3B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--chunk_width", type=int, default=128,
                       help="Fixed chunk width W")
    parser.add_argument("--max_inner_steps", type=int, default=4,
                       help="Maximum inner refinement steps K")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (number of chunks)")
    parser.add_argument("--num_steps", type=int, default=50000,
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate for sidecar")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Warmup steps")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--phase", type=str, default="a", choices=["a", "b", "c"],
                       help="Training phase")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code (required for Kimi-Linear)")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for TensorBoard logs")
    parser.add_argument("--eval_interval", type=int, default=1000,
                       help="Evaluation interval (steps)")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Checkpoint save interval (steps)")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize metrics tracker
    log_dir = Path(args.log_dir) / f"phase_{args.phase}"
    metrics_tracker = MetricsTracker(log_dir=str(log_dir), use_tensorboard=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    accelerator.print(f"Loading model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Wrap with recursive controller
    wrapper = ChunkRefineWrapper(
        base_model=base_model,
        layers_to_refine="all",
        use_latent_token=True,
        max_chunk_len=args.chunk_width,
    )
    
    # Create dummy dataset (replace with real data loading)
    accelerator.print("Creating dummy dataset...")
    dummy_data = create_dummy_data(num_samples=1000, seq_len=args.chunk_width)
    dataset = ChunkDataset(dummy_data, chunk_width=args.chunk_width)
    
    collator = ChunkCollator(
        chunk_width=args.chunk_width,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    def collate_fn(batch):
        return collator(batch, return_mask=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    
    # Optimizer (only for sidecar params)
    optimizer = torch.optim.AdamW(
        [
            {'params': wrapper.refine_cells.parameters(), 'lr': args.lr},
            {'params': wrapper.boundary.parameters(), 'lr': args.lr * 1.67},  # 5e-4
            {'params': wrapper.latent_token.parameters() if wrapper.latent_token else []},
        ],
        weight_decay=0.05,
    )
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_steps,
    )
    
    # Prepare with accelerator
    wrapper, optimizer, dataloader, scheduler = accelerator.prepare(
        wrapper, optimizer, dataloader, scheduler
    )
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    accelerator.print(f"Starting Phase {args.phase} training...")
    if args.phase == "a":
        train_phase_a(
            wrapper,
            dataloader,
            optimizer,
            scheduler,
            accelerator,
            num_steps=args.num_steps,
        )
    
    # Save checkpoint
    accelerator.print(f"Saving checkpoint to {output_dir}")
    unwrapped = accelerator.unwrap_model(wrapper)
    accelerator.save_state(output_dir / f"phase_{args.phase}_checkpoint")
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()

