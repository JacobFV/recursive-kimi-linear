"""Metrics tracking and evaluation for recursive generation training."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
import time


class MetricsTracker:
    """
    Comprehensive metrics tracker for training and evaluation.
    
    Tracks:
    - Loss components (final CE, masked CE, halt, ponder, stability)
    - Generation metrics (avg steps, commit rates, chunk lengths)
    - Performance metrics (throughput, memory)
    - Reasoning metrics (GSM8K, MATH-style problems)
    """
    
    def __init__(self, log_dir: str = "./logs", use_tensorboard: bool = True):
        """
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                print(f"✓ TensorBoard logging to {self.log_dir}")
            except ImportError:
                print("⚠ TensorBoard not available, using file logging only")
                self.use_tensorboard = False
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.current_step = 0
        self.start_time = time.time()
        
        # Evaluation results
        self.eval_results = {}
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if step is None:
            step = self.current_step
        
        self.metrics_history[tag].append((step, value))
        
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_losses(
        self,
        losses: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "train",
    ):
        """Log loss components."""
        for name, value in losses.items():
            self.log_scalar(f"{prefix}/loss/{name}", value, step)
        
        # Log total loss if present
        if "total" in losses:
            self.log_scalar(f"{prefix}/loss/total", losses["total"], step)
    
    def log_generation_metrics(
        self,
        avg_steps: float,
        commit_rate: float,
        avg_chunk_len: float,
        step: Optional[int] = None,
        prefix: str = "train",
    ):
        """Log generation-related metrics."""
        self.log_scalar(f"{prefix}/generation/avg_refine_steps", avg_steps, step)
        self.log_scalar(f"{prefix}/generation/commit_rate", commit_rate, step)
        self.log_scalar(f"{prefix}/generation/avg_chunk_length", avg_chunk_len, step)
    
    def log_performance_metrics(
        self,
        tokens_per_sec: float,
        memory_mb: float,
        step: Optional[int] = None,
        prefix: str = "train",
    ):
        """Log performance metrics."""
        self.log_scalar(f"{prefix}/performance/tokens_per_sec", tokens_per_sec, step)
        self.log_scalar(f"{prefix}/performance/memory_mb", memory_mb, step)
    
    def log_evaluation(
        self,
        results: Dict[str, float],
        step: Optional[int] = None,
        eval_name: str = "eval",
    ):
        """Log evaluation results."""
        self.eval_results[eval_name] = results
        
        for metric, value in results.items():
            self.log_scalar(f"evaluation/{eval_name}/{metric}", value, step)
    
    def update_step(self, step: int):
        """Update current step counter."""
        self.current_step = step
    
    def save_checkpoint_metrics(self, checkpoint_path: Path):
        """Save metrics to JSON file."""
        metrics_file = checkpoint_path / "metrics.json"
        
        data = {
            "metrics_history": dict(self.metrics_history),
            "eval_results": self.eval_results,
            "total_steps": self.current_step,
            "elapsed_time": time.time() - self.start_time,
        }
        
        with open(metrics_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Metrics saved to {metrics_file}")
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from logits and targets.
    
    Args:
        logits: [B, T, V] model logits
        targets: [B, T] target token IDs
        ignore_index: Token index to ignore
        
    Returns:
        Perplexity (exp of cross-entropy)
    """
    from torch.nn.functional import cross_entropy
    
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    loss = cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="mean")
    ppl = torch.exp(loss).item()
    
    return ppl


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute token-level accuracy."""
    preds = logits.argmax(dim=-1)
    mask = targets != -100
    correct = (preds == targets) & mask
    return correct.float().mean().item()


class ReasoningEvaluator:
    """Evaluator for reasoning tasks (GSM8K-style)."""
    
    def __init__(self):
        """Initialize with sample problems."""
        # Sample problems for testing (can be expanded)
        self.test_problems = [
            {
                "question": "Janet has 23 apples. She gives away 5 and eats 2. How many does she have left?",
                "answer": 16,
            },
            {
                "question": "A store has 40 books. They sell 12 and get 8 new ones. How many books now?",
                "answer": 36,
            },
        ]
    
    def evaluate(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        """
        Evaluate model on reasoning problems.
        
        Returns:
            Dictionary with accuracy and other metrics
        """
        correct = 0
        total = len(self.test_problems)
        
        for problem in self.test_problems:
            question = problem["question"]
            expected = problem["answer"]
            
            # Generate answer
            prompt = f"Question: {question}\nAnswer:"
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            
            if hasattr(model, "generate"):
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            elif hasattr(model, "generate_chunks"):
                output = model.generate_chunks(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            else:
                continue
            
            # Decode and extract number
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Simple extraction (look for last number in answer)
            import re
            numbers = re.findall(r'\d+', text.split("Answer:")[-1] if "Answer:" in text else text)
            if numbers:
                predicted = int(numbers[-1])
                if predicted == expected:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "reasoning_accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


def evaluate_model(
    model,
    tokenizer,
    eval_data: Optional[List[torch.Tensor]] = None,
    metrics_tracker: Optional[MetricsTracker] = None,
    step: Optional[int] = None,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_data: Optional list of token sequences for perplexity
        metrics_tracker: Optional tracker for logging
        step: Current step
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Perplexity evaluation
    if eval_data is not None and len(eval_data) > 0:
        model.eval()
        total_ppl = 0
        total_acc = 0
        count = 0
        
        with torch.no_grad():
            for batch in eval_data[:10]:  # Sample first 10 for speed
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch)
                else:
                    input_ids = batch
                
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                # Forward pass
                if hasattr(model, "forward"):
                    outputs = model(input_ids=input_ids, output_hidden_states=False)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                else:
                    continue
                
                # Compute metrics
                targets = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                
                if targets.size(1) > 0:
                    ppl = compute_perplexity(logits, targets)
                    acc = compute_accuracy(logits, targets)
                    
                    total_ppl += ppl
                    total_acc += acc
                    count += 1
        
        if count > 0:
            results["perplexity"] = total_ppl / count
            results["token_accuracy"] = total_acc / count
    
    # Reasoning evaluation
    try:
        evaluator = ReasoningEvaluator()
        reasoning_results = evaluator.evaluate(model, tokenizer)
        results.update(reasoning_results)
    except Exception as e:
        print(f"⚠ Reasoning evaluation failed: {e}")
        results["reasoning_accuracy"] = 0.0
    
    # Log to tracker
    if metrics_tracker:
        metrics_tracker.log_evaluation(results, step)
    
    model.train()
    return results

