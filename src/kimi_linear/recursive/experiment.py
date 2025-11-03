"""Experiment tracking and checkpointing system for recursive training."""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import yaml

from .config import RecursiveConfig


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    
    experiment_id: str
    experiment_name: str
    description: str
    created_at: str
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    # Training parameters
    phase: Optional[str] = None
    num_steps: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None
    
    # Model info
    base_model: Optional[str] = None
    config_path: Optional[str] = None
    
    # Results
    final_loss: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    
    # Notes and observations
    notes: List[str] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
        if self.issues is None:
            self.issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save metadata to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_file(cls, path: Path) -> 'ExperimentMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentTracker:
    """Tracks experiments with logging, checkpointing, and metadata."""
    
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: Path,
        description: str = "",
        config: Optional[RecursiveConfig] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Human-readable experiment name
            experiment_dir: Root directory for all experiment artifacts
            description: Experiment description
            config: RecursiveConfig used for this experiment
        """
        self.experiment_name = experiment_name
        self.experiment_dir = Path(experiment_dir)
        self.description = description
        self.config = config
        
        # Create experiment directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.notes_dir = self.experiment_dir / "notes"
        
        for d in [self.checkpoints_dir, self.logs_dir, self.metrics_dir, self.notes_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create metadata
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            experiment_name=experiment_name,
            description=description,
            created_at=datetime.now().isoformat(),
            python_version=self._get_python_version(),
            torch_version=torch.__version__,
            cuda_version=self._get_cuda_version(),
        )
        
        # Save initial metadata
        self.save_metadata()
        
        # Save config if provided
        if config:
            config.save(self.experiment_dir / "recursive_config.json")
            self.metadata.config_path = str(self.experiment_dir / "recursive_config.json")
            self.metadata.base_model = getattr(config, 'base_model_name', None)
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version if available."""
        try:
            if torch.cuda.is_available():
                return torch.version.cuda
        except:
            pass
        return None
    
    def update_metadata(self, **kwargs):
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                raise ValueError(f"Unknown metadata field: {key}")
        self.save_metadata()
    
    def add_note(self, note: str, step: Optional[int] = None):
        """Add a note/observation to the experiment."""
        timestamp = datetime.now().isoformat()
        if step is not None:
            note_entry = f"[Step {step}] {timestamp}: {note}"
        else:
            note_entry = f"{timestamp}: {note}"
        
        self.metadata.notes.append(note_entry)
        self.save_metadata()
        
        # Also save to notes file
        notes_file = self.notes_dir / "notes.txt"
        with open(notes_file, 'a') as f:
            f.write(note_entry + "\n")
    
    def add_issue(self, issue: str, step: Optional[int] = None):
        """Add an issue/problem encountered."""
        timestamp = datetime.now().isoformat()
        if step is not None:
            issue_entry = f"[Step {step}] {timestamp}: {issue}"
        else:
            issue_entry = f"{timestamp}: {issue}"
        
        self.metadata.issues.append(issue_entry)
        self.save_metadata()
        
        # Also save to notes file
        issues_file = self.notes_dir / "issues.txt"
        with open(issues_file, 'a') as f:
            f.write(issue_entry + "\n")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            step: Training step
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint_dir = self.checkpoints_dir / f"step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(model, 'save_pretrained'):
            # HuggingFace model
            model.save_pretrained(checkpoint_dir / "model")
        else:
            # PyTorch model
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # Save config
        if self.config:
            self.config.save(checkpoint_dir / "recursive_config.json")
        
        # Save checkpoint metadata
        checkpoint_meta = {
            "step": step,
            "is_best": is_best,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_dir / "checkpoint_metadata.json", 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)
        
        # Update experiment metadata if best
        if is_best:
            self.metadata.best_checkpoint_step = step
            if metrics and 'loss' in metrics:
                self.metadata.final_loss = metrics['loss']
            self.save_metadata()
        
        return checkpoint_dir
    
    def save_metrics(self, metrics: Dict[str, float], step: int):
        """Save metrics to JSON file."""
        metrics_file = self.metrics_dir / f"step_{step:06d}.json"
        # Ensure all values are JSON-serializable (convert numpy/torch types)
        cleaned_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                cleaned_metrics[k] = float(v)
            elif hasattr(v, 'item'):  # torch.Tensor
                cleaned_metrics[k] = float(v.item())
            else:
                cleaned_metrics[k] = v
        with open(metrics_file, 'w') as f:
            json.dump({"step": int(step), **cleaned_metrics}, f, indent=2)
    
    def save_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.experiment_dir / "experiment_metadata.json"
        self.metadata.save(metadata_file)
    
    def log_generation_sample(
        self,
        prompt: str,
        generated: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log a generation sample for later analysis."""
        samples_file = self.notes_dir / "generation_samples.jsonl"
        
        entry = {
            "step": step,
            "prompt": prompt,
            "generated": generated,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(samples_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "directory": str(self.experiment_dir),
            "num_notes": len(self.metadata.notes),
            "num_issues": len(self.metadata.issues),
            "best_checkpoint": self.metadata.best_checkpoint_step,
            "final_loss": self.metadata.final_loss,
        }
    
    def create_hf_card(self, hf_repo_path: Optional[str] = None) -> str:
        """Create a README card for HuggingFace model card."""
        card = f"""---
license: apache-2.0
tags:
- recursive-generation
- kimi-linear
- language-model
---

# {self.experiment_name}

## Experiment Details

- **Experiment ID**: {self.experiment_id}
- **Created**: {self.metadata.created_at}
- **Phase**: {self.metadata.phase or 'N/A'}

## Description

{self.description}

## Configuration

This model was trained with the following recursive configuration:
- Recursive enabled: {self.config.recursive_enabled if self.config else 'N/A'}
- Chunk width: {self.config.chunk_width if self.config else 'N/A'}
- Max inner steps: {self.config.max_inner_steps if self.config else 'N/A'}

## Training

- Steps: {self.metadata.num_steps or 'N/A'}
- Batch size: {self.metadata.batch_size or 'N/A'}
- Learning rate: {self.metadata.learning_rate or 'N/A'}

## Results

- Final loss: {self.metadata.final_loss or 'N/A'}
- Best checkpoint: Step {self.metadata.best_checkpoint_step or 'N/A'}

## Notes

{chr(10).join(self.metadata.notes[-5:]) if self.metadata.notes else 'No notes yet.'}

## Repository

Experiment artifacts: {hf_repo_path or 'N/A'}
"""
        return card

