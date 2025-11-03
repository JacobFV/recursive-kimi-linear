#!/usr/bin/env python3
"""Upload experiment results to HuggingFace Hub."""

import argparse
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi
import os
import json


def upload_experiment(
    experiment_dir: Path,
    hf_repo_id: str,
    hf_token: Optional[str] = None,
    private: bool = False,
):
    """
    Upload experiment artifacts to HuggingFace Hub.
    
    Args:
        experiment_dir: Directory containing experiment artifacts
        hf_repo_id: HuggingFace repo ID (username/repo-name)
        hf_token: HuggingFace token (or set HF_TOKEN env var)
        private: Whether to create private repo
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    # Get token
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF token required (--hf-token or HF_TOKEN env var)")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=hf_repo_id,
            private=private,
            repo_type="model",
            token=token,
        )
        print(f"Created new repo: {hf_repo_id}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Repo {hf_repo_id} already exists, uploading to it...")
        else:
            raise
    
    # Upload best checkpoint
    metadata_file = experiment_dir / "experiment_metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        best_step = metadata.get("best_checkpoint_step")
        if best_step is not None:
            checkpoint_dir = experiment_dir / "checkpoints" / f"step_{best_step:06d}"
            if checkpoint_dir.exists():
                print(f"Uploading best checkpoint from step {best_step}...")
                api.upload_folder(
                    folder_path=str(checkpoint_dir),
                    repo_id=hf_repo_id,
                    token=token,
                    repo_type="model",
                )
    
    # Upload config
    config_file = experiment_dir / "recursive_config.json"
    if config_file.exists():
        print("Uploading config...")
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo="recursive_config.json",
            repo_id=hf_repo_id,
            token=token,
            repo_type="model",
        )
    
    # Upload metadata
    if metadata_file.exists():
        print("Uploading metadata...")
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo="experiment_metadata.json",
            repo_id=hf_repo_id,
            token=token,
            repo_type="model",
        )
    
    # Upload notes
    notes_dir = experiment_dir / "notes"
    if notes_dir.exists():
        print("Uploading notes...")
        for note_file in notes_dir.glob("*.txt"):
            api.upload_file(
                path_or_fileobj=str(note_file),
                path_in_repo=f"notes/{note_file.name}",
                repo_id=hf_repo_id,
                token=token,
                repo_type="model",
            )
    
    # Create model card
    if metadata_file.exists():
        from kimi_linear.recursive.experiment import ExperimentTracker
        tracker = ExperimentTracker.from_metadata(metadata_file)  # Need to fix this
        card = tracker.create_hf_card(hf_repo_id)
        card_path = experiment_dir / "README.md"
        with open(card_path, 'w') as f:
            f.write(card)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            token=token,
            repo_type="model",
        )
    
    print(f"✓ Upload complete! View at: https://huggingface.co/{hf_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Experiment directory")
    parser.add_argument("--hf-repo", type=str, required=True,
                       help="HuggingFace repo ID (username/repo-name)")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                       help="Create private repo")
    
    args = parser.parse_args()
    
    upload_experiment(
        Path(args.experiment_dir),
        args.hf_repo,
        args.hf_token,
        args.private,
    )

