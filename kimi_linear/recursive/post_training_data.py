"""Data loader for post-training datasets from Post-Training-Data-Flywheel."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset


class FlywheelDataset(Dataset):
    """
    Dataset loader for Post-Training-Data-Flywheel datasets.
    
    Supports loading from HuggingFace Hub or local files.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_length: int = 2048,
        local_path: Optional[str] = None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name or local path identifier
            split: Dataset split to use
            max_length: Maximum sequence length
            local_path: Optional local path to dataset
        """
        self.max_length = max_length
        
        # Load dataset
        if local_path and Path(local_path).exists():
            print(f"Loading dataset from local path: {local_path}")
            # Handle local loading based on format
            # This is a placeholder - implement based on actual data format
            raise NotImplementedError("Local dataset loading not yet implemented")
        else:
            print(f"Loading dataset from HuggingFace: {dataset_name}")
            try:
                self.dataset = load_dataset(dataset_name, split=split)
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                raise
        
        print(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Extract text (adapt based on dataset structure)
        if "text" in example:
            text = example["text"]
        elif "instruction" in example and "output" in example:
            # Instruction following format
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            # Format as instruction following
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        elif "messages" in example:
            # Chat format
            messages = example["messages"]
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        else:
            # Fallback: use first text-like field
            text_fields = [k for k, v in example.items() if isinstance(v, str)]
            if text_fields:
                text = example[text_fields[0]]
            else:
                text = str(example)
        
        # Tokenize (will be done by collator)
        return {"text": text}
    
    def get_chunks(self, tokenizer, chunk_width: int = 128):
        """
        Get dataset as tokenized chunks for chunked training.
        
        Returns list of token sequences ready for ChunkCollator.
        """
        chunks = []
        for example in self.dataset:
            text = self._extract_text(example)
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Split into chunks
            for i in range(0, len(tokens), chunk_width):
                chunk = tokens[i:i + chunk_width]
                if len(chunk) >= chunk_width // 2:  # Minimum chunk size
                    chunks.append(torch.tensor(chunk))
        
        return chunks
    
    def _extract_text(self, example):
        """Extract text from example dict."""
        if "text" in example:
            return example["text"]
        elif "instruction" in example:
            inst = example.get("instruction", "")
            inp = example.get("input", "")
            out = example.get("output", "")
            if inp:
                return f"{inst}\n{inp}\n{out}"
            return f"{inst}\n{out}"
        elif "messages" in example:
            return "\n".join([m.get("content", "") for m in example["messages"]])
        return str(example)


def get_flywheel_datasets(
    dataset_names: List[str],
    split: str = "train",
    chunk_width: int = 128,
    tokenizer=None,
):
    """
    Load multiple Flywheel datasets and return as chunked sequences.
    
    Args:
        dataset_names: List of HuggingFace dataset names
        split: Dataset split
        chunk_width: Chunk width for training
        tokenizer: Tokenizer (if None, returns text only)
    
    Returns:
        List of token sequences ready for ChunkCollator
    """
    all_chunks = []
    
    for name in dataset_names:
        try:
            ds = FlywheelDataset(name, split=split)
            if tokenizer:
                chunks = ds.get_chunks(tokenizer, chunk_width)
                all_chunks.extend(chunks)
            else:
                # Return raw text examples
                for i in range(len(ds)):
                    all_chunks.append(ds[i])
        except Exception as e:
            print(f"⚠ Failed to load {name}: {e}")
            continue
    
    return all_chunks


# Recommended datasets from Post-Training-Data-Flywheel
RECOMMENDED_DATASETS = {
    "instruction_following": [
        "Open-Orca/OpenOrca",
        "Open-Orca/SlimOrca",
        "teknium/GPT4-LLM-Cleaned",
    ],
    "code": [
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "RLHFlow/CodeUltraFeedback-standard",
    ],
    "math": [
        "meta-math/MetaMathQA",
        "TIGER-Lab/MathInstruct",
        "openai/gsm8k",
    ],
}
