#!/usr/bin/env python3
"""Comprehensive model evaluation script for all testing phases."""

import torch
import argparse
from pathlib import Path
import sys
import json
from typing import Optional

# Add repo root to path (three levels up from scripts/evaluation/)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from kimi_linear.recursive.metrics import MetricsTracker, evaluate_model, ReasoningEvaluator
from kimi_linear.recursive import load_model_with_config, RecursiveConfig


def load_model(
    model_path: str,
    use_recursive: bool = False,
    recursive_config: Optional[RecursiveConfig] = None,
    config_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
):
    """
    Load model (regular or with recursive wrapper).
    
    Supports both legacy (use_recursive) and new config-based approach.
    """
    print(f"Loading model from {model_path}...")
    
    # Use new config-based loader if config provided
    if recursive_config is not None or config_path is not None:
        model, tokenizer, config = load_model_with_config(
            model_path,
            config=recursive_config,
            config_path=config_path,
            recursive_enabled=use_recursive if recursive_config is None and config_path is None else None,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print(f"  Recursive enabled: {config.recursive_enabled}")
        return model, tokenizer
    
    # Legacy mode: use_recursive flag
    if use_recursive:
        # Create default enabled config
        config = RecursiveConfig(recursive_enabled=True)
    else:
        config = RecursiveConfig(recursive_enabled=False)
    
    model, tokenizer, _ = load_model_with_config(
        model_path,
        config=config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    print(f"  Recursive enabled: {config.recursive_enabled}")
    return model, tokenizer


def run_perplexity_eval(
    model,
    tokenizer,
    eval_data: list,
    max_samples: Optional[int] = None,
):
    """Run perplexity evaluation."""
    print(f"\nRunning perplexity evaluation on {min(len(eval_data), max_samples or len(eval_data))} samples...")
    
    model.eval()
    total_ppl = 0
    total_acc = 0
    count = 0
    
    samples = eval_data[:max_samples] if max_samples else eval_data
    
    with torch.no_grad():
        for i, batch in enumerate(samples):
            if i % 10 == 0:
                print(f"  Processing sample {i+1}/{len(samples)}...")
            
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch)
            else:
                input_ids = batch
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            
            try:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                targets = input_ids[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                
                if targets.size(1) > 0:
                    from torch.nn.functional import cross_entropy
                    loss = cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )
                    ppl = torch.exp(loss).item()
                    
                    # Accuracy
                    preds = logits.argmax(dim=-1)
                    mask = targets != -100
                    acc = ((preds == targets) & mask).float().mean().item()
                    
                    total_ppl += ppl
                    total_acc += acc
                    count += 1
            except Exception as e:
                print(f"  ⚠ Error on sample {i}: {e}")
                continue
    
    if count > 0:
        avg_ppl = total_ppl / count
        avg_acc = total_acc / count
        return {"perplexity": avg_ppl, "token_accuracy": avg_acc}
    return {"perplexity": float("inf"), "token_accuracy": 0.0}


def run_reasoning_eval(model, tokenizer):
    """Run reasoning evaluation."""
    print("\nRunning reasoning evaluation...")
    
    evaluator = ReasoningEvaluator()
    results = evaluator.evaluate(model, tokenizer, max_new_tokens=256)
    
    return results


def run_generation_test(
    model,
    tokenizer,
    test_prompts: list,
    max_new_tokens: int = 128,
):
    """Test generation quality."""
    print(f"\nRunning generation test on {len(test_prompts)} prompts...")
    
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"  Prompt {i+1}: {prompt[:50]}...")
        
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        
        try:
            if hasattr(model, "generate_chunks"):
                output = model.generate_chunks(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    chunk_width=64,
                    max_inner_steps=4,
                    temperature=0.0,
                )
            elif hasattr(model, "generate"):
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
            else:
                continue
            
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generated": generated,
                "length": len(output[0]) - len(input_ids[0]),
            })
        except Exception as e:
            print(f"  ⚠ Error generating: {e}")
            results.append({
                "prompt": prompt,
                "generated": f"ERROR: {e}",
                "length": 0,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model at different stages")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (local or HuggingFace)")
    parser.add_argument("--stage", type=str, required=True,
                       choices=["baseline", "after_surgery", "finetuned_vanilla", "finetuned_recursive"],
                       help="Evaluation stage")
    parser.add_argument("--use_recursive", action="store_true",
                       help="Use recursive wrapper (legacy, use --recursive-config instead)")
    parser.add_argument("--recursive-config", type=str, default=None,
                       help="Path to RecursiveConfig JSON file")
    parser.add_argument("--output_dir", type=str, default="./results/eval",
                       help="Output directory for results")
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Skip perplexity evaluation")
    parser.add_argument("--skip_reasoning", action="store_true",
                       help="Skip reasoning evaluation")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Max samples for perplexity eval")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-select config based on stage if not provided
    config_path = args.recursive_config
    if config_path is None:
        # Map stage to default config file
        stage_config_map = {
            "baseline": "configs/baseline.json",
            "after_surgery": "configs/after_surgery.json",
            "finetuned_vanilla": "configs/baseline.json",
            "finetuned_recursive": "configs/recursive_phase_a.json",
        }
        default_config = stage_config_map.get(args.stage)
        if default_config and Path(default_config).exists():
            config_path = default_config
            print(f"Auto-selected config: {config_path}")
    
    print("=" * 60)
    print(f"Model Evaluation: {args.stage}")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Config path: {config_path or 'None (using defaults)'}")
    print(f"Use recursive (legacy): {args.use_recursive}")
    print()
    
    # Load model
    try:
        model, tokenizer = load_model(
            args.model_path,
            use_recursive=args.use_recursive,
            config_path=config_path,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Prepare test prompts
    test_prompts = [
        "The capital of France is",
        "What is 2 + 2?",
        "Explain how neural networks work:",
        "Write a Python function to reverse a string:",
    ]
    
    # Run evaluations
    results = {
        "stage": args.stage,
        "model_path": args.model_path,
        "use_recursive": args.use_recursive,
        "config_path": config_path,
    }
    
    # Perplexity (if we have data)
    if not args.skip_perplexity:
        # Create dummy eval data for testing
        # In practice, load from actual dataset
        print("\nNote: Using dummy data for perplexity eval")
        print("      Provide actual eval data for real evaluation")
        dummy_data = [
            torch.randint(1, 1000, (128,)) for _ in range(args.max_samples)
        ]
        ppl_results = run_perplexity_eval(model, tokenizer, dummy_data, args.max_samples)
        results["perplexity"] = ppl_results
    
    # Reasoning
    if not args.skip_reasoning:
        reasoning_results = run_reasoning_eval(model, tokenizer)
        results["reasoning"] = reasoning_results
    
    # Generation
    gen_results = run_generation_test(model, tokenizer, test_prompts)
    results["generation"] = gen_results
    
    # Save results
    output_file = output_dir / f"{args.stage}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"✓ Evaluation complete!")
    print(f"  Results saved to: {output_file}")
    print("=" * 60)
    
    # Print summary
    print("\nSummary:")
    if "perplexity" in results:
        print(f"  Perplexity: {results['perplexity'].get('perplexity', 'N/A'):.2f}")
        print(f"  Token Accuracy: {results['perplexity'].get('token_accuracy', 'N/A'):.4f}")
    if "reasoning" in results:
        print(f"  Reasoning Accuracy: {results['reasoning'].get('reasoning_accuracy', 'N/A'):.4f}")
    print(f"  Generated {len(results['generation'])} samples")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

