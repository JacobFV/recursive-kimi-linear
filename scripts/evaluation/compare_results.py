#!/usr/bin/env python3
"""Compare evaluation results across different stages."""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import sys


def load_results(stage: str, eval_dir: Path) -> Dict:
    """Load evaluation results for a stage."""
    result_file = eval_dir / f"{stage}_results.json"
    
    if not result_file.exists():
        print(f"⚠ Results file not found: {result_file}")
        return None
    
    with open(result_file, "r") as f:
        return json.load(f)


def format_metric(value: float, fmt: str = ".4f") -> str:
    """Format metric value."""
    if value is None or (isinstance(value, float) and (value == float("inf") or value != value)):
        return "N/A"
    return f"{value:{fmt}}"


def print_comparison_table(results_dict: Dict[str, Dict], metrics: List[str]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("Metrics Comparison")
    print("=" * 80)
    print(f"{'Metric':<30} " + " ".join([f"{stage:>15}" for stage in results_dict.keys()]))
    print("-" * 80)
    
    for metric in metrics:
        row = f"{metric:<30}"
        for stage in results_dict.keys():
            results = results_dict[stage]
            value = None
            
            # Navigate nested dicts
            if metric == "perplexity":
                value = results.get("perplexity", {}).get("perplexity")
            elif metric == "token_accuracy":
                value = results.get("perplexity", {}).get("token_accuracy")
            elif metric == "reasoning_accuracy":
                value = results.get("reasoning", {}).get("reasoning_accuracy")
            elif metric == "avg_generation_length":
                gen = results.get("generation", [])
                if gen:
                    lengths = [g.get("length", 0) for g in gen]
                    value = sum(lengths) / len(lengths) if lengths else 0
            
            row += f"{format_metric(value):>15} "
        print(row)
    
    print("=" * 80)


def print_stage_info(results_dict: Dict[str, Dict]):
    """Print detailed info for each stage."""
    print("\n" + "=" * 80)
    print("Stage Details")
    print("=" * 80)
    
    for stage, results in results_dict.items():
        print(f"\n[{stage}]")
        print(f"  Model: {results.get('model_path', 'N/A')}")
        print(f"  Recursive: {results.get('use_recursive', False)}")
        
        if "perplexity" in results:
            ppl = results["perplexity"]
            print(f"  Perplexity: {format_metric(ppl.get('perplexity'))}")
            print(f"  Token Accuracy: {format_metric(ppl.get('token_accuracy'))}")
        
        if "reasoning" in results:
            reasoning = results["reasoning"]
            print(f"  Reasoning Accuracy: {format_metric(reasoning.get('reasoning_accuracy'))}")
            print(f"  Correct: {reasoning.get('correct', 0)}/{reasoning.get('total', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across stages")
    parser.add_argument("stages", nargs="+",
                       help="Stages to compare (baseline, after_surgery, etc.)")
    parser.add_argument("--eval_dir", type=str, default="./results/eval",
                       help="Directory containing evaluation results")
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    # Load results
    results_dict = {}
    for stage in args.stages:
        results = load_results(stage, eval_dir)
        if results:
            results_dict[stage] = results
    
    if not results_dict:
        print("✗ No results found!")
        return 1
    
    # Print comparison
    metrics = [
        "perplexity",
        "token_accuracy",
        "reasoning_accuracy",
        "avg_generation_length",
    ]
    
    print_comparison_table(results_dict, metrics)
    print_stage_info(results_dict)
    
    # Calculate improvements
    if "baseline" in results_dict and len(results_dict) > 1:
        print("\n" + "=" * 80)
        print("Improvements over Baseline")
        print("=" * 80)
        
        baseline = results_dict["baseline"]
        baseline_ppl = baseline.get("perplexity", {}).get("perplexity")
        
        for stage, results in results_dict.items():
            if stage == "baseline":
                continue
            
            stage_ppl = results.get("perplexity", {}).get("perplexity")
            if baseline_ppl and stage_ppl:
                improvement = ((baseline_ppl - stage_ppl) / baseline_ppl) * 100
                print(f"{stage:30} PPL improvement: {improvement:+.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

