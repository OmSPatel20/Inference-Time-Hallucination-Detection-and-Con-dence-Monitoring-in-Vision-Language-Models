"""
10_experiment_runner.py — Run POPE evaluation across all experimental conditions.

Iterates over: quantization × POPE split
Runs script 07 for each combination and collects summaries.

Usage:
    python scripts/10_experiment_runner.py
    python scripts/10_experiment_runner.py --quick   # Only fp16 + random for testing
"""

import sys
import os
import argparse
import json
import time
import torch
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, load_model_and_processor

# Import POPE evaluation function
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pope_eval", os.path.join(os.path.dirname(__file__), "07_pope_evaluation.py")
)

# Also need to pre-import the detection modules it depends on
for script_name, alias in [
    ("04_entropy_detector", "_04_entropy_detector"),
    ("06_contrastive_decoder", "_06_contrastive_decoder"),
]:
    script_path = os.path.join(os.path.dirname(__file__), f"{script_name}.py")
    s = importlib.util.spec_from_file_location(alias, script_path)
    m = importlib.util.module_from_spec(s)
    sys.modules[alias] = m
    s.loader.exec_module(m)

pope_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pope_mod)
run_pope_evaluation = pope_mod.run_pope_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test: default quant + random only")
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    hw = config.get("hardware", {})
    default_quant = hw.get("default_quantization", "int4")
    use_contrastive = hw.get("enable_contrastive_detector", True) and not args.no_contrastive

    if args.quick:
        quantizations = [default_quant]
        splits = ["random"]
        max_samples = args.max_samples or 200
    else:
        quantizations = config["experiment"]["quantizations"]
        splits = config["experiment"]["pope_splits"]
        max_samples = args.max_samples

    all_summaries = []
    total_start = time.time()

    print("=" * 60)
    print("EXPERIMENT MATRIX")
    print("=" * 60)
    conditions = list(product(quantizations, splits))
    print(f"Conditions: {len(conditions)}")
    for q, s in conditions:
        print(f"  {q} × {s}")
    print()

    for quant in quantizations:
        print(f"\n{'#'*60}")
        print(f"# Loading model: {quant}")
        print(f"{'#'*60}")

        model, processor = load_model_and_processor(
            model_id=config["paths"]["model_id"],
            quantization=quant,
        )

        for split in splits:
            print(f"\n--- Running: {quant} × {split} ---")
            t0 = time.time()

            summary = run_pope_evaluation(
                model, processor, config,
                split=split,
                quantization=quant,
                max_samples=max_samples,
                use_contrastive=use_contrastive,
            )
            summary["wall_time_s"] = time.time() - t0
            all_summaries.append(summary)

        # Free model before loading next quantization
        del model
        torch.cuda.empty_cache()

    # Save combined results
    out_dir = config["paths"]["output_dir"]
    combined_file = os.path.join(out_dir, "experiment_matrix_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    total_time = time.time() - total_start

    # Print summary table
    print(f"\n{'='*70}")
    print(f"EXPERIMENT MATRIX RESULTS")
    print(f"{'='*70}")
    print(f"{'Quant':<8} {'Split':<14} {'Accuracy':<10} {'Halluc Rate':<12} {'Time':<8}")
    print(f"{'-'*70}")
    for s in all_summaries:
        print(f"{s['quantization']:<8} {s['split']:<14} "
              f"{s['accuracy']:<10.4f} {s['hallucination_rate']:<12.4f} "
              f"{s.get('wall_time_s', 0):<8.0f}s")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Results: {combined_file}")


if __name__ == "__main__":
    main()
