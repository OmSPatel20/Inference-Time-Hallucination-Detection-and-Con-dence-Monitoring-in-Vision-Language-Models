"""
09_confidence_monitor.py — Track confidence drift over sustained inference workloads.

Runs 2000+ consecutive inferences and logs per-request metrics to detect
whether hallucination behavior changes over time.

Usage:
    python scripts/09_confidence_monitor.py --quantization fp16 --num_samples 2000
"""

import sys
import os
import argparse
import time
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    load_pope_data, run_single_inference, build_prompt,
    get_gpu_memory_mb, get_gpu_memory_reserved_mb, get_system_stats,
    append_jsonl,
)

# Import entropy detector
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ent", os.path.join(os.path.dirname(__file__), "04_entropy_detector.py")
)
ent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ent_mod)
EntropyDetector = ent_mod.EntropyDetector


def extract_yes_no(answer: str) -> str:
    answer_lower = answer.lower().strip()
    if answer_lower.startswith("yes"):
        return "yes"
    elif answer_lower.startswith("no"):
        return "no"
    first_part = answer_lower[:20]
    if "yes" in first_part:
        return "yes"
    if "no" in first_part:
        return "no"
    return "unclear"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", default=None, help="fp16/int8/int4 (default: from config)")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--split", default="random")
    args = parser.parse_args()

    config = load_config()
    hw = config.get("hardware", {})
    quant = args.quantization or hw.get("default_quantization", "int4")
    num_samples = args.num_samples or config["experiment"]["num_drift_samples"]

    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    image_dir = config["paths"]["coco_images"]

    # Load POPE data — we'll cycle through it if num_samples > len(data)
    pope_data = load_pope_data(config["paths"]["pope_dir"], split=args.split)

    entropy_det = EntropyDetector(
        threshold=config["detection"]["entropy"]["initial_threshold"]
    )

    # Output
    out_dir = os.path.join(config["paths"]["output_dir"], "drift_monitor")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"drift_{quant}_{args.split}.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)

    print(f"\nConfidence drift monitor: {num_samples} consecutive inferences")
    print(f"Quantization: {quant}")
    print(f"Output: {out_file}\n")

    # Track running stats for windowed analysis
    window_size = 100
    recent_entropies = []
    recent_latencies = []
    recent_halluc_scores = []
    recent_correct = []

    start_time = time.time()

    for idx in tqdm(range(num_samples), desc="Drift monitor"):
        # Cycle through POPE data
        item = pope_data[idx % len(pope_data)]
        image_file = item["image"]
        question = item["text"]
        gt_label = item["label"]

        try:
            image = load_coco_image(image_dir, image_file)
        except FileNotFoundError:
            continue

        # Run inference
        t0 = time.perf_counter()
        result = run_single_inference(
            model, processor, image, question,
            max_new_tokens=20,
            output_scores=True,
        )
        t1 = time.perf_counter()

        pred_label = extract_yes_no(result["answer"])
        is_correct = (pred_label == gt_label)
        is_hallucination = (pred_label == "yes" and gt_label == "no")

        # Entropy detection
        entropy_result = entropy_det.detect(result["scores"])

        # System stats
        sys_stats = get_system_stats()

        # Build record
        record = {
            "idx": idx,
            "timestamp": time.time(),
            "elapsed_s": time.time() - start_time,
            "image": image_file,
            "question": question,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "is_hallucination": is_hallucination,
            # Entropy metrics
            "entropy_mean": entropy_result["entropy_mean"],
            "entropy_max": entropy_result["entropy_max"],
            "entropy_halluc_score": entropy_result["hallucination_score"],
            "top_prob_mean": entropy_result["top_prob_mean"],
            "high_entropy_ratio": entropy_result["high_entropy_ratio"],
            # Performance metrics
            "latency_ms": (t1 - t0) * 1000,
            "num_tokens": result["num_tokens"],
            # System metrics
            "gpu_mem_allocated_mb": sys_stats["gpu_mem_allocated_mb"],
            "gpu_mem_reserved_mb": sys_stats["gpu_mem_reserved_mb"],
            "cpu_percent": sys_stats["cpu_percent"],
            "ram_used_gb": sys_stats["ram_used_gb"],
        }

        append_jsonl(record, out_file)

        # Update running stats
        recent_entropies.append(entropy_result["entropy_mean"])
        recent_latencies.append((t1 - t0) * 1000)
        recent_halluc_scores.append(entropy_result["hallucination_score"])
        recent_correct.append(int(is_correct))

        # Keep window
        if len(recent_entropies) > window_size:
            recent_entropies.pop(0)
            recent_latencies.pop(0)
            recent_halluc_scores.pop(0)
            recent_correct.pop(0)

        # Log windowed stats every 200 samples
        if (idx + 1) % 200 == 0:
            w_entropy = np.mean(recent_entropies)
            w_latency = np.mean(recent_latencies)
            w_halluc = np.mean(recent_halluc_scores)
            w_acc = np.mean(recent_correct)
            gpu_mem = sys_stats["gpu_mem_allocated_mb"]

            tqdm.write(
                f"  [{idx+1:5d}] "
                f"entropy={w_entropy:.3f}  "
                f"halluc_score={w_halluc:.3f}  "
                f"acc={w_acc:.3f}  "
                f"latency={w_latency:.0f}ms  "
                f"gpu={gpu_mem:.0f}MB"
            )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Drift monitor complete: {num_samples} samples in {elapsed:.0f}s")
    print(f"Output: {out_file}")
    print(f"{'='*60}")
    print(f"\nNext: Analyze drift with scripts/11_analysis_plots.py")


if __name__ == "__main__":
    main()
