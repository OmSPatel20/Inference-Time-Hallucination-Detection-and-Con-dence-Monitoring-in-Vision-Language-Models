"""
07_pope_evaluation.py — Run POPE benchmark with all three detection methods.

For each yes/no question:
  1. Get the model's answer
  2. Compute entropy, attention, and contrastive scores
  3. Compare with ground truth to determine if the model hallucinated
  4. Save everything for analysis

Usage:
    python scripts/07_pope_evaluation.py --split random --quantization fp16
    python scripts/07_pope_evaluation.py --split adversarial --quantization int4 --max_samples 500
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
    get_gpu_memory_mb, save_jsonl, append_jsonl,
)
from _04_entropy_detector import EntropyDetector
from _06_contrastive_decoder import ContrastiveDecoder


def extract_yes_no(answer: str) -> str:
    """Extract yes/no from model answer. Returns 'yes', 'no', or 'unclear'."""
    answer_lower = answer.lower().strip()
    if answer_lower.startswith("yes"):
        return "yes"
    elif answer_lower.startswith("no"):
        return "no"
    # Check if yes/no appears anywhere in first 20 chars
    first_part = answer_lower[:20]
    if "yes" in first_part:
        return "yes"
    if "no" in first_part:
        return "no"
    return "unclear"


def run_pope_evaluation(
    model, processor, config,
    split: str = "random",
    quantization: str = "fp16",
    max_samples: int = None,
    use_attention: bool = False,  # Attention is expensive, make optional
    use_contrastive: bool = True,
):
    """
    Run POPE evaluation with detection methods.

    Args:
        split:           POPE split (random, popular, adversarial)
        quantization:    For labeling results
        max_samples:     Limit samples for testing
        use_attention:   Whether to run attention detector (slow + memory-heavy)
        use_contrastive: Whether to run contrastive decoder (2x inference cost)
    """
    pope_data = load_pope_data(
        config["paths"]["pope_dir"],
        split=split,
        max_samples=max_samples or config["experiment"]["num_pope_samples"],
    )
    image_dir = config["paths"]["coco_images"]

    # Initialize detectors
    entropy_det = EntropyDetector(
        threshold=config["detection"]["entropy"]["initial_threshold"]
    )
    contrastive_det = ContrastiveDecoder(
        noise_sigma=config["detection"]["contrastive"]["noise_sigma"],
        kl_threshold=config["detection"]["contrastive"]["kl_threshold"],
    ) if use_contrastive else None

    # Output file
    out_dir = os.path.join(config["paths"]["output_dir"], "pope_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"pope_{split}_{quantization}.jsonl")

    # Clear previous results
    if os.path.exists(out_file):
        os.remove(out_file)

    print(f"\nRunning POPE evaluation: split={split}, quant={quantization}")
    print(f"Samples: {len(pope_data)}")
    print(f"Detectors: entropy=True, attention={use_attention}, contrastive={use_contrastive}")
    print(f"Output: {out_file}\n")

    correct = 0
    total = 0
    hallucinations = 0

    for idx, item in enumerate(tqdm(pope_data, desc=f"POPE-{split}")):
        image_file = item["image"]
        question = item["text"]
        gt_label = item["label"]  # "yes" or "no"

        try:
            image = load_coco_image(image_dir, image_file)
        except FileNotFoundError:
            continue

        # --- Main inference with scores ---
        result = run_single_inference(
            model, processor, image, question,
            max_new_tokens=20,  # yes/no answers are short
            output_scores=True,
        )

        pred_label = extract_yes_no(result["answer"])
        is_correct = (pred_label == gt_label)
        is_hallucination = (pred_label == "yes" and gt_label == "no")

        correct += int(is_correct)
        total += 1
        hallucinations += int(is_hallucination)

        # --- Entropy detection ---
        entropy_result = entropy_det.detect(result["scores"])

        # --- Contrastive detection (optional) ---
        contrastive_result = {}
        if contrastive_det:
            try:
                contrastive_result = contrastive_det.detect(
                    model, processor, image, question,
                    perturbation="gaussian",
                    max_new_tokens=20,
                )
                # Remove verbose fields
                contrastive_result.pop("kl_divergences", None)
                contrastive_result.pop("answer_original", None)
                contrastive_result.pop("answer_perturbed", None)
            except Exception as e:
                contrastive_result = {"error": str(e)}

        # --- Save record ---
        record = {
            "idx": idx,
            "image": image_file,
            "question": question,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "answer_raw": result["answer"],
            "is_correct": is_correct,
            "is_hallucination": is_hallucination,
            "latency_ms": result["latency_ms"],
            "num_tokens": result["num_tokens"],
            "gpu_mem_mb": get_gpu_memory_mb(),
            # Entropy scores
            "entropy_mean": entropy_result["entropy_mean"],
            "entropy_max": entropy_result["entropy_max"],
            "entropy_halluc_score": entropy_result["hallucination_score"],
            "top_prob_mean": entropy_result["top_prob_mean"],
            "high_entropy_ratio": entropy_result["high_entropy_ratio"],
            # Contrastive scores
            "contrastive_kl_mean": contrastive_result.get("kl_mean", None),
            "contrastive_halluc_score": contrastive_result.get("hallucination_score", None),
            "contrastive_answer_match": contrastive_result.get("answer_match", None),
            # Metadata
            "quantization": quantization,
            "split": split,
        }
        append_jsonl(record, out_file)

        # Progress logging every 100 samples
        if (idx + 1) % 100 == 0:
            acc = correct / total if total > 0 else 0
            halluc_rate = hallucinations / total if total > 0 else 0
            tqdm.write(f"  [{idx+1}/{len(pope_data)}] acc={acc:.3f} halluc_rate={halluc_rate:.3f}")

    # --- Summary ---
    accuracy = correct / total if total > 0 else 0
    halluc_rate = hallucinations / total if total > 0 else 0

    summary = {
        "split": split,
        "quantization": quantization,
        "total_samples": total,
        "accuracy": accuracy,
        "hallucination_rate": halluc_rate,
        "correct": correct,
        "hallucinations": hallucinations,
    }

    summary_file = os.path.join(out_dir, f"summary_{split}_{quantization}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"POPE {split} ({quantization}) Results:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Hallucination rate: {halluc_rate:.4f}")
    print(f"  Total samples:     {total}")
    print(f"  Results: {out_file}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random", choices=["random", "popular", "adversarial"])
    parser.add_argument("--quantization", default=None, help="fp16/int8/int4 (default: from config)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_contrastive", action="store_true", help="Skip contrastive detection")
    parser.add_argument("--use_attention", action="store_true", help="Enable attention detection")
    args = parser.parse_args()

    config = load_config()
    hw = config.get("hardware", {})

    # Default quantization from config (int4 for laptop)
    quant = args.quantization or hw.get("default_quantization", "int4")

    # Respect hardware flags unless overridden by CLI
    use_contrastive = hw.get("enable_contrastive_detector", True) and not args.no_contrastive
    use_attention = args.use_attention and hw.get("enable_attention_detector", False)

    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    run_pope_evaluation(
        model, processor, config,
        split=args.split,
        quantization=quant,
        max_samples=args.max_samples,
        use_attention=use_attention,
        use_contrastive=use_contrastive,
    )


if __name__ == "__main__":
    # Handle the import issue: scripts use filenames starting with numbers
    # Python can't import "04_entropy_detector" directly
    # So we rename them in the import with aliases
    import importlib.util

    for script_name, alias in [
        ("04_entropy_detector", "_04_entropy_detector"),
        ("06_contrastive_decoder", "_06_contrastive_decoder"),
    ]:
        script_path = os.path.join(os.path.dirname(__file__), f"{script_name}.py")
        spec = importlib.util.spec_from_file_location(alias, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)

    main()
