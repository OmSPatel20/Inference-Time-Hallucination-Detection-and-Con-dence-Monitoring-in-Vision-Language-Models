"""
03_batched_pipeline.py — Batched inference at different quantization levels.

Measures throughput (tokens/sec) and memory usage across configurations.

Usage:
    python scripts/03_batched_pipeline.py
    python scripts/03_batched_pipeline.py --quantization int4 --batch_size 8
"""

import sys
import os
import argparse
import time
import json
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    build_prompt, get_gpu_memory_mb, get_system_stats, save_jsonl,
)


def run_batched_inference(model, processor, images, questions, max_new_tokens=128):
    """
    Run inference on a batch of image+question pairs.

    Args:
        images:    List of PIL images
        questions: List of question strings

    Returns:
        dict with answers, latency, throughput
    """
    prompts = [build_prompt(q) for q in questions]

    # Process batch
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t1 = time.perf_counter()

    # Decode each answer
    input_len = inputs["input_ids"].shape[1]
    answers = []
    total_tokens = 0
    for i in range(len(questions)):
        gen_ids = outputs[i, input_len:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        answers.append(answer)
        # Count non-pad tokens
        non_pad = (gen_ids != processor.tokenizer.pad_token_id).sum().item()
        total_tokens += non_pad

    elapsed = t1 - t0
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    return {
        "answers": answers,
        "total_tokens": total_tokens,
        "latency_s": elapsed,
        "throughput_tok_per_s": throughput,
        "gpu_mem_mb": get_gpu_memory_mb(),
    }


def benchmark_config(model, processor, image_dir, config, batch_size, num_batches=5):
    """Run multiple batches and report average metrics."""

    # Load a pool of images
    image_files = sorted(os.listdir(image_dir))[:batch_size * num_batches]
    if len(image_files) < batch_size:
        print(f"  WARNING: Only {len(image_files)} images available, need {batch_size}")
        return None

    question = "Is there a person in this image?"
    max_new_tokens = config["model"]["max_new_tokens"]

    results = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_files = image_files[start:end]

        images = [load_coco_image(image_dir, f) for f in batch_files]
        questions = [question] * batch_size

        r = run_batched_inference(model, processor, images, questions, max_new_tokens)
        results.append(r)

        # Print first batch answers as sanity check
        if batch_idx == 0:
            print(f"  Sample answer: {r['answers'][0][:80]}...")

    # Aggregate
    avg_throughput = sum(r["throughput_tok_per_s"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    gpu_mem = results[-1]["gpu_mem_mb"]

    return {
        "avg_throughput_tok_per_s": avg_throughput,
        "avg_latency_s": avg_latency,
        "gpu_mem_mb": gpu_mem,
        "num_batches": num_batches,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", default=None, help="fp16/int8/int4 (default: run all)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: run all)")
    args = parser.parse_args()

    config = load_config()
    image_dir = config["paths"]["coco_images"]

    # Determine which configs to run (respecting hardware limits)
    hw = config.get("hardware", {})
    max_bs = hw.get("max_batch_size", 32)

    if args.quantization:
        quants = [args.quantization]
    else:
        quants = config["experiment"]["quantizations"]

    if args.batch_size:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = [bs for bs in config["experiment"]["batch_sizes"] if bs <= max_bs]

    if not batch_sizes:
        batch_sizes = [1]

    print(f"Hardware: {hw.get('vram_gb', '?')}GB VRAM")
    print(f"Running: quantizations={quants}, batch_sizes={batch_sizes}")

    all_results = []

    for quant in quants:
        print(f"\n{'='*60}")
        print(f"QUANTIZATION: {quant}")
        print(f"{'='*60}")

        # Load model with this quantization
        model, processor = load_model_and_processor(
            model_id=config["paths"]["model_id"],
            quantization=quant,
        )

        for bs in batch_sizes:
            print(f"\n  Batch size: {bs}")

            # Warm up (first inference is always slower)
            images = [load_coco_image(image_dir, sorted(os.listdir(image_dir))[0])]
            _ = run_batched_inference(model, processor, images, ["test"], max_new_tokens=10)

            # Benchmark
            result = benchmark_config(model, processor, image_dir, config, bs, num_batches=5)
            if result:
                result["quantization"] = quant
                all_results.append(result)
                print(f"  Throughput: {result['avg_throughput_tok_per_s']:.1f} tok/s")
                print(f"  Latency:   {result['avg_latency_s']:.2f} s/batch")
                print(f"  GPU mem:   {result['gpu_mem_mb']:.0f} MB")

        # Free GPU memory before loading next quantization
        del model
        torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(config["paths"]["output_dir"], "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Quant':<8} {'Batch':<8} {'Tok/s':<12} {'Latency':<12} {'GPU MB':<10}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"{r['quantization']:<8} {r['batch_size']:<8} "
              f"{r['avg_throughput_tok_per_s']:<12.1f} "
              f"{r['avg_latency_s']:<12.2f} "
              f"{r['gpu_mem_mb']:<10.0f}")


if __name__ == "__main__":
    main()
