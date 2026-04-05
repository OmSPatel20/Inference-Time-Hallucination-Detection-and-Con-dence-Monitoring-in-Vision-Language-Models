"""
02_basic_inference.py — Load LLaVA and run a single image+question inference.

This is the "hello world" of the project. If this works, your setup is correct.

Usage:
    python scripts/02_basic_inference.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_config, load_model_and_processor, run_single_inference, load_coco_image


def main():
    config = load_config()

    # ------------------------------------------------------------------
    # Step 1: Load model (INT4 — fits 8GB VRAM)
    # ------------------------------------------------------------------
    quant = config.get("hardware", {}).get("default_quantization", "int4")
    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    # ------------------------------------------------------------------
    # Step 2: Load a test image from COCO
    # ------------------------------------------------------------------
    image_dir = config["paths"]["coco_images"]

    # Pick the first image in the directory
    images = sorted(os.listdir(image_dir))
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        print("Run scripts/01_download_data.py first.")
        return

    test_image_file = images[0]
    print(f"\nTest image: {test_image_file}")
    image = load_coco_image(image_dir, test_image_file)
    print(f"Image size: {image.size}")

    # ------------------------------------------------------------------
    # Step 3: Run inference with a simple question
    # ------------------------------------------------------------------
    questions = [
        "Is there a dog in this image?",
        "Describe this image in one sentence.",
        "What objects can you see in this image?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        result = run_single_inference(
            model, processor, image, q,
            max_new_tokens=config["model"]["max_new_tokens"],
        )

        print(f"Answer:   {result['answer']}")
        print(f"Tokens:   {result['num_tokens']}")
        print(f"Latency:  {result['latency_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # Step 4: Verify we can get logits (needed for entropy detection)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Verifying logit extraction...")
    print(f"{'='*60}")

    result = run_single_inference(
        model, processor, image,
        "Is there a person in this image?",
        max_new_tokens=20,
        output_scores=True,
    )

    print(f"Answer: {result['answer']}")
    print(f"Number of score tensors: {len(result['scores'])}")
    print(f"Each score tensor shape: {result['scores'][0].shape}")
    print(f"  → (batch_size, vocab_size) = {result['scores'][0].shape}")

    # ------------------------------------------------------------------
    # Step 5: Verify we can get attention weights (MEMORY WARNING)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Verifying attention extraction...")
    print("NOTE: Attention extraction uses a LOT of VRAM.")
    print("      On 8GB GPU, only safe with max_new_tokens <= 3")
    print(f"{'='*60}")

    result = run_single_inference(
        model, processor, image,
        "Is there a cat in this image?",
        max_new_tokens=3,  # Keep TINY — attention tensors are huge
        output_attentions=True,
    )

    print(f"Answer: {result['answer']}")
    if result["attentions"] is not None:
        # attentions is a tuple (one per generated token)
        # Each element is a tuple (one per layer)
        # Each layer tensor shape: (batch, num_heads, seq_len, seq_len)
        first_step = result["attentions"][0]  # First generated token
        print(f"Attention steps: {len(result['attentions'])}")
        print(f"Layers per step: {len(first_step)}")
        print(f"First layer shape: {first_step[0].shape}")
        print("  → (batch, heads, seq_len, seq_len)")
    else:
        print("WARNING: Attentions are None. Check model configuration.")

    print("\n✓ Basic inference working. Proceed to scripts/03_batched_pipeline.py")


if __name__ == "__main__":
    main()
