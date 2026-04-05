"""
05_attention_detector.py — Detect hallucinations via visual attention analysis.

Core idea: If the model isn't attending to image tokens when generating object names,
it's generating from language priors → hallucination.

Usage:
    python scripts/05_attention_detector.py
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    build_prompt,
)


class AttentionDetector:
    """
    Detect hallucinations by analyzing how much the model attends to
    image tokens vs text tokens during generation.

    In LLaVA-1.5, the input sequence is:
        [BOS] [system tokens] [IMAGE_TOKENS x 576] [question tokens] [generated tokens]

    We measure what fraction of attention goes to the image token positions.
    """

    def __init__(self, num_image_tokens: int = 576, layers: list = None,
                 min_visual_ratio: float = 0.15):
        """
        Args:
            num_image_tokens: Number of image tokens (576 for LLaVA-1.5 with 336px)
            layers:           Which layers to analyze (negative indices from end)
            min_visual_ratio: Below this ratio → flagged as hallucination
        """
        self.num_image_tokens = num_image_tokens
        self.layers = layers or [-1, -2, -3, -4]
        self.min_visual_ratio = min_visual_ratio

    def find_image_token_positions(self, input_ids: torch.Tensor, 
                                    image_token_id: int = 32000) -> tuple:
        """
        Find the start and end positions of image tokens in the input.

        LLaVA replaces the <image> placeholder with 576 image embedding tokens.
        After processing, these occupy consecutive positions in the sequence.

        Returns:
            (start_pos, end_pos) — slice of image token positions
        """
        # In the processed input, image tokens are typically inserted after
        # the system prompt. We can find them by looking for the image token ID
        # or by knowing the structure.
        #
        # For LLaVA-1.5-hf, after processor, image tokens start after the
        # initial text tokens. The safest approach: the total input length
        # minus question tokens gives us the image region.
        #
        # Simpler heuristic: image tokens are positions 1 to 576 (after BOS)
        # or we can detect them by the special token ID.

        ids = input_ids.cpu().numpy()

        # Look for image token ID (32000 in LLaVA-1.5)
        image_positions = np.where(ids == image_token_id)[0]

        if len(image_positions) > 0:
            start = int(image_positions[0])
            end = int(image_positions[-1]) + 1
            return start, end

        # Fallback: assume image tokens start after the first few text tokens
        # and span num_image_tokens positions
        # This is approximate but works for most LLaVA inputs
        start = 5  # After BOS + system tokens
        end = start + self.num_image_tokens
        return start, min(end, len(ids))

    def compute_visual_attention_ratio(
        self, attentions: tuple, input_ids: torch.Tensor,
        num_input_tokens: int,
    ) -> dict:
        """
        For each generated token, compute what fraction of attention
        goes to image tokens vs text tokens.

        Args:
            attentions:       Tuple of attention tuples from generate()
                              attentions[step][layer] shape: (batch, heads, 1, seq_len)
            input_ids:        Input token IDs (to find image token positions)
            num_input_tokens: Length of the input sequence (before generation)

        Returns:
            dict with per-token and aggregate visual attention metrics
        """
        img_start, img_end = self.find_image_token_positions(input_ids)
        num_img = img_end - img_start

        per_token_visual_ratios = []
        per_token_text_ratios = []

        num_steps = len(attentions)

        for step_idx in range(num_steps):
            step_attns = attentions[step_idx]  # Tuple of layer tensors

            # Average across selected layers
            visual_ratios_per_layer = []
            for layer_idx in self.layers:
                if abs(layer_idx) > len(step_attns):
                    continue

                # Shape: (batch, num_heads, seq_len, seq_len)
                # For generation step, the last token's attention is what matters
                attn = step_attns[layer_idx][0]  # Remove batch dim

                # Get attention from the last (newly generated) token
                # Shape: (num_heads, seq_len)
                last_token_attn = attn[:, -1, :]  # All heads, last query position

                # Average across heads
                avg_attn = last_token_attn.mean(dim=0).float().cpu().numpy()

                # Sum attention on image tokens
                img_attn = avg_attn[img_start:img_end].sum()
                total_attn = avg_attn.sum()

                if total_attn > 0:
                    visual_ratio = img_attn / total_attn
                else:
                    visual_ratio = 0.0

                visual_ratios_per_layer.append(visual_ratio)

            # Average across layers
            if visual_ratios_per_layer:
                step_visual_ratio = float(np.mean(visual_ratios_per_layer))
            else:
                step_visual_ratio = 0.0

            per_token_visual_ratios.append(step_visual_ratio)
            per_token_text_ratios.append(1.0 - step_visual_ratio)

        visual_ratios = np.array(per_token_visual_ratios)
        text_ratios = np.array(per_token_text_ratios)

        # Flag tokens with low visual attention
        flagged_mask = visual_ratios < self.min_visual_ratio
        low_visual_ratio = float(flagged_mask.mean()) if len(flagged_mask) > 0 else 0.0

        # Composite hallucination score
        # Higher = more likely hallucinating (less visual grounding)
        hallucination_score = 1.0 - visual_ratios.mean() if len(visual_ratios) > 0 else 1.0

        return {
            "visual_attn_mean": float(visual_ratios.mean()) if len(visual_ratios) > 0 else 0.0,
            "visual_attn_min": float(visual_ratios.min()) if len(visual_ratios) > 0 else 0.0,
            "visual_attn_std": float(visual_ratios.std()) if len(visual_ratios) > 0 else 0.0,
            "text_attn_mean": float(text_ratios.mean()) if len(text_ratios) > 0 else 1.0,
            "low_visual_ratio": low_visual_ratio,
            "hallucination_score": float(hallucination_score),
            "per_token_visual_ratios": visual_ratios.tolist(),
            "image_token_range": [int(img_start), int(img_end)],
            "num_image_tokens_found": int(num_img),
        }


def run_with_attention(model, processor, image, question, max_new_tokens=128):
    """
    Run inference and extract attention weights.
    
    NOTE: For long generations, attention tensors consume massive GPU memory.
    Keep max_new_tokens small (≤50) when extracting attention.
    """
    prompt = build_prompt(question)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "input_ids": inputs["input_ids"][0],
        "generated_ids": generated_ids,
        "attentions": outputs.attentions,
        "num_input_tokens": input_len,
    }


def main():
    config = load_config()

    # VRAM guard
    hw = config.get("hardware", {})
    if not hw.get("enable_attention_detector", True):
        print("=" * 60)
        print("ATTENTION DETECTOR IS DISABLED IN CONFIG")
        print(f"Your GPU has {hw.get('vram_gb', '?')}GB VRAM.")
        print("Attention extraction stores full attention tensors in VRAM")
        print("and will OOM on GPUs with < 16GB.")
        print()
        print("To force-run anyway (may crash):")
        print("  Set hardware.enable_attention_detector: true in config")
        print("  OR pass --force flag")
        print("=" * 60)

        # Still allow --force override
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()
        if not args.force:
            return

    quant = hw.get("default_quantization", "int4")
    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    image_dir = config["paths"]["coco_images"]
    image_file = sorted(os.listdir(image_dir))[0]
    image = load_coco_image(image_dir, image_file)

    detector = AttentionDetector(
        num_image_tokens=config["model"]["num_image_tokens"],
        layers=config["detection"]["attention"]["layers"],
        min_visual_ratio=config["detection"]["attention"]["min_visual_ratio"],
    )

    # --- Demo: Compare grounded vs. hallucination-prone questions ---
    test_cases = [
        "What color is the largest object in this image?",   # Grounded
        "Is there a dog in this image?",                      # May or may not be
        "What is this person's name?",                        # Hallucination-prone
    ]

    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"Q: {q}")

        # Use small max_new_tokens to manage memory
        result = run_with_attention(model, processor, image, q, max_new_tokens=30)

        if result["attentions"] is None or len(result["attentions"]) == 0:
            print("  WARNING: No attention weights returned.")
            continue

        detection = detector.compute_visual_attention_ratio(
            result["attentions"],
            result["input_ids"],
            result["num_input_tokens"],
        )

        print(f"A: {result['answer']}")
        print(f"  Visual attn mean: {detection['visual_attn_mean']:.4f}")
        print(f"  Visual attn min:  {detection['visual_attn_min']:.4f}")
        print(f"  Low-visual ratio: {detection['low_visual_ratio']:.4f}")
        print(f"  Halluc. score:    {detection['hallucination_score']:.4f}")
        print(f"  Image tokens:     [{detection['image_token_range'][0]}-"
              f"{detection['image_token_range'][1]}] "
              f"({detection['num_image_tokens_found']} tokens)")

        # Show per-token visual attention for first 10 tokens
        ratios = detection["per_token_visual_ratios"][:10]
        tokens = result["generated_ids"][:10]
        print(f"  Per-token visual attention (first 10):")
        for i, (ratio, tid) in enumerate(zip(ratios, tokens)):
            token_str = processor.tokenizer.decode([tid.item()])
            bar = "█" * int(ratio * 40)
            print(f"    [{i:2d}] {ratio:.3f} {bar:<20} '{token_str}'")

    print("\n✓ Attention detector working.")
    print("  NOTE: For full POPE evaluation, use max_new_tokens=20 to manage memory.")


if __name__ == "__main__":
    main()
