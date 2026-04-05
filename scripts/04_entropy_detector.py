"""
04_entropy_detector.py — Detect hallucinations via token-level entropy.

Core idea: When the model is confident, the softmax distribution is peaked (low entropy).
When it's uncertain or hallucinating, entropy is high.

Usage:
    python scripts/04_entropy_detector.py
    python scripts/04_entropy_detector.py --image_file COCO_val2014_000000000042.jpg --question "Is there a dog?"
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    run_single_inference,
)


class EntropyDetector:
    """
    Detect hallucinations by measuring token-level prediction entropy.

    High entropy → model is uncertain → possible hallucination
    Low entropy  → model is confident → likely grounded
    """

    def __init__(self, threshold: float = 1.5, percentile: int = 90):
        """
        Args:
            threshold:   Absolute entropy threshold for flagging tokens
            percentile:  Alternative: flag tokens above this percentile
        """
        self.threshold = threshold
        self.percentile = percentile

    def compute_token_entropies(self, scores: tuple) -> np.ndarray:
        """
        Compute entropy for each generated token from its logit distribution.

        Args:
            scores: Tuple of tensors, one per generation step.
                    Each tensor shape: (batch_size, vocab_size)

        Returns:
            numpy array of entropy values, shape (num_tokens,)
        """
        entropies = []
        for step_logits in scores:
            # step_logits shape: (batch_size, vocab_size)
            # Take first sample in batch
            logits = step_logits[0].float()  # (vocab_size,)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Compute entropy: H = -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()

            entropies.append(entropy)

        return np.array(entropies)

    def compute_token_top_probs(self, scores: tuple) -> np.ndarray:
        """
        Compute the probability of the most likely token at each step.
        Low top-1 probability = high uncertainty.

        Returns:
            numpy array of top-1 probabilities, shape (num_tokens,)
        """
        top_probs = []
        for step_logits in scores:
            logits = step_logits[0].float()
            probs = F.softmax(logits, dim=-1)
            top_p = probs.max().item()
            top_probs.append(top_p)
        return np.array(top_probs)

    def detect(self, scores: tuple, generated_ids: torch.Tensor = None,
               tokenizer=None) -> dict:
        """
        Run full entropy-based detection on one inference result.

        Args:
            scores:        Tuple of logit tensors from model.generate()
            generated_ids: Token IDs of generated text (for per-token reporting)
            tokenizer:     For decoding token IDs to strings

        Returns:
            dict with:
                - entropy_mean:       Mean entropy across all tokens
                - entropy_max:        Max entropy of any single token
                - entropy_std:        Standard deviation of entropies
                - high_entropy_ratio: Fraction of tokens above threshold
                - entropies:          Per-token entropy array
                - top_probs:          Per-token top-1 probability array
                - flagged_tokens:     List of (position, token_str, entropy) for flagged tokens
                - hallucination_score: Composite score (higher = more likely hallucination)
        """
        entropies = self.compute_token_entropies(scores)
        top_probs = self.compute_token_top_probs(scores)

        # Flag high-entropy tokens
        threshold = self.threshold
        if len(entropies) > 5:
            # Also consider percentile-based threshold
            pct_threshold = np.percentile(entropies, self.percentile)
            threshold = min(threshold, pct_threshold)

        flagged_mask = entropies > threshold
        high_entropy_ratio = flagged_mask.mean()

        # Decode flagged tokens
        flagged_tokens = []
        if generated_ids is not None and tokenizer is not None:
            for i in range(len(entropies)):
                if flagged_mask[i]:
                    token_str = tokenizer.decode([generated_ids[i].item()])
                    flagged_tokens.append({
                        "position": int(i),
                        "token": token_str,
                        "entropy": float(entropies[i]),
                    })

        # Composite hallucination score: weighted combination
        # Higher = more likely hallucination
        hallucination_score = (
            0.4 * (entropies.mean() / 5.0)           # Normalize to ~[0,1]
            + 0.3 * high_entropy_ratio                 # Fraction flagged
            + 0.3 * (1.0 - top_probs.mean())           # Low confidence
        )

        return {
            "entropy_mean": float(entropies.mean()),
            "entropy_max": float(entropies.max()),
            "entropy_std": float(entropies.std()),
            "entropy_median": float(np.median(entropies)),
            "top_prob_mean": float(top_probs.mean()),
            "top_prob_min": float(top_probs.min()),
            "high_entropy_ratio": float(high_entropy_ratio),
            "threshold_used": float(threshold),
            "num_tokens": int(len(entropies)),
            "flagged_tokens": flagged_tokens,
            "hallucination_score": float(hallucination_score),
            "entropies": entropies.tolist(),
            "top_probs": top_probs.tolist(),
        }

    def calibrate_threshold(self, all_entropies: list, all_labels: list) -> float:
        """
        Find the entropy threshold that maximizes F1 on a calibration set.

        Args:
            all_entropies: List of mean-entropy values per sample
            all_labels:    List of 0/1 labels (1 = hallucination)

        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import f1_score

        entropies = np.array(all_entropies)
        labels = np.array(all_labels)

        best_f1 = 0
        best_thresh = 0

        for thresh in np.linspace(entropies.min(), entropies.max(), 200):
            preds = (entropies > thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"Calibrated threshold: {best_thresh:.4f} (F1={best_f1:.4f})")
        self.threshold = best_thresh
        return best_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", default=None)
    parser.add_argument("--question", default="Is there a dog in this image?")
    args = parser.parse_args()

    config = load_config()
    quant = config.get("hardware", {}).get("default_quantization", "int4")
    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    image_dir = config["paths"]["coco_images"]
    if args.image_file:
        image_file = args.image_file
    else:
        image_file = sorted(os.listdir(image_dir))[0]

    image = load_coco_image(image_dir, image_file)
    detector = EntropyDetector(
        threshold=config["detection"]["entropy"]["initial_threshold"]
    )

    print(f"\nImage: {image_file}")
    print(f"Question: {args.question}")
    print()

    # Run inference with scores
    result = run_single_inference(
        model, processor, image, args.question,
        max_new_tokens=config["model"]["max_new_tokens"],
        output_scores=True,
    )

    # Run detection
    detection = detector.detect(
        result["scores"],
        result["generated_ids"],
        processor.tokenizer,
    )

    print(f"Answer: {result['answer']}")
    print(f"\n--- Entropy Detection Results ---")
    print(f"Mean entropy:        {detection['entropy_mean']:.4f}")
    print(f"Max entropy:         {detection['entropy_max']:.4f}")
    print(f"Std entropy:         {detection['entropy_std']:.4f}")
    print(f"Mean top-1 prob:     {detection['top_prob_mean']:.4f}")
    print(f"High entropy ratio:  {detection['high_entropy_ratio']:.4f}")
    print(f"Hallucination score: {detection['hallucination_score']:.4f}")
    print(f"Tokens generated:    {detection['num_tokens']}")

    if detection["flagged_tokens"]:
        print(f"\nFlagged tokens ({len(detection['flagged_tokens'])}):")
        for ft in detection["flagged_tokens"][:10]:
            print(f"  [{ft['position']:3d}] '{ft['token']}' → entropy={ft['entropy']:.3f}")

    # Quick demo: run on multiple questions to show contrast
    print(f"\n{'='*60}")
    print("Comparison: Questions with different hallucination risk")
    print(f"{'='*60}")

    test_questions = [
        "Describe the main colors in this image.",          # Low risk (visual)
        "Is there an elephant in this image?",              # Medium risk
        "What is the person thinking about?",               # High risk (hallucination-prone)
    ]

    for q in test_questions:
        r = run_single_inference(
            model, processor, image, q,
            max_new_tokens=50, output_scores=True,
        )
        d = detector.detect(r["scores"])
        print(f"\nQ: {q}")
        print(f"A: {r['answer'][:80]}...")
        print(f"   entropy_mean={d['entropy_mean']:.3f}  "
              f"halluc_score={d['hallucination_score']:.3f}")


if __name__ == "__main__":
    main()
