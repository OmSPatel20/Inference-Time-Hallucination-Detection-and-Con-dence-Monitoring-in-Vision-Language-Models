"""
06_contrastive_decoder.py — Detect hallucinations via visual contrastive decoding.

Core idea: Run inference with the original image AND a perturbed image.
If outputs are similar → model ignores vision → hallucination risk.
If outputs differ → model uses vision → grounded.

Usage:
    python scripts/06_contrastive_decoder.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    build_prompt,
)


class ContrastiveDecoder:
    """
    Detect hallucinations by comparing model outputs on original vs.
    perturbed images. If the model barely changes its answer when the
    image is degraded, it wasn't relying on vision in the first place.
    """

    def __init__(self, noise_sigma: float = 0.5, kl_threshold: float = 0.1):
        self.noise_sigma = noise_sigma
        self.kl_threshold = kl_threshold

    # ----- Perturbation methods -----

    def add_gaussian_noise(self, image: Image.Image, sigma: float = None) -> Image.Image:
        """Add Gaussian noise to image."""
        sigma = sigma or self.noise_sigma
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))

    def convert_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale and back to RGB."""
        return image.convert("L").convert("RGB")

    def reduce_resolution(self, image: Image.Image, factor: int = 4) -> Image.Image:
        """Downsample then upsample to lose detail."""
        w, h = image.size
        small = image.resize((w // factor, h // factor), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    def random_crop(self, image: Image.Image, keep_ratio: float = 0.7) -> Image.Image:
        """Random crop and resize back to original size."""
        w, h = image.size
        new_w = int(w * keep_ratio)
        new_h = int(h * keep_ratio)
        left = np.random.randint(0, w - new_w + 1)
        top = np.random.randint(0, h - new_h + 1)
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)

    def get_perturbation(self, image: Image.Image, method: str) -> Image.Image:
        """Apply a perturbation method by name."""
        methods = {
            "gaussian": self.add_gaussian_noise,
            "grayscale": self.convert_grayscale,
            "low_res": self.reduce_resolution,
            "crop": self.random_crop,
        }
        if method not in methods:
            raise ValueError(f"Unknown perturbation: {method}. Options: {list(methods.keys())}")
        return methods[method](image)

    # ----- Core detection -----

    def get_logits_for_image(self, model, processor, image, question, max_new_tokens=128):
        """
        Run inference and return per-step logits.

        Returns:
            (answer_str, list_of_logit_tensors)
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
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0, input_len:]
        answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer, outputs.scores

    def compute_kl_divergence(self, scores_original: tuple, scores_perturbed: tuple) -> np.ndarray:
        """
        Compute KL divergence between original and perturbed output distributions
        at each generation step.

        KL(P_orig || P_pert) measures how different the distributions are.
        Low KL → model gives same answer regardless of image → not using vision.
        High KL → model's answer depends on the image → visually grounded.

        Returns:
            numpy array of KL divergences, one per token
        """
        # Use the shorter sequence length
        num_steps = min(len(scores_original), len(scores_perturbed))
        kl_divs = []

        for step in range(num_steps):
            logits_orig = scores_original[step][0].float()
            logits_pert = scores_perturbed[step][0].float()

            # Convert to log-probabilities
            log_probs_orig = F.log_softmax(logits_orig, dim=-1)
            log_probs_pert = F.log_softmax(logits_pert, dim=-1)
            probs_orig = F.softmax(logits_orig, dim=-1)

            # KL(orig || pert) = sum(p_orig * (log_p_orig - log_p_pert))
            kl = F.kl_div(log_probs_pert, probs_orig, reduction="sum", log_target=False)
            kl_divs.append(kl.item())

        return np.array(kl_divs)

    def detect(self, model, processor, image: Image.Image, question: str,
               perturbation: str = "gaussian", max_new_tokens: int = 128) -> dict:
        """
        Run contrastive detection: compare original vs. perturbed image outputs.

        Args:
            model, processor: Loaded VLM
            image:            Original PIL image
            question:         Question string
            perturbation:     Perturbation method name
            max_new_tokens:   Max tokens to generate

        Returns:
            dict with detection results
        """
        # Generate with original image
        answer_orig, scores_orig = self.get_logits_for_image(
            model, processor, image, question, max_new_tokens
        )

        # Generate with perturbed image
        perturbed_image = self.get_perturbation(image, perturbation)
        answer_pert, scores_pert = self.get_logits_for_image(
            model, processor, perturbed_image, question, max_new_tokens
        )

        # Compute KL divergence per token
        kl_divs = self.compute_kl_divergence(scores_orig, scores_pert)

        # Answer match: did the answer change?
        answer_match = (
            answer_orig.lower().strip().startswith(answer_pert.lower().strip()[:3])
            or answer_pert.lower().strip().startswith(answer_orig.lower().strip()[:3])
        )

        # Hallucination score:
        # Low KL = same output regardless of image = not using vision = high halluc risk
        kl_mean = float(kl_divs.mean()) if len(kl_divs) > 0 else 0.0

        # Normalize: map KL to [0, 1] range with sigmoid-like function
        # kl_mean near 0 → score near 1 (hallucination)
        # kl_mean > 1 → score near 0 (grounded)
        hallucination_score = float(np.exp(-kl_mean))

        return {
            "answer_original": answer_orig,
            "answer_perturbed": answer_pert,
            "answer_match": answer_match,
            "perturbation": perturbation,
            "kl_mean": kl_mean,
            "kl_max": float(kl_divs.max()) if len(kl_divs) > 0 else 0.0,
            "kl_min": float(kl_divs.min()) if len(kl_divs) > 0 else 0.0,
            "kl_std": float(kl_divs.std()) if len(kl_divs) > 0 else 0.0,
            "low_kl_ratio": float((kl_divs < self.kl_threshold).mean()) if len(kl_divs) > 0 else 1.0,
            "hallucination_score": hallucination_score,
            "kl_divergences": kl_divs.tolist(),
            "num_steps_compared": int(len(kl_divs)),
        }

    def detect_multi_perturbation(self, model, processor, image, question,
                                   perturbations=None, max_new_tokens=128) -> dict:
        """
        Run contrastive detection with multiple perturbation types and aggregate.
        """
        perturbations = perturbations or ["gaussian", "grayscale", "low_res"]
        results = {}
        kl_means = []

        for pert in perturbations:
            r = self.detect(model, processor, image, question, pert, max_new_tokens)
            results[pert] = r
            kl_means.append(r["kl_mean"])

        # Aggregate: use minimum KL across perturbations (worst case)
        min_kl = min(kl_means)
        avg_kl = np.mean(kl_means)

        return {
            "per_perturbation": results,
            "aggregate_kl_mean": float(avg_kl),
            "aggregate_kl_min": float(min_kl),
            "hallucination_score": float(np.exp(-min_kl)),
        }


def main():
    config = load_config()
    quant = config.get("hardware", {}).get("default_quantization", "int4")
    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    image_dir = config["paths"]["coco_images"]
    image_file = sorted(os.listdir(image_dir))[0]
    image = load_coco_image(image_dir, image_file)

    detector = ContrastiveDecoder(
        noise_sigma=config["detection"]["contrastive"]["noise_sigma"],
        kl_threshold=config["detection"]["contrastive"]["kl_threshold"],
    )

    test_cases = [
        "What objects are visible in this image?",
        "Is there a giraffe in this image?",
        "What emotion is the person feeling?",
    ]

    for q in test_cases:
        print(f"\n{'='*60}")
        print(f"Q: {q}")

        result = detector.detect(
            model, processor, image, q,
            perturbation="gaussian",
            max_new_tokens=50,
        )

        print(f"A (original):  {result['answer_original'][:80]}")
        print(f"A (perturbed): {result['answer_perturbed'][:80]}")
        print(f"Answer match:  {result['answer_match']}")
        print(f"KL mean:       {result['kl_mean']:.4f}")
        print(f"Halluc. score: {result['hallucination_score']:.4f}")

    # Multi-perturbation demo
    print(f"\n{'='*60}")
    print("Multi-perturbation analysis:")
    q = "Is there a cat in this image?"
    result = detector.detect_multi_perturbation(
        model, processor, image, q, max_new_tokens=30,
    )
    print(f"Q: {q}")
    for pert, r in result["per_perturbation"].items():
        print(f"  {pert:12s}: KL={r['kl_mean']:.4f}  halluc={r['hallucination_score']:.4f}")
    print(f"  Aggregate:    KL={result['aggregate_kl_mean']:.4f}  "
          f"halluc={result['hallucination_score']:.4f}")

    print("\n✓ Contrastive decoder working.")


if __name__ == "__main__":
    main()
