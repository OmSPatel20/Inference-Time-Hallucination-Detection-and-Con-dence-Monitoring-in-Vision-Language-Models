"""
08_chair_evaluation.py — CHAIR (Caption Hallucination Assessment with Image Relevance).

Evaluates free-form image captions for hallucinated objects.

Usage:
    python scripts/08_chair_evaluation.py --quantization fp16 --num_samples 500
"""

import sys
import os
import re
import argparse
import json
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_config, load_model_and_processor, load_coco_image,
    run_single_inference, get_coco_image_objects, save_jsonl, append_jsonl,
)


# COCO 80 categories + common synonyms
COCO_OBJECTS = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "people",
               "guy", "lady", "gentleman", "player", "rider", "skier", "snowboarder",
               "surfer", "pedestrian"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "vehicle", "sedan", "suv", "truck"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus"],
    "train": ["train", "locomotive"],
    "truck": ["truck", "pickup"],
    "boat": ["boat", "ship", "vessel", "canoe", "kayak", "sailboat"],
    "traffic light": ["traffic light", "stoplight", "signal"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench", "seat"],
    "bird": ["bird", "pigeon", "seagull", "parrot", "eagle", "duck", "goose"],
    "cat": ["cat", "kitten", "kitty", "feline"],
    "dog": ["dog", "puppy", "canine", "pup", "hound"],
    "horse": ["horse", "pony", "stallion", "mare"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull", "calf"],
    "elephant": ["elephant"],
    "bear": ["bear", "teddy bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "bag", "rucksack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "bag"],
    "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage", "briefcase"],
    "frisbee": ["frisbee", "disc"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "baseball", "basketball", "football", "soccer ball", "tennis ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "glass", "goblet"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake"],
    "chair": ["chair", "seat"],
    "couch": ["couch", "sofa", "loveseat"],
    "potted plant": ["potted plant", "plant", "flower", "vase"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "computer", "notebook"],
    "mouse": ["mouse"],
    "remote": ["remote", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "phone", "cellphone", "mobile", "smartphone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink", "basin"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal"],
    "hair drier": ["hair drier", "hairdryer", "dryer"],
    "toothbrush": ["toothbrush"],
}

# Build reverse lookup: synonym → canonical category
SYNONYM_TO_CATEGORY = {}
for category, synonyms in COCO_OBJECTS.items():
    for syn in synonyms:
        SYNONYM_TO_CATEGORY[syn.lower()] = category


def extract_mentioned_objects(caption: str) -> set:
    """
    Extract COCO object categories mentioned in a caption.

    Returns set of canonical category names.
    """
    caption_lower = caption.lower()
    mentioned = set()

    # Check each synonym (longer phrases first to avoid partial matches)
    all_synonyms = sorted(SYNONYM_TO_CATEGORY.keys(), key=len, reverse=True)

    for synonym in all_synonyms:
        # Word boundary matching
        pattern = r'\b' + re.escape(synonym) + r'\b'
        if re.search(pattern, caption_lower):
            category = SYNONYM_TO_CATEGORY[synonym]
            mentioned.add(category)

    return mentioned


def compute_chair_metrics(captions_data: list) -> dict:
    """
    Compute CHAIR metrics across all captions.

    CHAIR_i = (total hallucinated objects) / (total mentioned objects)
    CHAIR_s = (sentences with >= 1 hallucination) / (total sentences)

    Args:
        captions_data: List of dicts with 'mentioned_objects', 'gt_objects', 'hallucinated_objects'

    Returns:
        dict with CHAIR_i, CHAIR_s, and per-category stats
    """
    total_mentioned = 0
    total_hallucinated = 0
    total_sentences = 0
    sentences_with_halluc = 0
    halluc_by_category = defaultdict(int)
    mention_by_category = defaultdict(int)

    for item in captions_data:
        mentioned = item["mentioned_objects"]
        hallucinated = item["hallucinated_objects"]

        total_mentioned += len(mentioned)
        total_hallucinated += len(hallucinated)
        total_sentences += 1

        if len(hallucinated) > 0:
            sentences_with_halluc += 1

        for obj in mentioned:
            mention_by_category[obj] += 1
        for obj in hallucinated:
            halluc_by_category[obj] += 1

    chair_i = total_hallucinated / max(total_mentioned, 1)
    chair_s = sentences_with_halluc / max(total_sentences, 1)

    # Per-category hallucination rate
    category_rates = {}
    for cat in mention_by_category:
        rate = halluc_by_category.get(cat, 0) / mention_by_category[cat]
        category_rates[cat] = {
            "mentions": mention_by_category[cat],
            "hallucinations": halluc_by_category.get(cat, 0),
            "rate": rate,
        }

    return {
        "CHAIR_i": chair_i,
        "CHAIR_s": chair_s,
        "total_mentioned": total_mentioned,
        "total_hallucinated": total_hallucinated,
        "total_sentences": total_sentences,
        "sentences_with_halluc": sentences_with_halluc,
        "category_rates": category_rates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", default=None, help="fp16/int8/int4 (default: from config)")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    hw = config.get("hardware", {})
    quant = quant or hw.get("default_quantization", "int4")
    num_samples = args.num_samples or config["experiment"]["num_chair_samples"]

    model, processor = load_model_and_processor(
        model_id=config["paths"]["model_id"],
        quantization=quant,
    )

    image_dir = config["paths"]["coco_images"]
    annotations_file = config["paths"]["coco_annotations"]

    # Load ground-truth objects per image
    print("Loading COCO annotations...")
    gt_objects_by_id = get_coco_image_objects(annotations_file)

    # Get image files
    image_files = sorted(os.listdir(image_dir))[:num_samples]

    # Output
    out_dir = os.path.join(config["paths"]["output_dir"], "chair_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"chair_{quant}.jsonl")
    if os.path.exists(out_file):
        os.remove(out_file)

    print(f"\nRunning CHAIR evaluation: {num_samples} images, quant={quant}")
    caption_prompt = "Describe this image in detail."

    captions_data = []

    for img_file in tqdm(image_files, desc="CHAIR"):
        try:
            image = load_coco_image(image_dir, img_file)
        except Exception:
            continue

        # Extract image ID from filename: COCO_val2014_000000XXXXXX.jpg
        try:
            image_id = int(img_file.split("_")[-1].split(".")[0])
        except ValueError:
            continue

        # Get ground truth objects
        gt_objects = gt_objects_by_id.get(image_id, set())

        # Generate caption
        result = run_single_inference(
            model, processor, image, caption_prompt,
            max_new_tokens=150,
            output_scores=True,
        )

        caption = result["answer"]
        mentioned = extract_mentioned_objects(caption)
        hallucinated = mentioned - gt_objects

        # Entropy score for this caption
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ent_det", os.path.join(os.path.dirname(__file__), "04_entropy_detector.py")
        )
        ent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ent_mod)
        entropy_det = ent_mod.EntropyDetector()
        entropy_result = entropy_det.detect(result["scores"])

        record = {
            "image_id": image_id,
            "image_file": img_file,
            "caption": caption,
            "mentioned_objects": sorted(list(mentioned)),
            "gt_objects": sorted(list(gt_objects)),
            "hallucinated_objects": sorted(list(hallucinated)),
            "num_mentioned": len(mentioned),
            "num_hallucinated": len(hallucinated),
            "has_hallucination": len(hallucinated) > 0,
            "entropy_mean": entropy_result["entropy_mean"],
            "entropy_halluc_score": entropy_result["hallucination_score"],
            "latency_ms": result["latency_ms"],
        }

        captions_data.append(record)
        append_jsonl(record, out_file)

    # Compute CHAIR metrics
    metrics = compute_chair_metrics(captions_data)

    summary = {
        "quantization": quant,
        "num_samples": len(captions_data),
        **metrics,
    }
    # Remove non-serializable
    summary.pop("category_rates", None)

    summary_file = os.path.join(out_dir, f"chair_summary_{quant}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CHAIR Results ({quant}):")
    print(f"  CHAIR_i:  {metrics['CHAIR_i']:.4f}  (object-level hallucination rate)")
    print(f"  CHAIR_s:  {metrics['CHAIR_s']:.4f}  (sentence-level hallucination rate)")
    print(f"  Total objects mentioned:     {metrics['total_mentioned']}")
    print(f"  Total objects hallucinated:  {metrics['total_hallucinated']}")
    print(f"  Captions with hallucination: {metrics['sentences_with_halluc']}/{metrics['total_sentences']}")
    print(f"\n  Results: {out_file}")
    print(f"  Summary: {summary_file}")

    # Top hallucinated categories
    if metrics["category_rates"]:
        print(f"\n  Most hallucinated categories:")
        sorted_cats = sorted(metrics["category_rates"].items(),
                              key=lambda x: x[1]["hallucinations"], reverse=True)
        for cat, stats in sorted_cats[:10]:
            if stats["hallucinations"] > 0:
                print(f"    {cat:20s}: {stats['hallucinations']}/{stats['mentions']} "
                      f"({stats['rate']:.2%})")


if __name__ == "__main__":
    main()
