"""
01_download_data.py — Download COCO images, POPE benchmark, and pre-cache the model.

Run this FIRST before anything else.

Usage:
    python scripts/01_download_data.py
"""

import os
import json
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  [SKIP] {desc or dest} already exists")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {desc or url} ...")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1024 / 1024
            print(f"\r    {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()  # newline after progress


def main():
    base_dir = Path("data")

    # ------------------------------------------------------------------
    # 1. COCO 2014 validation images (~6GB, 40,504 images)
    # ------------------------------------------------------------------
    print("\n=== Step 1: COCO 2014 Validation Images ===")
    coco_dir = base_dir / "coco"
    coco_zip = coco_dir / "val2014.zip"
    coco_images = coco_dir / "val2014"

    if not coco_images.exists():
        download_file(
            "http://images.cocodataset.org/zips/val2014.zip",
            str(coco_zip),
            "COCO val2014 images (~6GB)"
        )
        print("  Extracting...")
        with zipfile.ZipFile(str(coco_zip), "r") as zf:
            zf.extractall(str(coco_dir))
        # Clean up zip to save disk
        os.remove(str(coco_zip))
        print(f"  Done. Images at: {coco_images}")
    else:
        print(f"  [SKIP] COCO images already at {coco_images}")

    # ------------------------------------------------------------------
    # 2. COCO annotations (for CHAIR evaluation)
    # ------------------------------------------------------------------
    print("\n=== Step 2: COCO Annotations ===")
    ann_dir = coco_dir / "annotations"
    ann_zip = coco_dir / "annotations_trainval2014.zip"
    ann_file = ann_dir / "instances_val2014.json"

    if not ann_file.exists():
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            str(ann_zip),
            "COCO annotations (~252MB)"
        )
        print("  Extracting...")
        with zipfile.ZipFile(str(ann_zip), "r") as zf:
            zf.extractall(str(coco_dir))
        os.remove(str(ann_zip))
        print(f"  Done. Annotations at: {ann_dir}")
    else:
        print(f"  [SKIP] Annotations already at {ann_file}")

    # ------------------------------------------------------------------
    # 3. POPE benchmark
    # ------------------------------------------------------------------
    print("\n=== Step 3: POPE Benchmark ===")
    pope_dir = base_dir / "pope"
    pope_dir.mkdir(parents=True, exist_ok=True)

    pope_base_url = (
        "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco"
    )

    for split in ["random", "popular", "adversarial"]:
        dest = pope_dir / f"coco_pope_{split}.json"
        if not dest.exists():
            url = f"{pope_base_url}/coco_pope_{split}.json"
            download_file(url, str(dest), f"POPE {split} split")
        else:
            print(f"  [SKIP] POPE {split} already exists")

    # Verify POPE files
    for split in ["random", "popular", "adversarial"]:
        fpath = pope_dir / f"coco_pope_{split}.json"
        with open(fpath) as f:
            count = sum(1 for _ in f)
        print(f"  POPE {split}: {count} questions")

    # ------------------------------------------------------------------
    # 4. Pre-cache the LLaVA model (downloads ~14GB on first run)
    # ------------------------------------------------------------------
    print("\n=== Step 4: Pre-cache LLaVA-1.5-7B Model ===")
    print("  This downloads ~14GB of model weights on first run.")
    print("  Subsequent runs use the HuggingFace cache.")

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch

        model_id = "llava-hf/llava-1.5-7b-hf"

        print(f"  Downloading processor for {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("  Processor cached.")

        print(f"  Downloading model weights for {model_id}...")
        print("  (This will take 5-15 minutes depending on connection)")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        del model  # Free memory, we just wanted to cache it
        print("  Model weights cached.")

    except ImportError:
        print("  [WARN] transformers not installed, skipping model pre-cache.")
        print("         Model will download on first inference run.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  COCO images:      {coco_images}")
    print(f"  COCO annotations: {ann_dir}")
    print(f"  POPE benchmark:   {pope_dir}")
    print(f"  Model cache:      ~/.cache/huggingface/")
    print()
    print("Next step: python scripts/02_basic_inference.py")


if __name__ == "__main__":
    main()
