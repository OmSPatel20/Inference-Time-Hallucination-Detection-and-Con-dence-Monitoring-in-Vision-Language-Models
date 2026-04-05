"""
utils.py — Shared utilities for the VLM hallucination detection project.
Used by every other script.
"""

import os
import json
import time
import yaml
import torch
import psutil
import logging
from pathlib import Path
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes to console and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_and_processor(
    model_id: str,
    quantization: str = "fp16",
    device_map: str = "auto",
):
    """
    Load LLaVA-1.5-7B with specified quantization.

    Args:
        model_id:      HuggingFace model ID (e.g. "llava-hf/llava-1.5-7b-hf")
        quantization:  One of "fp16", "int8", "int4"
        device_map:    Device placement strategy

    Returns:
        (model, processor) tuple
    """
    print(f"Loading model: {model_id} | quantization: {quantization}")

    load_kwargs = {"device_map": device_map}

    if quantization == "fp16":
        load_kwargs["torch_dtype"] = torch.float16

    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config

    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config

    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    # VRAM safety check
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        if quantization == "fp16" and total_vram < 15:
            print(f"  WARNING: FP16 needs ~14GB VRAM, you have {total_vram:.1f}GB.")
            print(f"  Forcing INT4 quantization to avoid OOM.")
            quantization = "int4"
            load_kwargs.pop("torch_dtype", None)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
        elif quantization == "int8" and total_vram < 10:
            print(f"  WARNING: INT8 needs ~8GB VRAM + headroom, you have {total_vram:.1f}GB.")
            print(f"  Forcing INT4 quantization to avoid OOM.")
            quantization = "int4"
            load_kwargs.pop("quantization_config", None)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config

    model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    print(f"Model loaded. GPU memory: {get_gpu_memory_mb():.0f} MB")
    return model, processor


def flush_gpu():
    """Force free GPU memory. Call after deleting a model."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """
    Build the chat-style prompt that LLaVA expects.
    The <image> token is a placeholder the processor will fill.
    """
    return f"USER: <image>\n{question}\nASSISTANT:"


def run_single_inference(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 128,
    output_attentions: bool = False,
    output_scores: bool = False,
) -> dict:
    """
    Run inference on a single image + question.

    Returns dict with keys:
        - answer:         Decoded text string
        - input_ids:      Input token IDs
        - generated_ids:  Generated token IDs (new tokens only)
        - scores:         Tuple of logit tensors per step (if output_scores=True)
        - attentions:     Tuple of attention tensors per step (if output_attentions=True)
        - latency_ms:     Wall-clock time in milliseconds
        - num_tokens:     Number of generated tokens
    """
    prompt = build_prompt(question)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Move to model device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=output_scores,
            output_attentions=output_attentions,
            return_dict_in_generate=True,
        )
    t1 = time.perf_counter()

    generated_ids = outputs.sequences[0, input_len:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    result = {
        "answer": answer,
        "input_ids": inputs["input_ids"][0],
        "generated_ids": generated_ids,
        "latency_ms": (t1 - t0) * 1000,
        "num_tokens": len(generated_ids),
    }

    if output_scores:
        result["scores"] = outputs.scores  # tuple of (vocab_size,) tensors

    if output_attentions:
        result["attentions"] = outputs.attentions  # tuple of layer attention tensors

    return result


# ---------------------------------------------------------------------------
# POPE data loading
# ---------------------------------------------------------------------------

def load_pope_data(pope_dir: str, split: str = "random", max_samples: int = None) -> list:
    """
    Load POPE benchmark questions.

    Each item: {"question_id": int, "image": str, "text": str, "label": "yes"/"no"}
    """
    filename = f"coco_pope_{split}.json"
    filepath = os.path.join(pope_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"POPE file not found: {filepath}\n"
            f"Run 01_download_data.py first."
        )

    data = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} POPE questions (split={split})")
    return data


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def load_coco_image(image_dir: str, image_filename: str) -> Image.Image:
    """Load a COCO image by filename."""
    path = os.path.join(image_dir, image_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def get_coco_image_objects(annotations_file: str) -> dict:
    """
    Parse COCO annotations to get ground-truth object categories per image.

    Returns: {image_id: set of category names}
    """
    from pycocotools.coco import COCO

    coco = COCO(annotations_file)
    image_objects = {}

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cats = set()
        for ann in anns:
            cat = coco.loadCats(ann["category_id"])[0]["name"]
            cats.add(cat.lower())
        image_objects[img_id] = cats

    return image_objects


# ---------------------------------------------------------------------------
# System monitoring
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """Get reserved (cached) GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0.0


def get_system_stats() -> dict:
    """Get current system resource usage."""
    return {
        "gpu_mem_allocated_mb": get_gpu_memory_mb(),
        "gpu_mem_reserved_mb": get_gpu_memory_reserved_mb(),
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_jsonl(data: list, filepath: str):
    """Save a list of dicts as JSON lines."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} records to {filepath}")


def load_jsonl(filepath: str) -> list:
    """Load JSON lines file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def append_jsonl(item: dict, filepath: str):
    """Append a single JSON line to a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(item) + "\n")
