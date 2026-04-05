# 🔍 Inference-Time Hallucination Detection in Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Training-free hallucination detection for Vision-Language Models. Compares token entropy, visual contrastive decoding, and attention grounding on frozen LLaVA-1.5-7B across quantization levels and sustained inference. Evaluated on POPE and CHAIR benchmarks. Runs on consumer GPUs (RTX 4060 8GB).

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What This Project Does](#what-this-project-does)
- [Architecture](#architecture)
- [Detection Methods](#detection-methods)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Dashboard](#dashboard)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Expected Results](#expected-results)
- [Hardware Notes](#hardware-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Problem Statement

Vision-Language Models (VLMs) such as LLaVA, Qwen2-VL, and LLaMA 3.2 Vision have achieved impressive results on multimodal benchmarks, but they suffer from a critical reliability problem: **hallucination**. These models confidently describe objects that do not exist in the image, invent spatial relationships, and fabricate details that sound plausible but have no grounding in the visual input.

Unlike factual errors in text-only LLMs, VLM hallucinations arise from a **distinct failure mode** — the language decoder stops attending to image features and falls back on statistical language priors. The model generates what is likely to follow the text, not what is actually in the picture.

### Why Existing Solutions Fall Short

Most hallucination mitigation research focuses on **training-time interventions**:

- Preference optimization (DPO/RLHF on hallucination examples)
- Contrastive fine-tuning on curated datasets
- Specialized hallucination-aware training objectives

These approaches are **expensive** (require full fine-tuning), **model-specific** (tied to one architecture), and **impractical for deployment** (you cannot retrain every model you serve).

### The Gap This Project Addresses

**Inference-time detection** — methods that work on frozen, pre-trained models without any retraining — remains significantly underexplored. Specifically, three questions are unanswered in the current literature:

1. **How do inference-time detection methods compare under realistic serving conditions?** Most papers evaluate on single-sample, full-precision inference. Nobody tests what happens when you quantize the model, batch requests, or vary sequence length.

2. **Does hallucination behavior drift over sustained workloads?** In production, models run thousands of consecutive inferences. Do confidence patterns shift? Do detection methods degrade? No existing benchmark captures this.

3. **Can multiple detection signals be combined into a practical monitoring system?** Individual detection papers report metrics in isolation. Nobody has built an integrated pipeline that tracks multiple signals simultaneously.

This project addresses all three gaps.

---

## What This Project Does

We build an **instrumented VLM inference pipeline** with integrated hallucination detection and confidence monitoring. The system:

1. **Deploys LLaVA-1.5-7B** in an optimized inference pipeline with INT4/INT8/FP16 quantization support
2. **Implements three training-free detection methods** that run at inference time on frozen models
3. **Evaluates on standard benchmarks** (POPE for object hallucination, CHAIR for caption hallucination)
4. **Monitors confidence drift** over 1000+ consecutive inferences
5. **Provides a Streamlit dashboard** for interactive result exploration

No model weights are modified. No fine-tuning. Everything operates at inference time.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Input                                 │
│                  Image + Question                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  LLaVA-1.5-7B                                │
│         CLIP ViT-L/14@336px → MLP → Vicuna-7B               │
│              (FP16 / INT8 / INT4)                            │
└──────────┬──────────────┬──────────────┬─────────────────────┘
           │              │              │
        logits       attentions    generated text
           │              │              │
           ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│              Hallucination Detection Layer                    │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────────────┐ │
│  │   Token     │  │  Attention  │  │ Visual Contrastive    │ │
│  │  Entropy    │  │  Grounding  │  │    Decoding           │ │
│  │            │  │             │  │                       │ │
│  │ Per-token  │  │ Visual vs   │  │ Original vs perturbed │ │
│  │ softmax    │  │ text attn   │  │ image comparison      │ │
│  │ entropy    │  │ ratio       │  │                       │ │
│  └─────┬──────┘  └──────┬──────┘  └──────────┬────────────┘ │
│        └────────────────┼────────────────────┘              │
│                         ▼                                    │
│               Combined Detection Score                       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                Monitoring & Logging                           │
│                                                              │
│  • Per-request confidence scores     • GPU memory tracking   │
│  • Detection rate over time          • Latency profiling     │
│  • Confidence drift analysis         • JSONL logging         │
└──────────────────────────────────────────────────────────────┘
```

---

## Detection Methods

### 1. Token-Level Entropy

For each generated token, we compute the entropy of the softmax distribution over the vocabulary:

```
H(t) = -Σ p(x) · log p(x)
```

**High entropy** → the model is uncertain about what to generate → hallucination signal.
**Low entropy** → the model is confident → likely grounded.

We aggregate per-token entropies into a response-level score using mean entropy, max entropy, and the fraction of tokens above a calibrated threshold.

### 2. Visual Contrastive Decoding

We run inference twice — once with the **original image** and once with a **perturbed image** (Gaussian noise, grayscale, or resolution reduction). Then we compute the KL divergence between the two output distributions at each generation step.

**High KL** → the model's answer depends on the image → visually grounded.
**Low KL** → the model gives the same answer regardless of the image → hallucination risk.

### 3. Attention-Based Grounding *(disabled on <16GB VRAM)*

LLaVA uses self-attention where image tokens occupy positions 1–576 in the sequence. For each generated token, we measure what fraction of attention goes to image tokens vs. text tokens.

**Low visual attention ratio** during object mentions → the model is generating from language priors, not visual evidence.

> **Note:** This method stores full attention tensors in VRAM and requires ≥16GB. It is disabled by default on consumer GPUs and can be force-enabled with `--force`.

---

## Project Structure

```
vlm_project/
│
├── run.py                        # Single entry point — replaces all .bat/.sh
├── dashboard.py                  # Streamlit interactive results viewer
├── requirements.txt              # Python dependencies
│
├── configs/
│   └── experiment_config.yaml    # All hyperparameters (quantization, samples, thresholds)
│
├── scripts/
│   ├── utils.py                  # Shared code: model loading, I/O, monitoring
│   ├── 01_download_data.py       # Downloads COCO val2014, POPE benchmark, caches model
│   ├── 02_basic_inference.py     # Single image+question inference test
│   ├── 03_batched_pipeline.py    # Throughput benchmark across configurations
│   ├── 04_entropy_detector.py    # Detection method 1: token entropy
│   ├── 05_attention_detector.py  # Detection method 2: visual attention ratio
│   ├── 06_contrastive_decoder.py # Detection method 3: original vs perturbed image
│   ├── 07_pope_evaluation.py     # POPE benchmark with all detectors
│   ├── 08_chair_evaluation.py    # Free-form caption hallucination evaluation
│   ├── 09_confidence_monitor.py  # Long-running inference drift tracking
│   ├── 10_experiment_runner.py   # Full experimental matrix runner
│   └── 11_analysis_plots.py      # Generate all figures for the report
│
├── slurm/                        # HiPerGator (UF) cluster scripts
│   ├── setup_env.sh
│   └── run_experiment.slurm
│
├── outputs/                      # All results (created at runtime)
│   ├── pope_results/             # Per-question detection scores
│   ├── chair_results/            # Per-caption hallucination analysis
│   ├── drift_monitor/            # Time-series confidence logs
│   └── plots/                    # Generated figures
│
├── LAPTOP_README.md              # Hardware-specific notes for 8GB GPUs
└── PROJECT_GUIDE.md              # Detailed technical walkthrough
```

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU, 8GB VRAM | NVIDIA GPU, 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Disk | 25GB free | 40GB free |
| OS | Windows 10/11, Linux | Any |

**Tested on:**
- HP Omen, RTX 4060 (8GB), i7 14th gen, 16GB RAM, Windows 11
- UF HiPerGator, A100 (80GB), SLURM

### Software

- Python 3.10+
- CUDA 12.1+
- Anaconda/Miniconda

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/vlm-hallucination-detection.git
cd vlm-hallucination-detection
```

### Step 2: Create the conda environment

```bash
conda create -n vlm_halluc python=3.10 -y
conda activate vlm_halluc
```

### Step 3: Install dependencies

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 4: Verify installation

```bash
python run.py setup
```

You should see all green checkmarks:

```
  [12:00:00] ✓ Python 3.10.x
  [12:00:00] ✓ PyTorch 2.2.0
  [12:00:00] ✓ GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB VRAM)
  [12:00:01] ✓ All required packages installed
  [12:00:01] ✓ Streamlit 1.33.0 (dashboard available)
  [12:00:01] ✓ Free disk space: 45.2 GB
  [12:00:01] ✓ Config found
  [12:00:01] ✓ Environment OK — ready to run
```

---

## How to Run

### Quick reference

| Command | What it does | Time |
|---------|-------------|------|
| `python run.py setup` | Check environment and dependencies | ~10 sec |
| `python run.py download` | Download COCO images, POPE benchmark, cache model | ~20 min |
| `python run.py test` | Quick sanity check (200 POPE questions) | ~20 min |
| `python run.py full` | Run complete experiment pipeline | ~4-6 hours |
| `python run.py full --resume` | Resume from where it stopped | varies |
| `python run.py pope` | POPE evaluation only (3 splits × 1000 samples) | ~2-3 hours |
| `python run.py chair` | CHAIR caption evaluation only (200 captions) | ~30-45 min |
| `python run.py drift` | Confidence drift monitoring (1000 inferences) | ~1 hour |
| `python run.py plots` | Generate all analysis figures | ~1 min |
| `python run.py dashboard` | Launch Streamlit interactive dashboard | until Ctrl+C |
| `python run.py status` | Show what has been completed | ~1 sec |
| `python run.py clean` | Delete all outputs for a fresh start | ~1 sec |

### Recommended workflow

**First time (do this once):**

```bash
conda activate vlm_halluc
python run.py setup
python run.py download
python run.py test
```

If everything passes, proceed.

**Run full experiments (leave overnight):**

```bash
python run.py full
```

Close Chrome, Discord, and any GPU-heavy applications before running.

**Explore results:**

```bash
python run.py dashboard
```

Opens an interactive dashboard in your browser at `http://localhost:8501`.

**If it crashes midway:**

```bash
python run.py full --resume
```

This skips already-completed steps and picks up where it stopped.

### Running individual scripts directly

Each script in `scripts/` can be run independently with custom arguments:

```bash
# POPE evaluation with specific settings
python scripts/07_pope_evaluation.py --split adversarial --max_samples 500

# CHAIR with custom sample count
python scripts/08_chair_evaluation.py --num_samples 100

# Drift monitoring
python scripts/09_confidence_monitor.py --num_samples 2000

# Entropy detector on a specific image
python scripts/04_entropy_detector.py --image_file COCO_val2014_000000000042.jpg --question "Is there a dog?"
```

---

## Dashboard

The Streamlit dashboard provides five interactive tabs:

| Tab | Contents |
|-----|----------|
| **📊 Overview** | Accuracy, hallucination rates, CHAIR scores at a glance |
| **🎯 POPE Results** | ROC curves, entropy distributions, sample browser with filters |
| **📝 CHAIR Results** | Most hallucinated object categories, caption-level browser |
| **📈 Confidence Drift** | Time-series plots of entropy, latency, GPU memory, accuracy |
| **⚖️ Detection Comparison** | Side-by-side AUC comparison of entropy vs. contrastive methods |

Launch with:

```bash
python run.py dashboard
```

---

## Benchmarks & Evaluation

### POPE (Polling-based Object Probing Evaluation)

Binary yes/no questions about object presence in images. Three difficulty splits:

- **Random**: Negative objects sampled randomly from COCO
- **Popular**: Negative objects are the most frequent COCO categories (harder)
- **Adversarial**: Negative objects co-occur frequently with ground-truth objects (hardest)

### CHAIR (Caption Hallucination Assessment with Image Relevance)

Free-form image captioning evaluated against COCO ground-truth annotations:

- **CHAIR_i**: Fraction of mentioned objects that are hallucinated
- **CHAIR_s**: Fraction of captions containing at least one hallucination

### Confidence Drift

1000+ consecutive inferences with per-request logging of:

- Token entropy (mean, max, std)
- Top-1 probability
- Latency
- GPU memory (allocated and reserved)
- Prediction correctness

First-half vs. second-half statistical comparison reveals whether model behavior changes over sustained serving.

---

## Expected Results

Approximate numbers on LLaVA-1.5-7B (INT4):

| Metric | Random | Popular | Adversarial |
|--------|--------|---------|-------------|
| Accuracy | ~0.85 | ~0.80 | ~0.75 |
| Hallucination Rate | ~0.10 | ~0.15 | ~0.22 |
| Entropy Detection AUC | ~0.65 | ~0.62 | ~0.60 |

CHAIR (200 captions):

| Metric | Value |
|--------|-------|
| CHAIR_i | ~0.08-0.15 |
| CHAIR_s | ~0.35-0.50 |

Your exact numbers will vary based on which COCO images are sampled.

---

## Hardware Notes

### RTX 4060 (8GB VRAM)

- Uses **INT4 quantization only** — FP16 requires 14GB, INT8 requires ~10GB
- **Batch size 1** — no VRAM headroom for batching
- **Attention detector disabled** — attention tensors overflow 8GB
- Entropy + contrastive detection work fine
- `max_new_tokens` capped at 64 to limit KV cache

### A100 / RTX 4090 (40-80GB VRAM)

Edit `configs/experiment_config.yaml`:

```yaml
hardware:
  vram_gb: 80
  default_quantization: "fp16"
  max_batch_size: 32
  enable_attention_detector: true

experiment:
  quantizations: ["fp16", "int8", "int4"]
  batch_sizes: [1, 8, 32]
  num_pope_samples: 3000
  num_chair_samples: 500
  num_drift_samples: 2000
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Close Chrome/Discord. Reduce `max_new_tokens` to 32 in config. |
| `bitsandbytes` error on Windows | Run `pip install bitsandbytes-windows` |
| `FileNotFoundError: POPE` | Run `python run.py download` first |
| Model download stalls | Delete `~/.cache/huggingface/` and retry |
| `ModuleNotFoundError` | Run `conda activate vlm_halluc` |
| Streamlit won't launch | Run `pip install streamlit plotly` |
| Attention detector OOM | Expected on 8GB. Use `--force` for single-example demo only. |
| `run.py full` crashes at step X | Run `python run.py full --resume` to continue |

---

## Citation

If you use this project in your work:

```bibtex
@misc{vlm_hallucination_detection_2026,
  author       = {Om},
  title        = {Inference-Time Hallucination Detection in Vision-Language Models},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/YOUR_USERNAME/vlm-hallucination-detection}},
  note         = {University of Florida, MS in AI Systems}
}
```

### Key References

- [LLaVA-1.5](https://arxiv.org/abs/2310.03744) — Liu et al., 2023. Visual Instruction Tuning.
- [POPE](https://arxiv.org/abs/2305.10355) — Li et al., 2023. Evaluating Object Hallucination in LVLMs.
- [CHAIR](https://arxiv.org/abs/1809.02156) — Rohrbach et al., 2018. Object Hallucination in Image Captioning.
- [VADE](https://aclanthology.org/2025.findings-acl.773/) — Prabhakaran et al., 2025. Visual Attention Guided Hallucination Detection.
- [VTI](https://openreview.net/forum?id=LBl7Hez0fF) — ICLR 2025. Reducing Hallucinations via Latent Space Steering.
- [ECD](https://arxiv.org/abs/2504.12169) — 2025. Efficient Contrastive Decoding with Probabilistic Hallucination Detection.

---

## Acknowledgments

Built as a semester project for the **MS in AI Systems** program at the **University of Florida**, Herbert Wertheim College of Engineering.

---

<p align="center">
  <b>University of Florida · Spring 2026</b>
</p>
