# Inference-Time Hallucination Detection in VLMs — Complete Build Guide

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Download Data & Models](#3-download-data--models)
4. [Step 1: Basic VLM Inference](#4-step-1-basic-vlm-inference)
5. [Step 2: Batched & Quantized Pipeline](#5-step-2-batched--quantized-pipeline)
6. [Step 3: Hallucination Detection — Token Entropy](#6-step-3-hallucination-detection--token-entropy)
7. [Step 4: Hallucination Detection — Attention-Based](#7-step-4-hallucination-detection--attention-based)
8. [Step 5: Hallucination Detection — Visual Contrastive Decoding](#8-step-5-hallucination-detection--visual-contrastive-decoding)
9. [Step 6: POPE Evaluation](#9-step-6-pope-evaluation)
10. [Step 7: CHAIR Evaluation](#10-step-7-chair-evaluation)
11. [Step 8: Confidence Drift Monitoring](#11-step-8-confidence-drift-monitoring)
12. [Step 9: Multi-Condition Experiment Runner](#12-step-9-multi-condition-experiment-runner)
13. [Step 10: Analysis & Visualization](#13-step-10-analysis--visualization)
14. [HiPerGator SLURM Scripts](#14-hipergator-slurm-scripts)
15. [Project Timeline](#15-project-timeline)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Project Overview

**What you're building:** An instrumented VLM inference pipeline that detects hallucinations
at inference time using three training-free methods, evaluated under varying operational
conditions (batch size, quantization, sequence length).

**Architecture:**
```
Image + Question
       │
       ▼
┌─────────────────┐
│  LLaVA-1.5-7B   │ ◄── (FP16 / INT8 / INT4)
│  Inference Core  │
└────────┬────────┘
         │ logits, attentions, tokens
         ▼
┌─────────────────────────────────┐
│   Hallucination Detection Layer │
│  ┌───────┐ ┌───────┐ ┌───────┐ │
│  │Entropy│ │Attn   │ │Visual │ │
│  │Score  │ │Diverg.│ │Contst.│ │
│  └───┬───┘ └───┬───┘ └───┬───┘ │
│      └─────────┼─────────┘     │
│                ▼               │
│       Combined Score           │
└────────────────┬───────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Monitoring & Logging        │
│  - Per-request confidence       │
│  - Detection rate over time     │
│  - Latency / memory tracking    │
└─────────────────────────────────┘
```

**File Structure:**
```
vlm_project/
├── PROJECT_GUIDE.md          ← You are here
├── requirements.txt
├── configs/
│   └── experiment_config.yaml
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_basic_inference.py
│   ├── 03_batched_pipeline.py
│   ├── 04_entropy_detector.py
│   ├── 05_attention_detector.py
│   ├── 06_contrastive_decoder.py
│   ├── 07_pope_evaluation.py
│   ├── 08_chair_evaluation.py
│   ├── 09_confidence_monitor.py
│   ├── 10_experiment_runner.py
│   ├── 11_analysis_plots.py
│   └── utils.py
├── slurm/
│   ├── setup_env.sh
│   └── run_experiment.slurm
└── outputs/                  ← All results go here
```

---

## 2. Environment Setup

### On HiPerGator (UF):

```bash
# Step 1: Load modules
module load conda
module load cuda/12.1

# Step 2: Create conda environment
conda create -n vlm_halluc python=3.10 -y
conda activate vlm_halluc

# Step 3: Install PyTorch (CUDA 12.1)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install project dependencies
pip install transformers==4.40.0 \
            accelerate==0.28.0 \
            bitsandbytes==0.43.0 \
            Pillow==10.3.0 \
            scipy==1.13.0 \
            scikit-learn==1.4.2 \
            matplotlib==3.8.4 \
            seaborn==0.13.2 \
            pandas==2.2.1 \
            pyyaml==6.0.1 \
            tqdm==4.66.2 \
            pycocotools==2.0.7 \
            psutil==5.9.8

# Step 5: Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### On a local machine with GPU:

```bash
conda create -n vlm_halluc python=3.10 -y
conda activate vlm_halluc
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Download Data & Models

Run `scripts/01_download_data.py`. This downloads:
- MS-COCO 2014 validation images (~6GB)
- POPE benchmark annotations
- COCO object annotations (for CHAIR)
- LLaVA-1.5-7B model weights (~14GB, cached by HuggingFace)

**Estimated total disk: ~25GB**

---

## 4. Step 1: Basic VLM Inference

**Goal:** Load LLaVA-1.5-7B, pass it one image + question, get a text answer.
This is the foundation — everything else builds on this.

**File:** `scripts/02_basic_inference.py`

**What you should see:**
```
Loading model... (takes 30-60 seconds first time)
Question: Is there a dog in this image?
Answer: Yes, there is a dog in the image...
Tokens generated: 24
Time: 1.3s
```

**What to check:**
- Model loads without OOM errors
- Output is coherent English (not garbage tokens)
- GPU memory usage is ~14GB for FP16

---

## 5. Step 2: Batched & Quantized Pipeline

**Goal:** Run inference at different batch sizes (1, 4, 8, 16) and quantization
levels (FP16, INT8, INT4). Measure throughput and memory.

**File:** `scripts/03_batched_pipeline.py`

**Key insight:** Quantization reduces memory but may change hallucination behavior —
that's one of the things we're investigating.

**Expected results:**
| Config | GPU Memory | Throughput |
|--------|-----------|------------|
| FP16, batch=1  | ~14GB | ~15 tok/s |
| INT8, batch=1  | ~8GB  | ~12 tok/s |
| INT4, batch=1  | ~5GB  | ~10 tok/s |
| FP16, batch=8  | ~18GB | ~40 tok/s |
| INT4, batch=8  | ~8GB  | ~30 tok/s |

---

## 6. Step 3: Hallucination Detection — Token Entropy

**Goal:** For each generated token, compute the entropy of the softmax distribution.
High entropy = model is uncertain = possible hallucination.

**File:** `scripts/04_entropy_detector.py`

**How it works:**
1. During generation, capture the logit vector at each step
2. Convert logits → probabilities via softmax
3. Compute entropy: H = -Σ p(x) log p(x)
4. Flag tokens where entropy exceeds a threshold
5. Aggregate per-response: mean entropy, max entropy, fraction of high-entropy tokens

**Threshold calibration:** Use POPE random split to find the threshold that
maximizes F1 score, then evaluate on POPE adversarial split.

---

## 7. Step 4: Hallucination Detection — Attention-Based

**Goal:** Extract how much the model "looks at" the image vs. generating from
language priors. If the model ignores the image, it's likely hallucinating.

**File:** `scripts/05_attention_detector.py`

**How it works:**
1. Run inference with `output_attentions=True`
2. For each generated token, extract attention weights from all layers
3. Separate attention into: attention-on-image-tokens vs. attention-on-text-tokens
4. Compute visual attention ratio = sum(attn on image) / sum(attn on all)
5. Low visual attention ratio during object mentions → hallucination signal

**Important:** LLaVA uses self-attention (not cross-attention), so image tokens are
just the first N tokens in the sequence. You need to know how many image tokens
there are (576 for LLaVA-1.5 with 336px CLIP encoder: 24×24 patches).

---

## 8. Step 5: Hallucination Detection — Visual Contrastive Decoding

**Goal:** Generate answers from both the original image and a degraded version.
If the answer doesn't change much, the model is relying on language priors, not vision.

**File:** `scripts/06_contrastive_decoder.py`

**How it works:**
1. Run inference with original image → get logits_original
2. Run inference with perturbed image (Gaussian noise, σ=0.5) → get logits_perturbed
3. Compute KL divergence between the two distributions at each step
4. High KL divergence = model IS using visual info (good)
5. Low KL divergence = model ignoring the image (hallucination risk)

**Perturbation types tested:**
- Gaussian noise (σ = 0.3, 0.5, 0.8)
- Random crop (70% of image)
- Grayscale conversion
- Resolution reduction (336→84→336 resize)

---

## 9. Step 6: POPE Evaluation

**Goal:** Run all three detection methods on the POPE benchmark and compute
precision, recall, F1, accuracy, and ROC-AUC.

**File:** `scripts/07_pope_evaluation.py`

**POPE format:** Each line is a yes/no question about object presence:
```json
{"question_id": 1, "image": "COCO_val2014_000000xxx.jpg",
 "text": "Is there a dog in the image?", "label": "yes"}
```

**Evaluation protocol:**
1. For each question, run inference and get the model's yes/no answer
2. Simultaneously compute all three detection scores
3. Compare model answer vs. ground truth → hallucination label
4. For each detection method, compute ROC curve: detection score vs. hallucination label
5. Report AUC, and precision/recall/F1 at the optimal threshold

---

## 10. Step 7: CHAIR Evaluation

**Goal:** Evaluate free-form image captions for hallucinated objects.

**File:** `scripts/08_chair_evaluation.py`

**CHAIR metrics:**
- CHAIR_i = (hallucinated objects) / (all mentioned objects)
- CHAIR_s = (sentences with hallucination) / (all sentences)

**How it works:**
1. For 500 COCO val images, generate a caption: "Describe this image in detail."
2. Parse the caption to extract mentioned objects (using COCO's 80 categories)
3. Compare against ground-truth annotations
4. Any mentioned object not in ground truth = hallucination
5. Also record detection scores for each caption

---

## 11. Step 8: Confidence Drift Monitoring

**Goal:** Run 2000+ consecutive inferences and track whether hallucination
detection scores drift over time.

**File:** `scripts/09_confidence_monitor.py`

**What we're looking for:**
- Does mean entropy increase after 500+ inferences? (thermal drift)
- Does visual attention ratio degrade over long sessions?
- Does KV-cache accumulation affect detection reliability?

**Logging:** Each inference logs a JSON line:
```json
{"idx": 0, "timestamp": 1711234567.89, "entropy_mean": 0.43,
 "entropy_max": 2.1, "visual_attn_ratio": 0.31,
 "contrastive_kl": 0.87, "latency_ms": 134, "gpu_mem_mb": 14200,
 "answer": "yes", "question_id": 12345}
```

---

## 12. Step 9: Multi-Condition Experiment Runner

**Goal:** Run the full evaluation under every experimental condition.

**File:** `scripts/10_experiment_runner.py`

**Experimental matrix:**
| Variable | Values |
|----------|--------|
| Quantization | FP16, INT8, INT4 |
| Batch size | 1, 8, 32 |
| POPE split | random, popular, adversarial |

That's 3 × 3 × 3 = 27 conditions. Each runs ~3000 POPE questions.
Estimated total: ~8-12 GPU-hours on A100.

---

## 13. Step 10: Analysis & Visualization

**Goal:** Generate all figures and tables for the final report.

**File:** `scripts/11_analysis_plots.py`

**Plots generated:**
1. ROC curves for each detection method (per POPE split)
2. Bar chart: F1 score across quantization levels
3. Heatmap: Detection AUC across (quantization × batch size)
4. Time series: Confidence drift over 2000 inferences
5. Scatter: Entropy vs. visual attention ratio (colored by hallucination)
6. Table: CHAIR scores across conditions

---

## 14. HiPerGator SLURM Scripts

See `slurm/` directory. Request:
- 1× A100 80GB (or 1× A100 40GB for INT4/INT8)
- 32GB RAM
- 8 hours for full experiment matrix
- 2 hours for single-condition runs

---

## 15. Project Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Environment setup, data download, basic inference | Scripts 01-02 working |
| 3-4 | Batched pipeline, entropy detector | Scripts 03-04, initial POPE numbers |
| 5-6 | Attention detector, contrastive decoder | Scripts 05-06 working |
| 7-8 | POPE + CHAIR full evaluation | Scripts 07-08, baseline results |
| 9-10 | Multi-condition experiments | Script 10, full experimental matrix |
| 11-12 | Confidence drift experiments | Script 09, drift analysis |
| 13-14 | Analysis, visualization, writing | Script 11, all figures |
| 15 | Final report + presentation | PDF report |

---

## 16. Troubleshooting

**OOM on GPU:** Use INT4 quantization or reduce batch size.
```python
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, load_in_4bit=True, device_map="auto"
)
```

**Slow first inference:** Normal. HuggingFace downloads + compiles on first run.
Subsequent runs use cache at `~/.cache/huggingface/`.

**POPE file not found:** Check that `01_download_data.py` completed successfully.
POPE files should be at `data/pope/coco_pope_{random,popular,adversarial}.json`.

**Attention extraction returns None:** You must pass `output_attentions=True` to
the `model.generate()` call AND use `return_dict_in_generate=True`. See script 05.

**Mismatched image token count:** LLaVA-1.5-7B with CLIP ViT-L/14@336px produces
576 image tokens (24×24 patches). Verify with:
```python
vision_outputs = model.vision_tower(pixel_values)
print(f"Image tokens: {vision_outputs.last_hidden_state.shape[1]}")
```
