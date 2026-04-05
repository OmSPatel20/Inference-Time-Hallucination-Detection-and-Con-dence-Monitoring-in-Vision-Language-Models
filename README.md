# Inference-Time-Hallucination-Detection-and-Con-dence-Monitoring-in-Vision-Language-Models
Training-free hallucination detection for Vision-Language Models. Compares token entropy, visual contrastive decoding, and attention grounding on frozen LLaVA-1.5-7B across quantization levels and sustained inference. Evaluated on POPE and CHAIR benchmarks.

Vision-Language Models (VLMs) like LLaVA, Qwen2-VL, and LLaMA 3.2 Vision can answer questions about images, describe scenes, and reason across text and vision. But they hallucinate — they confidently describe objects that aren't there, invent details the image doesn't support, and fabricate answers that sound right but aren't grounded in what the model actually sees.
This isn't the same problem as factual errors in text-only LLMs. VLM hallucinations come from a specific failure: the language decoder stops paying attention to the image and falls back on language priors. The model generates what's statistically likely to follow the text, not what's actually in the picture.
Most existing solutions attack this during training — preference optimization, contrastive fine-tuning, curated hallucination datasets. These approaches are expensive, model-specific, and useless if you're deploying a frozen pre-trained model you can't retrain.
This project takes a different approach: detect hallucinations at inference time, with no retraining, on frozen models.
We implement and compare three training-free detection methods:

Token-level entropy: Flag tokens where the model's prediction distribution is uncertain
Visual contrastive decoding: Compare outputs from the original image vs. a degraded version — if the answer doesn't change, the model isn't using vision
Attention-based grounding: Measure how much the model attends to image tokens vs. text tokens during generation

We evaluate these methods on standard benchmarks (POPE, CHAIR) and go further than existing work by studying how detection reliability changes under real operational conditions — different quantization levels (FP16, INT8, INT4), batch sizes, and sustained inference workloads. We also track confidence drift over thousands of consecutive inferences to test whether hallucination patterns shift during long-running deployment.
The pipeline is built around LLaVA-1.5-7B, runs on consumer GPUs (tested on RTX 4060 8GB in INT4), and includes a monitoring layer that logs per-request confidence scores, detection flags, latency, and memory usage.
Built as a semester project for the MS in AI Systems program at the University of Florida.

# How to Use
Step 2: Create the environment (one-time, ~10 min)
conda create -n vlm_halluc python=3.10 -y
conda activate vlm_halluc
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate==0.28.0 bitsandbytes==0.43.0 Pillow==10.3.0 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pandas==2.2.1 pyyaml==6.0.1 tqdm==4.66.2 pycocotools==2.0.7 psutil==5.9.8
Verify it worked:
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
Should print:
True
NVIDIA GeForce RTX 4060 Laptop GPU

Step 3: Download data (~20 min, needs ~25GB disk space)
python scripts/01_download_data.py
This downloads COCO images (6GB), COCO annotations (252MB), POPE benchmark (tiny), and caches the LLaVA model weights (14GB into ~/.cache/huggingface/). You only do this once.

Step 4: Verify basic inference works (~2 min)
python scripts/02_basic_inference.py
You should see the model load in INT4, answer 3 questions about a COCO image, and print logit/attention verification. If this OOMs, something else is eating your GPU memory — close Chrome, Discord, etc.

Step 5: Test entropy detector (~2 min)
python scripts/04_entropy_detector.py
Shows per-token entropy for different questions on the same image. You should see higher entropy for hallucination-prone questions like "What is the person thinking about?"

Step 6: Test contrastive decoder (~3 min)
python scripts/06_contrastive_decoder.py
Runs each question twice (original image + noisy image), shows KL divergence. Low KL = model ignoring the image.

Step 7: Quick POPE test (~15 min)
python scripts/07_pope_evaluation.py --split random --max_samples 200 --no_contrastive
Runs 200 yes/no questions, prints accuracy and hallucination rate. If this works, everything is solid.

Step 8: Full POPE evaluation (~2-3 hours)
Close everything except Anaconda Prompt. Then:
python scripts/10_experiment_runner.py
Runs 1000 POPE questions × 3 splits (random, popular, adversarial) with entropy + contrastive detection. This is the core experiment.

Step 9: CHAIR evaluation (~30-45 min)
python scripts/08_chair_evaluation.py
Generates 200 image captions, checks for hallucinated objects against COCO ground truth.

Step 10: Confidence drift monitor (~1 hour)
python scripts/09_confidence_monitor.py
Runs 1000 consecutive inferences, logs entropy/latency/memory at every step.

Step 11: Generate all plots (~1 min)
python scripts/11_analysis_plots.py
Creates ROC curves, entropy scatter plots, drift time series, heatmaps — all saved to outputs/plots/.

Step 12: Check your results
dir outputs\plots
You should see:
roc_curves.png
entropy_analysis.png
confidence_drift.png
detection_heatmap.png
quantization_comparison.png
results_table.tex
The .tex file is a LaTeX table you can copy-paste into your final report.

Or, run steps 3-11 all at once overnight:
conda activate vlm_halluc
run_all_laptop.bat full
Go to sleep, check outputs/plots/ in the morning.
