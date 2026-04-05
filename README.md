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
