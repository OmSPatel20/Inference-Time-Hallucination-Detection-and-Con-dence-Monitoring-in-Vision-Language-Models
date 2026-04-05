# Running on RTX 4060 Laptop (8GB VRAM)

## What changed from the original design

| Original (A100 80GB) | Your laptop (RTX 4060 8GB) |
|---|---|
| FP16, INT8, INT4 quantizations | **INT4 only** — FP16 needs 14GB, INT8 needs 10GB |
| Batch sizes 1, 8, 32 | **Batch size 1 only** — no VRAM headroom for batching |
| 3 detection methods | **2 methods** — attention detector disabled (OOM) |
| 3000 POPE samples/split | **1000 samples/split** — ~3x faster |
| 500 CHAIR captions | **200 captions** |
| 2000 drift samples | **1000 drift samples** |
| ~8h total on A100 | **~4-6h total on RTX 4060** |

## Is INT4-only a problem for the project?

No. Here's why:

1. **The project's research question is about inference-time detection methods, not quantization effects.** The quantization comparison (FP16 vs INT8 vs INT4) was a secondary analysis. The core contribution — comparing entropy, attention, and contrastive detection under operational conditions — works perfectly with a single quantization level.

2. **You still have two strong detection methods.** Entropy detection (the strongest signal in most papers) and visual contrastive decoding both work fine in INT4. Attention detection is the weakest of the three anyway.

3. **If your professor asks about quantization effects,** you can mention it as a limitation and say: "Full quantization comparison requires 16GB+ VRAM, which I plan to run on HiPerGator for the final report." Then run 1-2 conditions on HiPerGator for the paper.

## Setup (Windows)

```
# Open Anaconda Prompt
cd path\to\vlm_project
setup_env_windows.bat
```

## Quick test (~20 min)

```
conda activate vlm_halluc
run_all_laptop.bat quick
```

This runs:
1. Downloads COCO images + POPE benchmark (~6GB, one-time)
2. Loads LLaVA-1.5-7B in INT4 (~5GB VRAM)
3. Runs 3 test inferences
4. Runs entropy detector on a sample
5. Runs 200 POPE questions

If everything passes → you're ready for the full run.

## Full run (~4-6 hours)

```
run_all_laptop.bat full
```

**Leave this running overnight.** Close other GPU-heavy apps (games, Chrome with hardware acceleration, etc).

## Memory tips

- **Close Chrome** — it uses GPU memory for acceleration
- **Close Discord/Slack** — same reason
- **Don't game while running** — obviously
- **Monitor GPU:** Open a second terminal and run `nvidia-smi -l 5` to watch memory every 5 seconds
- **If you OOM:** The scripts have an auto-fallback that forces INT4, but if the KV cache still overflows:
  - Reduce `max_new_tokens` from 64 to 32 in `configs/experiment_config.yaml`
  - Run with `--max_samples 500` to cut the POPE evaluation in half

## Expected VRAM usage

| Phase | VRAM |
|---|---|
| Model load (INT4) | ~4.5 GB |
| Inference (1 image) | ~5.5 GB |
| Inference + scores | ~6.0 GB |
| Inference + attentions | ~7.5+ GB ← **danger zone** |
| Contrastive (2x inference) | ~6.0 GB (sequential, not parallel) |

## File locations after full run

```
outputs/
├── benchmark_results.json         # Throughput numbers
├── experiment_matrix_results.json # All POPE results combined
├── pope_results/
│   ├── pope_random_int4.jsonl     # Per-question results
│   ├── pope_popular_int4.jsonl
│   ├── pope_adversarial_int4.jsonl
│   └── summary_*.json
├── chair_results/
│   ├── chair_int4.jsonl           # Per-caption results
│   └── chair_summary_int4.json
├── drift_monitor/
│   └── drift_int4_random.jsonl    # Per-inference time series
└── plots/
    ├── roc_curves.png
    ├── quantization_comparison.png
    ├── entropy_analysis.png
    ├── confidence_drift.png
    ├── detection_heatmap.png
    └── results_table.tex          # Copy-paste into your LaTeX report
```

## Getting quantization comparison data (optional, for the report)

If you want FP16/INT8 numbers for the final report, run these on HiPerGator:

```bash
# On HiPerGator (A100 GPU)
python scripts/07_pope_evaluation.py --split adversarial --quantization fp16 --max_samples 1000
python scripts/07_pope_evaluation.py --split adversarial --quantization int8 --max_samples 1000
```

Then copy the JSONL files back to your laptop and re-run `scripts/11_analysis_plots.py` — it auto-detects all result files.
