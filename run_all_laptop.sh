#!/bin/bash
# run_all_laptop.sh — Run the full project on a laptop with RTX 4060 (8GB VRAM)
#
# Everything runs in INT4 quantization, batch size 1, reduced sample counts.
# Estimated total time: ~4-6 hours
#
# Usage:
#   bash run_all_laptop.sh          # Full run
#   bash run_all_laptop.sh quick    # Quick sanity check (~20 min)

set -e

MODE=${1:-full}

echo "=============================================="
echo "VLM Hallucination Detection — Laptop Mode"
echo "Hardware: RTX 4060 (8GB) + INT4 quantization"
echo "Mode: $MODE"
echo "=============================================="

# Activate conda env
# conda activate vlm_halluc

case $MODE in
    quick)
        echo ""
        echo "=== QUICK TEST (~20 min) ==="
        echo ""

        echo "[1/4] Downloading data..."
        python scripts/01_download_data.py

        echo "[2/4] Testing basic inference..."
        python scripts/02_basic_inference.py

        echo "[3/4] Testing entropy detector..."
        python scripts/04_entropy_detector.py

        echo "[4/4] Quick POPE eval (200 samples)..."
        python scripts/07_pope_evaluation.py --split random --max_samples 200 --no_contrastive

        echo ""
        echo "=== QUICK TEST DONE ==="
        echo "If everything above passed, run: bash run_all_laptop.sh full"
        ;;

    full)
        echo ""
        echo "=== FULL RUN (~4-6 hours) ==="
        echo ""

        # Step 1: Data
        echo "[1/7] Downloading data..."
        python scripts/01_download_data.py

        # Step 2: Basic verification
        echo "[2/7] Verifying inference..."
        python scripts/02_basic_inference.py

        # Step 3: Throughput benchmark (INT4 only, batch=1 only)
        echo "[3/7] Benchmarking throughput..."
        python scripts/03_batched_pipeline.py

        # Step 4: POPE evaluation across all 3 splits (~2-3 hours)
        echo "[4/7] POPE evaluation (1000 samples × 3 splits)..."
        python scripts/10_experiment_runner.py

        # Step 5: CHAIR evaluation (~30-45 min)
        echo "[5/7] CHAIR evaluation (200 captions)..."
        python scripts/08_chair_evaluation.py

        # Step 6: Confidence drift (~1 hour)
        echo "[6/7] Confidence drift monitoring (1000 inferences)..."
        python scripts/09_confidence_monitor.py

        # Step 7: Generate all plots
        echo "[7/7] Generating analysis plots..."
        python scripts/11_analysis_plots.py

        echo ""
        echo "=============================================="
        echo "FULL RUN COMPLETE"
        echo "Results:  outputs/"
        echo "Plots:    outputs/plots/"
        echo "=============================================="
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_all_laptop.sh [quick|full]"
        exit 1
        ;;
esac
