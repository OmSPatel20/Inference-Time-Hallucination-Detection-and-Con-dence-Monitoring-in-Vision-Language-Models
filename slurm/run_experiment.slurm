#!/bin/bash
#SBATCH --job-name=vlm_halluc
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=a100:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --account=YOUR_GROUP_HERE
#    ^^^ CHANGE THIS to your HiPerGator group account

# Usage:
#   sbatch slurm/run_experiment.slurm                     # Full experiment matrix
#   sbatch slurm/run_experiment.slurm --export=MODE=quick  # Quick test
#   sbatch slurm/run_experiment.slurm --export=MODE=drift  # Drift monitoring only

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Setup
module load conda cuda/12.1
conda activate vlm_halluc

cd $SLURM_SUBMIT_DIR
mkdir -p outputs/logs

MODE=${MODE:-full}

case $MODE in
    quick)
        echo "=== QUICK TEST MODE ==="
        python scripts/10_experiment_runner.py --quick --max_samples 100
        ;;

    drift)
        echo "=== DRIFT MONITORING MODE ==="
        python scripts/09_confidence_monitor.py --quantization fp16 --num_samples 2000
        python scripts/09_confidence_monitor.py --quantization int4 --num_samples 2000
        ;;

    pope_only)
        echo "=== POPE EVALUATION ONLY ==="
        for quant in fp16 int8 int4; do
            for split in random popular adversarial; do
                echo "Running: $quant x $split"
                python scripts/07_pope_evaluation.py \
                    --split $split --quantization $quant --max_samples 3000
            done
        done
        ;;

    chair_only)
        echo "=== CHAIR EVALUATION ONLY ==="
        for quant in fp16 int8 int4; do
            python scripts/08_chair_evaluation.py --quantization $quant --num_samples 500
        done
        ;;

    full)
        echo "=== FULL EXPERIMENT MATRIX ==="

        # Step 1: Download data (if not already done)
        python scripts/01_download_data.py

        # Step 2: Benchmark throughput
        python scripts/03_batched_pipeline.py

        # Step 3: Full POPE evaluation matrix
        python scripts/10_experiment_runner.py

        # Step 4: CHAIR evaluation
        for quant in fp16 int4; do
            python scripts/08_chair_evaluation.py --quantization $quant --num_samples 500
        done

        # Step 5: Confidence drift
        python scripts/09_confidence_monitor.py --quantization fp16 --num_samples 2000

        # Step 6: Generate plots
        python scripts/11_analysis_plots.py
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Options: quick, drift, pope_only, chair_only, full"
        exit 1
        ;;
esac

echo "Job completed: $(date)"
