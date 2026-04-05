#!/bin/bash
# setup_env.sh — Run this ONCE to set up the conda environment on HiPerGator
#
# Usage:
#   bash slurm/setup_env.sh

set -e

echo "=== Setting up VLM Hallucination Detection Environment ==="

module load conda
module load cuda/12.1

# Create environment
conda create -n vlm_halluc python=3.10 -y
conda activate vlm_halluc

# PyTorch with CUDA
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
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

# Verify
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import transformers
print(f'Transformers: {transformers.__version__}')
print('Setup complete!')
"

echo "=== Environment setup complete ==="
echo "Next: sbatch slurm/run_experiment.slurm"
