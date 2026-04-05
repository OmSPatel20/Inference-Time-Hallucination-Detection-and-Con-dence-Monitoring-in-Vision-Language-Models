@echo off
REM setup_env_windows.bat — One-time environment setup for Windows laptop
REM
REM Prerequisites:
REM   - Anaconda or Miniconda installed
REM   - NVIDIA GPU drivers installed (check: nvidia-smi)
REM   - CUDA 12.1 compatible GPU (RTX 4060 = yes)
REM
REM Usage:
REM   Open Anaconda Prompt, cd to project folder, then:
REM   setup_env_windows.bat

echo === Setting up VLM Hallucination Detection Environment ===
echo.

REM Step 1: Verify GPU
echo [1/4] Checking GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if errorlevel 1 (
    echo ERROR: nvidia-smi failed. Install NVIDIA drivers first.
    pause
    exit /b 1
)
echo.

REM Step 2: Create conda env
echo [2/4] Creating conda environment...
conda create -n vlm_halluc python=3.10 -y
if errorlevel 1 (
    echo ERROR: conda create failed.
    pause
    exit /b 1
)

REM Step 3: Install packages
echo [3/4] Installing packages (this takes 5-10 minutes)...
call conda activate vlm_halluc

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate==0.28.0 bitsandbytes==0.43.0 Pillow==10.3.0 scipy==1.13.0 scikit-learn==1.4.2 matplotlib==3.8.4 seaborn==0.13.2 pandas==2.2.1 pyyaml==6.0.1 tqdm==4.66.2 pycocotools==2.0.7 psutil==5.9.8

REM Step 4: Verify
echo [4/4] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NO GPU'); import transformers; print(f'Transformers: {transformers.__version__}')"

echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. conda activate vlm_halluc
echo   2. run_all_laptop.bat quick
echo.
pause
