#!/usr/bin/env bash
# Setup dependencies on MSI remote machine for Strate IV training.
#
# Usage: ssh msi "bash -s" < scripts/setup_msi.sh

set -euo pipefail

echo "=== Setting up Strate IV dependencies on MSI ==="

# Ensure we have a working Python environment
if command -v conda &> /dev/null; then
    echo "Using conda environment"
    conda activate financial_ia 2>/dev/null || conda create -n financial_ia python=3.11 -y && conda activate financial_ia
fi

# Install PyTorch (CUDA)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install pytorch-lightning>=2.0 gymnasium>=1.0 stable-baselines3>=2.0
pip install dacite pyyaml tensorboard einops numpy pandas

# Install project in editable mode
cd ~/Financial_IA
pip install -e ".[dev]"

# Verify installation
python -c "
import torch
import gymnasium
import stable_baselines3
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Gymnasium: {gymnasium.__version__}')
print(f'SB3: {stable_baselines3.__version__}')
"

echo "=== Setup complete ==="
