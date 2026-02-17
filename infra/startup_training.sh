#!/bin/bash
# Startup script for Financial-IA GPU Training VM (Deep Learning VM)
set -euo pipefail

LOG="/var/log/financial-ia-setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Financial-IA Training VM Setup ==="
echo "Date: $(date -u)"

# Deep Learning VM already has conda/pip/PyTorch/CUDA
# Install additional project dependencies
pip install --upgrade pip
pip install \
    google-cloud-storage \
    google-cloud-logging \
    pytorch-lightning \
    gymnasium \
    stable-baselines3 \
    einops \
    dacite \
    pyyaml \
    tensorboard \
    pyarrow \
    pandas \
    tqdm

# Verify GPU
echo "=== GPU Status ==="
nvidia-smi || echo "WARNING: nvidia-smi not available yet (driver may still be installing)"

echo "=== Setup complete ==="
