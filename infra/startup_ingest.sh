#!/bin/bash
# Startup script for Financial-IA Data Ingest VM
set -euo pipefail

LOG="/var/log/financial-ia-setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Financial-IA Ingest VM Setup ==="
echo "Date: $(date -u)"

# System updates
apt-get update -y
apt-get install -y python3-pip python3-venv git htop tmux

# Create project directory
PROJECT_DIR="/opt/financial-ia"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install \
    google-cloud-storage \
    google-cloud-logging \
    pyarrow \
    pandas \
    aiohttp \
    tqdm \
    requests

echo "=== Setup complete ==="
echo "Activate with: source /opt/financial-ia/.venv/bin/activate"
