#!/usr/bin/env bash
# Sync code to MSI and retrieve logs/checkpoints.
#
# Usage:
#   ./scripts/sync_to_msi.sh push     # Push code to MSI
#   ./scripts/sync_to_msi.sh pull     # Pull logs/checkpoints from MSI
#   ./scripts/sync_to_msi.sh both     # Push code, pull results

set -euo pipefail

# Configuration — edit these for your MSI setup
MSI_HOST="${MSI_HOST:-msi}"
MSI_DIR="${MSI_DIR:-~/Financial_IA}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ACTION="${1:-both}"

push_code() {
    echo "=== Pushing code to ${MSI_HOST}:${MSI_DIR} ==="
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.eggs' \
        --exclude '*.egg-info' \
        --exclude 'tb_logs/' \
        --exclude 'checkpoints/' \
        --exclude 'data/trajectory_buffer/' \
        --exclude 'outputs/' \
        --exclude '.venv' \
        --exclude 'venv' \
        "${LOCAL_DIR}/" "${MSI_HOST}:${MSI_DIR}/"
    echo "=== Push complete ==="
}

pull_results() {
    echo "=== Pulling logs/checkpoints from ${MSI_HOST}:${MSI_DIR} ==="

    # Pull TensorBoard logs
    mkdir -p "${LOCAL_DIR}/tb_logs/strate_iv/"
    rsync -avz --progress \
        "${MSI_HOST}:${MSI_DIR}/tb_logs/strate_iv/" \
        "${LOCAL_DIR}/tb_logs/strate_iv/"

    # Pull checkpoints
    mkdir -p "${LOCAL_DIR}/checkpoints/strate_iv/"
    rsync -avz --progress \
        "${MSI_HOST}:${MSI_DIR}/tb_logs/strate_iv/checkpoints/" \
        "${LOCAL_DIR}/checkpoints/strate_iv/" 2>/dev/null || true

    echo "=== Pull complete ==="
}

case "$ACTION" in
    push)
        push_code
        ;;
    pull)
        pull_results
        ;;
    both)
        push_code
        pull_results
        ;;
    *)
        echo "Usage: $0 {push|pull|both}"
        exit 1
        ;;
esac
