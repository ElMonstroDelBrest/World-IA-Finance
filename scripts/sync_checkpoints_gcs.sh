#!/bin/bash
###############################################################################
# Financial-IA — GCS Checkpoint Sync (Cron-friendly)
#
# Syncs local checkpoints and logs to GCS bucket every run.
# Designed to run as a cron job every 10 minutes on Spot VMs.
#
# Setup (on the VM):
#   chmod +x scripts/sync_checkpoints_gcs.sh
#
#   # Run once manually:
#   ./scripts/sync_checkpoints_gcs.sh
#
#   # Setup cron (every 10 min):
#   (crontab -l 2>/dev/null; echo "*/10 * * * * /home/daniel/Financial_IA/scripts/sync_checkpoints_gcs.sh >> /tmp/gcs_sync.log 2>&1") | crontab -
#
#   # Or run in background loop:
#   ./scripts/sync_checkpoints_gcs.sh --loop &
###############################################################################

set -euo pipefail

BUCKET="${GCS_BUCKET:-gs://financial-ia-datalake}"
VERSION="${TRAIN_VERSION:-v4}"
PROJECT_DIR="${PROJECT_DIR:-/home/daniel/Financial_IA}"
SYNC_INTERVAL=600  # 10 minutes (for --loop mode)

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] GCS checkpoint sync starting..."

sync_to_gcs() {
    # Sync Strate II checkpoints (versioned)
    if [ -d "${PROJECT_DIR}/checkpoints/strate_ii" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/checkpoints/strate_ii/" \
            "${BUCKET}/${VERSION}/checkpoints/strate_ii/" \
            2>&1 | tail -3
        echo "  Synced strate_ii checkpoints → ${VERSION}/"
    fi

    # Sync Strate IV checkpoints (versioned)
    if [ -d "${PROJECT_DIR}/tb_logs/strate_iv" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/tb_logs/strate_iv/" \
            "${BUCKET}/${VERSION}/checkpoints/strate_iv/" \
            2>&1 | tail -3
        echo "  Synced strate_iv checkpoints → ${VERSION}/"
    fi

    # Explicitly sync VecNormalize from best_model/
    if [ -f "${PROJECT_DIR}/tb_logs/strate_iv/best_model/vecnormalize.pkl" ]; then
        gsutil cp \
            "${PROJECT_DIR}/tb_logs/strate_iv/best_model/vecnormalize.pkl" \
            "${BUCKET}/${VERSION}/checkpoints/strate_iv/best_model/vecnormalize.pkl" \
            2>/dev/null
        echo "  Synced vecnormalize.pkl → ${VERSION}/best_model/"
    fi

    # Sync TensorBoard logs (versioned)
    if [ -d "${PROJECT_DIR}/tb_logs" ]; then
        gsutil -m rsync -r \
            "${PROJECT_DIR}/tb_logs/" \
            "${BUCKET}/${VERSION}/tb_logs/" \
            2>&1 | tail -3
        echo "  Synced TensorBoard logs → ${VERSION}/"
    fi

    # Sync training log
    if [ -f "${PROJECT_DIR}/training_h100.log" ]; then
        gsutil cp "${PROJECT_DIR}/training_h100.log" \
            "${BUCKET}/${VERSION}/logs/training_h100.log" \
            2>/dev/null
    fi

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Sync complete (${VERSION})."
}

# Main logic
if [ "${1:-}" = "--loop" ]; then
    echo "Running in loop mode (every ${SYNC_INTERVAL}s)..."
    while true; do
        sync_to_gcs || echo "  WARNING: sync failed, will retry"
        sleep "$SYNC_INTERVAL"
    done
else
    sync_to_gcs
fi
