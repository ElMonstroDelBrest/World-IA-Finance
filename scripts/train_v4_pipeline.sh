#!/bin/bash
###############################################################################
# Financial-IA v4 — Full Training Pipeline (H100)
#
# Prerequisites:
#   - Strate I checkpoint exists (10 epochs VQ-VAE)
#   - Tokens pre-computed in data/tokens_full/
#   - Strate II can auto-resume from last.ckpt
#
# This script:
#   1. Fixes data symlinks
#   2. Resumes/finishes Strate II training (epoch 34 → 300)
#   3. Pre-computes trajectory buffer v4 (600 episodes, 16 futures)
#   4. Trains Strate IV PPO with VecNormalize fixes
#   5. Starts GCS sync
#
# Usage:
#   chmod +x scripts/train_v4_pipeline.sh
#   nohup ./scripts/train_v4_pipeline.sh > training_v4.log 2>&1 &
###############################################################################

set -euo pipefail

PROJECT_DIR="${HOME}/Financial_IA"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

LOG="training_v4.log"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Financial-IA v4 Pipeline Starting ==="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
log "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

# Ensure 'python' exists (some Deep Learning VMs only have python3)
if ! command -v python &>/dev/null; then
    alias python=python3
    export PATH="$(dirname $(which python3)):$PATH"
fi
PYTHON=$(which python3 2>/dev/null || which python)

# ─── Step 0: Fix data symlinks ──────────────────────────────────────────────
log "Step 0: Setting up data paths..."

# Strate II config references data/tokens/ but real data is in data/tokens_full/
if [ ! -L "data/tokens" ] && [ -d "data/tokens_full" ]; then
    ln -sfn tokens_full data/tokens
    log "  Symlinked data/tokens → data/tokens_full"
fi

# Strate I/precompute references data/binance_1h_subset/ but we have data/ohlcv_full/
if [ ! -L "data/binance_1h_subset" ] && [ -d "data/ohlcv_full" ]; then
    ln -sfn ohlcv_full data/binance_1h_subset
    log "  Symlinked data/binance_1h_subset → data/ohlcv_full"
fi

# Also create data/binance_1h/ symlink (referenced in plan)
if [ ! -L "data/binance_1h" ] && [ -d "data/ohlcv_full" ]; then
    ln -sfn ohlcv_full data/binance_1h
    log "  Symlinked data/binance_1h → data/ohlcv_full"
fi

# Find best Strate I checkpoint
STRATE_I_CKPT=$(find checkpoints -path '*strate-i-epoch=09*' -name '*.ckpt' | head -1)
log "  Strate I checkpoint: $STRATE_I_CKPT"

# Find best/last Strate II checkpoint
STRATE_II_LAST="checkpoints/strate_ii/last.ckpt"
log "  Strate II last checkpoint: $STRATE_II_LAST"

# ─── Step 1: Train Strate II (Fin-JEPA) — Fresh start ────────────────────────
log "=== Step 1: Strate II (Fin-JEPA) — Fresh v4 training ==="

# Clean old checkpoints to avoid resume conflicts (compiled vs non-compiled keys)
if [ -f "checkpoints/strate_ii/last.ckpt" ]; then
    mkdir -p checkpoints/strate_ii_backup_pre_v4
    mv checkpoints/strate_ii/last*.ckpt checkpoints/strate_ii_backup_pre_v4/ 2>/dev/null || true
    log "  Moved old last.ckpt to backup (incompatible with torch.compile)"
fi

# Full training: 300 epochs on 5373 token sequences
# Config has batch_size: 32 (dev); for full dataset with --compile on H100
$PYTHON scripts/train_strate_ii.py \
    --config configs/strate_ii.yaml \
    --compile \
    --no_resume \
    2>&1 | tee -a "$LOG"

log "  Strate II training complete."

# Find the best Strate II checkpoint (lowest val loss)
STRATE_II_BEST=$(find checkpoints/strate_ii -name '*.ckpt' ! -name 'last*' \
    -printf '%f\t%p\n' | sort | head -1 | cut -f2)
if [ -z "$STRATE_II_BEST" ]; then
    STRATE_II_BEST="checkpoints/strate_ii/last.ckpt"
fi
log "  Best Strate II checkpoint: $STRATE_II_BEST"

# ─── Step 2: Pre-compute trajectory buffer v4 ───────────────────────────────
log "=== Step 2: Trajectory buffer v4 (600 episodes, 16 futures) ==="

$PYTHON scripts/precompute_trajectories.py \
    --strate_i_checkpoint "$STRATE_I_CKPT" \
    --strate_ii_checkpoint "$STRATE_II_BEST" \
    --token_dir data/tokens/ \
    --ohlcv_dir data/ohlcv_full/ \
    --output_dir data/trajectory_buffer_v4/ \
    --n_episodes 600 \
    --stratified \
    2>&1 | tee -a "$LOG"

log "  Trajectory buffer v4 complete: $(ls data/trajectory_buffer_v4/*.pt 2>/dev/null | wc -l) episodes"

# ─── Step 3: Train Strate IV — PPO ──────────────────────────────────────────
log "=== Step 3: Strate IV (PPO) — 1M timesteps ==="

$PYTHON scripts/train_strate_iv.py \
    --config configs/strate_iv.yaml \
    --buffer_dir data/trajectory_buffer_v4/ \
    --no_resume \
    2>&1 | tee -a "$LOG"

log "  Strate IV training complete."

# ─── Step 4: Verify outputs ─────────────────────────────────────────────────
log "=== Step 4: Verification ==="

# Check best model + vecnormalize exist together
BEST_DIR="tb_logs/strate_iv/best_model"
if [ -f "$BEST_DIR/best_model.zip" ] && [ -f "$BEST_DIR/vecnormalize.pkl" ]; then
    log "  best_model.zip + vecnormalize.pkl present in $BEST_DIR"
else
    log "  WARNING: Missing files in $BEST_DIR!"
    ls -la "$BEST_DIR/" 2>/dev/null
fi

# Run a quick demo
$PYTHON scripts/demo_results.py \
    --model_path "$BEST_DIR/best_model.zip" \
    --buffer_dir data/trajectory_buffer_v4/ \
    --output outputs/demo_v4.png \
    --n_demos 5 \
    2>&1 | tee -a "$LOG"

log "  Demo plots saved to outputs/"

# ─── Step 5: Sync to GCS ────────────────────────────────────────────────────
log "=== Step 5: Syncing to GCS (v4 prefix) ==="

export TRAIN_VERSION=v4
bash scripts/sync_checkpoints_gcs.sh 2>&1 | tee -a "$LOG"

log "=== v4 Pipeline Complete ==="
log "Outputs:"
log "  Model:       $BEST_DIR/best_model.zip"
log "  VecNormalize: $BEST_DIR/vecnormalize.pkl"
log "  Buffer:      data/trajectory_buffer_v4/"
log "  Demos:       outputs/demo_v4*.png"
