#!/bin/bash
###############################################################################
# v5 Data Preparation — Local (no GPU needed)
#
# Phase 1a: Download all Binance Futures pairs (~299 pairs, ~1-2h)
# Phase 1b: Convert parquet -> .pt tensors (~5 min)
#
# Run in background:
#   nohup ./scripts/run_v5_local_dataprep.sh > dataprep_v5.log 2>&1 &
###############################################################################

set -euo pipefail

PROJECT_DIR="/home/daniel/Documents/Financial_IA"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PYTHONPATH="$PROJECT_DIR"

LOG="dataprep_v5.log"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== v5 Data Preparation Starting ==="
log "Python: $($PYTHON --version)"

# ─── Phase 1a: Download all Binance Futures pairs ───────────────────────────
log "=== Phase 1a: Downloading all Binance USDT-M Futures pairs ==="
log "  Output: data/raw/1h/"
log "  This takes ~1-2 hours..."

$PYTHON scripts/download_massive_data.py \
    --interval 1h \
    --output_dir data/raw \
    --max_concurrent 12 \
    2>&1 | tee -a "$LOG"

N_PARQUETS=$(ls data/raw/1h/*.parquet 2>/dev/null | wc -l)
log "  Download complete: ${N_PARQUETS} parquet files"

# ─── Phase 1b: Convert parquet -> .pt ────────────────────────────────────────
log "=== Phase 1b: Converting parquet -> .pt tensors ==="

$PYTHON scripts/convert_parquet_to_pt.py \
    --input_dir data/raw/1h \
    --output_dir data/ohlcv_v5 \
    2>&1 | tee -a "$LOG"

N_PT=$(ls data/ohlcv_v5/*.pt 2>/dev/null | wc -l)
log "  Conversion complete: ${N_PT} .pt files"

# ─── Summary ────────────────────────────────────────────────────────────────
TOTAL_SIZE=$(du -sh data/ohlcv_v5/ 2>/dev/null | cut -f1)
log ""
log "=== Data Preparation Complete ==="
log "  Parquets:  ${N_PARQUETS} files in data/raw/1h/"
log "  Tensors:   ${N_PT} files in data/ohlcv_v5/"
log "  Size:      ${TOTAL_SIZE:-N/A}"
log ""
log "Next: copy data/ohlcv_v5/ to H100 and run ./scripts/train_v5_pipeline.sh --skip-download --start-phase=2"
