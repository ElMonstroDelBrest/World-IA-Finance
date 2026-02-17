"""Ingest Binance klines CSV files into OHLCV tensors for Strate I.

Reads CSV files from klines directories (1h, 5m, 1m), concatenates
monthly files per pair, extracts OHLCV columns, saves as .pt tensors.

Binance CSV format (no header):
  open_time, O, H, L, C, volume, close_time, quote_volume,
  trades, taker_buy_vol, taker_buy_quote_vol, ignore

Usage:
    # Ingest all 1h pairs:
    python scripts/ingest_binance.py \
        --klines_dir /mnt/wd/trading/klines_1h \
        --output_dir data/binance_1h

    # Ingest specific pairs only:
    python scripts/ingest_binance.py \
        --klines_dir /mnt/wd/trading/klines_1h \
        --output_dir data/binance_1h \
        --pairs BTCUSDT ETHUSDT DOGEUSDT
"""

import argparse
import csv
from pathlib import Path

import torch


def read_klines_csv(csv_path: Path) -> list[list[float]]:
    """Read a single Binance klines CSV file.

    Returns list of [O, H, L, C, V] rows, sorted by open_time.
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
                v = float(row[5])
                t = int(row[0])
                rows.append((t, o, h, l, c, v))
            except (ValueError, IndexError):
                continue
    # Sort by timestamp
    rows.sort(key=lambda x: x[0])
    # Return OHLCV only (drop timestamp)
    return [[r[1], r[2], r[3], r[4], r[5]] for r in rows]


def ingest_pair(pair_dir: Path) -> torch.Tensor | None:
    """Ingest all monthly CSV files for a single pair.

    Args:
        pair_dir: Directory containing monthly CSV files (e.g., 2024-02.csv).

    Returns:
        (T, 5) float32 tensor or None if no data.
    """
    csv_files = sorted(pair_dir.glob("*.csv"))
    if not csv_files:
        return None

    all_rows = []
    for csv_file in csv_files:
        rows = read_klines_csv(csv_file)
        all_rows.extend(rows)

    if not all_rows:
        return None

    tensor = torch.tensor(all_rows, dtype=torch.float32)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Ingest Binance klines to OHLCV tensors")
    parser.add_argument("--klines_dir", type=str, required=True, help="Path to klines directory")
    parser.add_argument("--output_dir", type=str, default="data/binance_1h", help="Output directory")
    parser.add_argument("--pairs", nargs="*", default=None, help="Specific pairs to ingest (default: all)")
    args = parser.parse_args()

    klines_path = Path(args.klines_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get pair directories
    if args.pairs:
        pair_dirs = [klines_path / p for p in args.pairs]
        pair_dirs = [d for d in pair_dirs if d.exists()]
    else:
        pair_dirs = sorted([d for d in klines_path.iterdir() if d.is_dir()])

    print(f"Ingesting {len(pair_dirs)} pairs from {klines_path}")

    stats = {"total": 0, "skipped": 0, "timesteps": 0}

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        tensor = ingest_pair(pair_dir)

        if tensor is None or tensor.shape[0] < 32:
            print(f"  SKIP {pair_name}: insufficient data")
            stats["skipped"] += 1
            continue

        out_file = output_path / f"{pair_name}.pt"
        torch.save(tensor, out_file)
        stats["total"] += 1
        stats["timesteps"] += tensor.shape[0]
        print(f"  {pair_name}: {tensor.shape[0]} candles → {out_file.name}")

    print(f"\nDone: {stats['total']} pairs ingested, {stats['skipped']} skipped")
    print(f"Total timesteps: {stats['timesteps']:,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
