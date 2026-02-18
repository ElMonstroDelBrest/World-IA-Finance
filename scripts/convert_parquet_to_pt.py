#!/usr/bin/env python3
"""Convert downloaded parquet OHLCV files to .pt tensors for Strate I.

Reads each data/raw/1h/{PAIR}.parquet, extracts OHLCV columns (float32),
and saves as data/ohlcv_v5/{PAIR}.pt with shape (T, 5).

Usage:
    PYTHONPATH=. python scripts/convert_parquet_to_pt.py
    PYTHONPATH=. python scripts/convert_parquet_to_pt.py --input_dir data/raw/1h --output_dir data/ohlcv_v5
"""

import argparse
from pathlib import Path

import pandas as pd
import torch


def convert_parquet_to_pt(input_dir: str, output_dir: str) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {in_path}")
        return

    print(f"Found {len(parquet_files)} parquet files in {in_path}")
    ohlcv_cols = ["open", "high", "low", "close", "volume"]

    for pf in parquet_files:
        pair = pf.stem  # e.g. BTCUSDT
        df = pd.read_parquet(pf)

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Extract OHLCV columns
        missing = [c for c in ohlcv_cols if c not in df.columns]
        if missing:
            print(f"  SKIP {pair}: missing columns {missing}")
            continue

        tensor = torch.tensor(df[ohlcv_cols].values, dtype=torch.float32)  # (T, 5)
        torch.save(tensor, out_path / f"{pair}.pt")
        print(f"  {pair}: {tensor.shape[0]} candles -> {out_path / pair}.pt")

    print(f"\nDone. {len(list(out_path.glob('*.pt')))} files in {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet OHLCV to .pt tensors")
    parser.add_argument("--input_dir", default="data/raw/1h", help="Input parquet directory")
    parser.add_argument("--output_dir", default="data/ohlcv_v5", help="Output .pt directory")
    args = parser.parse_args()
    convert_parquet_to_pt(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
