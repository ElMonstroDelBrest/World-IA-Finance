#!/usr/bin/env python3
"""Cloud-native ingestion: Binance tick data → GCS Parquet.

Streams ZIP files from data.binance.vision directly into GCS as Parquet,
without writing to local disk. Optimized for GCP internal bandwidth (>10 Gbps).

Architecture:
    data.binance.vision (ZIP) → memory → CSV extract → Parquet → GCS upload

Data source:
    https://data.binance.vision/data/futures/um/daily/trades/{SYMBOL}/{file}.zip
    Each ZIP contains a single CSV with columns:
    id, price, qty, quoteQty, time, isBuyerMaker

Usage:
    # Ingest all USDT-M futures trades (full history):
    python scripts/gcp_ingest.py --bucket financial-ia-datalake

    # Specific pairs:
    python scripts/gcp_ingest.py --bucket financial-ia-datalake --pairs BTCUSDT ETHUSDT

    # Specific date range:
    python scripts/gcp_ingest.py --bucket financial-ia-datalake --start 2024-01-01 --end 2024-12-31

    # Dry run (list what would be downloaded):
    python scripts/gcp_ingest.py --bucket financial-ia-datalake --dry_run

    # Ingest klines (1h OHLCV) instead of trades:
    python scripts/gcp_ingest.py --bucket financial-ia-datalake --data_type klines --interval 1h
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import PurePosixPath

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from google.cloud import storage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_DATA_BASE = "https://data.binance.vision/data/futures/um/daily"
BINANCE_KLINES_BASE = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_TRADES_BASE = "https://data.binance.vision/data/futures/um/daily/trades"

# Exchange info for pair discovery
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

TRADE_COLUMNS = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker"]
TRADE_DTYPES = {
    "id": "int64",
    "price": "float64",
    "qty": "float64",
    "quoteQty": "float64",
    "time": "int64",
    "isBuyerMaker": "bool",
}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]

MAX_WORKERS = 16
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def get_usdt_perpetuals(min_age_days: int = 180) -> list[str]:
    """Fetch all active USDT-M perpetual futures symbols from Binance."""
    resp = requests.get(EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    now_ms = int(time.time() * 1000)
    pairs = []

    for s in data["symbols"]:
        if (
            s["contractType"] == "PERPETUAL"
            and s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
        ):
            onset_ms = s.get("onboardDate", now_ms)
            age_days = (now_ms - onset_ms) / 86_400_000
            if age_days >= min_age_days:
                pairs.append(s["symbol"])

    return sorted(pairs)


def get_pair_start_date(symbol: str) -> date:
    """Get approximate start date for a pair by probing Binance data archives."""
    # Try progressively older dates until we get a 404
    probe_date = date(2019, 9, 1)
    today = date.today()

    while probe_date < today:
        url = f"{BINANCE_TRADES_BASE}/{symbol}/{symbol}-trades-{probe_date}.zip"
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return probe_date
        except requests.RequestException:
            pass
        probe_date += timedelta(days=90)  # Jump by quarters

    return today - timedelta(days=365)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def blob_exists(bucket: storage.Bucket, blob_name: str) -> bool:
    """Check if a blob already exists in GCS (for checkpointing)."""
    return bucket.blob(blob_name).exists()


def upload_parquet_to_gcs(
    bucket: storage.Bucket,
    blob_name: str,
    table: pa.Table,
) -> int:
    """Upload a PyArrow Table as Parquet directly to GCS. Returns byte size."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    data = buf.getvalue()

    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="application/octet-stream")

    return len(data)


# ---------------------------------------------------------------------------
# Download + transform
# ---------------------------------------------------------------------------

def download_zip_to_memory(url: str) -> bytes | None:
    """Download a ZIP file entirely in memory. Returns bytes or None on error."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 404:
                return None  # File doesn't exist (e.g., pair not trading that day)
            else:
                log.warning(f"HTTP {resp.status_code} for {url}, retry {attempt+1}")
        except requests.RequestException as e:
            log.warning(f"Error fetching {url}: {e}, retry {attempt+1}")
        time.sleep(2 ** attempt)

    return None


def zip_csv_to_arrow_trades(zip_bytes: bytes) -> pa.Table | None:
    """Extract CSV from ZIP bytes and convert to Arrow Table (trades)."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    names=TRADE_COLUMNS,
                    dtype=TRADE_DTYPES,
                    header=0,
                )
    except Exception as e:
        log.warning(f"Failed to parse ZIP: {e}")
        return None

    if df.empty:
        return None

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    return pa.Table.from_pandas(df, preserve_index=False)


def zip_csv_to_arrow_klines(zip_bytes: bytes) -> pa.Table | None:
    """Extract CSV from ZIP bytes and convert to Arrow Table (klines)."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file, names=KLINE_COLUMNS, header=0)
    except Exception as e:
        log.warning(f"Failed to parse ZIP: {e}")
        return None

    if df.empty:
        return None

    # Keep OHLCV columns, convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return pa.Table.from_pandas(df, preserve_index=False)


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_one_day(
    bucket: storage.Bucket,
    symbol: str,
    day: date,
    data_type: str,
    interval: str,
    prefix: str,
) -> dict:
    """Download one day's data, convert to Parquet, upload to GCS.

    Returns stats dict.
    """
    day_str = day.strftime("%Y-%m-%d")

    if data_type == "trades":
        url = f"{BINANCE_TRADES_BASE}/{symbol}/{symbol}-trades-{day_str}.zip"
        gcs_path = f"{prefix}/trades/{symbol}/{day_str}.parquet"
    else:
        url = f"{BINANCE_KLINES_BASE}/{symbol}/{interval}/{symbol}-{interval}-{day_str}.zip"
        gcs_path = f"{prefix}/klines/{interval}/{symbol}/{day_str}.parquet"

    # Checkpoint: skip if already uploaded
    if blob_exists(bucket, gcs_path):
        return {"symbol": symbol, "date": day_str, "status": "exists", "rows": 0, "bytes": 0}

    # Download ZIP in memory
    zip_bytes = download_zip_to_memory(url)
    if zip_bytes is None:
        return {"symbol": symbol, "date": day_str, "status": "missing", "rows": 0, "bytes": 0}

    # Parse to Arrow
    if data_type == "trades":
        table = zip_csv_to_arrow_trades(zip_bytes)
    else:
        table = zip_csv_to_arrow_klines(zip_bytes)

    if table is None or table.num_rows == 0:
        return {"symbol": symbol, "date": day_str, "status": "empty", "rows": 0, "bytes": 0}

    # Upload Parquet to GCS
    n_bytes = upload_parquet_to_gcs(bucket, gcs_path, table)

    return {
        "symbol": symbol,
        "date": day_str,
        "status": "ok",
        "rows": table.num_rows,
        "bytes": n_bytes,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_date_range(start: date, end: date) -> list[date]:
    """Generate list of dates from start to end (inclusive)."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def run_ingest(args):
    """Main ingestion orchestrator."""
    # GCS client (uses Application Default Credentials)
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    # Verify bucket exists
    if not bucket.exists():
        log.error(f"Bucket {args.bucket} does not exist. Create it first (terraform apply).")
        return

    # Discover pairs
    if args.pairs:
        pairs = args.pairs
    else:
        log.info("Discovering all USDT-M perpetual futures...")
        pairs = get_usdt_perpetuals(min_age_days=180)

    log.info(f"Pairs: {len(pairs)}")

    # Date range
    end_date = date.today() - timedelta(days=1)  # Yesterday (today may be incomplete)
    if args.end:
        end_date = min(end_date, datetime.strptime(args.end, "%Y-%m-%d").date())

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = date(2020, 1, 1)  # Default: from 2020

    dates = generate_date_range(start_date, end_date)
    log.info(f"Date range: {start_date} → {end_date} ({len(dates)} days)")

    # Build task list: (symbol, date) pairs
    tasks = [(symbol, d) for symbol in pairs for d in dates]
    log.info(f"Total tasks: {len(tasks):,} ({len(pairs)} pairs × {len(dates)} days)")

    if args.dry_run:
        print(f"\nWould process {len(tasks):,} files")
        print(f"  Pairs: {len(pairs)}")
        print(f"  Days: {len(dates)}")
        print(f"  Data type: {args.data_type}")
        if args.data_type == "klines":
            print(f"  Interval: {args.interval}")
        print(f"  GCS prefix: gs://{args.bucket}/{args.prefix}/")
        print(f"\nFirst 10 pairs: {pairs[:10]}")
        return

    # Execute in parallel
    total_rows = 0
    total_bytes = 0
    n_ok = 0
    n_exists = 0
    n_missing = 0
    n_errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for symbol, d in tasks:
            fut = executor.submit(
                process_one_day,
                bucket, symbol, d,
                args.data_type, args.interval, args.prefix,
            )
            futures[fut] = (symbol, d)

        with tqdm(total=len(futures), desc="Ingesting", unit="file") as pbar:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    status = result["status"]
                    if status == "ok":
                        n_ok += 1
                        total_rows += result["rows"]
                        total_bytes += result["bytes"]
                    elif status == "exists":
                        n_exists += 1
                    elif status == "missing":
                        n_missing += 1
                    else:
                        n_errors += 1
                except Exception as e:
                    n_errors += 1
                    sym, d = futures[fut]
                    log.error(f"Exception for {sym}/{d}: {e}")

                pbar.update(1)
                pbar.set_postfix(
                    ok=n_ok, skip=n_exists, miss=n_missing, err=n_errors,
                    MB=f"{total_bytes/1e6:.0f}",
                )

    # Summary
    print("\n" + "=" * 60)
    print(f"INGESTION COMPLETE — {args.data_type}")
    print("=" * 60)
    print(f"  Uploaded:     {n_ok:,} files ({total_rows:,} rows, {total_bytes/1e9:.2f} GB)")
    print(f"  Skipped:      {n_exists:,} files (already in GCS)")
    print(f"  Missing:      {n_missing:,} files (pair not trading that day)")
    print(f"  Errors:       {n_errors:,} files")
    print(f"  Destination:  gs://{args.bucket}/{args.prefix}/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Binance Futures data to GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bucket", type=str, required=True,
                        help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default="raw",
                        help="GCS path prefix (default: raw)")
    parser.add_argument("--data_type", type=str, default="trades",
                        choices=["trades", "klines"],
                        help="Data type to ingest (default: trades)")
    parser.add_argument("--interval", type=str, default="1h",
                        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                        help="Kline interval (only for --data_type klines)")
    parser.add_argument("--pairs", nargs="+", type=str, default=None,
                        help="Specific pairs (default: all)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--dry_run", action="store_true",
                        help="List tasks without executing")
    args = parser.parse_args()

    run_ingest(args)


if __name__ == "__main__":
    main()
