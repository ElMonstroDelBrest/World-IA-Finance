"""Pretokenize and analyze token distributions for BTC vs volatile altcoin.

Usage:
    PYTHONPATH=. python scripts/analyze_tokens.py \
        --checkpoint checkpoints/strate-i-epoch=09-val/loss/total=-0.0575.ckpt \
        --data_dir data/binance_1h_subset \
        --pairs BTCUSDT PEPEUSDT
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from src.strate_i.config import load_config as load_strate_i_config
from src.strate_i.data.transforms import compute_log_returns, extract_patches
from src.strate_i.lightning_module import StrateILightningModule


def tokenize_pair(tokenizer, data_path: Path, patch_length: int, stride: int) -> torch.Tensor:
    """Tokenize a single pair's OHLCV data."""
    ohlcv = torch.load(data_path, weights_only=True)
    log_ret = compute_log_returns(ohlcv)
    patches = extract_patches(log_ret, patch_length, stride)
    if patches.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        # Process in batches to avoid OOM
        all_tokens = []
        batch_size = 256
        for i in range(0, patches.shape[0], batch_size):
            batch = patches[i:i + batch_size]
            tokens = tokenizer.tokenize(batch)
            all_tokens.append(tokens)
        return torch.cat(all_tokens)


def compute_stats(tokens: torch.Tensor, num_codes: int = 1024) -> dict:
    """Compute distribution statistics for a token sequence."""
    if tokens.numel() == 0:
        return {}

    counts = Counter(tokens.tolist())
    total = tokens.numel()

    # Top tokens
    top_20 = counts.most_common(20)

    # Distribution metrics
    probs = torch.zeros(num_codes)
    for idx, count in counts.items():
        probs[idx] = count / total

    used_codes = (probs > 0).sum().item()
    entropy = -(probs[probs > 0] * torch.log2(probs[probs > 0])).sum().item()
    max_entropy = np.log2(num_codes)

    return {
        "total_tokens": total,
        "unique_codes": used_codes,
        "utilization": used_codes / num_codes,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy,
        "top_20": top_20,
        "counts": counts,
        "probs": probs,
    }


def print_report(pair_name: str, stats: dict):
    """Print a formatted distribution report."""
    print(f"\n{'='*60}")
    print(f"  {pair_name} — Token Distribution Report")
    print(f"{'='*60}")
    print(f"  Total tokens:       {stats['total_tokens']:,}")
    print(f"  Unique codes used:  {stats['unique_codes']}/{1024} ({stats['utilization']:.1%})")
    print(f"  Entropy:            {stats['entropy']:.2f} / {stats['max_entropy']:.2f} bits ({stats['normalized_entropy']:.1%})")
    print(f"\n  Top 20 tokens:")
    print(f"  {'Rank':>4}  {'Token':>6}  {'Count':>6}  {'Freq':>7}  {'Bar'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*30}")
    for rank, (token, count) in enumerate(stats['top_20'], 1):
        freq = count / stats['total_tokens']
        bar = '█' * int(freq * 200)
        print(f"  {rank:>4}  {token:>6}  {count:>6}  {freq:>6.2%}  {bar}")


def print_comparison(name_a: str, stats_a: dict, name_b: str, stats_b: dict):
    """Compare two pairs' token distributions."""
    print(f"\n{'='*60}")
    print(f"  Comparison: {name_a} vs {name_b}")
    print(f"{'='*60}")

    # Shared tokens
    codes_a = set(stats_a['counts'].keys())
    codes_b = set(stats_b['counts'].keys())
    shared = codes_a & codes_b
    only_a = codes_a - codes_b
    only_b = codes_b - codes_a

    print(f"  Shared codes:     {len(shared)}")
    print(f"  Only in {name_a}:  {len(only_a)}")
    print(f"  Only in {name_b}:  {len(only_b)}")
    print(f"  Jaccard similarity: {len(shared) / len(codes_a | codes_b):.3f}")

    # Jensen-Shannon divergence
    p = stats_a['probs']
    q = stats_b['probs']
    m = 0.5 * (p + q)
    # Avoid log(0)
    mask = m > 0
    kl_pm = torch.zeros_like(p)
    kl_qm = torch.zeros_like(q)
    kl_pm[p > 0] = (p[p > 0] * torch.log2(p[p > 0] / m[p > 0])).clamp(min=0)
    kl_qm[q > 0] = (q[q > 0] * torch.log2(q[q > 0] / m[q > 0])).clamp(min=0)
    jsd = 0.5 * kl_pm.sum() + 0.5 * kl_qm.sum()
    print(f"  Jensen-Shannon divergence: {jsd:.4f} bits")

    # Top tokens overlap
    top_a = set(t for t, _ in stats_a['top_20'][:10])
    top_b = set(t for t, _ in stats_b['top_20'][:10])
    overlap = top_a & top_b
    print(f"  Top-10 overlap:    {len(overlap)} tokens in common")
    if overlap:
        print(f"    Shared top tokens: {sorted(overlap)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i_binance.yaml")
    parser.add_argument("--data_dir", type=str, default="data/binance_1h_subset")
    parser.add_argument("--pairs", nargs="+", default=["BTCUSDT", "PEPEUSDT"])
    args = parser.parse_args()

    config = load_strate_i_config(args.strate_i_config)

    # Load model
    model = StrateILightningModule.load_from_checkpoint(
        args.checkpoint, config=config
    )
    model.eval()
    tokenizer = model.tokenizer

    patch_length = config.patch.patch_length
    stride = config.patch.stride

    # Tokenize each pair
    all_stats = {}
    for pair in args.pairs:
        data_path = Path(args.data_dir) / f"{pair}.pt"
        if not data_path.exists():
            print(f"Warning: {data_path} not found, skipping")
            continue

        print(f"Tokenizing {pair}...")
        tokens = tokenize_pair(tokenizer, data_path, patch_length, stride)
        stats = compute_stats(tokens, num_codes=config.codebook.num_codes)
        all_stats[pair] = stats
        print_report(pair, stats)

    # Comparison if we have 2+ pairs
    pairs = list(all_stats.keys())
    if len(pairs) >= 2:
        print_comparison(pairs[0], all_stats[pairs[0]], pairs[1], all_stats[pairs[1]])

    # Save token files for Strate II
    output_dir = Path("data/tokens")
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs:
        data_path = Path(args.data_dir) / f"{pair}.pt"
        if not data_path.exists():
            continue
        tokens = tokenize_pair(tokenizer, data_path, patch_length, stride)
        seq_len = 64
        # Create multiple sequences of seq_len from the token stream
        n_seqs = tokens.shape[0] // seq_len
        for i in range(n_seqs):
            chunk = tokens[i * seq_len : (i + 1) * seq_len]
            # Weekend mask: all zeros for crypto (24/7 trading)
            weekend_mask = torch.zeros(seq_len, dtype=torch.float32)
            torch.save(
                {"token_indices": chunk, "weekend_mask": weekend_mask},
                output_dir / f"{pair}_seq{i:04d}.pt",
            )
        print(f"\nSaved {n_seqs} sequences for {pair} to {output_dir}/")


if __name__ == "__main__":
    main()
