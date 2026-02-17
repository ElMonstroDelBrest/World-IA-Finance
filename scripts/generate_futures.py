"""Generate N divergent OHLCV future trajectories (Strate III Multiverse).

Usage:
    PYTHONPATH=. python scripts/generate_futures.py \
        --strate_ii_checkpoint checkpoints/strate_ii/best.ckpt \
        --strate_i_checkpoint checkpoints/strate_i_best.ckpt \
        --pair BTCUSDT \
        --data_dir data/binance_1h_subset \
        --n_samples 32 \
        --output_dir outputs/multiverse/

    # Smoke test with synthetic data (no checkpoints needed):
    PYTHONPATH=. python scripts/generate_futures.py --smoke_test
"""

import argparse
from pathlib import Path

import torch


def smoke_test(n_samples: int = 8, output_dir: str = "outputs/multiverse/"):
    """Run a smoke test with random weights — no checkpoints needed."""
    from src.strate_i.decoder import Decoder
    from src.strate_i.revin import RevIN
    from src.strate_ii.jepa import FinJEPA
    from src.strate_ii.multiverse import MultiverseGenerator

    print("=== Smoke test: generating futures with random weights ===")

    jepa = FinJEPA(
        num_codes=1024, codebook_dim=64, d_model=128, d_state=16,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=64, pred_hidden_dim=256, pred_n_layers=2,
        pred_z_dim=32,
    )
    jepa.eval()

    decoder = Decoder(latent_dim=64, hidden_channels=128, out_channels=5,
                      patch_length=16, n_layers=4, kernel_size=3)
    decoder.eval()

    revin = RevIN(n_channels=5)

    generator = MultiverseGenerator(jepa, decoder, revin)

    B, S, N_tgt = 1, 48, 8
    tokens = torch.randint(0, 1024, (B, S))
    target_pos = torch.arange(S, S + N_tgt).clamp(max=63).unsqueeze(0)
    # Fake OHLCV context: BTC-like prices
    context_ohlcv = torch.randn(B, S * 16, 5).abs() * 100 + 30000

    result = generator.generate(
        tokens, None, target_pos, context_ohlcv, n_samples=n_samples,
    )

    print(f"  Latents shape:      {result['latents'].shape}")
    print(f"  Codebook Z shape:   {result['codebook_z'].shape}")
    print(f"  Patches norm shape: {result['patches_norm'].shape}")
    print(f"  Patches real shape: {result['patches_real'].shape}")
    print(f"  OHLCV shape:        {result['ohlcv'].shape}")

    ohlcv = result["ohlcv"]  # (N, B, N_tgt, patch_len, 5)
    print(f"\n  OHLCV sample 0, patch 0, first 3 candles:")
    for t in range(min(3, ohlcv.shape[3])):
        o, h, l, c, v = ohlcv[0, 0, 0, t].tolist()
        print(f"    t={t}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} V={v:.0f}")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path / "smoke_test.pt")
    print(f"\n  Saved to {out_path / 'smoke_test.pt'}")

    # Visualization
    _plot_futures(result, out_path / "smoke_test.png")
    print("=== Smoke test passed ===")


def generate_from_checkpoints(args):
    """Generate futures from trained checkpoints."""
    from src.strate_i.config import load_config as load_strate_i_config
    from src.strate_i.lightning_module import StrateILightningModule
    from src.strate_i.decoder import Decoder
    from src.strate_i.revin import RevIN
    from src.strate_ii.config import load_config as load_strate_ii_config
    from src.strate_ii.lightning_module import StrateIILightningModule
    from src.strate_ii.multiverse import MultiverseGenerator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Strate II
    strate_ii_config = load_strate_ii_config(args.strate_ii_config)
    strate_ii = StrateIILightningModule.load_from_checkpoint(
        args.strate_ii_checkpoint, config=strate_ii_config, strict=False,
    )
    jepa = strate_ii.jepa.to(device).eval()

    # Load Strate I for decoder + codebook
    strate_i_config = load_strate_i_config(args.strate_i_config)
    strate_i = StrateILightningModule.load_from_checkpoint(
        args.strate_i_checkpoint, config=strate_i_config,
    )
    decoder = strate_i.tokenizer.vqvae.decoder.to(device).eval()
    revin = RevIN(n_channels=5).to(device)

    # Load codebook into JEPA
    codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
    jepa.load_codebook(codebook_weights.to(device))

    generator = MultiverseGenerator(jepa, decoder, revin)

    # Load data
    token_path = Path(args.data_dir) / f"{args.pair}_tokens.pt"
    if not token_path.exists():
        print(f"Token file not found: {token_path}")
        print("Looking for any available token files...")
        available = list(Path(args.data_dir).glob("*_tokens.pt"))
        if available:
            token_path = available[0]
            print(f"Using: {token_path}")
        else:
            raise FileNotFoundError(f"No token files found in {args.data_dir}")

    data = torch.load(token_path, weights_only=True)
    if isinstance(data, dict):
        tokens = data.get("token_indices", data.get("tokens"))
        ohlcv_raw = data.get("ohlcv", data.get("context_ohlcv"))
    else:
        tokens = data
        ohlcv_raw = None

    # Take last seq_len tokens as context
    seq_len = strate_ii_config.embedding.seq_len
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    tokens = tokens[:, -seq_len:].to(device)
    B, S = tokens.shape

    # Target positions: predict the next N_tgt positions
    N_tgt = min(args.n_targets, seq_len - S) if S < seq_len else 8
    target_pos = torch.arange(S, S + N_tgt).clamp(max=seq_len - 1)
    target_pos = target_pos.unsqueeze(0).expand(B, -1).to(device)

    # Context OHLCV for RevIN stats
    if ohlcv_raw is None:
        print("Warning: No OHLCV context found. Using dummy context for RevIN stats.")
        context_ohlcv = torch.ones(B, S * 16, 5, device=device) * 50000
    else:
        if ohlcv_raw.dim() == 2:
            ohlcv_raw = ohlcv_raw.unsqueeze(0)
        context_ohlcv = ohlcv_raw[:, -(S * 16):].to(device)

    # Generate futures
    print(f"Generating {args.n_samples} futures for {args.pair}...")
    result = generator.generate(
        tokens, None, target_pos, context_ohlcv, n_samples=args.n_samples,
    )

    # Save
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_name = f"{args.pair}_{args.n_samples}futures"
    torch.save(result, out_path / f"{save_name}.pt")
    print(f"Saved to {out_path / f'{save_name}.pt'}")

    # Visualize
    _plot_futures(result, out_path / f"{save_name}.png", title=args.pair)
    print("Done.")


def _plot_futures(result: dict, save_path: Path, title: str = "Multiverse Futures"):
    """Plot OHLCV fan chart showing N future scenarios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed, skipping visualization.")
        return

    ohlcv = result["ohlcv"]  # (N, B, N_tgt, patch_len, 5)
    N, B, N_tgt, patch_len, _ = ohlcv.shape

    # Flatten patches into continuous time series (close prices, channel 3)
    # Shape: (N, B, N_tgt * patch_len)
    close_prices = ohlcv[..., 3].reshape(N, B, N_tgt * patch_len)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot each sample's close price trajectory for batch 0
    time = np.arange(N_tgt * patch_len)
    for i in range(N):
        prices = close_prices[i, 0].cpu().numpy()
        alpha = max(0.1, 0.5 / (N / 8))
        ax.plot(time, prices, alpha=alpha, color="steelblue", linewidth=0.8)

    # Plot median trajectory
    median_prices = close_prices[:, 0].median(dim=0).values.cpu().numpy()
    ax.plot(time, median_prices, color="darkred", linewidth=2, label="Median")

    # Confidence bands
    q05 = close_prices[:, 0].quantile(0.05, dim=0).cpu().numpy()
    q95 = close_prices[:, 0].quantile(0.95, dim=0).cpu().numpy()
    ax.fill_between(time, q05, q95, alpha=0.15, color="steelblue", label="5-95% CI")

    ax.set_title(f"{title} — {N} Future Scenarios")
    ax.set_xlabel("Time steps (within predicted patches)")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate multiverse futures (Strate III)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test with random weights")
    parser.add_argument("--strate_ii_checkpoint", type=str, default=None)
    parser.add_argument("--strate_ii_config", type=str, default="configs/strate_ii.yaml")
    parser.add_argument("--strate_i_checkpoint", type=str, default=None)
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i_binance.yaml")
    parser.add_argument("--pair", type=str, default="BTCUSDT")
    parser.add_argument("--data_dir", type=str, default="data/tokens/")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--n_targets", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/multiverse/")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(n_samples=args.n_samples, output_dir=args.output_dir)
    else:
        if args.strate_ii_checkpoint is None or args.strate_i_checkpoint is None:
            parser.error("--strate_ii_checkpoint and --strate_i_checkpoint required "
                         "(or use --smoke_test)")
        generate_from_checkpoints(args)


if __name__ == "__main__":
    main()
