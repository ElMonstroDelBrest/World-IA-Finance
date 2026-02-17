"""Pre-compute trajectory buffer for Strate IV RL training.

Usage:
    # Smoke test with random weights (no checkpoints needed):
    PYTHONPATH=. python scripts/precompute_trajectories.py --smoke_test

    # From trained checkpoints:
    PYTHONPATH=. python scripts/precompute_trajectories.py \
        --strate_ii_checkpoint checkpoints/strate_ii/best.ckpt \
        --strate_i_checkpoint checkpoints/strate_i_best.ckpt \
        --token_dir data/tokens/ \
        --ohlcv_dir data/binance_1h_subset/ \
        --output_dir data/trajectory_buffer/
"""

import argparse
from pathlib import Path

import torch


def smoke_test(
    n_episodes: int = 10,
    n_futures: int = 4,
    n_tgt: int = 8,
    output_dir: str = "data/trajectory_buffer/",
):
    """Smoke test: generate buffer entries with random weights."""
    from src.strate_i.decoder import Decoder
    from src.strate_i.revin import RevIN
    from src.strate_ii.jepa import FinJEPA
    from src.strate_ii.multiverse import MultiverseGenerator
    from src.strate_iv.trajectory_buffer import TrajectoryPrecomputer, TrajectoryEntry

    print("=== Smoke test: pre-computing trajectory buffer with random weights ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build models with random weights
    jepa = FinJEPA(
        num_codes=1024, codebook_dim=64, d_model=128, d_state=16,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=64, pred_hidden_dim=256, pred_n_layers=2,
        pred_z_dim=32,
    ).to(device).eval()

    decoder = Decoder(
        latent_dim=64, hidden_channels=128, out_channels=5,
        patch_length=16, n_layers=4, kernel_size=3,
    ).to(device).eval()

    revin = RevIN(n_channels=5).to(device)
    generator = MultiverseGenerator(jepa, decoder, revin)

    precomputer = TrajectoryPrecomputer(
        generator=generator, jepa=jepa,
        n_futures=n_futures, n_tgt=n_tgt,
    )

    # Generate synthetic episodes
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    S = 48  # context sequence length
    for i in range(n_episodes):
        tokens = torch.randint(0, 1024, (S,), device=device)
        weekend_mask = torch.zeros(S, device=device)
        # Mark weekends every 7th position
        weekend_mask[5::7] = 1.0
        weekend_mask[6::7] = 1.0

        context_ohlcv = torch.rand(S * 16, 5, device=device) * 100 + 30000

        entry = precomputer.compute_entry(tokens, weekend_mask, context_ohlcv)

        save_dict = {
            "context_tokens": entry.context_tokens,
            "weekend_mask": entry.weekend_mask,
            "context_ohlcv": entry.context_ohlcv,
            "future_ohlcv": entry.future_ohlcv,
            "future_latents": entry.future_latents,
            "revin_means": entry.revin_means,
            "revin_stds": entry.revin_stds,
            "last_close": entry.last_close,
            "h_x_pooled": entry.h_x_pooled,
        }
        torch.save(save_dict, out_path / f"episode_{i:05d}.pt")

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_episodes}] episodes saved")

    print(f"\n  Buffer saved to {out_path} ({n_episodes} episodes)")
    print(f"  Each entry: {n_futures} futures, {n_tgt} targets")

    # Quick validation
    from src.strate_iv.trajectory_buffer import TrajectoryBuffer
    buf = TrajectoryBuffer(str(out_path))
    print(f"  Loaded back: {len(buf)} entries")
    entry = buf.sample()
    print(f"  Sample entry shapes:")
    print(f"    h_x_pooled:      {entry.h_x_pooled.shape}")
    print(f"    future_ohlcv:    {entry.future_ohlcv.shape}")
    print(f"    future_latents:  {entry.future_latents.shape}")
    print(f"    revin_stds:      {entry.revin_stds.shape}")

    print("=== Smoke test passed ===")


def precompute_from_checkpoints(args):
    """Pre-compute buffer from trained checkpoints.

    Token files are named {PAIR}_seq{NNNN}.pt (e.g. BTCUSDT_seq0003.pt).
    OHLCV files are {PAIR}.pt (e.g. BTCUSDT.pt) in ohlcv_dir.
    Each token sequence i covers OHLCV[i*seq_len*patch_len : (i+1)*seq_len*patch_len].
    """
    from src.strate_i.config import load_config as load_strate_i_config
    from src.strate_i.lightning_module import StrateILightningModule
    from src.strate_i.revin import RevIN
    from src.strate_ii.config import load_config as load_strate_ii_config
    from src.strate_ii.lightning_module import StrateIILightningModule
    from src.strate_ii.multiverse import MultiverseGenerator
    from src.strate_ii.data.token_dataset import TokenSequenceDataset
    from src.strate_iv.config import load_config as load_strate_iv_config
    from src.strate_iv.trajectory_buffer import TrajectoryPrecomputer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configs
    strate_iv_config = load_strate_iv_config(args.config)

    # Load Strate II
    # Detect z_dim from checkpoint to handle pre-Strate III checkpoints (z_dim=0)
    strate_ii_config = load_strate_ii_config(args.strate_ii_config)
    ckpt_state = torch.load(args.strate_ii_checkpoint, weights_only=False, map_location="cpu")
    pred_in_dim = ckpt_state["state_dict"]["jepa.predictor.mlp.0.weight"].shape[1]
    ckpt_z_dim = pred_in_dim - strate_ii_config.mamba2.d_model * 2
    if ckpt_z_dim != strate_ii_config.predictor.z_dim:
        print(f"  Checkpoint z_dim={ckpt_z_dim} (config has {strate_ii_config.predictor.z_dim}), overriding")
        from dataclasses import replace
        pred_cfg = replace(strate_ii_config.predictor, z_dim=ckpt_z_dim)
        strate_ii_config = replace(strate_ii_config, predictor=pred_cfg)
    del ckpt_state

    strate_ii = StrateIILightningModule.load_from_checkpoint(
        args.strate_ii_checkpoint, config=strate_ii_config,
        strict=False, weights_only=False,
    )
    jepa = strate_ii.jepa.to(device).eval()

    # Load Strate I for decoder
    strate_i_config = load_strate_i_config(args.strate_i_config)
    strate_i = StrateILightningModule.load_from_checkpoint(
        args.strate_i_checkpoint, config=strate_i_config,
        weights_only=False,
    )
    decoder = strate_i.tokenizer.vqvae.decoder.to(device).eval()
    revin = RevIN(n_channels=5).to(device)

    # Load codebook
    codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
    jepa.load_codebook(codebook_weights.to(device))

    generator = MultiverseGenerator(jepa, decoder, revin)

    seq_len = strate_ii_config.embedding.seq_len
    patch_len = strate_i_config.patch.patch_length  # 16

    precomputer = TrajectoryPrecomputer(
        generator=generator,
        jepa=jepa,
        n_futures=strate_iv_config.buffer.n_futures,
        n_tgt=strate_iv_config.env.n_tgt,
    )

    # Load token dataset
    dataset = TokenSequenceDataset(
        token_dir=args.token_dir,
        seq_len=seq_len,
    )

    # Build OHLCV lookup: token file name → (pair, seq_idx) → OHLCV slice
    ohlcv_dir = Path(args.ohlcv_dir)
    token_dir = Path(args.token_dir)
    token_files = sorted(token_dir.glob("*.pt"))

    # Cache loaded OHLCV tensors per pair
    ohlcv_cache: dict[str, torch.Tensor] = {}

    def ohlcv_lookup(idx):
        """Load the OHLCV slice corresponding to token sequence idx."""
        fname = token_files[idx].stem  # e.g. "BTCUSDT_seq0003"
        pair, seq_part = fname.rsplit("_seq", 1)
        seq_idx = int(seq_part)

        # Load and cache full OHLCV for this pair
        if pair not in ohlcv_cache:
            ohlcv_path = ohlcv_dir / f"{pair}.pt"
            if ohlcv_path.exists():
                ohlcv_cache[pair] = torch.load(ohlcv_path, weights_only=True)
            else:
                print(f"  Warning: OHLCV not found for {pair}, using dummy")
                ohlcv_cache[pair] = torch.ones(seq_len * patch_len * 20, 5) * 50000

        full_ohlcv = ohlcv_cache[pair]  # (T, 5)

        # Each token sequence covers seq_len patches of patch_len candles
        candles_per_seq = seq_len * patch_len  # 64 * 16 = 1024
        start = seq_idx * candles_per_seq
        end = start + candles_per_seq

        # Clamp to available data
        end = min(end, full_ohlcv.shape[0])
        start = min(start, max(0, end - candles_per_seq))

        return full_ohlcv[start:end]

    n_episodes = args.n_episodes or strate_iv_config.buffer.n_episodes
    output_dir = args.output_dir or strate_iv_config.buffer.buffer_dir

    print(f"Token files: {len(token_files)}")
    print(f"OHLCV dir: {ohlcv_dir}")
    print(f"Episodes to generate: {n_episodes}")
    print(f"Futures per episode: {strate_iv_config.buffer.n_futures}")
    print(f"Targets per episode: {strate_iv_config.env.n_tgt}")
    print()

    precomputer.run(
        dataset=dataset,
        ohlcv_lookup=ohlcv_lookup,
        output_dir=output_dir,
        n_episodes=n_episodes,
        device=str(device),
    )

    print("Pre-computation complete.")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Strate IV trajectory buffer")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test with random weights")
    parser.add_argument("--config", type=str, default="configs/strate_iv.yaml")
    parser.add_argument("--strate_ii_checkpoint", type=str, default=None)
    parser.add_argument("--strate_ii_config", type=str, default="configs/strate_ii.yaml")
    parser.add_argument("--strate_i_checkpoint", type=str, default=None)
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i_binance.yaml")
    parser.add_argument("--token_dir", type=str, default="data/tokens/")
    parser.add_argument("--ohlcv_dir", type=str, default="data/binance_1h_subset/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir from config")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Override n_episodes from config")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        if args.strate_ii_checkpoint is None or args.strate_i_checkpoint is None:
            parser.error("--strate_ii_checkpoint and --strate_i_checkpoint required "
                         "(or use --smoke_test)")
        precompute_from_checkpoints(args)


if __name__ == "__main__":
    main()
