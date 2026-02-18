"""MultiverseGenerator: full pipeline from JEPA latents to OHLCV candles.

Orchestrates: FinJEPA (stochastic) → output_proj → Strate I Decoder → RevIN denorm → OHLCV.

Usage:
    generator = MultiverseGenerator(jepa, decoder, revin)
    result = generator.generate(token_indices, weekend_mask, target_positions, context_ohlcv)
    # result["ohlcv"]: (N, B, N_tgt, patch_len, 5) — N future scenarios as OHLCV candles
"""

from __future__ import annotations

import torch
from torch import Tensor


class MultiverseGenerator:
    """Generates N divergent OHLCV future trajectories from a market state.

    Pipeline:
        1. JEPA stochastic predictor → N latent trajectories (d_model)
        2. output_proj → codebook space (codebook_dim)
        3. Strate I decoder → normalized log-return patches (patch_len, 5)
        4. RevIN denormalize → real-scale log-returns
        5. exp + cumprod → OHLCV candles

    Args:
        jepa: Trained FinJEPA model (with stochastic predictor).
        decoder: Strate I Decoder (frozen).
        revin: Strate I RevIN module (for denormalization).
    """

    def __init__(self, jepa, decoder, revin):
        self.jepa = jepa
        self.decoder = decoder
        self.revin = revin

    @torch.no_grad()
    def generate(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None,
        target_positions: Tensor,
        context_ohlcv: Tensor,
        n_samples: int = 16,
    ) -> dict[str, Tensor]:
        """Generate N future OHLCV trajectories.

        Args:
            token_indices: (B, S) context token indices.
            weekend_mask: (B, S) apathy mask (or None).
            target_positions: (B, N_tgt) positions to predict.
            context_ohlcv: (B, T, 5) raw OHLCV context for RevIN stats.
                T = number of time steps in the context window.
                Channels: [Open, High, Low, Close, Volume].
            n_samples: Number of divergent futures.

        Returns:
            dict with keys:
                "latents": (N, B, N_tgt, d_model) raw latent predictions
                "codebook_z": (N, B, N_tgt, codebook_dim) projected to codebook space
                "patches_norm": (N, B, N_tgt, patch_len, 5) decoded normalized patches
                "patches_real": (N, B, N_tgt, patch_len, 5) denormalized log-returns
                "ohlcv": (N, B, N_tgt, patch_len, 5) reconstructed OHLCV candles
        """
        device = token_indices.device

        # 1. Generate N latent trajectories via JEPA stochastic predictor
        h_futures = self.jepa.generate_futures(
            token_indices, weekend_mask, target_positions, n_samples=n_samples,
        )  # (N, B, N_tgt, d_model)

        N, B, N_tgt, d_model = h_futures.shape

        # 2. Project to codebook space
        z_futures = self.jepa.project_to_codebook_space(h_futures)  # (N, B, N_tgt, codebook_dim)
        codebook_dim = z_futures.shape[-1]

        # 3. Decode each patch via Strate I decoder (expects (B, D) → (B, L, C))
        z_flat = z_futures.reshape(N * B * N_tgt, codebook_dim)
        patches_norm = self.decoder(z_flat)  # (N*B*N_tgt, patch_len, 5)
        patch_len = patches_norm.shape[1]
        patches_norm = patches_norm.reshape(N, B, N_tgt, patch_len, 5)

        # 4. Denormalize via RevIN stats estimated from context
        means, stds = self._estimate_context_stats(context_ohlcv)  # (B, 1, 5) each
        # Broadcast: (N, B, N_tgt, patch_len, 5) with (1, B, 1, 1, 5)
        means_5d = means.unsqueeze(0).unsqueeze(2)  # (1, B, 1, 1, 5)
        stds_5d = stds.unsqueeze(0).unsqueeze(2)    # (1, B, 1, 1, 5)
        patches_real = patches_norm * stds_5d + means_5d

        # 5. Reconstruct OHLCV candles from log-returns
        ohlcv = self._log_returns_to_ohlcv(patches_real, context_ohlcv)

        return {
            "latents": h_futures,
            "codebook_z": z_futures,
            "patches_norm": patches_norm,
            "patches_real": patches_real,
            "ohlcv": ohlcv,
        }

    def _estimate_context_stats(
        self, context_ohlcv: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Estimate log-return mean and std from the context OHLCV.

        Computes log-returns from consecutive closes in context_ohlcv,
        then returns per-channel mean and std as RevIN proxies.

        Args:
            context_ohlcv: (B, T, 5) raw OHLCV. Channels: O, H, L, C, V.

        Returns:
            means: (B, 1, 5) per-channel mean of log-returns.
            stds: (B, 1, 5) per-channel std of log-returns.
        """
        eps = 1e-8
        # Use only the last 64 candles (4 patches) for stats — more representative
        # of the current regime than the full context (~2048 candles)
        recent_window = min(64, context_ohlcv.shape[1])
        recent_ohlcv = context_ohlcv[:, -recent_window:, :]

        # Compute log-returns for OHLC channels (0-3)
        prices = recent_ohlcv[:, :, :4].clamp(min=eps)  # (B, W, 4)
        log_returns_ohlc = torch.log(prices[:, 1:] / prices[:, :-1])  # (B, W-1, 4)

        # Compute log1p for volume channel (4)
        volume = recent_ohlcv[:, :, 4:5].clamp(min=0)  # (B, W, 1)
        # Use diff of log1p(volume) as "log-return" proxy for volume
        log_vol = torch.log1p(volume)
        log_returns_vol = log_vol[:, 1:] - log_vol[:, :-1]  # (B, W-1, 1)

        log_returns = torch.cat([log_returns_ohlc, log_returns_vol], dim=-1)  # (B, T-1, 5)

        means = log_returns.mean(dim=1, keepdim=True)  # (B, 1, 5)
        stds = log_returns.std(dim=1, keepdim=True).clamp(min=eps)  # (B, 1, 5)

        return means, stds

    def _log_returns_to_ohlcv(
        self, patches_real: Tensor, context_ohlcv: Tensor
    ) -> Tensor:
        """Convert denormalized log-return patches to OHLCV candles.

        Flattens all patches into a single timeline and applies a global
        cumprod so that prices are C0-continuous across patch boundaries.

        Args:
            patches_real: (N, B, N_tgt, patch_len, 5) denormalized log-returns.
            context_ohlcv: (B, T, 5) raw OHLCV context.

        Returns:
            (N, B, N_tgt, patch_len, 5) OHLCV candles with absolute prices.
        """
        N, B, N_tgt, patch_len, _ = patches_real.shape
        T_total = N_tgt * patch_len

        # Last close price from context as anchor
        last_close = context_ohlcv[:, -1, 3]  # (B,)
        # (1, B, 1)
        anchor = last_close.reshape(1, B, 1)

        # OHLC channels (0-3): flatten patches into continuous timeline
        log_returns_ohlc = patches_real[..., :4]  # (N, B, N_tgt, patch_len, 4)
        # Flatten: (N, B, N_tgt * patch_len, 4)
        lr_flat = log_returns_ohlc.reshape(N, B, T_total, 4)
        price_ratios = torch.exp(lr_flat)

        # Global cumprod across entire timeline (dim=2) — no discontinuities
        cum_ratios = torch.cumprod(price_ratios, dim=2)  # (N, B, T_total, 4)

        # Multiply by anchor to get absolute prices
        prices_flat = anchor.unsqueeze(-1) * cum_ratios  # (N, B, T_total, 4)

        # Reshape back to (N, B, N_tgt, patch_len, 4)
        prices = prices_flat.reshape(N, B, N_tgt, patch_len, 4)

        # Volume channel (4): inverse of log1p normalization
        mean_volume = context_ohlcv[:, :, 4].mean(dim=1)  # (B,)
        mean_volume = mean_volume.reshape(1, B, 1, 1).expand(N, B, N_tgt, patch_len)
        volume = mean_volume * torch.expm1(patches_real[..., 4]).clamp(min=0)

        ohlcv = torch.cat([prices, volume.unsqueeze(-1)], dim=-1)
        return ohlcv
