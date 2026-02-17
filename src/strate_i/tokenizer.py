import torch
import torch.nn as nn
from torch import Tensor

from .config import StrateIConfig
from .revin import RevIN, StatsStore
from .spherical_vqvae import SphericalVQVAE


class TopologicalTokenizer(nn.Module):
    """Public facade: RevIN + SphericalVQVAE. API for Strate II."""

    def __init__(self, config: StrateIConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(
            n_channels=config.patch.n_channels,
            eps=config.revin.eps,
            affine=config.revin.affine,
        )
        self.vqvae = SphericalVQVAE(
            enc_config=config.encoder,
            dec_config=config.decoder,
            codebook_config=config.codebook,
        )
        self.stats_store = StatsStore()

    def forward(
        self, patches: Tensor, patch_ids: list[str] | None = None
    ) -> dict[str, Tensor]:
        """(B, L, C) -> dict with x_hat, z_e, z_q, indices, losses, revin stats."""
        x_norm, means, stds = self.revin.normalize(patches)
        vq_out = self.vqvae(x_norm)

        if patch_ids is not None:
            for i, pid in enumerate(patch_ids):
                self.stats_store.store(pid, means[i], stds[i])

        return {**vq_out, "revin_means": means, "revin_stds": stds}

    @torch.no_grad()
    def tokenize(self, patches: Tensor) -> Tensor:
        """(B, L, C) -> (B,) token indices."""
        x_norm, _, _ = self.revin.normalize(patches)
        return self.vqvae.encode(x_norm)

    @torch.no_grad()
    def reconstruct(self, patches: Tensor) -> Tensor:
        """Full forward + RevIN denormalize. (B, L, C) -> (B, L, C)."""
        x_norm, means, stds = self.revin.normalize(patches)
        vq_out = self.vqvae(x_norm)
        return self.revin.denormalize(vq_out["x_hat"], means, stds)
