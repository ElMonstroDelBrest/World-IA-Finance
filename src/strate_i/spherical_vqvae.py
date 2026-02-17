import torch
import torch.nn as nn
from torch import Tensor

from src.common.math_utils import l2_normalize
from .config import EncoderConfig, DecoderConfig, CodebookConfig
from .encoder import Encoder
from .decoder import Decoder
from .codebook import SphericalCodebook


class SphericalVQVAE(nn.Module):
    def __init__(
        self,
        enc_config: EncoderConfig,
        dec_config: DecoderConfig,
        codebook_config: CodebookConfig,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=enc_config.in_channels,
            hidden_channels=enc_config.hidden_channels,
            latent_dim=enc_config.latent_dim,
            n_layers=enc_config.n_layers,
            dilation_base=enc_config.dilation_base,
            kernel_size=enc_config.kernel_size,
        )
        self.codebook = SphericalCodebook(
            num_codes=codebook_config.num_codes,
            latent_dim=codebook_config.latent_dim,
            ema_decay=codebook_config.ema_decay,
            eps=codebook_config.eps,
            dead_threshold=codebook_config.dead_threshold,
            commitment_weight=codebook_config.commitment_weight,
        )
        self.decoder = Decoder(
            latent_dim=dec_config.latent_dim,
            hidden_channels=dec_config.hidden_channels,
            out_channels=dec_config.out_channels,
            patch_length=dec_config.patch_length,
            n_layers=dec_config.n_layers,
            kernel_size=dec_config.kernel_size,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """(B, L, C) -> dict with x_hat, z_e, z_q, indices, losses, metrics."""
        z_e = self.encoder(x)
        vq_out = self.codebook(z_e)
        x_hat = self.decoder(vq_out["z_q"])
        return {**vq_out, "x_hat": x_hat, "z_e": z_e}

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """(B, L, C) -> (B,) token indices."""
        z_e = self.encoder(x)
        return self.codebook.encode(z_e)

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """(B,) indices -> (B, L, C) reconstructed patches."""
        emb = l2_normalize(self.codebook.embeddings, dim=-1)
        z_q = emb[indices]
        return self.decoder(z_q)
