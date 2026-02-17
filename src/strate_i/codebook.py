"""Spherical VQ codebook with EMA updates (van den Oord 2017)."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.common.math_utils import l2_normalize


class SphericalCodebook(nn.Module):
    def __init__(
        self,
        num_codes: int = 1024,
        latent_dim: int = 64,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: int = 2,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_threshold = dead_threshold
        self.commitment_weight = commitment_weight

        # Buffers (not parameters — updated via EMA, not gradient)
        self.register_buffer("embeddings", l2_normalize(torch.randn(num_codes, latent_dim)))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_weight", self.embeddings.clone())
        self.register_buffer("usage_count", torch.zeros(num_codes))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _kmeans_init(self, z_e: Tensor):
        """Initialize codebook from first batch (1-iter K-means: random subset)."""
        n = z_e.size(0)
        k = min(self.num_codes, n)
        indices = torch.randperm(n, device=z_e.device)[:k]
        selected = l2_normalize(z_e[indices], dim=-1)

        if k < self.num_codes:
            # Pad with perturbed copies if batch smaller than codebook
            repeats = (self.num_codes + k - 1) // k
            padded = selected.repeat(repeats, 1)[:self.num_codes]
            padded = l2_normalize(padded + torch.randn_like(padded) * 0.01, dim=-1)
            selected = padded

        self.embeddings.data.copy_(selected)
        self.ema_weight.data.copy_(selected)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _revive_dead_codes(self, z_e: Tensor):
        """Replace dead codes (usage < threshold) with random z_e from batch."""
        dead_mask = self.usage_count < self.dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        n_dead = int(n_dead)
        n_replace = min(n_dead, z_e.size(0))
        rand_idx = torch.randperm(z_e.size(0), device=z_e.device)[:n_replace]
        replacements = l2_normalize(z_e[rand_idx], dim=-1).to(self.embeddings.dtype)
        # Only replace up to n_replace dead codes
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_replace]
        self.embeddings.data[dead_indices] = replacements
        self.ema_weight.data[dead_indices] = replacements
        self.ema_count[dead_indices] = 1.0
        self.usage_count[dead_indices] = 0.0

    def forward(self, z_e: Tensor) -> dict[str, Tensor]:
        """Quantize z_e and compute losses.

        Args:
            z_e: (B, D) encoder outputs (already L2-normalized by encoder).

        Returns:
            dict with z_q, indices, commitment_loss, codebook_loss, perplexity, utilization.
        """
        if not self.initialized and self.training:
            self._kmeans_init(z_e)

        # Ensure everything is on the sphere
        z_e_norm = l2_normalize(z_e, dim=-1)
        emb_norm = l2_normalize(self.embeddings, dim=-1)

        # Cosine similarity via matmul (both normalized): (B, K)
        sim = z_e_norm @ emb_norm.T
        indices = sim.argmax(dim=-1)  # (B,)
        z_q = emb_norm[indices]  # (B, D)

        # EMA update
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(indices, self.num_codes).float()  # (B, K)
                batch_count = onehot.sum(0)  # (K,)
                batch_sum = onehot.T @ z_e_norm.float()  # (K, D) — cast to float32 for EMA buffers

                self.ema_count.mul_(self.ema_decay).add_(batch_count, alpha=1 - self.ema_decay)
                self.ema_weight.mul_(self.ema_decay).add_(batch_sum, alpha=1 - self.ema_decay)

                # Laplace smoothing
                n = self.ema_count.sum()
                count_smooth = (self.ema_count + self.eps) / (n + self.num_codes * self.eps) * n
                self.embeddings.data.copy_(
                    l2_normalize(self.ema_weight / count_smooth.unsqueeze(-1), dim=-1)
                )

                # Track usage for dead code detection
                self.usage_count.add_(batch_count)
                self._revive_dead_codes(z_e_norm)

        # Losses
        commitment_loss = self.commitment_weight * F.mse_loss(z_e_norm, z_q.detach())
        codebook_loss = F.mse_loss(z_e_norm.detach(), z_q)  # monitoring only

        # STE: gradients flow through z_e
        z_q_ste = z_e_norm + (z_q - z_e_norm).detach()

        # Per-batch perplexity and utilization
        with torch.no_grad():
            onehot = F.one_hot(indices, self.num_codes).float()
            avg_probs = onehot.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            utilization = (avg_probs > 0).float().mean()

        return {
            "z_q": z_q_ste,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "utilization": utilization,
        }

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Return codebook indices for inference. (B, D) -> (B,)."""
        x = l2_normalize(x, dim=-1)
        sim = x @ l2_normalize(self.embeddings, dim=-1).T
        return sim.argmax(dim=-1)

    @torch.no_grad()
    def reset_usage(self):
        """Reset usage counts (call at epoch boundary)."""
        self.usage_count.zero_()
