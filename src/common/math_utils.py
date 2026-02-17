import torch
from torch import Tensor


def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    return x / (norm + eps)


def cosine_similarity(a: Tensor, b: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    a_norm = l2_normalize(a, dim=dim, eps=eps)
    b_norm = l2_normalize(b, dim=dim, eps=eps)
    return torch.sum(a_norm * b_norm, dim=dim)
