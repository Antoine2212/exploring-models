import torch

def to_2d_chunks(x: torch.Tensor) -> torch.Tensor:
    # (..., head_dim) -> (..., n_freq, 2)
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    return x.view(*x.shape[:-1], head_dim // 2, 2)

def chunk_norms(x: torch.Tensor) -> torch.Tensor:
    return to_2d_chunks(x).norm(dim=-1)

def mean_chunk_norms_over_tokens(x: torch.Tensor) -> torch.Tensor:
    # x: (batch, seq, heads, head_dim)
    return chunk_norms(x).mean(dim=(0, 1))

def mean_chunk_norms_over_tokens_kv(x: torch.Tensor) -> torch.Tensor:
    # x: (batch, seq, kv_heads, head_dim)
    return chunk_norms(x).mean(dim=(0, 1))

def high_freq_ratio(norms: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    n = norms.shape[-1]
    k = max(1, int(n * frac))
    return norms[..., -k:].sum(dim=-1) / norms.sum(dim=-1).clamp_min(1e-8)

def low_freq_ratio(norms: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    n = norms.shape[-1]
    k = max(1, int(n * frac))
    return norms[..., :k].sum(dim=-1) / norms.sum(dim=-1).clamp_min(1e-8)