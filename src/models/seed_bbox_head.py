import torch
import torch.nn as nn

class SeedBBoxHead(nn.Module):
    """
    Permutation-invariant bbox head over variable-length sets of seeds.
    Expects concatenated seeds across the batch + batch_idx for segment pooling.
    """
    def __init__(self, d_feat=64, use_coords=True):
        super().__init__()
        self.use_coords = use_coords
        d_in = d_feat + (3 if use_coords else 0)

        self.embed = nn.Sequential(
            nn.Linear(d_in, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 4)  # δmean(3), δscale_raw(1)
        )

    def forward_batch(self, feats_all: torch.Tensor, centers_all: torch.Tensor,
                      batch_idx: torch.Tensor, B: int) -> torch.Tensor:
        """
        feats_all:   (N, 64)
        centers_all: (N, 3)
        batch_idx:   (N,)  in [0..B-1]
        returns:     (B, 4)
        """
        x = torch.cat([feats_all, centers_all], dim=-1) if self.use_coords else feats_all
        h = self.embed(x)                                # (N,128)

        # segment mean: sum / count
        D = h.size(-1)
        sums = torch.zeros(B, D, device=h.device, dtype=h.dtype)
        sums.index_add_(0, batch_idx, h)                 # per-batch sum
        counts = torch.bincount(batch_idx, minlength=B).clamp_min(1).view(B, 1).to(h.dtype)
        h_mean = sums / counts                           # (B,128)

        # segment max (PyTorch ≥2.0): scatter_reduce
        h_max = torch.full((B, D), -float("inf"), device=h.device, dtype=h.dtype)
        # expand indices to (N, D)
        idx_exp = batch_idx.view(-1, 1).expand(-1, D)
        h_max = h_max.scatter_reduce(0, idx_exp, h, reduce='amax', include_self=True)

        # segment var via E[x^2] - (E[x])^2
        sums2 = torch.zeros(B, D, device=h.device, dtype=h.dtype)
        sums2.index_add_(0, batch_idx, h * h)
        ex2   = sums2 / counts
        h_var = (ex2 - h_mean * h_mean).clamp_min(0)

        g = torch.cat([h_mean, h_max, h_var], dim=-1)    # (B, 384)
        d = self.head(g)                                  # (B, 4)
        return d