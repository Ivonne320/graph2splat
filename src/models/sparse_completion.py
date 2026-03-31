"""
Sparse-encoder + dense-decoder completion model.

Architecture
------------
Input : dense [B, 1+feat_out, G, G, G]  (same interface as UNetCompletionModel)

1. Extract occupied seed voxels → SLAT SparseTensor.
2. Sparse encoder via SLAT SparseConv3d wrappers (backend-agnostic: spconv or torchsparse):
     Level 0 – SubMConv (stride 1) ×2  → [N, c1],   resolution G
     Level 1 – SparseConv (stride 2)   → [N', c2],  resolution G/2
     Level 2 – SparseConv (stride 2)   → [N'', c3], resolution G/4
     Level 3 – SparseConv (stride 2)   → [N''', c4],resolution G/8
3. Densify each level via pure-PyTorch scatter → 4 dense feature maps.
4. Dense decoder (reuses DoubleConv / Up from unet3d_completion):
     bottleneck at G/8, three up-blocks with skip connections → G.
5. Output 1×1×1 conv → [B, out_channels, G, G, G]

Output: dense [B, 1+latent_dim, G, G, G]  (identical interface to UNetCompletionModel)

Memory advantage over plain UNet: the encoder stores [N, C] with N << G³
(only occupied seed voxels), so levels 0–3 use a tiny fraction of the memory
a full dense grid would require.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..modules.sparse.basic import SparseTensor
from ..modules.sparse.conv import SparseConv3d
from ..modules.sparse.norm import SparseGroupNorm
from ..modules.sparse.nonlinearity import SparseReLU

from .unet3d_completion import DinoCompressor, DoubleConv, Up, OutConv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparse_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """
    One sparse conv + GroupNorm + ReLU block.
    stride=1  → SubMConv3d path (same sparsity pattern); pass padding=None to
                trigger that branch in the SLAT SparseConv3d wrapper.
    stride=2  → strided SparseConv3d (downsamples & expands active set);
                pass padding=0 — spconv does not accept padding=None for
                strided convolutions.
    """
    num_groups = min(32, out_channels)
    padding = None if stride == 1 else 0
    return nn.Sequential(
        SparseConv3d(in_channels, out_channels,
                     kernel_size=3 if stride == 1 else 2,
                     stride=stride, padding=padding),
        SparseGroupNorm(num_groups, out_channels),
        SparseReLU(inplace=True),
    )


def sparse_to_dense(sp: SparseTensor, G: int) -> torch.Tensor:
    """
    Scatter a SLAT SparseTensor to a dense [B, C, G, G, G] tensor.
    Pure PyTorch — no backend-specific .dense() call.
    Coords: [N, 4]  (batch, x, y, z) in [0, G).
    """
    feats = sp.feats                      # [N, C]
    coords = sp.coords.long()             # [N, 4]
    N, C = feats.shape
    B = sp.shape[0]
    device = feats.device

    b = coords[:, 0].clamp(0, B - 1)
    x = coords[:, 1].clamp(0, G - 1)
    y = coords[:, 2].clamp(0, G - 1)
    z = coords[:, 3].clamp(0, G - 1)

    lin = b * (G * G * G) + x * (G * G) + y * G + z  # [N]

    dense_flat = torch.zeros(B * G * G * G, C, device=device, dtype=feats.dtype)
    dense_flat.index_put_((lin,), feats, accumulate=False)
    return dense_flat.view(B, G, G, G, C).permute(0, 4, 1, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# Sparse encoder
# ---------------------------------------------------------------------------

class SparseEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * m for m in (1, 2, 4, 8))
        self.channels: Tuple[int, int, int, int] = (c1, c2, c3, c4)

        self.enc0a = _sparse_block(in_channels, c1, stride=1)
        self.enc0b = _sparse_block(c1,          c1, stride=1)

        self.down1 = _sparse_block(c1, c2, stride=2)
        self.enc1  = _sparse_block(c2, c2, stride=1)

        self.down2 = _sparse_block(c2, c3, stride=2)
        self.enc2  = _sparse_block(c3, c3, stride=1)

        self.down3 = _sparse_block(c3, c4, stride=2)
        self.enc3  = _sparse_block(c4, c4, stride=1)

    def _dense_to_sparse(self, x_dense: torch.Tensor) -> Optional[SparseTensor]:
        """Extract occupied voxels (channel-0 > 0) into a SLAT SparseTensor."""
        B, C = x_dense.shape[:2]
        coord_list: List[torch.Tensor] = []
        feat_list: List[torch.Tensor] = []
        for b in range(B):
            xyz = (x_dense[b, 0] > 0).nonzero(as_tuple=False).int()  # [N_b, 3]
            if xyz.shape[0] == 0:
                continue
            b_col = xyz.new_full((xyz.shape[0], 1), b)
            coord_list.append(torch.cat([b_col, xyz], dim=1))         # [N_b, 4]
            feat_list.append(x_dense[b, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]].T.float())
        if not coord_list:
            return None
        return SparseTensor(
            feats=torch.cat(feat_list, dim=0).contiguous(),
            coords=torch.cat(coord_list, dim=0).contiguous(),
        )

    def forward(
        self, x_dense: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x_dense.shape[0]
        G = x_dense.shape[-1]
        c1, c2, c3, c4 = self.channels
        device, dtype = x_dense.device, x_dense.dtype

        sp = self._dense_to_sparse(x_dense)
        if sp is None:
            return (
                torch.zeros(B, c1, G,    G,    G,    device=device, dtype=dtype),
                torch.zeros(B, c2, G//2, G//2, G//2, device=device, dtype=dtype),
                torch.zeros(B, c3, G//4, G//4, G//4, device=device, dtype=dtype),
                torch.zeros(B, c4, G//8, G//8, G//8, device=device, dtype=dtype),
            )

        sp0 = self.enc0b(self.enc0a(sp))  # G  — processed but not densified (too large)
        sp1 = self.enc1(self.down1(sp0))  # G/2
        sp2 = self.enc2(self.down2(sp1))  # G/4
        sp3 = self.enc3(self.down3(sp2))  # G/8

        # d0 (full-res [B, c1, G, G, G]) is the dominant memory cost and is not returned.
        # The decoder skips the finest-scale skip connection.
        return (
            sparse_to_dense(sp1, G // 2),
            sparse_to_dense(sp2, G // 4),
            sparse_to_dense(sp3, G // 8),
        )


# ---------------------------------------------------------------------------
# Dense decoder
# ---------------------------------------------------------------------------

class SparseDenseDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        out_channels: int = 1,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * m for m in (1, 2, 4, 8))
        kw = dict(use_instance_norm=use_instance_norm, dropout_p=dropout_p)

        self.bottleneck = DoubleConv(c4, c4, **kw)
        self.up3 = Up(c4, c3, c3, **kw)              # G/8 → G/4
        self.up2 = Up(c3, c2, c2, **kw)              # G/4 → G/2
        # No full-res skip (d0 at G is too large); upsample G/2 → G with a plain DoubleConv
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            DoubleConv(c2, c1, **kw),
        )
        self.outc = OutConv(c1, out_channels)

    def forward(self, d1, d2, d3) -> torch.Tensor:
        x = self.bottleneck(d3)
        x = self.up3(x, d2)
        x = self.up2(x, d1)
        x = self.up1(x)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Full network + wrapper
# ---------------------------------------------------------------------------

class SparseCompletionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 65,
        base_channels: int = 32,
        out_channels: int = 1,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SparseEncoder(in_channels, base_channels)
        self.decoder = SparseDenseDecoder(base_channels, out_channels, use_instance_norm, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, d2, d3 = self.encoder(x)
        return self.decoder(d1, d2, d3)


class SparseCompletionModel(nn.Module):
    """
    Drop-in replacement for UNetCompletionModel.
    Identical interface: forward takes [B, 1+feat_out, G, G, G] → [B, out_channels, G, G, G].
    """

    def __init__(
        self,
        feat_in: int = 1024,
        feat_mid: int = 256,
        feat_out: int = 64,
        base_channels: int = 32,
        out_channels: int = 1,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_compressor = DinoCompressor(d_in=feat_in, d_mid=feat_mid, d_out=feat_out)
        self.sparse_net = SparseCompletionNet(
            in_channels=1 + feat_out,
            base_channels=base_channels,
            out_channels=out_channels,
            use_instance_norm=use_instance_norm,
            dropout_p=dropout_p,
        )
        self.compressed_channels = feat_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparse_net(x)
