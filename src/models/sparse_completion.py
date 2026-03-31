"""
Sparse-encoder + dense-decoder completion model.

Architecture
------------
Input : dense [B, 1+feat_out, G, G, G]  (same interface as UNetCompletionModel)

1. Extract occupied seed voxels from channel-0 (occupancy mask).
2. Sparse encoder (spconv):
     Level 0 – SubMConv3d ×2             → feats [N, c1],  resolution G
     Level 1 – SparseConv3d s=2 + SubM   → feats [N', c2], resolution G/2
     Level 2 – SparseConv3d s=2 + SubM   → feats [N'', c3], resolution G/4
     Level 3 – SparseConv3d s=2 + SubM   → feats [N''', c4], resolution G/8
3. Densify each level via spconv .dense() → 4 dense feature maps.
4. Dense decoder (reuses DoubleConv / Up from unet3d_completion):
     bottleneck on c4 at G/8
     up-block + skip: G/8→G/4, G/4→G/2, G/2→G
5. Output 1×1×1 conv → [B, out_channels, G, G, G]

Output: dense [B, 1+latent_dim, G, G, G]  (identical interface to UNetCompletionModel)

The model therefore slots into the existing training pipeline by simply
replacing `create_model()` — no changes to data loading or loss computation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from .unet3d_completion import DinoCompressor, DoubleConv, Up, OutConv


# ---------------------------------------------------------------------------
# Sparse encoder
# ---------------------------------------------------------------------------

class SparseEncoder(nn.Module):
    """
    Process seed voxels with sparse convolutions, then densify at each scale
    so the dense decoder can use them as skip features.

    The indice_key scheme follows spconv conventions:
      - 'subm*'   : SubMConv3d layers (preserve sparsity pattern)
      - 'spconv*' : strided SparseConv3d layers (downsample + expand sparsity)
    """

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * m for m in (1, 2, 4, 8))
        self.channels: Tuple[int, int, int, int] = (c1, c2, c3, c4)

        # ---- Level 0: full resolution ----------------------------------------
        self.enc0 = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, c1, 3, bias=False, indice_key="subm0a"),
            nn.BatchNorm1d(c1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(c1, c1, 3, bias=False, indice_key="subm0b"),
            nn.BatchNorm1d(c1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        # ---- Level 1: G/2 ----------------------------------------------------
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(c1, c2, 2, stride=2, bias=False, indice_key="spconv1"),
            nn.BatchNorm1d(c2, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(c2, c2, 3, bias=False, indice_key="subm1"),
            nn.BatchNorm1d(c2, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        # ---- Level 2: G/4 ----------------------------------------------------
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(c2, c3, 2, stride=2, bias=False, indice_key="spconv2"),
            nn.BatchNorm1d(c3, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(c3, c3, 3, bias=False, indice_key="subm2"),
            nn.BatchNorm1d(c3, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        # ---- Level 3: G/8 ----------------------------------------------------
        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(c3, c4, 2, stride=2, bias=False, indice_key="spconv3"),
            nn.BatchNorm1d(c4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(c4, c4, 3, bias=False, indice_key="subm3"),
            nn.BatchNorm1d(c4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def _build_sparse_input(
        self, x_dense: torch.Tensor, G: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract occupied voxel coordinates and features from a dense grid.

        Returns
        -------
        all_coords : int32 [N, 4]  – (batch, x, y, z)
        all_feats  : float32 [N, C]
        """
        B, C = x_dense.shape[:2]
        coord_list, feat_list = [], []
        for b in range(B):
            occ_b = x_dense[b, 0]                              # [G, G, G]
            xyz = (occ_b > 0).nonzero(as_tuple=False).int()    # [N_b, 3]
            if xyz.shape[0] == 0:
                continue
            b_col = xyz.new_full((xyz.shape[0], 1), b)
            coord_list.append(torch.cat([b_col, xyz], dim=1))  # [N_b, 4]
            # Gather feature vectors at each occupied position
            feats_b = x_dense[b, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]].T  # [N_b, C]
            feat_list.append(feats_b)
        if not coord_list:
            return None, None
        return (
            torch.cat(coord_list, dim=0).contiguous(),        # [N, 4]  int32
            torch.cat(feat_list, dim=0).float().contiguous(), # [N, C]  float32
        )

    def forward(
        self, x_dense: torch.Tensor, G: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C = x_dense.shape[:2]
        device = x_dense.device
        c1, c2, c3, c4 = self.channels

        all_coords, all_feats = self._build_sparse_input(x_dense, G)

        if all_coords is None:
            # No occupied seeds — return zero-filled dense tensors
            return (
                torch.zeros(B, c1, G,     G,     G,     device=device),
                torch.zeros(B, c2, G//2,  G//2,  G//2,  device=device),
                torch.zeros(B, c3, G//4,  G//4,  G//4,  device=device),
                torch.zeros(B, c4, G//8,  G//8,  G//8,  device=device),
            )

        sp = spconv.SparseConvTensor(all_feats, all_coords, (G, G, G), B)

        sp0 = self.enc0(sp)         # level 0: G
        sp1 = self.down1(sp0)       # level 1: G/2
        sp2 = self.down2(sp1)       # level 2: G/4
        sp3 = self.down3(sp2)       # level 3: G/8

        # .dense() → [B, C, D, H, W]  (zeros at non-occupied positions)
        d0 = sp0.dense()  # [B, c1, G,    G,    G   ]
        d1 = sp1.dense()  # [B, c2, G/2,  G/2,  G/2 ]
        d2 = sp2.dense()  # [B, c3, G/4,  G/4,  G/4 ]
        d3 = sp3.dense()  # [B, c4, G/8,  G/8,  G/8 ]

        return d0, d1, d2, d3


# ---------------------------------------------------------------------------
# Dense decoder
# ---------------------------------------------------------------------------

class SparseDenseDecoder(nn.Module):
    """
    Standard UNet decoder.  Receives 4 dense feature maps from the sparse
    encoder and reconstructs the full G³ prediction volume.
    """

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

        self.bottleneck = DoubleConv(c4, c4, **kw)          # G/8
        self.up3 = Up(c4, c3, c3, **kw)                     # G/8 → G/4
        self.up2 = Up(c3, c2, c2, **kw)                     # G/4 → G/2
        self.up1 = Up(c2, c1, c1, **kw)                     # G/2 → G
        self.outc = OutConv(c1, out_channels)

    def forward(
        self,
        d0: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        d3: torch.Tensor,
    ) -> torch.Tensor:
        x = self.bottleneck(d3)
        x = self.up3(x, d2)
        x = self.up2(x, d1)
        x = self.up1(x, d0)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class SparseCompletionNet(nn.Module):
    """Sparse encoder + dense decoder backbone."""

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
        G = x.shape[-1]
        d0, d1, d2, d3 = self.encoder(x, G)
        return self.decoder(d0, d1, d2, d3)


class SparseCompletionModel(nn.Module):
    """
    Drop-in replacement for UNetCompletionModel.

    Owns DinoCompressor (identical to the UNet variant) + SparseCompletionNet.
    The forward signature is identical: takes a dense [B, 1+feat_out, G, G, G]
    tensor and returns [B, out_channels, G, G, G].
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
