import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoCompressor(nn.Module):
    """Simple MLP to compress 1024-D features into fewer channels."""

    def __init__(self, d_in: int = 1024, d_mid: int = 256, d_out: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.SiLU(),
            nn.Linear(d_mid, d_out),
        )
        self.ln = nn.LayerNorm(d_out)

    @torch.no_grad()
    def _l2norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self._l2norm(feats)
        x = self.mlp(x)
        return self.ln(x)


class DoubleConv(nn.Module):
    """(conv => norm => ReLU) * 2, with optional Dropout3d.

    Args:
        use_instance_norm: replace BatchNorm3d with InstanceNorm3d (affine=True).
            InstanceNorm normalises per-sample so train/eval behaviour is identical,
            which avoids the running-stats drift seen with BatchNorm at single-sample
            inference.
        dropout_p: if > 0, append a Dropout3d layer after the second ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        def _norm(c: int) -> nn.Module:
            return nn.InstanceNorm3d(c, affine=True) if use_instance_norm else nn.BatchNorm3d(c)

        layers: list = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout3d(p=dropout_p))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, use_instance_norm=use_instance_norm, dropout_p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with skip connection then double conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_channels * 2, out_channels, use_instance_norm=use_instance_norm, dropout_p=dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3DCompletion(nn.Module):
    """
    Compact 3D U-Net tailored for voxel completion.
    Input shape: (B, Cin, G, G, G) where Cin≈seed mask + feature channels.
    Output: logits (B,1,G,G,G).
    """

    def __init__(
        self,
        in_channels: int = 65,
        base_channels: int = 32,
        out_channels: int = 1,
        use_instance_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        kw = dict(use_instance_norm=use_instance_norm, dropout_p=dropout_p)
        self.inc = DoubleConv(in_channels, c1, **kw)
        self.down1 = Down(c1, c2, **kw)
        self.down2 = Down(c2, c3, **kw)
        self.down3 = Down(c3, c4, **kw)
        self.down4 = Down(c4, c5, **kw)
        self.bottleneck = DoubleConv(c5, c5, **kw)
        self.up1 = Up(c5, c4, c4, **kw)
        self.up2 = Up(c4, c3, c3, **kw)
        self.up3 = Up(c3, c2, c2, **kw)
        self.up4 = Up(c2, c1, c1, **kw)
        self.outc = OutConv(c1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bottleneck(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetCompletionModel(nn.Module):
    """
    Wrapper that owns both the feature compressor and the UNet backbone so they
    are saved/restored together.
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
        extra_in_channels: int = 0,
    ) -> None:
        super().__init__()
        self.feature_compressor = DinoCompressor(d_in=feat_in, d_mid=feat_mid, d_out=feat_out)
        self.unet = UNet3DCompletion(
            in_channels=1 + feat_out + extra_in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            use_instance_norm=use_instance_norm,
            dropout_p=dropout_p,
        )
        self.compressed_channels = feat_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
