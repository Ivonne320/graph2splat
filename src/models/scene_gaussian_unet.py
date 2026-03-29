import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGatedConv(nn.Module):
    """3D convolution gated by a guidance mask."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.mask_conv = nn.Conv3d(1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        gate = torch.sigmoid(self.mask_conv(mask))
        return self.act(self.norm(feat * gate))


class MaskAwareDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block1 = MaskGatedConv(in_ch, out_ch)
        self.block2 = MaskGatedConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, mask)
        x = self.block2(x, mask)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = MaskAwareDoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(x)
        mask = F.max_pool3d(mask, kernel_size=2, stride=2)
        x = self.conv(x, mask)
        return x, mask


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.conv = MaskAwareDoubleConv(skip_ch * 2, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        mask: torch.Tensor,
        skip_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        mask = F.interpolate(mask, scale_factor=2, mode="nearest")
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        if diffX or diffY or diffZ:
            x = F.pad(
                x,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )
            mask = F.pad(
                mask,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        mask_cat = torch.cat([skip_mask, mask], dim=1).amax(dim=1, keepdim=True)
        x = self.conv(x, mask_cat)
        return x, mask_cat


class SceneGaussianUNet(nn.Module):
    """3D UNet that predicts Gaussian parameters per voxel."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        dino_feat_dim: int,
        dino_grid_channels: int,
        gaussians_per_voxel: int,
        param_dim: int,
    ) -> None:
        super().__init__()
        self.dino_feat_dim = dino_feat_dim
        self.dino_grid_channels = dino_grid_channels if dino_feat_dim > 0 else 0
        if self.dino_grid_channels > 0:
            self.dino_projector = nn.Linear(dino_feat_dim, self.dino_grid_channels, bias=False)
        else:
            self.dino_projector = None

        self.gaussians_per_voxel = gaussians_per_voxel
        self.param_dim = param_dim

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        self.inc = MaskAwareDoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        self.bottleneck = MaskAwareDoubleConv(c5, c5)
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)
        out_ch = gaussians_per_voxel * param_dim
        self.out = nn.Conv3d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x, mask)
        x2, m2 = self.down1(x1, mask)
        x3, m3 = self.down2(x2, m2)
        x4, m4 = self.down3(x3, m3)
        x5, m5 = self.down4(x4, m4)
        bottleneck = self.bottleneck(x5, m5)
        x, m = self.up1(bottleneck, x4, m5, m4)
        x, m = self.up2(x, x3, m, m3)
        x, m = self.up3(x, x2, m, m2)
        x, _ = self.up4(x, x1, m, mask)
        return self.out(x)

    def project_dino_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if self.dino_projector is None:
            raise RuntimeError("DINO projector is not initialised.")
        return self.dino_projector(feats)
