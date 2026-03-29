from typing import Optional

import torch
import torch.nn as nn

from src.models.backbones.sparse_decoder import SLatGaussianDecoder
from src.modules import sparse as sp
from src.representations import Gaussian


class SLatGaussianDecoderRgbSkip(SLatGaussianDecoder):
    """
    Decoder that injects a per-voxel RGB skip into the Gaussian DC features.
    """

    def __init__(self, *args, rgb_skip_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb_skip_proj = nn.LazyLinear(3)
        self.rgb_skip_scale = nn.Parameter(torch.tensor(float(rgb_skip_scale)))
        self._rgb_skip: Optional[sp.SparseTensor] = None

    def forward_with_rgb_skip(
        self, x: sp.SparseTensor, rgb_skip: sp.SparseTensor
    ) -> list[Gaussian]:
        self._rgb_skip = rgb_skip
        out = super().forward(x)
        self._rgb_skip = None
        return out

    def to_representation(self, x: sp.SparseTensor) -> list[Gaussian]:
        ret = []
        num_gaussians = self.rep_config["num_gaussians"]
        rgb_skip = self._rgb_skip

        for i in range(x.shape[0]):
            representation = Gaussian(
                sh_degree=self.rep_config["sh_degree"],
                aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
            )

            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
            rgb_add = None
            if rgb_skip is not None:
                rgb_feats = rgb_skip.feats[rgb_skip.layout[i]]
                rgb_proj = torch.tanh(self.rgb_skip_proj(rgb_feats))
                rgb_add = (
                    rgb_proj.unsqueeze(1)
                    .repeat(1, num_gaussians, 1)
                    .reshape(-1, 1, 3)
                )

            for k, v in self.layout.items():
                if k == "_xyz":
                    offset = x.feats[x.layout[i]][
                        :, v["range"][0] : v["range"][1]
                    ].reshape(-1, *v["shape"])
                    offset = offset * self.rep_config["lr"][k]
                    if self.rep_config["perturb_offset"]:
                        offset = offset + self.offset_perturbation
                    offset = torch.nan_to_num(offset, nan=0.0, posinf=1e3, neginf=-1e3)
                    offset = (
                        torch.tanh(offset)
                        / self.resolution
                        * 0.5
                        * self.rep_config["voxel_size"]
                    )
                    _xyz = xyz.unsqueeze(1) + offset
                    setattr(representation, k, _xyz.flatten(0, 1))
                else:
                    feats = (
                        x.feats[x.layout[i]][:, v["range"][0] : v["range"][1]]
                        .reshape(-1, *v["shape"])
                        .flatten(0, 1)
                    )
                    feats = feats * self.rep_config["lr"][k]
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
                    if k == "_features_dc" and rgb_add is not None:
                        feats = feats + rgb_add.to(dtype=feats.dtype) * self.rgb_skip_scale
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret
