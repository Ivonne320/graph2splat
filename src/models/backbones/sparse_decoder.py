# The code has been adapted from https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_vae/decoder_gs.py

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones.base import SparseTransformerBase
from src.modules import sparse as sp
from src.representations import Gaussian
from utils.random_utils import hammersley_sequence

_REPRESENTATION_CONFIG = {
    "perturb_offset": True,
    "voxel_size": 1.5,
    "num_gaussians": 32,
    "2d_filter_kernel_size": 0.1,
    "3d_filter_kernel_size": 9e-4,
    "scaling_bias": 4e-3,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
    "lr": {
        "_xyz": 1.0,
        "_features_dc": 1.0,
        "_features_rest": 1.0,  
        "_scaling": 1.0,
        "_rotation": 0.1,
        "_opacity": 1.0,
    },
}


class SLatGaussianDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config or _REPRESENTATION_CONFIG
        self._calc_layout()
        self.out_layer = sp.SparseLinear(model_channels, self.out_channels)

        self._build_perturbation()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _build_perturbation(self) -> None:
        perturbation = [
            hammersley_sequence(3, i, self.rep_config["num_gaussians"])
            for i in range(self.rep_config["num_gaussians"])
        ]
        perturbation = torch.tensor(perturbation).float() * 2 - 1
        perturbation = perturbation / self.rep_config["voxel_size"]
        perturbation = torch.atanh(perturbation).to(self.device)
        self.register_buffer("offset_perturbation", perturbation)

    def _calc_layout(self) -> None:
        sh_degree = 0
        num_sh_rest = (sh_degree + 1)**2 - 1  # exclude DC (1)
        num_features_rest = num_sh_rest * 3  # RGB per SH coef
        
        self.layout = {
            "_xyz": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_features_dc": {
                "shape": (self.rep_config["num_gaussians"], 1, 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            # "_features_rest": {
            #     "shape": (self.rep_config["num_gaussians"], num_sh_rest, 3),
            #     "size": self.rep_config["num_gaussians"] * num_features_rest,
            # },
            "_scaling": {
                "shape": (self.rep_config["num_gaussians"], 3),
                "size": self.rep_config["num_gaussians"] * 3,
            },
            "_rotation": {
                "shape": (self.rep_config["num_gaussians"], 4),
                "size": self.rep_config["num_gaussians"] * 4,
            },
            "_opacity": {
                "shape": (self.rep_config["num_gaussians"], 1),
                "size": self.rep_config["num_gaussians"],
            },
        }
        start = 0
        for k, v in self.layout.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.out_channels = start

    def to_representation(self, x: sp.SparseTensor) -> List[Gaussian]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            representation = Gaussian(
                # sh_degree=1,
                sh_degree=0,
                aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                # mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
                mininum_kernel_size=0.002,
                scaling_bias=self.rep_config["scaling_bias"],
                opacity_bias=self.rep_config["opacity_bias"],
                scaling_activation=self.rep_config["scaling_activation"],
            )

            xyz = (x.coords[x.layout[i]][:, 1:].float() + 0.5) / self.resolution
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
                    if k == "_opacity":
                        # Clamp or squash to valid range to avoid NaNs in loss
                        # feats = torch.sigmoid(feats)  # in (0,1)
                        feats = torch.clamp(feats, -10, 10)
                    elif k == "_scaling":
                        # Prevent negative or zero scaling (could break splatting)
                        feats = torch.clamp(feats, min=-10, max=5)
                    elif k == "_rotation":
                        feats = feats

                    # Optional NaN/Inf guard for all feats
                    feats = torch.nan_to_num(feats, nan=0.0, posinf=1e3, neginf=-1e3)
                    setattr(representation, k, feats)
            ret.append(representation)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[Gaussian]:
        h = super().forward(x)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return self.to_representation(h)
