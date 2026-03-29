import json
import os
from typing import Any, List, Optional

import torch
from safetensors.torch import load_file
from torch import nn

from configs import AutoencoderConfig
from src.models.backbones import SLatEncoder
from src.models.backbones.sparse_decoder_rgb_skip import SLatGaussianDecoderRgbSkip
from src.modules.sparse.basic import SparseTensor
from src.representations.gaussian.gaussian_model import Gaussian

SCRATCH = os.environ.get("SCRATCH", "/scratch")


class LatentAutoencoderRgbSkip(nn.Module):
    """
    Latent autoencoder with RGB skip injected into Gaussian DC features.
    """

    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        downsample: bool = False,
        load_pretrained: bool = True,
    ) -> None:
        super(LatentAutoencoderRgbSkip, self).__init__()
        self.cfg = cfg
        self._rgb_skip: Optional[SparseTensor] = None

        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)

        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["slat_encoder"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            configs["args"]["resolution"]=128
            self.encoder = SLatEncoder(**configs["args"])
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder = self.encoder.to(device)
        else:
            self.encoder = SLatEncoder(
                resolution=128,
                in_channels=1024,
                model_channels=768,
                latent_channels=16,
                num_blocks=12,
                num_heads=12,
                use_fp16=True,
            ).to(device)

        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["slat_decoder_gs"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            configs["args"]["representation_config"]["sh_degree"] = self.cfg.sh_degree
            if self.cfg.sh_degree > 0:
                configs["args"]["representation_config"]["lr"]["_features_rest"] = 1.0
            configs["args"]["representation_config"]["num_gaussians"] = 8
            configs["args"]["representation_config"]["scaling_bias"] = 8e-4
            configs["args"]["resolution"]=128
            net = SLatGaussianDecoderRgbSkip(**configs["args"])
            # net.load_state_dict(state_dict)
        else:
            net = SLatGaussianDecoderRgbSkip(
                resolution=128,
                model_channels=768,
                latent_channels=16,
                num_blocks=12,
                num_heads=12,
                num_head_channels=64,
                use_fp16=True,
            ).to(device)

        self.decoder = net.to(device)

    def encode(self, data_dict: dict[str, Any]) -> SparseTensor:
        data_dict = data_dict["scene_graphs"]
        voxel_sparse_tensor = data_dict["tot_obj_splat"]
        self._rgb_skip = voxel_sparse_tensor
        return self.encoder(voxel_sparse_tensor)

    def decode(self, code: SparseTensor) -> List[Gaussian]:
        if self._rgb_skip is not None:
            out = self.decoder.forward_with_rgb_skip(code, self._rgb_skip)
            self._rgb_skip = None
            return out
        return self.decoder(code)
