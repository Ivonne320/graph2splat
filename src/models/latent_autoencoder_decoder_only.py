import json
import os
from typing import Any, List

import torch
from safetensors.torch import load_file
from torch import nn

from configs import AutoencoderConfig
from src.models.backbones import (
    SLatEncoder,
    SLatGaussianDecoder,
    SparseStructureDecoder,
    SparseStructureEncoder,
)
from src.modules.sparse.basic import SparseTensor
from src.representations.gaussian.gaussian_model import Gaussian

SCRATCH = os.environ.get("SCRATCH", "/scratch")


class LatentAutoencoderDecoderOnly(nn.Module):
    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        downsample: bool = False,
        load_pretrained: bool = True,
    ) -> None:
        super(LatentAutoencoderDecoderOnly, self).__init__()
        self.cfg = cfg
        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)

        # if load_pretrained:
        #     path = trellis_pipeline["args"]["models"]["slat_encoder"]
        #     with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
        #         configs = json.load(f)
        #     configs["args"]["resolution"]=128
        #     state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
        #     self.encoder = SLatEncoder(**configs["args"])
        #     self.encoder.load_state_dict(state_dict, strict=False)
        #     self.encoder = self.encoder.to(device)
        # else:
        #     self.encoder = SLatEncoder(
        #         resolution=128,
        #         in_channels=1024,
        #         model_channels=768,
        #         latent_channels=16,
        #         num_blocks=12,
        #         num_heads=12,
        #         use_fp16=True,
        #     ).to(device)

        # if load_pretrained:
        #     path = trellis_pipeline["args"]["models"]["slat_decoder_gs"]
        #     with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
        #         configs = json.load(f)
        #     state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
        #     configs["args"]["representation_config"]["sh_degree"] =  self.cfg.sh_degree
        #     # configs["args"]["representation_config"]["sh_degree"] = 1
        #     # self.cfg.sh_degree = 1
        #     # configs["args"]["representation_config"]["lr"]["_scaling"] = 0.1
        #     if self.cfg.sh_degree > 0:
        #         configs["args"]["representation_config"]["lr"]["_features_rest"] = 1.0
        #     # configs["args"]["representation_config"]["num_gaussians"]=self.cfg.num_gaussians
        #     configs["args"]["representation_config"]["num_gaussians"]=8
        #     # configs["args"]["representation_config"]["perturb_offset"] = False
        #     # configs["args"]["representation_config"]["voxel_size"]=2
        #     configs["args"]["representation_config"]["scaling_bias"]= 8e-4
        #     configs["args"]["resolution"]=128
        #     # configs["args"]["representation_config"]["3d_filter_kernel_size"] = 5e-3

            
        #     net = SLatGaussianDecoder(**configs["args"])
        #     # net.load_state_dict(state_dict)

        # else:
        net = SLatGaussianDecoder(
            resolution=128,
            model_channels=768,
            latent_channels=1027,
            num_blocks=12,
            num_heads=12,
            num_head_channels=64,
            use_fp16=True,
        ).to(device)

        self.decoder = net.to(device)

    # def encode(self, data_dict: dict[str, Any]) -> SparseTensor:
    #     data_dict = data_dict["scene_graphs"]
    #     voxel_sparse_tensor = data_dict["tot_obj_splat"]
    #     return self.encoder(voxel_sparse_tensor)

    def decode(self, code: SparseTensor) -> List[Gaussian]:
        return self.decoder(code)
