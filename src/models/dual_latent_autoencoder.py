import json
import os
import copy
from typing import Any, List, Optional

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
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.backbones.sparse_decoder import _REPRESENTATION_CONFIG

SCRATCH = os.environ.get("SCRATCH", "/scratch")

config_sparse = copy.deepcopy(_REPRESENTATION_CONFIG)
config_sparse["num_gaussians"] = 32

config_complex = copy.deepcopy(_REPRESENTATION_CONFIG)
config_complex["num_gaussians"] = 64


class DualDecoderAutoencoder(LatentAutoencoder):
    def __init__(self, cfg, device, load_pretrained: bool = True):
        super().__init__(cfg, device)
        self.cfg = cfg
        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)
        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["slat_decoder_gs"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs_sparse = json.load(f)
                configs_complex = copy.deepcopy(configs_sparse)

            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            config_sparse = configs_sparse["args"]
            config_complex = configs_complex["args"]
            config_complex["representation_config"]["num_gaussians"]=64
            
                                   
            self.decoder_easy = SLatGaussianDecoder( 
                                                    **config_sparse
                                                    ).to(device)
            # define with same args
            self.decoder_complex = SLatGaussianDecoder(
                                                    **config_complex
                                                    ).to(device)

    def decode(self, x, use_complex_mask: Optional[torch.Tensor] = None):
        if use_complex_mask is None:
            # default behavior: decode everything with `decoder_sparse`
            return self.decoder_easy(x)
        
        # Optionally, split x into two groups
        mask = use_complex_mask.bool()  # shape [N]

        x_complex = SparseTensor(
            coords=x.coords[mask],
            feats=x.feats[mask],
            shape=x.shape,
        )

        x_easy = SparseTensor(
            coords=x.coords[~mask],
            feats=x.feats[~mask],
            shape=x.shape,
        )
        batch_size = int(x.coords[:, 0].max().item()) + 1
        # sparse_out = self.decoder_easy(x_easy)
        sparse_out = self.decoder_easy(x_easy)
        complex_out = self.decoder_complex(x_complex)
        
        # Reassemble
        # sparse_idx = (~use_complex_mask).nonzero(as_tuple=True)[0]
        # complex_idx = (use_complex_mask).nonzero(as_tuple=True)[0]
        # for i, idx in enumerate(sparse_idx):
        #     out[idx] = sparse_out[i]
        # for i, idx in enumerate(complex_idx):
        #     out[idx] = complex_out[i]
        merged = Gaussian(sh_degree=0,
                aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                mininum_kernel_size=config_sparse["3d_filter_kernel_size"],
                scaling_bias=config_sparse["scaling_bias"],
                opacity_bias=config_sparse["opacity_bias"],
                scaling_activation=config_sparse["scaling_activation"],)
        
        merged._xyz = torch.cat([sparse_out[0]._xyz, complex_out[0]._xyz], dim=0)
        merged._features_dc = torch.cat([sparse_out[0]._features_dc, complex_out[0]._features_dc], dim=0)
        # merged._features_rest = torch.cat([sparse_out[0]._features_rest, complex_out[0]._features_rest], dim=0)
        merged._opacity = torch.cat([sparse_out[0]._opacity, complex_out[0]._opacity], dim=0)
        merged._scaling = torch.cat([sparse_out[0]._scaling, complex_out[0]._scaling], dim=0)
        merged._rotation = torch.cat([sparse_out[0]._rotation, complex_out[0]._rotation], dim=0)

        return [merged], complex_out[0] 
        # return sparse_out

