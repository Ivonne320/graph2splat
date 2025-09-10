import json
import os, os.path as osp
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors.torch import load_file
import torch.nn as nn
import torch.nn.functional as F

from src.models.sparse_structure_flow import *
from src.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
from configs import AutoencoderConfig

SCRATCH = os.environ.get("SCRATCH", "/scratch")

class StructureModel(nn.Module):
    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        downsample: bool = False,
        load_pretrained: bool = True,
    ) -> None:
        super(StructureModel, self).__init__()
        self.cfg = cfg
        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)
            
        # Instantiate Structure Encoder
        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["sparse_structure_encoder"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            self.encoder = SparseStructureEncoder(**configs["args"])
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder = self.encoder.to(device)
        else:
            self.encoder = SparseStructureEncoder(
                in_channels = 1,
                latent_channels = 8,
                num_res_blocks = 2,
                num_res_blocks_middle = 2,
                channels = [32, 128, 512],
                use_fp16 = True
            ).to(device)
            
        # Instantiate Structure Decoder

        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["sparse_structure_decoder"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            self.decoder = SparseStructureDecoder(**configs["args"])
            self.decoder.load_state_dict(state_dict, strict=False)
            self.decoder = self.decoder.to(device)
        else:
            self.decoder = SparseStructureDecoder(
                out_channels=1,
                latent_channels=8,
                num_res_blocks=2,
                num_res_blocks_middle=2,
                channels=[512, 128, 32],
                use_fp16=True                
            ).to(device)
            
        # Instantiate Flow
                
        if load_pretrained:          
            path = trellis_pipeline["args"]["models"]["sparse_structure_flow_model"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            self.flow = SparseStructureFlowModel(**configs["args"])
            self.flow.load_state_dict(state_dict, strict=False)
            self.flow = self.flow.to(device)

        else:
            self.flow = SparseStructureFlowModel(
                resolution=16,
                in_channels=8,
                out_channels=8,
                model_channels=1024,
                cond_channels=1024,
                num_blocks=24,
                num_heads=16,
                mlp_ratio=4,
                patch_size=1,
                pe_mode='ape',
                qk_rms_norm=True,
                use_fp16=True
            ).to(device)

    # def encode(self, data_dict: dict[str, Any]) -> SparseTensor:
    #     data_dict = data_dict["scene_graphs"]
    #     voxel_sparse_tensor = data_dict["tot_obj_splat"]
    #     return self.encoder(voxel_sparse_tensor)

    # def decode(self, code: SparseTensor) -> List[Gaussian]:
    #     return self.decoder(code)
