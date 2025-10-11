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
from src.models.cond_adapter import CondAdapter
from src.models.seed_bbox_head import SeedBBoxHead

SCRATCH = os.environ.get("SCRATCH", "/scratch")

class DinoCompressor(nn.Module):
    def __init__(self, d_in=1024, d_mid=256, d_out=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid), nn.SiLU(),
            nn.Linear(d_mid, d_out)
        )
        self.ln = nn.LayerNorm(d_out)

    @torch.no_grad()
    def _l2norm(self, x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, feats):  # feats: (M,1024)
        x = self._l2norm(feats)
        x = self.mlp(x)
        x = self.ln(x)
        return x
    
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
        
        self.comp = DinoCompressor(d_in=1024, d_mid=256, d_out=64).to(device)
        self.seed_bbox = SeedBBoxHead(d_feat=64, use_coords=True).to(device)
        C_out = 64
        # Instantiate Structure Encoder
        if load_pretrained:
            path = trellis_pipeline["args"]["models"]["sparse_structure_encoder"]
            with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
                configs = json.load(f)
            state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
            self.encoder = SparseStructureEncoder(**configs["args"])
            self.encoder.load_state_dict(state_dict, strict=False)
            self.encoder = self.encoder.to(device)
            
            # adjust encoder input layer
            ch0 = self.encoder.channels[0]
            self.encoder.input_layer = nn.Conv3d(1 + C_out, ch0, 3, padding=1).to(device)
        else:
            self.encoder = SparseStructureEncoder(
                in_channels = 1,
                latent_channels = 8,
                num_res_blocks = 2,
                num_res_blocks_middle = 2,
                channels = [32, 128, 512],
                use_fp16 = True
            ).to(device)
            ch0 = self.encoder.channels[0]
            self.encoder.input_layer = nn.Conv3d(1 + C_out, ch0, 3, padding=1).to(device)
            
        # self.global_pool = nn.AdaptiveAvgPool3d(1).to(device) 
        # self.box_head = nn.Sequential(
        #     nn.Linear(self.encoder.channels[-1], 256), nn.SiLU(),
        #     nn.Linear(256, 128), nn.SiLU(),
        #     nn.Linear(128, 4)  # δmean(3), δscale_raw(1)
        # ).to(device)   
        
        
            
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
            
            
    def forward_bbox(self, feat_3d, mean_seed0, scale_seed0):
        glob = self.global_pool(feat_3d).flatten(1)
        d = self.box_head(glob)
        dmean, dscale_raw = d[:, :3], d[:, 3:4]
        scale_pred = scale_seed0 * (1.0 + F.softplus(dscale_raw))
        mean_pred  = mean_seed0 + dmean
        return mean_pred, scale_pred
    
    @torch.no_grad()
    def _idx_to_centers(self, idx: torch.Tensor, G: int) -> torch.Tensor:
        # idx: (M,3) -> centers in [-0.5, 0.5]
        return (idx.float() + 0.5) / G - 0.5
    
    def forward_bbox_from_seeds_batch(
        self,
        feats_comp_list: list[torch.Tensor],  # length B, (Mi,64) fp32
        idx_list: list[torch.Tensor],         # length B, (Mi,3)  int
        G: int,
        mean_seed0: torch.Tensor,             # (B,3)
        scale_seed0: torch.Tensor             # (B,1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          mean_pred:  (B,3)
          scale_pred: (B,)
        """
        B = len(feats_comp_list)
        # Concatenate seeds across batch
        if B == 0:
            return mean_seed0, scale_seed0.view(-1)

        # Handle empty Mi robustly
        feats_all  = torch.cat([f if f.numel() else f.new_zeros((0, f.shape[-1])) for f in feats_comp_list], dim=0)
        idx_all    = torch.cat([i if i.numel() else i.new_zeros((0, 3)) for i in idx_list], dim=0)
        batch_idx  = torch.cat([torch.full((feats_comp_list[b].shape[0],), b, device=feats_all.device, dtype=torch.long)
                                for b in range(B)], dim=0) if feats_all.numel() else torch.empty(0, dtype=torch.long, device=mean_seed0.device)

        # Centers in seed canonical
        centers_all = self._idx_to_centers(idx_all, G) if idx_all.numel() else feats_all.new_zeros((0,3))

        # If a sample has zero seeds, we still want a row; scatter ops above handle it via bincount(minlength=B)
        d = self.seed_bbox.forward_batch(feats_all, centers_all, batch_idx, B)  # (B,4)

        dmean, dscale_raw = d[:, :3], d[:, 3:4]
        scale_base = scale_seed0.clamp_min(1e-6)          # (B,1)
        scale_pred = scale_base * (1.0 + F.softplus(dscale_raw))
        mean_pred  = mean_seed0 + dmean
        return mean_pred, scale_pred.view(-1)

