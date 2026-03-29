import argparse
import json
from argparse import Namespace
import logging
from safetensors.torch import load_file
import os
import os.path as osp
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

from gaussian_renderer import render
from scene.cameras import MiniCam

# from src.datasets import Scan3RPatchObjectModifiedDataset
from src.datasets import Scan3RSceneBatchDataset
from src.representations.gaussian.gaussian_model import Gaussian
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SCRATCH = os.environ.get("SCRATCH", "/scratch")

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets.loaders import get_train_val_data_loader, get_val_dataloader
from src.engine import EpochBasedTrainer
from src.models.latent_autoencoder import LatentAutoencoder
# from src.models.structure_model import StructureModel
from src.models.structure_model_with_bbox_head import StructureModel
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r
from utils.gaussian_splatting import GaussianSplat
# from utils.loss_utils import l1_loss, ssim
# from utils.graphics_utils import getProjectionMatrix
from utils.visualisation import save_vox_as_ply, side_by_side, slice_mosaic
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SCRATCH = os.environ.get("SCRATCH", "/scratch")

class FlowEulerCfgSampler:
    """Minimal Euler sampler with classifier-free guidance."""

    def __init__(self, sigma_min: float) -> None:
        self.sigma_min = sigma_min

    def _pred_velocity(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_scalar: float,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        cfg_strength: float,
        t_scale: float,
    ) -> torch.Tensor:
        bsz = x_t.shape[0]
        t = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.float32)
        t_model = (t * t_scale).to(dtype=torch.float32)
        cond_rep = cond
        neg_rep = neg_cond if neg_cond is not None else cond
        if cond is not None and cond.shape[0] == 1 and bsz > 1:
            cond_rep = cond.repeat(bsz, *([1] * (cond.dim() - 1)))
        if neg_rep is not None and neg_rep.shape[0] == 1 and bsz > 1:
            neg_rep = neg_rep.repeat(bsz, *([1] * (neg_rep.dim() - 1)))
        pred = model(x_t, t_model, cond_rep)
        if cfg_strength == 0.0 or neg_cond is None:
                return pred
        neg_pred = model(x_t, t_model, neg_rep)
        return (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        steps: int,
        cfg_strength: float,
        t_scale: float,
    ) -> torch.Tensor:
        sample = noise
        t_vals = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
        for idx in range(steps):
            t_curr = t_vals[idx].item()
            t_prev = t_vals[idx + 1].item()
            v = self._pred_velocity(model, sample, t_curr, cond, neg_cond, cfg_strength, t_scale)
            sample = sample - (t_curr - t_prev) * v
        return sample


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        
        self.cfg = cfg
        self.cfg.data.preload_slat = False
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules
        scene_graph_cfg = getattr(self.cfg.data, "scene_graph", None)
        self.seed_feat_dim: int = (
            getattr(scene_graph_cfg, "feat_dim", 1024) if scene_graph_cfg is not None else 1024
        )
        
        # TODO update configs
        self.G = 64
        self.use_cond = True
        self.bbox_warmup_epochs = 500
        self.recon_weight_after_warmup = 1.0
        self.use_predicted_box_for_remap = True
        self.flow_loss_weight: float = getattr(cfg.train.loss, "flow_loss_weight", 1.0)
        self.flow_cond_max_ctx: int = getattr(cfg.train.loss, "flow_cond_max_ctx", 2048)
        self.flow_cond_rand_fill: int = getattr(cfg.train.loss, "flow_cond_rand_fill", 0)
        # self.flow_cond_use_gt: bool = getattr(cfg.train.loss, "flow_cond_use_gt", False)
        self.flow_cond_use_gt = False
        # self.enable_flow_sampling: bool = getattr(
        #     cfg.train.loss, "flow_sampling_eval", False
        # )
        self.enable_flow_sampling = True
        self.flow_sampling_interval: int = getattr(cfg.train.loss, "flow_sampling_interval", 50)
        flow_sampling_steps_cfg: int = getattr(cfg.train.loss, "flow_sampling_steps", 50)
        self.flow_sampling_steps_train: int = getattr(
            cfg.train.loss, "flow_sampling_steps_train", flow_sampling_steps_cfg
        )
        self.flow_sampling_steps_eval: int = getattr(
            cfg.train.loss, "flow_sampling_steps_eval", flow_sampling_steps_cfg
        )
        self.flow_sampling_batch: int = getattr(cfg.train.loss, "flow_sampling_batch", 2)
        self.flow_sigma_min: float = getattr(cfg.train.loss, "flow_sigma_min", 0.0001)
        self.flow_t_scale: float = getattr(cfg.train.loss, "flow_t_scale", 1000.0)
        self.flow_t_schedule: Dict[str, Any] = getattr(
            cfg.train.loss,
            "flow_t_schedule",
            {"name": "uniform", "mean": 0.0, "std": 1.0},
        )
        self.flow_p_uncond: float = getattr(cfg.train.loss, "flow_p_uncond", 0.0)
        self.flow_cfg_strength: float = getattr(cfg.train.loss, "flow_cfg_strength", 0.0)
        self.flow_sampler = FlowEulerCfgSampler(self.flow_sigma_min)
        # deterministic controls
        self.flow_train_use_fixed_noise: bool = getattr(
            cfg.train.loss, "flow_train_use_fixed_noise", False
        )
        self.flow_train_fixed_seed: int = getattr(
            cfg.train.loss, "flow_train_fixed_seed", 0
        )
        self.flow_sampling_use_fixed_noise: bool = getattr(
            cfg.train.loss, "flow_sampling_use_fixed_noise", False
        )
        self.flow_sampling_fixed_seed: int = getattr(
            cfg.train.loss, "flow_sampling_fixed_seed", 0
        )
        self._flow_train_generator: Optional[torch.Generator] = None
        self._flow_sampling_generator: Optional[torch.Generator] = None

        # image patch metadata for conditioning
        img_enc_cfg = self.cfg.data.img_encoding
        if img_enc_cfg.img_rotate:
            self.image_w = img_enc_cfg.resize_h
            self.image_h = img_enc_cfg.resize_w
            self.patch_w = img_enc_cfg.patch_h
            self.patch_h = img_enc_cfg.patch_w
        else:
            self.image_w = self.cfg.data.img.w
            self.image_h = self.cfg.data.img.h
            self.patch_w = img_enc_cfg.patch_w
            self.patch_h = img_enc_cfg.patch_h
        self.num_image_patches = self.patch_h * self.patch_w
        self.img_patch_feat_dim = self.cfg.autoencoder.encoder.img_patch_feat_dim
        self.patch_feature_folder = osp.join(
            self.cfg.data.root_dir, "files", img_enc_cfg.feature_dir
        )
        self.patch_feature_cache: Dict[str, Dict[Any, Any]] = {}
        self.image_cond_model_name: str = getattr(cfg.train.loss, "image_cond_model", "dinov2_vitg14")
        self.image_cond_resize: int = getattr(cfg.train.loss, "image_cond_resize", 518)
        self.dino_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._image_encoder = None
        self._image_token_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self.use_dino_disk_cache: bool = getattr(cfg.train.loss, "use_dino_cache", True)
        cache_dir = getattr(cfg.train.loss, "dino_cache_dir", osp.join(self.cfg.output_dir, "dino_cache"))
        self.dino_cache_dir = cache_dir
        if self.use_dino_disk_cache:
            os.makedirs(self.dino_cache_dir, exist_ok=True)
                
        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight

        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(

            cfg, dataset = Scan3RSceneBatchDataset
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        self.cond_feat_dim: int = self._resolve_cond_feat_dim()
        self.occ_cond_feat_dim: int = 4 + self.seed_feat_dim
        self.flow_cond_dim: Optional[int] = None
        # model
        model = self.create_model()
        self.register_model(model)
        self.has_flow = True
        self.flow_patch_size: int = getattr(self.model.flow, "patch_size", 1) if self.has_flow else 1
        self.flow_cond_dim = getattr(self.model.flow, "cond_channels", self.cond_feat_dim)

        flow_params = list(self.model.flow.parameters())
        adapter_params: List[nn.Parameter] = []
        if getattr(self.model, "cond_adapter", None) is not None:
            adapter_params += list(self.model.cond_adapter.parameters())
        if getattr(self.model, "occ_cond_adapter", None) is not None:
            adapter_params += list(self.model.occ_cond_adapter.parameters())
        optimizer = optim.AdamW(
            flow_params + adapter_params,
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
            eps=1e-3,
        )
        self.register_optimizer(optimizer)

        # scheduler
        if cfg.train.optim.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                cfg.train.optim.lr_decay_steps,
                gamma=cfg.train.optim.lr_decay,
            )
        elif cfg.train.optim.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.train.optim.T_max,
                eta_min=cfg.train.optim.lr_min,
                T_mult=cfg.train.optim.T_mult,
                last_epoch=-1,
            )
        elif cfg.train.optim.scheduler == "linear":
            scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: (
                    1.0
                    if epoch <= cfg.train.optim.sched_start_epoch
                    else (
                        1.0
                        if epoch >= cfg.train.optim.sched_end_epoch
                        else (
                            1
                            - (epoch - cfg.train.optim.sched_start_epoch)
                            / (
                                cfg.train.optim.sched_end_epoch
                                - cfg.train.optim.sched_start_epoch
                            )
                        )
                        + (cfg.train.optim.end_lr / cfg.train.optim.lr)
                        * (epoch - cfg.train.optim.sched_start_epoch)
                        / (
                            cfg.train.optim.sched_end_epoch
                            - cfg.train.optim.sched_start_epoch
                        )
                    )
                ),
            )
        else:
            scheduler = None

        if scheduler is not None:
            self.register_scheduler(scheduler)

        self.logger.info("Initialisation Complete")

    def create_model(self) -> StructureModel:
        
        model = StructureModel(
            cfg=self.cfg.autoencoder,
            device=self.device,
            cond_feat_dim=self.cond_feat_dim,
            occ_cond_feat_dim=self.occ_cond_feat_dim,
            flow_cond_dim=self.flow_cond_dim,
        )

        model.load_state_dict(
            torch.load(
                "/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_structure_model_gt2gt/2025-11-13_2scenes_gt2gt/snapshots/epoch-10.pth.tar", map_location=self.device
            )["model"], strict=False
        )
        json_path = f"{SCRATCH}/TRELLIS-image-large/pipeline.json"
        with open(json_path, "r") as f:
            trellis_pipeline = json.load(f)
        path = trellis_pipeline["args"]["models"]["sparse_structure_flow_model"]
        with open(f"{SCRATCH}/TRELLIS-image-large/{path}.json", "r") as f:
            flow_configs = json.load(f)
        flow_state_dict = load_file(f"{SCRATCH}/TRELLIS-image-large/{path}.safetensors")
        model.flow.load_state_dict(flow_state_dict, strict=False)
        
        # model.load_state_dict(
        #     torch.load(
        #         "/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_structure_model/2025-11-5_100_scenes_continue_turvsky_lambda_8/snapshots/epoch-31.pth.tar", map_location=self.device
        #     )["model"]
        # )
        # self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
        for param in model.decoder.parameters():
            param.requires_grad = False
        model.decoder.eval()

        
        return model
    
    # TODO: check config; build loss & training pipeline; adapt dataloader; prepare overfitting data; visualization

    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        root_dir_scratch = '/cluster/scratch/wangyih/3RScan'
        base = osp.join(root_dir_scratch, "files", "gs_annotations", scene_id, "scene_level_structure")
        p = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not osp.exists(p):
            cands = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_") and f.endswith(".npz")]
            if not cands: raise FileNotFoundError(f"No aligned pack in {base}")
            p = osp.join(base, cands[0])
        return np.load(p)
    
    def _build_occ_cond(self, pack, expect_G: int, use_cond: bool):
        G = int(pack["G"])
        if G != expect_G: raise ValueError(f"G mismatch: expected {expect_G}, got {G}")
        occ = np.zeros((G,G,G), dtype=np.uint8)
        gt  = pack["vox_idx_gt_occ"]
        if gt.size: occ[gt[:,0], gt[:,1], gt[:,2]] = 1
        cond = None
        if use_cond:
            if "feats" in pack and pack["feats"].size:
                cond = pack["feats"].astype(np.float32).mean(axis=0)  # (1024,)
            else:
                cond = np.zeros((1024,), dtype=np.float32)
            cond = torch.from_numpy(cond).unsqueeze(0)
        occ = torch.from_numpy(occ).unsqueeze(0).float()     # (1,G,G,G)
        cond = None if cond is None else cond
        return occ, cond
    
    def _scene_id(self, scene_ids):
        x = scene_ids
        # while isinstance(x, (list,tuple)) and len(x)>0: x = x[0]
        x = x[0]
        return str(x)

    def _choose_frame(self, scene_id, B):
        root_dir = self.cfg.data.root_dir
        scene_dir = osp.join(root_dir, "scenes")
        frame_ids = scan3r.load_frame_idxs(scene_dir, scene_id)
        if len(frame_ids)> 60:
            frame_ids = frame_ids[:60]
        if len(frame_ids) >= B:
            fids = random.sample(frame_ids, B)
        else: fids = ['000000', '000000', '000000', '000000']
        
        # fids = ['000009', '000009', '000009', '000009', '000009', '000009']
        return fids
    
    def _rasterize_idx(self, idx_np: np.ndarray, G: int) -> torch.Tensor:
        """(M,3) -> (1,G,G,G) float {0,1}"""
        occ = np.zeros((G,G,G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:,0], idx_np[:,1], idx_np[:,2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()
    
    def _indices_to_occ(self, idx_t: torch.Tensor, G: int) -> torch.Tensor:
        """Indices tensor -> (1,1,G,G,G) occupancy on same device"""
        occ = torch.zeros(1, 1, G, G, G, device=idx_t.device, dtype=torch.float32)
        if idx_t.numel() > 0:
            occ[0, 0, idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]] = 1.0
        return occ
        
    def _make_batch(self, data_dict: Dict[str, Any]):
        sg = data_dict["scene_graphs"]
        scene_ids_arr = sg["scene_ids"]
        scene_ids = [sid[0] for sid in scene_ids_arr]
        frame_ids_seq = sg.get("frame_ids")
        image_frames = sg.get("image_frames", {})
        G = self.G

        occ_gt_list = []
        mean_gt_list, scale_gt_list = [], []
        mean_seed0_list, scale_seed0_list = [], []
        frame_sel_list: List[str] = []
        scene_sel_list: List[str] = []

        seed_idx_tensors = []
        feats_tensors = []

        for idx, sid in enumerate(scene_ids):
            if frame_ids_seq is not None and idx < len(frame_ids_seq):
                fid = str(frame_ids_seq[idx])
            else:
                frames = image_frames.get(sid, [])
                if frames:
                    fid = str(frames[0])
                else:
                    frame_ids = scan3r.load_frame_idxs(
                        osp.join(self.cfg.data.root_dir, "scenes"), sid
                    )
                    fid = frame_ids[0] if frame_ids else "000000"

            pack = self._load_aligned_pack(self.root_dir, sid, fid)
            occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0 = \
                self._build_inputs(pack, G)

            seed_idx_tensors.append(torch.from_numpy(seed_idx).int())
            feats_tensors.append(torch.from_numpy(feats).float())

            occ_gt_list.append(occ_gt.unsqueeze(0))          # (1,1,G,G,G)
            mean_gt_list.append(mean_gt.unsqueeze(0))        # (1,3)
            scale_gt_list.append(scale_gt.unsqueeze(0))      # (1,1)
            mean_seed0_list.append(mean_seed0.unsqueeze(0))  # (1,3)
            scale_seed0_list.append(scale_seed0.unsqueeze(0))# (1,1)
            frame_sel_list.append(fid)
            scene_sel_list.append(sid)

        occ_gt      = torch.cat(occ_gt_list,      0).to(self.device)        # (B,1,G,G,G)
        mean_gt     = torch.cat(mean_gt_list,     0).to(self.device)        # (B,3)
        scale_gt    = torch.cat(scale_gt_list,    0).to(self.device)        # (B,1)
        mean_seed0  = torch.cat(mean_seed0_list,  0).to(self.device)        # (B,3)
        scale_seed0 = torch.cat(scale_seed0_list, 0).to(self.device)        # (B,1)

        return (
            occ_gt,
            seed_idx_tensors,
            feats_tensors,
            mean_gt,
            scale_gt,
            mean_seed0,
            scale_seed0,
            frame_sel_list,
            scene_sel_list,
        )

    
    def scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int):
        """
        idx_t:  (M,3) int on device
        feat_t: (M,C) float on device
        returns:
        grid_feats: (1,C,G,G,G) float
        seed_occ:   (1,1,G,G,G) float binary
        """
        if idx_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else 64
            grid_feats = torch.zeros(1, C, G, G, G, device=feat_t.device, dtype=feat_t.dtype)
            seed_occ   = torch.zeros(1, 1, G, G, G, device=feat_t.device, dtype=feat_t.dtype)
            return grid_feats, seed_occ

        M, C = feat_t.shape
        lin = (idx_t[:,0]*G*G + idx_t[:,1]*G + idx_t[:,2]).long()  # (M,)

        Csum = torch.zeros(C, G*G*G, device=feat_t.device, dtype=feat_t.dtype)
        cnt  = torch.zeros(G*G*G,   device=feat_t.device, dtype=feat_t.dtype)

        Csum.index_add_(1, lin, feat_t.T)                      # sum features
        cnt.index_add_(0, lin, torch.ones(M, device=feat_t.device, dtype=feat_t.dtype))

        mask = cnt > 0
        Csum[:, mask] = Csum[:, mask] / cnt[mask]

        grid_feats = Csum.view(C, G, G, G).unsqueeze(0)        # (1,C,G,G,G)

        seed_occ = torch.zeros(1,1,G,G,G, device=feat_t.device, dtype=feat_t.dtype)
        uniq = torch.unique(lin)
        seed_occ.view(1,1,-1)[0,0,uniq] = 1.0
        return grid_feats, seed_occ
        
    
    def _build_inputs(self, pack, expect_G: int):
        G = int(pack["G"])
        if G != expect_G:
            raise ValueError(f"G mismatch: expected {expect_G}, got {G}")

        occ_gt  = self._rasterize_idx(pack["vox_idx_gt_occ"], G)  # (1,G,G,G)

        seed_idx = pack.get("seed_idx", np.zeros((0,3), np.int32))
        feats    = pack.get("feats",    np.zeros((0,1024), np.float32))

        # mean_gt   = torch.from_numpy(pack["mean"].astype(np.float32))            # (3,)
        mean_gt = torch.from_numpy(pack["mean_gt"].astype(np.float32)) 
        scale_gt  = torch.tensor(float(pack["scale_gt"]), dtype=torch.float32).view(1)  # (1,)

        mean_seed0  = torch.from_numpy(pack["seed_box_init_mean"].astype(np.float32)) # (3,)
        scale_seed0 = torch.tensor(float(pack["seed_box_init_scale"]), dtype=torch.float32).view(1)

        return occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0

    def _format_gt_input(self, occ_gt: torch.Tensor) -> torch.Tensor:
        """
        Convert (B,1,G,G,G) occupancy into encoder input by concatenating dummy feature channels.
        """
        B = occ_gt.shape[0]
        feat_ch = self.model.encoder.input_layer.in_channels - 1
        zeros = torch.zeros(B, feat_ch, self.G, self.G, self.G, device=occ_gt.device, dtype=occ_gt.dtype)
        # return torch.cat([occ_gt, zeros], dim=1)
        return occ_gt
    
    def _empty_flow_cond_tokens(
        self, G: int, patch_size: int, feat_dim: int, batch_size: int = 1
    ) -> torch.Tensor:
        gp = G // max(1, patch_size)
        target_L = max(1, min(self.flow_cond_max_ctx, gp * gp * gp))
        return torch.zeros(
            batch_size,
            target_L,
            self.flow_cond_dim,
            device=self.device,
            dtype=torch.float16,
        )

    def _build_occ_cond_tokens(
        self,
        idx: torch.Tensor,
        feats: Optional[torch.Tensor],
        G: int,
        patch_size: int,
    ) -> torch.Tensor:
        """
        Build (1, Lctx, feat_dim) conditioning tokens for the flow using remapped seed occupancy.
        Aggregates per patch, keeps the densest patches and pads/truncates to a fixed context.
        """
        feat_dim_out = self.occ_cond_feat_dim
        if idx is None or idx.numel() == 0:
            return self._empty_flow_cond_tokens(G, patch_size, feat_dim_out, batch_size=1)

        patch = max(1, patch_size)
        seeds = idx.detach().cpu().long()
        gp = max(1, G // patch)
        Lq = gp * gp * gp
        target_L = max(1, min(self.flow_cond_max_ctx, Lq))

        patch_coords = (seeds // patch).clamp(min=0, max=gp - 1)
        patch_id = (
            patch_coords[:, 0] * gp * gp
            + patch_coords[:, 1] * gp
            + patch_coords[:, 2]
        )

        counts = torch.zeros(Lq, dtype=torch.float32)
        counts.index_add_(
            0,
            patch_id,
            torch.ones_like(patch_id, dtype=torch.float32),
        )
        feat_sum = torch.zeros(Lq, self.seed_feat_dim, dtype=torch.float32)
        if feats is not None and feats.numel() > 0:
            feats_cpu = feats.detach().cpu().float()
            feat_sum.index_add_(0, patch_id, feats_cpu)

        patch_ids = torch.arange(Lq, dtype=torch.long)
        coords_all = torch.stack(
            [
                patch_ids // (gp * gp),
                (patch_ids // gp) % gp,
                patch_ids % gp,
            ],
            dim=1,
        ).float()
        if gp > 1:
            coords_norm_all = (coords_all / (gp - 1) - 0.5) * 2.0
        else:
            coords_norm_all = torch.zeros_like(coords_all)

        feat_mean_all = feat_sum.clone()
        nonzero_mask = counts > 0
        if self.seed_feat_dim > 0:
            denom = counts.clone()
            denom[denom == 0] = 1.0
            feat_mean_all = feat_sum / denom.unsqueeze(1)
            feat_mean_all[~nonzero_mask] = 0.0

        cond_tokens_full = torch.cat(
            [counts.unsqueeze(1), coords_norm_all, feat_mean_all], dim=1
        )
        # if target_L < Lq:
        #     # keep the densest patches so conditioning focuses on occupied regions
        #     top_idx = torch.argsort(counts, descending=True)
        #     top_idx = top_idx[:target_L]
        #     cond_tokens = cond_tokens_full[top_idx]
        # else:
        #     pad = target_L - Lq
        #     if pad > 0:
        #         pad_tokens = torch.zeros(pad, feat_dim_out, dtype=cond_tokens_full.dtype)
        #         cond_tokens = torch.cat([cond_tokens_full, pad_tokens], dim=0)
        #     else:
        #         cond_tokens = cond_tokens_full
        cond_tokens = cond_tokens_full

        cond_tokens = cond_tokens.to(device=self.device, dtype=torch.float32)
        # cond_tokens = self.model.occ_cond_adapter(cond_tokens).to(dtype=torch.float16)
        return cond_tokens.unsqueeze(0)
    
    def _load_patch_feature_scene(self, scene_id: str) -> Optional[Dict[Any, Any]]:
        if scene_id in self.patch_feature_cache:
            return self.patch_feature_cache[scene_id]
        path = osp.join(self.patch_feature_folder, f"{scene_id}.pkl")
        if not osp.exists(path):
            return None
        data = common.load_pkl_data(path)
        self.patch_feature_cache[scene_id] = data
        return data

    def _dino_cache_path(self, scene_id: str, frame_id: str) -> str:
        safe_scene = str(scene_id)
        safe_frame = str(frame_id)
        return osp.join(self.dino_cache_dir, safe_scene, f"{safe_frame}.pt")

    def _get_patch_tokens(self, scene_id: str, frame_id: str) -> Optional[torch.Tensor]:
        feat_dict = self._load_patch_feature_scene(scene_id)
        if feat_dict is None:
            return None
        keys = [
            frame_id,
            frame_id.lstrip("0") if isinstance(frame_id, str) else frame_id,
        ]
        try:
            keys.append(int(frame_id))
        except Exception:
            pass
        if isinstance(frame_id, str):
            try:
                keys.append(str(int(frame_id)))
            except Exception:
                pass
        feat = None
        for key in keys:
            if key in feat_dict:
                feat = feat_dict[key]
                break
        if feat is None:
            return None
        arr = np.asarray(feat)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2 and arr.shape[0] != self.num_image_patches:
            arr = arr.T
        arr = arr.reshape(-1, arr.shape[-1])
        return torch.from_numpy(arr.astype(np.float32))

    def _build_image_cond_tokens(self, scene_id: str, frame_id: str) -> Optional[torch.Tensor]:
        return self._encode_image_cond_tokens(scene_id, frame_id)

    def _resolve_cond_feat_dim(self) -> int:
        explicit_dim = getattr(self.cfg.train.loss, "image_cond_dim", None)
        if explicit_dim is not None:
            return int(explicit_dim)
        model_dim_map = {
            "dinov2_vitl14": 1024,
            "dinov2_vitl14_reg": 1024,
            "dinov2_vitg14": 1536,
            "dinov2_vitb14": 768,
            "dinov2_vits14": 384,
        }
        return model_dim_map.get(self.image_cond_model_name, self.img_patch_feat_dim)

    def _ensure_image_encoder(self) -> None:
        if self._image_encoder is None:
            self.logger.info(f"Loading Dinov2 image encoder ({self.image_cond_model_name})")
            model = torch.hub.load(
                "facebookresearch/dinov2",
                self.image_cond_model_name,
                pretrained=True,
            )
            model = model.eval().to(self.device)
            self._image_encoder = model

    def _resolve_frame_path(self, scene_id: str, frame_id: str) -> Optional[str]:
        base_dir = osp.join(self.root_dir, "scenes", scene_id, "sequence")
        fid = str(frame_id)
        candidates = [fid]
        try:
            fid_int = int(fid)
            candidates.extend([str(fid_int), f"{fid_int:06d}"])
        except Exception:
            pass
        candidates.append(fid.zfill(6))
        seen = []
        for cand in candidates:
            if cand in seen:
                continue
            seen.append(cand)
            img_name = f"frame-{cand}.color.jpg"
            path = osp.join(base_dir, img_name)
            if osp.exists(path):
                return path
        return None

    def _load_frame_image_tensor(self, scene_id: str, frame_id: str) -> Optional[torch.Tensor]:
        img_path = self._resolve_frame_path(scene_id, frame_id)
        if img_path is None:
            self.logger.warning("Image not found for %s frame %s", scene_id, frame_id)
            return None
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            self.logger.warning("Failed to open image %s: %s", img_path, exc)
            return None
        img = img.resize((self.image_cond_resize, self.image_cond_resize), Image.LANCZOS)
        arr = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        arr = arr.permute(2, 0, 1)
        arr = self.dino_normalize(arr)
        return arr

    def _encode_image_cond_tokens(self, scene_id: str, frame_id: str) -> Optional[torch.Tensor]:
        cache_key = (scene_id, frame_id)
        tokens: Optional[torch.Tensor] = None

        cached = self._image_token_cache.get(cache_key)
        if cached is not None:
            tokens = cached.to(self.device)
        elif self.use_dino_disk_cache:
            disk_path = self._dino_cache_path(scene_id, frame_id)
            if osp.exists(disk_path):
                try:
                    disk_payload = torch.load(disk_path, map_location="cpu")
                    if isinstance(disk_payload, dict) and "tokens" in disk_payload:
                        tokens = disk_payload["tokens"].float()
                    elif isinstance(disk_payload, torch.Tensor) and disk_payload.ndim == 2:
                        tokens = disk_payload.float()
                    else:
                        self.logger.warning(
                            "Legacy Dinov2 cache for %s/%s stores adapted tokens; re-encoding image.",
                            scene_id,
                            frame_id,
                        )
                        tokens = None
                    if tokens is not None:
                        self._image_token_cache[cache_key] = tokens.clone()
                        tokens = tokens.to(self.device)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to load Dinov2 cache for %s/%s: %s",
                        scene_id,
                        frame_id,
                        exc,
                    )

        if tokens is None:
            img_tensor = self._load_frame_image_tensor(scene_id, frame_id)
            if img_tensor is None:
                return None
            self._ensure_image_encoder()
            with torch.no_grad():
                image = img_tensor.unsqueeze(0).to(self.device)
                feats = self._image_encoder(image, is_training=True)["x_prenorm"]
                feats = F.layer_norm(feats, feats.shape[-1:])
            tokens = feats.squeeze(0).to(self.device, dtype=torch.float32)
            L = tokens.shape[0]
            target_L = max(1, min(self.flow_cond_max_ctx, L))
            # if L >= target_L:
            #     tokens = tokens[:target_L]
            # else:
            #     pad = target_L - L
            #     tokens = torch.cat(
            #         [tokens, torch.zeros(pad, tokens.shape[1], device=tokens.device)],
            #         dim=0,
            #     )
            tokens_cpu = tokens.detach().cpu()
            self._image_token_cache[cache_key] = tokens_cpu.clone()
            if self.use_dino_disk_cache:
                disk_path = self._dino_cache_path(scene_id, frame_id)
                try:
                    os.makedirs(osp.dirname(disk_path), exist_ok=True)
                    torch.save({"tokens": tokens_cpu}, disk_path)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to save Dinov2 cache for %s/%s: %s",
                        scene_id,
                        frame_id,
                        exc,
                    )
        cond_tokens = self.model.cond_adapter(tokens)
        cond_tokens = cond_tokens.unsqueeze(0).to(dtype=torch.float16)
        return cond_tokens

    def _apply_cfg_dropout(
        self, cond_tokens: Optional[torch.Tensor], neg_tokens: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if cond_tokens is None or neg_tokens is None:
            return cond_tokens
        if self.flow_p_uncond <= 0.0:
            return cond_tokens
        mask = (
            torch.rand(cond_tokens.shape[0], device=cond_tokens.device)
            < self.flow_p_uncond
        )
        if mask.any():
            cond_tokens = cond_tokens.clone()
            cond_tokens[mask] = neg_tokens[mask]
        return cond_tokens
    
    def _get_flow_train_generator(self) -> Optional[torch.Generator]:
        if not self.flow_train_use_fixed_noise:
            return None
        if self._flow_train_generator is None:
            device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            gen = torch.Generator(device=device_type)
            gen.manual_seed(int(self.flow_train_fixed_seed))
            self._flow_train_generator = gen
        return self._flow_train_generator

    def _get_flow_sampling_generator(self) -> Optional[torch.Generator]:
        if not self.flow_sampling_use_fixed_noise:
            return None
        if self._flow_sampling_generator is None:
            device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            gen = torch.Generator(device=device_type)
            gen.manual_seed(int(self.flow_sampling_fixed_seed))
            self._flow_sampling_generator = gen
        return self._flow_sampling_generator
    
    def _sample_flow_t(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        schedule = (self.flow_t_schedule or {}).get("name", "logit_normal")
        if schedule.lower() == "uniform":
            return torch.rand(batch_size, device=self.device, generator=generator)
        mean = (self.flow_t_schedule or {}).get("mean", 0.0)
        std = (self.flow_t_schedule or {}).get("std", 1.0)
        rand = torch.randn(batch_size, device=self.device, generator=generator)
        return torch.sigmoid(rand * std + mean)

    def _kld(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def _idx_to_centers(self, idx: torch.Tensor, G: int) -> torch.Tensor:
        # idx: (M,3) int -> (M,3) in [-0.5,0.5], fp32
        return (idx.float() + 0.5) / G - 0.5

    def _centers_to_idx(self, centers: torch.Tensor, G: int) -> torch.Tensor:
        # centers: (M,3) in [-0.5,0.5] -> (M,3) int clipped
        idx = torch.floor((centers + 0.5) * G).long()
        return torch.clamp(idx, 0, G - 1)

    def _compute_mean_iou(
        self, probs: torch.Tensor, gt: torch.Tensor, thresholds=(0.3, 0.5, 0.7)
    ) -> Dict[float, torch.Tensor]:
        metrics: Dict[float, torch.Tensor] = {}
        gt_bin = (gt > 0.5).float()
        for thr in thresholds:
            pr = (probs >= thr).float()
            inter = (pr * gt_bin).sum(dim=(1, 2, 3, 4))
            union = pr.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4)) - inter
            metrics[thr] = (inter / (union + 1e-6)).mean()
        return metrics

    def _sample_latents_with_flow(
        self,
        cond_tokens: torch.Tensor,
        neg_tokens: torch.Tensor,
        latent_shape: Tuple[int, ...],
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = max(1, int(steps if steps is not None else self.flow_sampling_steps_eval))
        sampler_gen = self._get_flow_sampling_generator()
        noise = torch.randn(
            latent_shape,
            device=self.device,
            dtype=torch.float32,
            generator=sampler_gen,
        )
        samples = self.flow_sampler.sample(
            self.model.flow,
            noise,
            cond_tokens,
            neg_tokens,
            steps=steps,
            cfg_strength=self.flow_cfg_strength,
            t_scale=self.flow_t_scale,
        )
        return samples
    
    def focal_bce_with_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha= None,   # weight for positives (class=1)
        mask= None,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Focal BCE on logits. Supports scalar or tensor alpha (broadcastable to logits).
        logits:  any shape
        targets: same shape, {0,1} (or [0,1] soft labels)
        mask:    same shape (1 keeps, 0 ignores)
        """
        # Standard BCE (stable, on logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = p if y=1 else (1-p), computed stably
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # (1 - p_t)^gamma
        mod = (1.0 - p_t).clamp_min(eps).pow(gamma)

        if alpha is not None:
            # alpha is weight for positives; build alpha_t that matches targets
            # alpha can be a scalar or broadcastable tensor (e.g., (B,1,1,1,1))
            alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
            loss = alpha_t * mod * ce
        else:
            loss = mod * ce

        if mask is not None:
            loss = loss * mask

        if reduction == "mean":
            denom = (mask.sum() if mask is not None else torch.numel(loss)).clamp_min(1.0)
            return loss.sum() / denom
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _remap_seed_idx_with_bbox(
        self,
        seed_idx: torch.Tensor,      # (M,3) int, seed canonical (built with mean_seed0/scale_seed0)
        mean_src: torch.Tensor,      # (B,3)   mean_seed0 for sample b
        scale_src: torch.Tensor,     # (B,1)   scale_seed0 for sample b
        mean_dst: torch.Tensor,      # (B,3)   mean to map into (pred or GT)
        scale_dst: torch.Tensor,     # (B,1)   scale to map into (pred or GT)
        b: int,                      # batch index
        G: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Map seed_idx (seed canonical) -> dst canonical by going through the shared
        'world_can' frame. All means/scales are in the *same rotated frame*
        (you already applied R_frame in your pack building).
            seed_can -> world_can (mean_src, scale_src)
                    -> dst_can   (mean_dst, scale_dst)
        """
        # 1) seed idx -> seed centers
        c_seed = self._idx_to_centers(seed_idx, G)                        # (M,3)

        # 2) seed centers -> world_can
        s_src = torch.clamp(scale_src[b].view(1,1), min=eps)              # (1,1)
        m_src = mean_src[b].view(1,3)                                     # (1,3)
        world_can = c_seed * (2.0 * s_src) + m_src                        # (M,3)

        # 3) world_can -> dst centers
        s_dst = torch.clamp(scale_dst[b].view(1,1), min=eps)
        m_dst = mean_dst[b].view(1,3)
        c_dst = (world_can - m_dst) / (2.0 * s_dst)                       # (M,3) in ~[-0.5,0.5]

        # 4) centers -> idx in dst canonical
        idx_dst = self._centers_to_idx(c_dst, G)                          # (M,3) int
        return idx_dst

    def remap_occ(self, occ_src, mean_src, scale_src, mean_dst, scale_dst, G):
        B = occ_src.shape[0]
        occ_dst = torch.zeros_like(occ_src)
        for b in range(B):
            idx_src = (occ_src[b,0] > 0.5).nonzero(as_tuple=False)
            if idx_src.numel() == 0: 
                continue
            idx_dst = self._remap_seed_idx_with_bbox(
                seed_idx=idx_src, mean_src=mean_src, scale_src=scale_src,
                mean_dst=mean_dst,   scale_dst=scale_dst, b=b, G=G
            )
            occ_dst[b,0, idx_dst[:,0], idx_dst[:,1], idx_dst[:,2]] = 1.0
        return occ_dst


    def freeze_encoder(self) -> None:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            
    def get_extrinsics_by_frame_id(self, scene_id, frame_id, frames, img_poses):
        for obj_id in frames[scene_id]:
            if frame_id in frames[scene_id][obj_id]:
                pose_idx = frames[scene_id][obj_id].index(frame_id)
                extrinsics = img_poses[scene_id][obj_id][pose_idx]
                return extrinsics
        else:
            raise ValueError(f"Frame {frame_id} not found in scene {scene_id}.")
        
    def get_unique_frame_ids(self, frames, scene_id):
        frame_id_set = {
            frame_id
            for obj_id in frames[scene_id]
            for frame_id in frames[scene_id][obj_id]
        }
        return list(frame_id_set)
    
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, StructureModel)
        # data_dict["scene_graphs"]["tot_obj_splat"] =  data_dict["scene_graphs"]["tot_obj_splat"][0]
        # occ_gt, occ_vis = self._make_batch(data_dict)
        (
            occ_gt,
            seed_idx_list,
            feats_list,
            mean_gt,
            scale_gt,
            mean_seed0,
            scale_seed0,
            frame_ids_sel,
            scene_ids,
        ) = self._make_batch(data_dict)
        B, _, G, _, _ = occ_gt.shape
        flow_enabled = self.has_flow

        mean_dst, scale_dst = mean_gt, scale_gt
        recon_w = self.recon_weight_after_warmup
        occ_gt_aligned = self.remap_occ(
            occ_src=occ_gt,
            mean_src=mean_gt,
            scale_src=scale_gt.view(B, 1),
            mean_dst=mean_gt,
            scale_dst=scale_gt.view(B, 1),
            G=self.G,
        )

        cond_tok_list = [] if flow_enabled else None
        neg_tok_list = [] if flow_enabled else None
        if flow_enabled and cond_tok_list is not None:
            for b in range(B):
                cond_tok = self._build_image_cond_tokens(scene_ids[b], frame_ids_sel[b])
                if cond_tok is None:
                    cond_tok = self._empty_flow_cond_tokens(
                        self.G, self.flow_patch_size, self.cond_feat_dim, batch_size=1
                    )
                cond_tok_list.append(cond_tok)
                neg_tok_list.append(torch.zeros_like(cond_tok))

        occ_vis = torch.zeros_like(occ_gt)
        cond_tokens = None
        cond_tokens_pos = None
        neg_tokens = None
        if flow_enabled and cond_tok_list is not None:
            if cond_tok_list:
                cond_tokens_pos = torch.cat(cond_tok_list, dim=0)
                neg_tokens = torch.cat(neg_tok_list, dim=0)
            else:
                feat_dim = self.cond_feat_dim
                cond_tokens_pos = self._empty_flow_cond_tokens(
                    self.G, self.flow_patch_size, feat_dim, batch_size=B
                )
                neg_tokens = torch.zeros_like(cond_tokens_pos)
            # cond_tokens = self._apply_cfg_dropout(cond_tokens_pos, neg_tokens)
            cond_tokens = cond_tokens_pos

        x_gt_input = self._format_gt_input(occ_gt_aligned.float())
        with torch.no_grad():
            _, mu, _logvar, _ = self.model.encoder(
                x_gt_input, sample_posterior=False, return_raw=True, return_feat=True
            )
            logits_gt = self.model.decoder(mu)

        sample_logits = None
        sample_iou_metrics: Dict[float, torch.Tensor] = {}
        should_sample = (
            self.enable_flow_sampling
            and flow_enabled
            and cond_tokens_pos is not None
            and self.flow_sampling_interval > 0
            and (
                (iteration % self.flow_sampling_interval == 0)
                or (self.inner_iteration == 1)
            )
        )
        if should_sample:
            with torch.no_grad():
                B_samp = min(self.flow_sampling_batch, cond_tokens_pos.shape[0])
                if B_samp > 0:
                    cond_samp = cond_tokens_pos[:B_samp]
                    neg_samp = neg_tokens[:B_samp] if neg_tokens is not None else torch.zeros_like(cond_samp)
                    latent_shape = (B_samp,) + tuple(mu.shape[1:])
                    sample_latent = self._sample_latents_with_flow(
                        cond_samp,
                        neg_samp,
                        latent_shape,
                        steps=self.flow_sampling_steps_train,
                    )
                    latent_mse = ((sample_latent - mu)**2).mean()
                    sample_logits = self.model.decoder(sample_latent)
                    sample_probs = torch.sigmoid(sample_logits)
                    sample_gt = occ_gt_aligned[:B_samp].float()
                    sample_iou_metrics = self._compute_mean_iou(sample_probs, sample_gt)
        flow_loss = torch.tensor(0.0, device=self.device)
        if flow_enabled and cond_tokens is not None:
            Bz = mu.shape[0]
            train_gen = self._get_flow_train_generator()
            t = self._sample_flow_t(Bz, generator=train_gen)
            eps = torch.randn(
                mu.shape,
                device=mu.device,
                dtype=mu.dtype,
                generator=train_gen,
            )
            t_b = t.view(Bz, 1, 1, 1, 1)
            sigma = self.flow_sigma_min + (1.0 - self.flow_sigma_min) * t_b
            zt = (1.0 - t_b) * mu + sigma * eps
            v_target = ((1.0 - self.flow_sigma_min) * eps - mu).detach()
            cond_tokens = cond_tokens.to(device=self.device, dtype=torch.float16)
            # uncond_tokens = (
            #     neg_tokens.to(device=self.device, dtype=torch.float16)
            #     if neg_tokens is not None
            #     else torch.zeros_like(cond_tokens)
            # )
            t_input = (t * self.flow_t_scale).to(self.device)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                v_pred_cond = self.model.flow(zt, t_input, cond_tokens)
                # v_pred_uncond = self.model.flow(zt, t_input, uncond_tokens)
            loss_cond = F.mse_loss(v_pred_cond.float(), v_target.float())
            # loss_uncond = F.mse_loss(v_pred_uncond.float(), v_target.float())
            # flow_loss = 0.5 * (loss_cond + loss_uncond)
            flow_loss = loss_cond

        loss = self.flow_loss_weight * flow_loss
        with torch.no_grad():
            probs = torch.sigmoid(logits_gt.float())
            iou_metrics = self._compute_mean_iou(probs, occ_gt_aligned.float())
            iou_03 = iou_metrics[0.3]
            iou_05 = iou_metrics[0.5]
            iou_07 = iou_metrics[0.7]
       
        # -----------------------------------------------------------------------------------------------
        
        
        # x_in_list = []
        # seed_occ_list = []
       
        # # Build per-sample inputs on device
        # for b in range(B):
        #     idx_np = seed_idx_list[b]
        #     feats_np = feats_list[b]
        #     idx_t   = torch.from_numpy(idx_np).to(self.device).int()        # (Mb,3)
        #     feats_t = torch.from_numpy(feats_np).to(self.device).float()     # (Mb,1024)

        #     # compress per-seed
        #     with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        #         feats_comp = self.model.comp(feats_t)                        # (Mb,64) fp16→ln→fp16

        #     # scatter mean into grid
        #     grid_dino, seed_occ = self.scatter_voxel_mean(idx_t, feats_comp.float(), self.G)  # (1,64,G,G,G), (1,1,G,G,G)
        #     x_in = torch.cat([seed_occ, grid_dino], dim=1)                   # (1,1+64,G,G,G)

        #     x_in_list.append(x_in)
        #     seed_occ_list.append(seed_occ)

        # x_in   = torch.cat(x_in_list, dim=0)     # (B,65,G,G,G)
        # occ_vis= torch.cat(seed_occ_list, dim=0) # (B,1,G,G,G)


        
        # # with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        # with torch.amp.autocast('cuda', enabled=False):
        #     # --- VAE using TRELLIS enc/dec ---
        #     # z, mu, logvar = self.model.encoder(occ_vis, sample_posterior=False, return_raw=True)
        #     z, mu, logvar, feat3d = self.model.encoder(x_in, sample_posterior=False, return_raw=True, return_feat=True)
        #     z_data = self._reparameterize(mu, logvar)       # (B,zc,8,8,8)
            
        # with torch.amp.autocast('cuda', enabled=False):
        #     logits = self.model.decoder(z_data)             # (B,1,G,G,G)
        #     mean_pred, scale_pred = self.model.forward_bbox(feat3d.float(), mean_seed0.float(), scale_seed0.float())

            
        # logits32 = logits.float(); occ_gt32 = occ_gt.float(); occ_vis32 = occ_vis.float()
        # occ_gt32 = occ_gt.float()
        # occ_vis32= occ_vis.float()

        # mask_complete = (occ_vis32 == 0).float()
        # bce = F.binary_cross_entropy_with_logits(logits32, occ_gt32, reduction='none')
        # # vae_rec = F.binary_cross_entropy_with_logits(logits, occ_1g)
        # vae_rec = (bce * mask_complete).sum() / (mask_complete.sum() + 1e-6)
        # vae_kld = self._kld(mu.float(), logvar.float())
        # flow_logits_viz = None        
        # # bbox loss (canonical frame)
        # mean_pred  = mean_pred.view(B, 3)
        # mean_gt    = mean_gt.view(B, 3)
        # # scale_pred = scale_pred[0]
        # scale_pred = scale_pred.view(B)          
        # scale_gt   = scale_gt.view(B)
         
        # L_box_mean  = F.l1_loss(mean_pred,  mean_gt, reduction='mean')
        # L_box_scale = F.l1_loss(torch.log(scale_pred+1e-6), torch.log(scale_gt+1e-6), reduction='mean')
        # L_box = L_box_mean + L_box_scale
    
        # mask = occ_vis32.to(logits32.device).bool()

      
        
        # loss = (
        #     1 * vae_rec
        #     + 0.001 * vae_kld
        #     + 0.5 * L_box
        #     # + 0.1*L_seed
        #     # + 0.1*L_shrink
        # )
        
        
        # if epoch > self.bbox_warmup_epochs:
        # ----------------------------------------------------------------------------------------
        loss_dict = {
            "loss": loss*100,
            "flow_loss": flow_loss,
            # "IoU @ 0.3": iou_03.item(),
            "IoU @ 0.5": iou_05.item(),
            # "IoU @ 0.7": iou_07.item(),
        }
        if sample_iou_metrics:
            loss_dict["IoU_flow @ 0.3"] = sample_iou_metrics.get(0.3, torch.tensor(0.0)).item()
            loss_dict["IoU_flow @ 0.5"] = sample_iou_metrics.get(0.5, torch.tensor(0.0)).item()
            loss_dict["IoU_flow @ 0.7"] = sample_iou_metrics.get(0.7, torch.tensor(0.0)).item()
            loss_dict["latent_mse"] = latent_mse
        # else:
        #     loss_dict = {
        #     "loss": loss * 100,
        #     "L_box": L_box,
        #     "L_box_mean": L_box_mean,
        #     "L_box_scale": L_box_scale,
        #     # "L_seed": L_seed,
        #     # "L_shrink": L_shrink,
        #     "scale_pred": scale_pred.mean().detach().item(),
        #     }
        
        display_logits = sample_logits if sample_logits is not None else logits_gt
        viz_B = min(display_logits.shape[0], 4)
        output_dict = {
            "logits": display_logits[:viz_B].detach().float().cpu(),
            "occ_gt": occ_gt_aligned[:viz_B].detach().float().cpu(),
            "occ_vis": occ_vis[:viz_B].detach().float().cpu(),
        }
        output_dict["sample_logits"] = display_logits[:viz_B].detach().float().cpu()
        # ------------------------------------------------------------------------------------
        # loss = (
        #      L_box
        # )
        # output_dict = {}
        # output_dict["sample_logits"] = None  # (≤2,1,G,G,G)
        # loss_dict = {
        # "loss": loss * 100,
        # "L_box": L_box,
        # "L_box_mean": L_box_mean,
        # "L_box_scale": L_box_scale,
        # "scale_pred": scale_pred.mean().detach().item(),
        # }
        
        
        return output_dict, loss_dict
    
    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):

        # self._save_embeddings(epoch, iteration, data_dict, output_dict)
        pass
    
    def _save_embeddings(self, epoch, iteration, data_dict, output_dict):
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        embeddings = output_dict["embeddings"]
        os.makedirs(f"{self.cfg.output_dir}/embeddings", exist_ok=True)
        for i in range(embeddings.shape[0]):
            scene_id = scene_ids[i][0]
            obj_id = obj_ids[i]
            torch.save(
                embeddings[i],
                f"{self.cfg.output_dir}/embeddings/{scene_id}_{obj_id}.pt",
            )

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        # self._save_embeddings(epoch, iteration, data_dict, output_dict)
        pass
    
    def set_eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        # self.perceptual_loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        self.training = True
        self.model.train()
        self.model.encoder.eval()
        self.model.decoder.eval()
        # self.perceptual_loss.train()
        torch.set_grad_enabled(True)

    def visualize(
        self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"
    ) -> None:
        
        """
        Visuals for structure training:
        - Save GT and Pred occupancy as PLY (voxel centers).
        - Save slice mosaics (GT | PR) as a PNG and TensorBoard image.
        - (Optional) Save a sample from the flow as PLY/PNG for quick sanity.
        """
        # if epoch > self.bbox_warmup_epochs:
        # if (epoch % 20)!=0:
        #     return
        outdir = f"{self.cfg.output_dir}/events"
        os.makedirs(outdir, exist_ok=True)
        
        # ---- unpack ----
        logits_b11ggg: torch.Tensor = output_dict["logits"]        # (B,1,G,G,G)
        gt_b11ggg: torch.Tensor     = output_dict["occ_gt"]        # (B,1,G,G,G) or (B,1,G,G,G) float
        input_b11ggg: torch.Tensor  = output_dict["occ_vis"]
        G: int                      = 64
        thr: float                  = 0.5
        B = logits_b11ggg.shape[0]
        
        sample_logits = output_dict.get("sample_logits", None)  # (B,1,G,G,G) or None
        
        
        # ---- save first few items ----
        max_items = min(4, B)
        for i in range(max_items):
            logits_1 = logits_b11ggg[i,0]               # (G,G,G)
            gt_1     = gt_b11ggg[i,0]                   # (G,G,G)
            input_1 = input_b11ggg[i,0]

            # point clouds
            pr_idx = (logits_1.sigmoid() > thr).nonzero(as_tuple=False)
            gt_idx = (gt_1 > 0.5).nonzero(as_tuple=False)
            input_idx = (input_1 > 0.5).nonzero(as_tuple=False)
            save_vox_as_ply(gt_idx, G, f"{outdir}/{mode}_gt_struct_{i}_completion.ply")
            save_vox_as_ply(pr_idx, G, f"{outdir}/{mode}_pred_struct_{i}_completion.ply")
            save_vox_as_ply(input_idx, G, f"{outdir}/{mode}_input_struct_{i}_completion.ply")
            # slice mosaic PNG
            sbs = side_by_side(gt_1, logits_1, max_slices=6)  # (2,H,W)
            save_image(sbs, f"{outdir}/{mode}_slices_{i}.png", normalize=True)

            # TensorBoard (channels-first: we’ll make it NCHW)
            self.writer.add_image(
                f"{mode}/slices_{i}_gt_pred",
                sbs.unsqueeze(1),  # (2,1,H,W)
                global_step=epoch,
                dataformats="NCHW",
            )
        if sample_logits is not None:
            for i in range(min(2, sample_logits.shape[0])):
                sm = sample_logits[i,0]  # (G,G,G) logits or probs
                sm_idx = (sm.sigmoid() > thr).nonzero(as_tuple=False)
                save_vox_as_ply(sm_idx, G, f"{outdir}/{mode}_sample_struct_{i}.ply")
                s = slice_mosaic(sm, max_slices=6)  # (1,H,W)
                save_image(s, f"{outdir}/{mode}_sample_slices_{i}.png", normalize=True)
                self.writer.add_image(
                    f"{mode}/sample_slices_{i}",
                    s.unsqueeze(0),   # (1,1,H,W)
                    global_step=epoch,
                    dataformats="NCHW",
                )
    
            
        def _to_set(idx: torch.Tensor) -> set:
            # idx: (K,3) long
            if idx.numel() == 0:
                return set()
            return set(map(tuple, idx.cpu().numpy().astype(int).tolist()))

        for i in range(max_items):
            logits_1 = logits_b11ggg[i, 0]  # (G,G,G), logits
            gt_1     = gt_b11ggg[i, 0]      # (G,G,G), {0,1} or float in [0,1]

            pr_idx = (logits_1.sigmoid() > thr).nonzero(as_tuple=False).long()  # (Mp,3)
            gt_idx = (gt_1 > 0.5).nonzero(as_tuple=False).long()                # (Mg,3)

            # --- metrics via sets ---
            S_pr = _to_set(pr_idx)
            S_gt = _to_set(gt_idx)
            S_tp = S_pr & S_gt
            S_fp = S_pr - S_gt
            S_fn = S_gt - S_pr

            vox_pr = len(S_pr)
            vox_gt = len(S_gt)
            tp     = len(S_tp)
            fp     = len(S_fp)
            fn     = len(S_fn)
            union  = tp + fp + fn
            iou    = float(tp / (union + 1e-8))
            same   = (fp == 0 and fn == 0)

            print(f"[viz] pred vox={vox_pr} gt vox={vox_gt} TP={tp} FP={fp} FN={fn} IoU={iou:.4f} all_equal={same}")
        # else:
        #     pass
        # pass


def parse_args(
    parser: argparse.ArgumentParser = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument(
        "--config", dest="config", default="", type=str, help="configuration name"
    )
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--snapshot", default=None, help="load from snapshot")
    parser.add_argument(
        "--load_encoder", default=None, help="name of pretrained encoder"
    )
    parser.add_argument("--epoch", type=int, default=None, help="load epoch")
    parser.add_argument("--log_steps", type=int, default=1, help="logging steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for ddp")

    args, unknown_args = parser.parse_known_args()
    return parser, args, unknown_args


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    parser, args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
