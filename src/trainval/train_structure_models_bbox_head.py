import argparse
from argparse import Namespace
import logging
import os
import os.path as osp
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from gaussian_renderer import render
from scene.cameras import MiniCam

from src.datasets import Scan3RPatchObjectModifiedDataset
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
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getProjectionMatrix
from utils.visualisation import save_vox_as_ply, side_by_side, slice_mosaic
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        
        self.cfg = cfg
        self.cfg.data.preload_slat = False
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules
        
        # TODO update configs
        self.G = 64
        self.use_cond = True
        self.bbox_warmup_epochs = 500
        self.recon_weight_after_warmup = 1.0
        self.use_predicted_box_for_remap = True
                
        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight

        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(

            cfg, dataset = Scan3RPatchObjectModifiedDataset
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
            eps=1e-3,
            # fused=False
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
        
        model = StructureModel(cfg=self.cfg.autoencoder, device=self.device)

        # model.load_state_dict(
        #     torch.load(
        #         "/mnt/hdd4tb/trainings/training_structural_model/2025-09-29_00-48-40/snapshots/epoch-3000.pth.tar", map_location=self.device
        #     )["model"]
        # )
        model.load_state_dict(
            torch.load(
                "/mnt/hdd4tb/trainings/training_structural_model/2025-09-28_21-28-48_bbox_pretrained/snapshots/epoch-2000.pth.tar", map_location=self.device
            )["model"]
        )
        # self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        # model.eval()
        
        return model
    
    # TODO: check config; build loss & training pipeline; adapt dataloader; prepare overfitting data; visualization

    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        base = osp.join(root_dir, "files", "gs_annotations", scene_id, "scene_level_structure_uncan")
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
        if len(frame_ids)> 50:
            frame_ids = frame_ids[:50]
        fids = random.sample(frame_ids, B)
        # fids = ['000009', '000009', '000009', '000009', '000009', '000009']
        return fids
    
    def _rasterize_idx(self, idx_np: np.ndarray, G: int) -> torch.Tensor:
        """(M,3) -> (1,G,G,G) float {0,1}"""
        occ = np.zeros((G,G,G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:,0], idx_np[:,1], idx_np[:,2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()
        
    def _make_batch(self, data_dict: Dict[str, Any]):
        # B = len(data_dict["scene_graphs"]["scene_ids"])
        B = 6
        frames_all = data_dict["scene_graphs"].get("obj_img_top_frames", {})
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        scene_id = scene_ids[0][0]
        G = self.G
        fids = self._choose_frame(scene_id, B)
        occ_gt_list = []
        seed_idx_list, feats_list = [], []
        mean_gt_list, scale_gt_list = [], []
        mean_seed0_list, scale_seed0_list = [], []

        for b in range(B):
            sid = scene_id
            # frames_for_scene = frames_all.get(sid, [])
            # fid = self._choose_frame(sid)
            fid = fids[b]
            pack = self._load_aligned_pack('/mnt/hdd4tb/3RScan/', sid, fid)

            occ_gt, seed_idx, feats, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._build_inputs(pack, G)
            occ_gt_list.append(occ_gt.unsqueeze(0))     # (1,1,G,G,G)
            seed_idx_list.append(seed_idx)              # (Mi,3) numpy
            feats_list.append(feats)                    # (Mi,1024) numpy
            mean_gt_list.append(mean_gt.unsqueeze(0))   # (1,3)
            scale_gt_list.append(scale_gt.unsqueeze(0)) # (1,1)
            mean_seed0_list.append(mean_seed0.unsqueeze(0))
            scale_seed0_list.append(scale_seed0.unsqueeze(0))

        occ_gt   = torch.cat(occ_gt_list,   0).to(self.device)           # (B,1,G,G,G)
        mean_gt  = torch.cat(mean_gt_list,  0).to(self.device).squeeze(1)    # (B,3)
        scale_gt = torch.cat(scale_gt_list, 0).to(self.device)   # (B,1)
        mean_seed0  = torch.cat(mean_seed0_list,  0).to(self.device).squeeze(1) # (B,3)
        scale_seed0 = torch.cat(scale_seed0_list, 0).to(self.device)# (B,1)

        # Keep idx/feats as Python lists, convert inside train_step per sample
        return occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0
    
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
        data_dict["scene_graphs"]["tot_obj_splat"] =  data_dict["scene_graphs"]["tot_obj_splat"][0]
        # occ_gt, occ_vis = self._make_batch(data_dict)
        occ_gt, seed_idx_list, feats_list, mean_gt, scale_gt, mean_seed0, scale_seed0 = self._make_batch(data_dict)
        B, _, G, _, _ = occ_gt.shape
        
        feats_comp_list, idx_t_list = [], []
        for b in range(B):
            idx_np   = seed_idx_list[b]
            feats_np = feats_list[b]
            idx_t    = torch.from_numpy(idx_np).to(self.device).long()       # (Mb,3)
            feats_t  = torch.from_numpy(feats_np).to(self.device).float()    # (Mb,1024)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                feats_comp = self.model.comp(feats_t).float()                # (Mb,64) -> fp32 for bbox
            feats_comp_list.append(feats_comp)
            idx_t_list.append(idx_t)

        # 2) Batched bbox (decoupled from encoder)
        mean_pred, scale_pred = self.model.forward_bbox_from_seeds_batch(
            feats_comp_list=feats_comp_list,   # list[(Mi,64)]
            idx_list=idx_t_list,               # list[(Mi,3)]
            G=self.G,
            mean_seed0=mean_seed0,             # (B,3)
            scale_seed0=scale_seed0            # (B,1)
        )
        
        L_box_mean  = F.l1_loss(mean_pred, mean_gt, reduction='mean')
        L_box_scale = F.l1_loss(torch.log(scale_pred.clamp_min(1e-6)),
                                torch.log(scale_gt.view(B).clamp_min(1e-6)), reduction='mean')
        L_box = L_box_mean + L_box_scale
        
        # -----------------------------------------------------------------------------------------
        mean_dst, scale_dst = mean_pred.detach(), scale_pred.detach().view(B,1)
        # mean_dst, scale_dst = mean_gt, scale_gt
        recon_w = self.recon_weight_after_warmup
        occ_gt_aligned = self.remap_occ(
                                        occ_src=occ_gt,
                                        mean_src=mean_gt,               scale_src=scale_gt.view(B,1),
                                        mean_dst=mean_pred.detach(),    scale_dst=scale_pred.detach().view(B,1),
                                        G=self.G
                                    )
        x_dst_list, occ_dst_seed_list = [], []
        for b in range(B):
            idx_dst = self._remap_seed_idx_with_bbox(
                seed_idx=idx_t_list[b],     # (Mi,3) in seed canonical
                mean_src=mean_seed0, scale_src=scale_seed0,
                mean_dst=mean_dst, scale_dst=scale_dst,
                b=b, G=self.G
            )
            grid_dino_dst, seed_occ_dst = self.scatter_voxel_mean(idx_dst.int(), feats_comp_list[b], self.G)
            x_dst = torch.cat([seed_occ_dst, grid_dino_dst], dim=1)         # (1,65,G,G,G)
            x_dst_list.append(x_dst); occ_dst_seed_list.append(seed_occ_dst)

        x_in   = torch.cat(x_dst_list,        dim=0)  # (B,65,G,G,G)  -> structure input
        occ_vis= torch.cat(occ_dst_seed_list, dim=0)  # (B,1,G,G,G)  -> mask for completion

        # # 4) Structure path (unchanged)
        with torch.amp.autocast('cuda', enabled=False):
            z, mu, logvar, feat3d = self.model.encoder(x_in, sample_posterior=False, return_raw=True, return_feat=True)
            z_data = self._reparameterize(mu, logvar)
            logits = self.model.decoder(z_data)
            # logits = self.model.decoder(z)
        # losses (safe BCE)
        logits32 = torch.clamp(logits.float(), -30.0, 30.0)
        # occ_gt32 = occ_gt.float(); occ_vis32 = occ_vis.float()
        occ_gt32 = occ_gt_aligned.float(); occ_vis32 = occ_vis.float()
        # mask_complete = (occ_vis32 == 0).float()
        mask_complete = torch.ones_like(occ_gt32)
        bce = F.binary_cross_entropy_with_logits(logits32, occ_gt32, reduction='none')
        vae_rec = (bce * mask_complete).sum() / (mask_complete.sum() + 1e-6)
        # vae_rec = bce
        vae_kld = self._kld(mu.float(), logvar.float())
        print("vae_rec before loss calculation: ", vae_rec)
        loss = recon_w * vae_rec + 1e-3 * recon_w * vae_kld + 0.5 * L_box
        # loss = recon_w * vae_rec + 1e-3 * recon_w * vae_kld 
        print("vae_rec after loss calculation: ", vae_rec)
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
        "loss": loss * 100,
        "vae_rec": vae_rec.detach().cpu(),
        "vae_kld": vae_kld,
        "L_box": L_box,
        "L_box_mean": L_box_mean,
        "L_box_scale": L_box_scale,
        # "L_seed": L_seed,
        # "L_shrink": L_shrink,
        "scale_pred": scale_pred.mean().detach().item(),
        }
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
        
        viz_B = min(logits.shape[0], 4)
        output_dict = {
            "logits": logits[:viz_B].detach().float().cpu(),   # (B,1,G,G,G)
            "occ_gt": occ_gt_aligned[:viz_B].detach().float().cpu(),   # (B,1,G,G,G)
            "occ_vis": occ_vis[:viz_B].detach().float().cpu(), 
        }
        # # # if flow_logits_viz is not None:
        # # #     # match visualize() key
        
        output_dict["sample_logits"] = None  # (≤2,1,G,G,G)
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
