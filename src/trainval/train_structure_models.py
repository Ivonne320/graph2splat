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
from src.models.structure_model import StructureModel
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
        #         "/home/yihan/graph2splat/pretrained/training_structure_models/2025-09-09_19-26-52_transforming/snapshots/epoch-3000.pth.tar", map_location=self.device
        #     )["model"]
        # )
        # self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        # model.eval()
        
        return model
    
    # TODO: check config; build loss & training pipeline; adapt dataloader; prepare overfitting data; visualization

    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        base = osp.join(root_dir, "files", "gs_annotations", scene_id, "scene_level")
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

    def _choose_frame(self, frames_for_scene):
        # if isinstance(frames_for_scene, dict):
        #     for _, arr in frames_for_scene.items():
        #         if isinstance(arr,(list,tuple)) and len(arr)>0: return str(arr[0])
        # if isinstance(frames_for_scene,(list,tuple)) and len(frames_for_scene)>0:
        #     return str(frames_for_scene[0])
        return "000000"
    
    def _rasterize_idx(self, idx_np: np.ndarray, G: int) -> torch.Tensor:
        """(M,3) -> (1,G,G,G) float {0,1}"""
        occ = np.zeros((G,G,G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:,0], idx_np[:,1], idx_np[:,2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()
        
    # def _make_batch(self, data_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     B = len(data_dict["scene_graphs"]["scene_ids"])
       
    #     frames_all = data_dict["scene_graphs"].get("obj_img_top_frames", {})
    #     scene_ids = data_dict["scene_graphs"]["scene_ids"]

    #     occ_list, cond_list = [], []
    #     for b in range(B):
    #         sid = self._scene_id(scene_ids[b])
    #         frames_for_scene = frames_all.get(sid, [])
    #         fid = self._choose_frame(frames_for_scene)
    #         pack = self._load_aligned_pack('/home/yihan/3RScan_structure_models/', sid, fid)
    #         occ, cond = self._build_occ_cond(pack, self.G, self.use_cond)
    #         occ_list.append(occ)
    #         if self.use_cond: cond_list.append(cond)

    #     occ = torch.cat(occ_list, dim=0).unsqueeze(1).to(self.device)   # (B,1,G,G,G)
    #     cond = None
    #     if self.use_cond:
    #         cond = torch.stack(cond_list, dim=0).to(self.device)         # (B,1024)
    #     return occ, cond
    
    def _make_batch(self, data_dict: Dict[str, Any]):
        B = len(data_dict["scene_graphs"]["scene_ids"])
        frames_all = data_dict["scene_graphs"].get("obj_img_top_frames", {})
        scene_ids = data_dict["scene_graphs"]["scene_ids"]

        occ_gt_list, occ_vis_list, cond_tok_list = [], [], []
        G = self.G
        patch_size = getattr(self.model.flow, "patch_size", 1)

        for b in range(B):
            sid = self._scene_id(scene_ids[b])
            frames_for_scene = frames_all.get(sid, [])
            fid = self._choose_frame(frames_for_scene)
            pack = self._load_aligned_pack('/home/yihan/3RScan_structure_models/', sid, fid)

            occ_gt, occ_vis, cond_tokens = self._build_occ_and_cond_tokens(pack, G, patch_size)
            occ_gt_list.append(occ_gt.unsqueeze(0))
            occ_vis_list.append(occ_vis.unsqueeze(0))
            cond_tok_list.append(cond_tokens)   # already (1,Lq,C)

        occ_gt  = torch.cat(occ_gt_list,  dim=0).to(self.device)   # (B,1,G,G,G)
        occ_vis = torch.cat(occ_vis_list, dim=0).to(self.device)   # (B,1,G,G,G)
        cond    = torch.cat(cond_tok_list, dim=0).to(self.device)  # (B,Lq,Ccond)
        return occ_gt, occ_vis, cond
        
    def _build_flow_cond_tokens_from_pack(self, pack: dict, G: int, patch_size: int,
                                            max_ctx: int = 2048, rand_fill: int = 0):
        """
        Build (1, Lctx, 1024) conditioning tokens for flow **without** GT.
        Assumes pack["feats"] is aligned 1:1 with pack["seed_idx"].
        """
        device = self.device
        feat_dim = 1024
        seeds_np = pack.get("seed_idx", np.zeros((0,3), np.int32))
        feats_np = pack.get("feats", np.zeros((0, feat_dim), np.float32))

        # Handle empty case
        if seeds_np.size == 0 or feats_np.size == 0:
            gp = G // max(1, patch_size)
            Lq = gp * gp * gp
            Lctx = min(max_ctx, Lq)
            return torch.zeros(1, Lctx, feat_dim, device=device, dtype=torch.float16)

        # Sanity: M seeds ↔ M feats
        seeds = torch.as_tensor(seeds_np, dtype=torch.long)      # (M,3) CPU
        feats = torch.as_tensor(feats_np, dtype=torch.float32)   # (M,1024) CPU
        assert seeds.ndim == 2 and seeds.shape[1] == 3, f"seed_idx shape {seeds.shape} must be (M,3)"
        assert feats.ndim == 2 and feats.shape[1] == feat_dim, f"feats shape {feats.shape} must be (M,1024)"
        assert seeds.shape[0] == feats.shape[0], "seed_idx and feats must have same length"

        # Patch binning like training
        gp = G // max(1, patch_size)
        Lq = gp * gp * gp
        pcoords = (seeds // patch_size).clamp(min=0, max=gp-1)   # (M,3) CPU
        patch_id = (pcoords[:,0] * gp * gp + pcoords[:,1] * gp + pcoords[:,2])  # (M,)

        # Aggregate on CPU
        sum_feats = torch.zeros(Lq, feat_dim, dtype=feats.dtype)    # CPU
        sum_feats.index_add_(0, patch_id, feats)
        counts = torch.zeros(Lq, 1, dtype=feats.dtype)              # CPU
        counts.index_add_(0, patch_id, torch.ones_like(patch_id, dtype=feats.dtype).unsqueeze(1))

        non_empty = (counts.squeeze(1) > 0)
        if non_empty.sum() == 0:
            Lctx = min(max_ctx, Lq)
            return torch.zeros(1, Lctx, feat_dim, device=device, dtype=torch.float16)

        sum_feats = sum_feats[non_empty]                            # (Lne,1024)
        counts    = counts[non_empty]                               # (Lne,1)
        cond_tokens = (sum_feats / counts.clamp_min(1)).to(device=self.device, dtype=torch.float16)  # (Lne,1024)

        # Cap to max_ctx, prefer densest bins
        Lne = cond_tokens.shape[0]
        if Lne > max_ctx:
            counts_1d = counts.squeeze(1)   # CPU
            topk = max_ctx - min(rand_fill, max_ctx)
            topk_idx = torch.topk(counts_1d, k=topk, largest=True).indices
            if rand_fill > 0 and (Lne - topk) > 0:
                all_idx = torch.arange(Lne)
                mask = torch.ones(Lne, dtype=torch.bool); mask[topk_idx] = False
                rest = all_idx[mask]
                perm = torch.randperm(rest.numel())[:rand_fill]
                sel_idx = torch.cat([topk_idx, rest[perm]], dim=0)
            else:
                sel_idx = topk_idx
            cond_tokens = cond_tokens[sel_idx.to(cond_tokens.device)]

        return cond_tokens.unsqueeze(0)  # (1,Lctx,1024), fp16

    
    def _build_occ_and_cond_tokens(self, pack, expect_G: int, patch_size: int):
        """Return (occ_gt, occ_vis, cond_tokens)"""
        G = int(pack["G"])
        if G != expect_G:
            raise ValueError(f"G mismatch: expected {expect_G}, got {G}")

        occ_gt  = self._rasterize_idx(pack["vox_idx_gt_occ"], G)                  # (1,G,G,G)
        occ_vis = self._rasterize_idx(pack.get("seed_idx", np.zeros((0,3), np.int32)), G)

        cond_tokens = self._build_flow_cond_tokens_from_pack(pack, G, patch_size) # (1,Lq,Ccond)
        return occ_gt, occ_vis, cond_tokens

    def _kld(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std



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
        # with torch.no_grad():
        # occ_1g, cond = self._make_batch(data_dict)
        occ_gt, occ_vis, cond_tokens = self._make_batch(data_dict)  # shapes: (B,1,G,G,G), (B,1,G,G,G), (B,Lq,Ccond)
        
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
             # --- VAE using TRELLIS enc/dec ---
            # mu, logvar = self.model.encoder(occ_1g)         # (B,zc,8,8,8) each
            # z, mu, logvar = self.model.encoder(
            #     occ_1g, sample_posterior=False, return_raw=True
            # )
            
            z, mu, logvar = self.model.encoder(occ_vis, sample_posterior=False, return_raw=True)
            z_data = self._reparameterize(mu, logvar)       # (B,zc,8,8,8)
            logits = self.model.decoder(z_data)             # (B,1,G,G,G)
            
        logits32 = logits.float(); occ_gt32 = occ_gt.float(); occ_vis32 = occ_vis.float()

        mask_complete = (occ_vis32 == 0).float()
        bce = F.binary_cross_entropy_with_logits(logits32, occ_gt32, reduction='none')
        # vae_rec = F.binary_cross_entropy_with_logits(logits, occ_1g)
        vae_rec = (bce * mask_complete).sum() / (mask_complete.sum() + 1e-6)
        vae_kld = self._kld(mu.float(), logvar.float())
            
            # --- Rectified Flow matching on z (straight-line path) ---
        B = z_data.size(0)
        t = torch.rand(B, device=self.device)
        eps = torch.randn_like(z_data)
        zt = (1 - t)[:,None,None,None,None] * z_data + t[:,None,None,None,None] * eps
        # v_target = (z_data - eps)
        v_target = (z_data - eps).detach()
        cond_tokens = cond_tokens.to(device=self.device, dtype=torch.float16)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            v_pred = self.model.flow(zt, t, cond_tokens)
        flow_loss = F.mse_loss(v_pred.float(), v_target.float())

        loss = (
            1 * vae_rec
            + 0.001 * vae_kld
            + 1 * flow_loss
        )

        loss_dict = {
            "loss": loss * 100,
            "vae_rec": vae_rec,
            "vae_kld": vae_kld,
            "flow_loss": flow_loss,
        }
        
        
        viz_B = min(logits.shape[0], 4)
        output_dict = {
            "logits": logits[:viz_B].detach().float().cpu(),   # (B,1,G,G,G)
            "occ_gt": occ_gt[:viz_B].detach().float().cpu(),   # (B,1,G,G,G)
            "occ_vis": occ_vis[:viz_B].detach().float().cpu(), 
        }
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

        def _centers_from_idx(idx: torch.Tensor, G: int) -> np.ndarray:
            if idx.numel() == 0:
                return np.zeros((0,3), dtype=np.float32)
            c = (idx.float() + 0.5) / float(G) - 0.5
            return c.cpu().numpy().astype(np.float32)

        def _write_ply_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {xyz.shape[0]}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(xyz, rgb.astype(np.uint8)):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

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
