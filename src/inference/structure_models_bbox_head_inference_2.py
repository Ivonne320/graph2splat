import argparse
import logging
import os
import os.path as osp

import numpy as np
import time
import torch

from configs import Config, update_configs
from src.datasets import Scan3RPatchObjectModifiedDataset
from utils import common, torch_util
import copy
# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SCRATCH = os.environ.get("SCRATCH", "/scratch")

import numpy as np
import torch
import random
import cv2
import matplotlib.cm as cm
import open3d as o3d
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

from configs import Config, update_configs
from src.datasets.loaders import get_train_val_data_loader, get_val_dataloader
# from src.models.structure_model import StructureModel
from src.models.structure_model_with_bbox_head import StructureModel
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r
from utils.gaussian_splatting import GaussianSplat
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getProjectionMatrix
from preprocessing.voxel_anno.voxelise_features_scene_inference_structure_models import _dilate_voxels_from_idx, _prep_image_for_dino, _project_and_sample_dino, _save_featured_voxel, unproject_frame_to_world, _voxelize_points_normed
from preprocessing.voxel_anno.voxelise_features_scene_final import _dilate_voxels_idx
from utils.visualisation import save_vox_as_ply, side_by_side, slice_mosaic, save_voxel_as_ply
from utils.canonicalize import align_gravity_with_plane, yaw_canonicalize_xy
import torch.nn.functional as F
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_LOGGER = logging.getLogger(__name__)

def _write_points_and_edges_ply(path, verts_xyz, verts_rgb, edges_v1v2, edges_rgb):
    Nv, Ne = verts_xyz.shape[0], edges_v1v2.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {Nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {Ne}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x,y,z),(r,g,b) in zip(verts_xyz, verts_rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        for (i,j),(r,g,b) in zip(edges_v1v2, edges_rgb):
            f.write(f"{int(i)} {int(j)} {int(r)} {int(g)} {int(b)}\n")

def _accum_bbox_lines_world(verts_xyz, verts_rgb, edges_v1v2, edges_rgb,
                            mu_can, s_can, R_can, color_rgb):
    signs = np.array([[-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
                      [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]], np.float32)
    corners_can = mu_can[None, :] + float(s_can) * signs           # (8,3)
    # back to world: X_world = R^T · X_can    (no extra translation)
    corners_w = (R_can.T @ corners_can.T).T

    base = verts_xyz.shape[0]
    verts_xyz = np.vstack([verts_xyz, corners_w])
    verts_rgb = np.vstack([verts_rgb, np.tile(np.array(color_rgb, np.uint8), (8, 1))])

    EDGES = np.array([[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]], dtype=np.int32)
    edges_v1v2 = np.vstack([edges_v1v2, EDGES + base])
    edges_rgb  = np.vstack([edges_rgb,  np.tile(np.array(color_rgb, np.uint8), (len(EDGES), 1))])
    return verts_xyz, verts_rgb, edges_v1v2, edges_rgb

class StructureCompletionPipeline:
    def __init__(self, cfg: Config, visualize=False, split="train"):
        self.cfg = cfg
        self.cfg.data.preload_slat = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.G = 64
        self.use_cond = True
        # self.dataset = Scan3RPatchObjectModifiedDataset(cfg, split)
        self.vis = visualize
        self.sample_features = True
        self.model_dir = '/mnt/hdd4tb/3RScan'
        self.output_dir = osp.join(cfg.data.root_dir, cfg.inference.output_dir)
        self.dino_model, self.dino_transform = self._build_dino()
        
    def _build_dino(self):
        # load once
        dino = torch.hub.load(
            "/home/yihan/.cache/torch/hub/facebookresearch_dinov3_main",
            'dinov3_vitl16', source='local', pretrained=False
        )
        ckpt_path = "/home/yihan/.cache/torch/hub/facebookresearch_dinov3_main/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        dino.load_state_dict(state_dict, strict=False)
        dino.eval().to(self.device)

        tfm = transforms.Compose(
            [
                transforms.Resize((518, 518)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return dino, tfm
    
    @torch.no_grad()
    def _get_dino_embedding(self, images: torch.Tensor, model, transform, device):
        """
        images: (B,3,H,W) in [0,1]
        returns:
        emb:   (B, C, H_p, W_p)
        sizes: (H_in, W_in, H_p, W_p)
        """
        # preprocess with your transform
        imgs = images.reshape(-1, 3, images.shape[-2], images.shape[-1]).cpu()
        inp  = transform(imgs).cuda()         # (B,3,H_in,W_in)

        model.eval()
        out = model(inp, is_training=True)    # dict with patch tokens

        # 1) take patch tokens directly (B, N, C)
        if "x_norm_patchtokens" not in out:
            # fallback if needed
            reg = getattr(model, "num_register_tokens", 0)
            tok = out["x_prenorm"][:, 1 + reg : ]         # (B, N, C)
        else:
            tok = out["x_norm_patchtokens"]               # (B, N, C)

        # 2) compute patch grid from actual input + patch size
        H_in, W_in = int(inp.shape[-2]), int(inp.shape[-1])
        p = getattr(model, "patch_size", None)
        if p is None and hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
            p = model.patch_embed.patch_size
        p = int(p[0] if isinstance(p, (tuple, list)) else p)
        H_p, W_p = H_in // p, W_in // p
        assert tok.shape[1] == H_p * W_p, f"N={tok.shape[1]} != {H_p*W_p} (H_in={H_in}, W_in={W_in}, p={p})"

        # 3) (B,N,C) -> (B,C,H_p,W_p)
        emb = tok.permute(0, 2, 1).contiguous().view(tok.size(0), tok.size(2), H_p, W_p)
        return emb, (H_in, W_in, H_p, W_p)

    def load_model(self):
        model = StructureModel(cfg=self.cfg.autoencoder, device=self.device)
        # model.load_state_dict(torch.load('/mnt/hdd4tb/trainings/training_structural_model_less_lr/2025-09-30_12-10-45/snapshots/epoch-22000.pth.tar',  map_location=self.device)["model"])
        model.load_state_dict(torch.load('/mnt/hdd4tb/trainings/training_structure_model_uncan/2025-10-05_21-12-20/snapshots/epoch-22000.pth.tar',  map_location=self.device)["model"])       
        model.eval()
        return model
    
    def _rasterize_idx(self, idx_np: np.ndarray, G: int) -> torch.Tensor:
        """(M,3) -> (1,G,G,G) float {0,1}"""
        occ = np.zeros((G,G,G), dtype=np.uint8)
        if idx_np.size:
            occ[idx_np[:,0], idx_np[:,1], idx_np[:,2]] = 1
        return torch.from_numpy(occ).unsqueeze(0).float()
    
    def _idx_to_world(self, idx_xyz: torch.Tensor, G: int, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        idx_xyz: (N,3) int/long voxel indices in {0..G-1}
        mean, scale: (3,) np arrays from pack; world = ((idx+0.5)/G - 0.5) * (2*scale) + mean
        Returns: (N,3) world coords (np.float32)
        """
        idx = idx_xyz.detach().cpu().float().numpy()  # (N,3)
        centers_normed = (idx + 0.5) / G - 0.5
        world = centers_normed * (2.0 * scale[None, :]) + mean[None, :]
        return world.astype(np.float32) 
    
    def _world_to_cam(self, Xw: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
        """
        Xw: (N,3), T_cw: (4,4) camera<-world
        Returns Xc: (N,3)
        """
        N = Xw.shape[0]
        homog = np.concatenate([Xw, np.ones((N,1), dtype=np.float32)], axis=1)  # (N,4)
        Xc_h = (T_cw @ homog.T).T  # (N,4)
        return Xc_h[:, :3]   
    
    def _project(self, K: np.ndarray, Xc: np.ndarray, img_hw: tuple) -> tuple:
        """
        K: (3,3), Xc: (N,3) camera coords, img_hw=(H,W).
        Returns: uvs (M,2) int, depth (M,), mask (M,) valid
        """
        x, y, z = Xc[:,0], Xc[:,1], Xc[:,2]
        valid = z > 1e-6
        fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
        u = (fx * (x/z) + cx).astype(np.float32)
        v = (fy * (y/z) + cy).astype(np.float32)
        H, W = img_hw
        valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
        d = z[valid].astype(np.float32)
        return np.stack([u, v], axis=1), d, valid
    def _paint_points(self, rgb: np.ndarray, uvd: tuple, color=(0,255,0), radius=1) -> np.ndarray:
        """
        Paint points colored by depth (closer = warmer, farther = cooler).
        uvd: (uvs, depth) from _project.
        """
        uvs, depth = uvd
        H, W = rgb.shape[:2]
        zbuf = np.full((H, W), np.inf, dtype=np.float32)
        out = rgb.copy()

        # Normalize depths to [0,1] for colormap
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / max(d_max - d_min, 1e-6)

        # Map to RGB colors
        cmap = cm.get_cmap('jet')
        colors = (cmap(depth_norm)[:, :3] * 255).astype(np.uint8)  # (N,3)

        for (u, v), z, c in zip(uvs, depth, colors):
            if z < zbuf[v, u]:
                zbuf[v, u] = z
                cv2.circle(out, (int(u), int(v)), radius, tuple(int(x) for x in c.tolist()), thickness=-1)

        return out
    # ---------- Canonicalization from SEED ONLY ----------

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

    # ---------- Index remap between two canonical AABBs ----------
    def _remap_idx_between_boxes(self, idx_src, G, mu_src, s_src, mu_dst, s_dst):
        """
        idx_src: (M,3) int indices under (mu_src, s_src). Return (M,3) indices under (mu_dst, s_dst).
        All in the SAME canonical orientation.
        """
        idx_src = np.asarray(idx_src, np.int32)
        x0 = (idx_src.astype(np.float32) + 0.5) / G - 0.5
        X  = x0 * (2.0 * float(s_src)) + np.asarray(mu_src, np.float32)
        x1 = (X - np.asarray(mu_dst, np.float32)) / (2.0 * float(s_dst))
        i1 = np.floor((x1 + 0.5) * G).astype(np.int32)
        return np.clip(i1, 0, G-1)

    # ---------- Feature scatter: mean per voxel (compressed features) ----------
    def _scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int):
        """
        idx_t:  (M,3) int on device
        feat_t: (M,C) float on device
        returns: grid_feats (1,C,G,G,G) and seed_occ (1,1,G,G,G)
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

        Csum.index_add_(1, lin, feat_t.T)
        cnt.index_add_(0, lin, torch.ones(M, device=feat_t.device, dtype=feat_t.dtype))

        mask = cnt > 0
        Csum[:, mask] = Csum[:, mask] / cnt[mask]

        grid_feats = Csum.view(C, G, G, G).unsqueeze(0)        # (1,C,G,G,G)
        seed_occ = torch.zeros(1,1,G,G,G, device=feat_t.device, dtype=feat_t.dtype)
        uniq = torch.unique(lin)
        seed_occ.view(1,1,-1)[0,0,uniq] = 1.0
        return grid_feats, seed_occ
    
    def _bbox_iou_can(self, mu1, s1, mu2, s2):
        """
        3D IoU of two axis-aligned cubes in canonical frame (units = meters in your rotated world).
        Each cube is [mu - s, mu + s] isotropically.
        """
        a0 = np.asarray(mu1, np.float32) - float(s1)
        a1 = np.asarray(mu1, np.float32) + float(s1)
        b0 = np.asarray(mu2, np.float32) - float(s2)
        b1 = np.asarray(mu2, np.float32) + float(s2)

        inter_len = np.maximum(0.0, np.minimum(a1, b1) - np.maximum(a0, b0))
        inter = float(np.prod(inter_len))
        vol1 = float((2.0 * float(s1)) ** 3)
        vol2 = float((2.0 * float(s2)) ** 3)
        iou = inter / (vol1 + vol2 - inter + 1e-8)
        return iou

    def _voxel_iou_from_indices(self, idx_pred_np, idx_gt_np):
        """
        idx_pred_np / idx_gt_np: (K,3) int arrays in the SAME canonical.
        Returns IoU, Precision, Recall and TP/FP/FN.
        """
        if idx_pred_np.size == 0 and idx_gt_np.size == 0:
            return dict(iou=1.0, precision=1.0, recall=1.0, tp=0, fp=0, fn=0)
        A = set(map(tuple, np.asarray(idx_pred_np, np.int32)))
        B = set(map(tuple, np.asarray(idx_gt_np,   np.int32)))
        TP = len(A & B); FP = len(A - B); FN = len(B - A)
        union = TP + FP + FN
        iou = TP / (union + 1e-8)
        prec = TP / (TP + FP + 1e-8)
        rec  = TP / (TP + FN + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec)
        return dict(iou=float(iou), precision=float(prec), recall=float(rec), f1 = float(f1),
                    tp=int(TP), fp=int(FP), fn=int(FN))

    def inference(self, idx):
        """
        Two-pass refinement:
        Pass A: seed prior -> bbox head predicts (mu_hat, s_hat)
        Pass B: re-index seeds to (mu_hat, s_hat) -> final logits
        """
        root_dir = self.cfg.data.root_dir
        scenes_dir = osp.join(root_dir, "scenes")
        frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=idx)
        if len(frame_idxs)>50:
            frame_idxs = frame_idxs[:50]
        # fid = []
        frame_idxs = ['000000']
        # --- Load calib & image ---
        for fid in frame_idxs:
            extrinsics = scan3r.load_frame_poses(data_dir=root_dir, scan_id=idx, frame_idxs=frame_idxs)
            K_rgb  = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=idx)["intrinsic_mat"]
            K_depth= scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=idx, type='depth')["intrinsic_mat"]
            rgb_path = f"{root_dir}/scenes/{idx}/sequence/frame-{fid}.color.jpg"
            with Image.open(rgb_path) as img:
                rgb_ref = np.asarray(img.convert("RGB"), dtype=np.uint8)
                
            # --- 1) Lift points from one frame (seed) ---
            Wp_world = unproject_frame_to_world(fid, extrinsics, K_rgb, K_depth, root_dir, idx)

            Rg = align_gravity_with_plane(Wp_world)
            Wp_g = (Rg @ Wp_world.T).T
            Rz = yaw_canonicalize_xy(Wp_g)
            R = Rz @ Rg
            # ------not doing canonicalization
            R =  np.eye(3, dtype=np.float32)
            
            Wp_can = (R @ Wp_world.T).T
            
            # --- 3) Seed prior box (mu0, s0), index seeds (Pass A input) ---
            G = self.G
            # PAD_VOX_SEED = 3
            mu0 = Wp_can.mean(0).astype(np.float32)
            base = float(np.max(np.abs(Wp_can - mu0)))
            base = max(base, 1e-6)
            # s0 = base / (1.0 - 2.0*PAD_VOX_SEED/float(G))  # isotropic scalar

            # x_norm_seed = (Wp_can - mu0[None,:]) / (2.0 * s0)
            x_norm_seed = (Wp_can - mu0[None,:]) / (2.0 * base)
            idx_seed_0 = np.floor((x_norm_seed + 0.5) * G).astype(np.int32)
            # idx_seed_0 = np.clip(idx_seed_0, 0, G-1)
            idx_seed_0 = np.unique(np.clip(idx_seed_0, 0, G-1), axis=0)
            # idx_seed_0 = _dilate_voxels_from_idx(idx_seed_0, grid_size=G)
            idx_seed_0 = _dilate_voxels_idx(idx_seed_0)
                    
            # --- 4) DINO features at seed voxels: sample ONCE (tokens attached to points) ---
            centers_normed_0 = (idx_seed_0.astype(np.float32) + 0.5)/G - 0.5
            # centers_can_0    = centers_normed_0 * (2.0*s0) + mu0[None,:]
            centers_can_0    = centers_normed_0 * (2.0*base) + mu0[None,:]
            centers_world_0  = (R.T @ centers_can_0.T).T

            frame_tokens, (H_in,W_in,H_p,W_p) = self._get_dino_embedding(
                _prep_image_for_dino(rgb_ref), self.dino_model, self.dino_transform, self.device
            )
            T_wc = np.linalg.inv(extrinsics[fid]).astype(np.float32)
            idx_keep, tokens = _project_and_sample_dino(
                voxel_world=torch.from_numpy(centers_world_0).float(),
                T_wc=T_wc, K=K_rgb, img_hw=rgb_ref.shape[:2], dino_tokens=frame_tokens
            )
            idx_kept_0 = idx_seed_0[idx_keep]
            feats_0 = tokens.astype(np.float32)                # (M,1024)
            feats_t = torch.from_numpy(feats_0).to(self.device).float()
        
            feats_comp_list, idx_t_list = [], []
            # --- compress + scatter (Pass A input) ---
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                feats_comp = self.model.comp(feats_t)          # (M,C_out)
            feats_comp_list.append(feats_comp)
            idx_t_list.append(torch.from_numpy(idx_kept_0).to(self.device).int())
            mean_seed0_t  = torch.from_numpy(mu0).to(self.device).float().view(1, 3)   # (1,3)
            scale_seed0_t = torch.tensor(base, device=self.device, dtype=torch.float32).view(1, 1)  # (1,1)
            mean_pred, scale_pred = self.model.forward_bbox_from_seeds_batch(
                feats_comp_list=feats_comp_list,   # list[(Mi,64)]
                idx_list=idx_t_list,               # list[(Mi,3)]
                G=self.G,
                mean_seed0=mean_seed0_t,             # (B,3)
                scale_seed0=scale_seed0_t            # (B,1)
            )
            B = 1
            mean_dst, scale_dst = mean_pred.detach(), scale_pred.detach().view(B,1)
            idx_dst = self._remap_seed_idx_with_bbox(
                    seed_idx=idx_t_list[0],     # (Mi,3) in seed canonical
                    mean_src=mean_seed0_t, scale_src=scale_seed0_t,
                    mean_dst=mean_dst, scale_dst=scale_dst,
                    b=0, G=self.G
                )
            
            grid_dino_0, seed_occ_0 = self._scatter_voxel_mean(
                idx_dst.int(), feats_comp.float(), G
            )
            x_in_0 = torch.cat([seed_occ_0, grid_dino_0], dim=1)   # (1,1+C_out,G,G,G)
            
            # --- 5) Pass A → bbox prediction ---
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                z_A, mu_A, logvar_A, feat3d_A = self.model.encoder(x_in_0, sample_posterior=False, return_raw=True, return_feat=True)
                logits_A = self.model.decoder(z_A)
            mu_pred_np = mean_pred[0].detach().cpu().numpy().astype(np.float32)
            s_pred_sc  = float(scale_pred[0].detach().cpu().item())
            pr_idx = (logits_A[0,0].sigmoid() > 0.5).nonzero(as_tuple=False).cpu().numpy().astype(np.int32)
            x_norm = (pr_idx.astype(np.float32) + 0.5)/G - 0.5
            X_can  = x_norm * (2.0*s_pred_sc) + mu_pred_np[None,:]
            X_world = (R.T @ X_can.T).T
            
            using_gt_mesh = False
            if using_gt_mesh:
                scenes_dir = os.path.join(self.cfg.data.root_dir, "scenes")
                mesh = scan3r.load_ply_mesh(data_dir=scenes_dir, scan_id=idx,
                                            label_file_name="labels.instances.annotated.v2.ply")
                annos = scan3r.load_ply_data(data_dir=scenes_dir, scan_id=idx,
                                            label_file_name="labels.instances.annotated.v2.ply")["vertex"]["objectId"]
                objects_info_file = osp.join(self.cfg.data.root_dir, "files", "objects.json")
                all_obj_info = common.load_json(objects_info_file)
                obj_data = next(
                    obj_data
                    for obj_data in all_obj_info["scans"]
                    if obj_data["scan"] == idx
                )
                object_ids = [int(obj["id"]) for obj in obj_data["objects"]]

                mask = np.isin(annos, object_ids)
                V_world_gt = np.asarray(mesh.vertices, np.float32)[mask]          # (N,3)
                V_can_gt   = (R @ V_world_gt.T).T                                  # rotate to canonical

                mu_gt = V_can_gt.mean(0).astype(np.float32)
                s_gt  = float(max(np.max(np.abs(V_can_gt - mu_gt[None,:])), 1e-6))

                # voxelize GT in its own (mu_gt, s_gt) box
                x_norm_gt = (V_can_gt - mu_gt[None,:]) / (2.0 * s_gt)
                idx_mesh  = np.floor((x_norm_gt + 0.5) * G).astype(np.int32)
                idx_mesh  = np.unique(np.clip(idx_mesh, 0, G-1), axis=0)
                idx_mesh  = _dilate_voxels_idx(idx_mesh)                            # keep same dilation as seeds

                # remap GT voxels → predicted canonical
                gt_idx_pred = self._remap_idx_between_boxes(
                    idx_mesh, G, mu_src=mu_gt, s_src=s_gt, mu_dst=mu_pred_np, s_dst=s_pred_sc
                )
                # mu_gt = V_world_gt.mean(0).astype(np.float32)
                x_norm = (pr_idx.astype(np.float32) + 0.5)/G - 0.5
                X_can  = x_norm * (2.0*s_pred_sc) + mu_pred_np[None,:]
                X_world = (R.T @ X_can.T).T                    # canonical -> world  ✅

                # Build a WORLD box (mu_world, s_world), e.g. from GT vertices in **world**
                mu_world = V_world_gt.mean(0).astype(np.float32)
                s_world  = float(max(np.max(np.abs(V_world_gt - mu_world[None,:])), 1e-6))

                # Re-voxelize **X_world** into the WORLD box to get WORLD indices
                x_norm_world = (X_world - mu_world[None,:]) / (2.0 * s_world)
                # pr_idx_world, _ = _voxelize_points_normed(x_norm_world, grid_size=G)
                pr_idx_world = np.floor((x_norm_world + 0.5) * G).astype(np.int32)
                pr_idx_world = np.clip(pr_idx_world, 0, G-1)


                                
            if self.vis:
                self._visualize_world_completion(idx, X_world, pr_idx, R, mu_pred_np, s_pred_sc)
            if self.sample_features:
                self._save_completed_features(idx, fid, X_world, pr_idx, extrinsics, K_rgb, mu_pred_np, s_pred_sc, R)
                # self._save_completed_features(idx, fid, X_world, pr_idx_world, extrinsics, K_rgb, mu_world, s_world, R)
                # self._save_completed_features(idx, fid, V_world_gt,gt_idx_pred, extrinsics, K_rgb, mu_pred_np, s_pred_sc, R)
            print(f'inference done for {fid}')
            
    @torch.no_grad()
    def _visualize_world_completion(self, scan_id, X_world, pr_idx, R, mu_pred, s_pred):
        outdir = "vis/structure_inference_bbox_head_2"; os.makedirs(outdir, exist_ok=True)
        G = self.G
        # save_vox_as_ply(torch.from_numpy(pr_idx), G, f"{outdir}/pred_struct_completion.ply")
        # 1) completion points as base vertices (gray)
        verts_xyz = X_world.astype(np.float32)
        verts_rgb = np.full((verts_xyz.shape[0],3), 200, np.uint8)
        edges_v1v2 = np.zeros((0,2), np.int32)
        edges_rgb  = np.zeros((0,3), np.uint8)

        # 2) GT bbox (compute from GT mesh in SAME canonical as R)
        scenes_dir = os.path.join(self.cfg.data.root_dir, "scenes")
        mesh = scan3r.load_ply_mesh(data_dir=scenes_dir, scan_id=scan_id,
                                    label_file_name="labels.instances.annotated.v2.ply")
        annos = scan3r.load_ply_data(data_dir=scenes_dir, scan_id=scan_id,
                                    label_file_name="labels.instances.annotated.v2.ply")["vertex"]["objectId"]
        objects_info_file = osp.join(self.cfg.data.root_dir, "files", "objects.json")
        all_obj_info = common.load_json(objects_info_file)
        obj_data = next(
            obj_data
            for obj_data in all_obj_info["scans"]
            if obj_data["scan"] == scan_id
        )
        object_ids = [int(obj["id"]) for obj in obj_data["objects"]]
        
        
        mask = np.isin(annos, object_ids)
        V_world_gt = np.asarray(mesh.vertices, np.float32)[mask]
        V_can_gt   = (R @ V_world_gt.T).T
        mu_gt = V_can_gt.mean(0).astype(np.float32)
        s_gt  = float(np.max(np.abs(V_can_gt - mu_gt[None,:])))
        x_norm_mesh = (V_can_gt - mu_gt[None, :]) / (2.0 * s_gt + 1e-12)  # center by mu_gt
        idx_mesh = np.floor((x_norm_mesh + 0.5) * G).astype(np.int32)
        idx_mesh = np.unique(idx_mesh, axis=0)
        idx_mesh = torch.from_numpy(idx_mesh)
        idx_mesh = torch.from_numpy(_dilate_voxels_idx(idx_mesh))
        # save_vox_as_ply(idx_mesh, G, f"{outdir}/world_gt.ply")
        # 3) Append predicted bbox (red) and GT bbox (green)
        verts_xyz, verts_rgb, edges_v1v2, edges_rgb = _accum_bbox_lines_world(
            verts_xyz, verts_rgb, edges_v1v2, edges_rgb, mu_pred, s_pred, R, (255,0,0)
        )
        verts_xyz, verts_rgb, edges_v1v2, edges_rgb = _accum_bbox_lines_world(
            verts_xyz, verts_rgb, edges_v1v2, edges_rgb, mu_gt, s_gt, R, (0,255,0)
        )
        # 4) One file: points + colored edges
        out_ply = f"{outdir}/pred_struct_completion_with_bbox.ply"
        _write_points_and_edges_ply(out_ply, verts_xyz, verts_rgb, edges_v1v2, edges_rgb)
        
        print(f"[vis] Saved completion + boxes (pred:red, gt:green) → {out_ply}")
        # Paint onto a couple frames
        root_dir = self.cfg.data.root_dir
        target_fids = ["000003","000010"]
        for target_fid in target_fids:
            extrinsics_all = scan3r.load_frame_poses(data_dir=root_dir, scan_id=scan_id, frame_idxs=[target_fid])
            T_wc = extrinsics_all[target_fid]; T_cw = np.linalg.inv(T_wc).astype(np.float32)
            K = scan3r.load_intrinsics(data_dir=osp.join(root_dir, "scenes"), scan_id=scan_id)["intrinsic_mat"]
            rgb_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{target_fid}.color.jpg"
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            Xc = self._world_to_cam(X_world, T_cw)
            uv, depth, valid_mask = self._project(K, Xc, rgb.shape[:2])
            overlay = self._paint_points(rgb, (uv, depth), radius=3)
            out_path = f"{outdir}/reproj_{scan_id}_{target_fid}.png"
            Image.fromarray(overlay).save(out_path)
            print(f"[vis] Saved overlay → {out_path}")

    @torch.no_grad()
    def _save_completed_features(self, scan_id, fid, X_world, pr_idx, extrinsics, K_rgb, mu_pred, s_pred, R_can):
        root_dir = self.cfg.data.root_dir
        scenes_dir = osp.join(root_dir, "scenes")
        # DINO sampling on completed voxels
        ref_rgb_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{fid}.color.jpg"
        with Image.open(ref_rgb_path) as img:
            ref_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        frame_tokens, _ = self._get_dino_embedding(_prep_image_for_dino(ref_rgb),
                                                self.dino_model, self.dino_transform, self.device)
        T_wc = np.linalg.inv(extrinsics[fid]).astype(np.float32)
        idx_keep, tokens = _project_and_sample_dino(
            voxel_world=torch.from_numpy(X_world).float(),
            T_wc=T_wc, K=K_rgb, img_hw=ref_rgb.shape[:2], dino_tokens=frame_tokens
        )
        feat_dim = 1024
        pts_normed = (X_world - mu_pred) / (2.0*s_pred)
        vox_idx, _ = _voxelize_points_normed(pts_normed, grid_size=64)
        M = pr_idx.shape[0]
        # M = vox_idx.shape[0]
        feats_full = np.zeros((M, feat_dim), dtype=np.float32)
        feats_full[idx_keep] = tokens.astype(np.float32)

        # Save npz
        scene_output_dir = osp.join(self.model_dir, "files", "gs_annotations", scan_id, "scene_level_single_frame_uncan", fid)
        os.makedirs(scene_output_dir, exist_ok=True)
        voxel_path = osp.join(scene_output_dir, "voxel_output_dense.npz")
        payload_full = np.concatenate([pr_idx.astype(np.float32), feats_full], axis=1)
        # payload_full = np.concatenate([vox_idx.astype(np.float32), feats_full], axis=1)
        # _save_featured_voxel(torch.from_numpy(payload_full), output_file=voxel_path)
        np.savez(
            voxel_path,
            arr_0=payload_full,
            # R_can=R_can
        )
        mean_scale_path = osp.join(scene_output_dir, "mean_scale_dense.npz")
        np.savez(mean_scale_path, mean=mu_pred, scale=s_pred)
        # Optional PLY
        save_voxel_as_ply(payload_full, "vis/structure_inference_bbox_head_2/completed_featured_voxel_world.ply", show_color=True)
    
    
    @torch.no_grad()
    def _predict_one(self, scan_id, fid=None, thr=0.5, mesh=None, annos=None, obj_data=None):
        """
        Returns:
        mu_pred (3,), s_pred (float),
        pr_idx (N,3) predicted voxel indices in PREDICTED canonical,
        mu_gt (3,), s_gt (float), gt_idx_pred (K,3) GT voxels remapped into PREDICTED canonical,
        R (3,3) used rotation (in case you want world-space).
        """
        root_dir = self.cfg.data.root_dir
        scenes_dir = osp.join(root_dir, "scenes")
        frame_idxs = scan3r.load_frame_idxs(data_dir=scenes_dir, scan_id=scan_id)
        if fid is None:
            fid = frame_idxs[0]

        # --- calib & image ---
        extrinsics = scan3r.load_frame_poses(data_dir=root_dir, scan_id=scan_id, frame_idxs=frame_idxs)
        K_rgb  = scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id)["intrinsic_mat"]
        K_depth= scan3r.load_intrinsics(data_dir=scenes_dir, scan_id=scan_id, type='depth')["intrinsic_mat"]
        rgb_path = f"{root_dir}/scenes/{scan_id}/sequence/frame-{fid}.color.jpg"
        with Image.open(rgb_path) as img:
            rgb_ref = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # --- seed lift & canonical rotation from seeds ---
        t0 = time.time()
        Wp_world = unproject_frame_to_world(fid, extrinsics, K_rgb, K_depth, root_dir, scan_id)
        Rg = align_gravity_with_plane(Wp_world)
        Wp_g = (Rg @ Wp_world.T).T
        Rz = yaw_canonicalize_xy(Wp_g)
        # print('canonicalization takes', time.time()-t0);t0 = time.time()
        
        R  = Rz @ Rg
        Wp_can = (R @ Wp_world.T).T

        # --- seed prior (mu0, s0=base) & seed voxelization ---
        G = self.G
        mu0  = Wp_can.mean(0).astype(np.float32)
        base = float(np.max(np.abs(Wp_can - mu0))); base = max(base, 1e-6)
        x_norm_seed = (Wp_can - mu0[None,:]) / (2.0 * base)
        idx_seed_0  = np.floor((x_norm_seed + 0.5) * G).astype(np.int32)
        # idx_seed_0  = np.clip(idx_seed_0, 0, G-1)
        idx_seed_0 = np.unique(np.clip(idx_seed_0, 0, G-1),axis=0)
        idx_seed_0  = _dilate_voxels_idx(idx_seed_0)
        # print('dilation idx seed takes', time.time()-t0); t0 = time.time()

        # --- DINO features at seed voxels (single frame) ---
        centers_normed_0 = (idx_seed_0.astype(np.float32) + 0.5) / G - 0.5
        centers_can_0    = centers_normed_0 * (2.0 * base) + mu0[None,:]
        centers_world_0  = (R.T @ centers_can_0.T).T

        frame_tokens, _ = self._get_dino_embedding(_prep_image_for_dino(rgb_ref),
                                                self.dino_model, self.dino_transform, self.device)
        T_wc = np.linalg.inv(extrinsics[fid]).astype(np.float32)
        idx_keep, tokens = _project_and_sample_dino(
            voxel_world=torch.from_numpy(centers_world_0).float(),
            T_wc=T_wc, K=K_rgb, img_hw=rgb_ref.shape[:2], dino_tokens=frame_tokens
        )
        idx_kept_0 = idx_seed_0[idx_keep]
        feats_t = torch.from_numpy(tokens.astype(np.float32)).to(self.device).float()
        # print('dino sampling takes', time.time()-t0); t0 = time.time()
        # --- compress & bbox predict (batch size 1) ---
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            feats_comp = self.model.comp(feats_t)  # (M,C_out)

        feats_comp_list = [feats_comp]
        idx_t_list      = [torch.from_numpy(idx_kept_0).to(self.device).long()]
        mean_seed0_t  = torch.from_numpy(mu0).to(self.device).float().view(1,3)
        scale_seed0_t = torch.tensor(base, device=self.device, dtype=torch.float32).view(1,1)

        mean_pred, scale_pred = self.model.forward_bbox_from_seeds_batch(
            feats_comp_list=feats_comp_list,
            idx_list=idx_t_list,
            G=G,
            mean_seed0=mean_seed0_t,
            scale_seed0=scale_seed0_t
        )
        mu_pred = mean_pred[0].detach().cpu().numpy().astype(np.float32)
        s_pred  = float(scale_pred[0].detach().cpu().item())
        # print('bbox head takes', time.time()-t0); t0 = time.time()
        # --- scatter seeds remapped to predicted canonical & decode logits ---
        idx_dst = self._remap_seed_idx_with_bbox(
            seed_idx=idx_t_list[0],
            mean_src=mean_seed0_t, scale_src=scale_seed0_t,
            mean_dst=mean_pred,   scale_dst=scale_pred.view(1,1),
            b=0, G=G
        )
        grid_dino, seed_occ = self._scatter_voxel_mean(idx_dst.int(), feats_comp.float(), G)
        # print('remapping and scatter takes', time.time()-t0); t0 = time.time()
        x_in = torch.cat([seed_occ, grid_dino], dim=1)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            z, mu, logvar, feat3d = self.model.encoder(x_in, sample_posterior=False, return_raw=True, return_feat=True)
            logits = self.model.decoder(z)  # (1,1,G,G,G)

        # --- predicted voxels (pred canonical) ---
        pr = (logits[0,0].float().sigmoid().cpu().numpy() > float(thr))
        pr_idx = np.stack(np.nonzero(pr), axis=1).astype(np.int32)  # (N,3)

        # --- GT: build mesh-canonical box/vox, then remap GT→pred canonical for fair IoU ---
        scenes_dir = os.path.join(self.cfg.data.root_dir, "scenes")
        # mesh = scan3r.load_ply_mesh(data_dir=scenes_dir, scan_id=scan_id,
        #                             label_file_name="labels.instances.annotated.v2.ply")
        # annos = scan3r.load_ply_data(data_dir=scenes_dir, scan_id=scan_id,
        #                             label_file_name="labels.instances.annotated.v2.ply")["vertex"]["objectId"]
        # objects_info_file = osp.join(self.cfg.data.root_dir, "files", "objects.json")
        # all_obj_info = common.load_json(objects_info_file)
        # obj_data = next(x for x in all_obj_info["scans"] if x["scan"] == scan_id)
        mesh_one = copy.deepcopy(mesh)
        object_ids = [int(obj["id"]) for obj in obj_data["objects"]]

        mask = np.isin(annos, object_ids)
        V_world_gt = np.asarray(mesh_one.vertices, np.float32)[mask]
        V_can_gt   = (R @ V_world_gt.T).T
        mu_gt = V_can_gt.mean(0).astype(np.float32)
        s_gt  = float(np.max(np.abs(V_can_gt - mu_gt[None,:])))

        x_norm_gt = (V_can_gt - mu_gt[None,:]) / (2.0 * s_gt + 1e-12)
        idx_mesh  = np.floor((x_norm_gt + 0.5) * G).astype(np.int32)
        idx_mesh  = np.unique(np.clip(idx_mesh, 0, G-1), axis=0)
        idx_mesh = _dilate_voxels_idx(idx_mesh)

        # remap GT vox -> predicted canonical (drop OOB)
        gt_idx_pred = self._remap_idx_between_boxes(
            idx_mesh, G, mu_src=mu_gt, s_src=s_gt, mu_dst=mu_pred, s_dst=s_pred
        )

        return dict(mu_pred=mu_pred, s_pred=s_pred,
                    pr_idx=pr_idx, logits=logits[0,0].detach().cpu(),
                    mu_gt=mu_gt, s_gt=s_gt,
                    gt_idx_pred=gt_idx_pred, R=R)
    def run(self, scene_id=None):
        if scene_id is not None:
            self.inference(scene_id)
        else:
            for idx in range(len(self.dataset)):
                self.inference(idx)
                
    @torch.no_grad()
    def evaluate(self, scene_ids, thr=0.5, n_frames="all", frame_policy="first"):
        """
        scene_ids: iterable of scan ids
        thr: voxel threshold for logits.sigmoid()
        n_frames: "all" or int
        frame_policy: "first" or "random" (only used when n_frames is int)
        Returns:
        {
            "summary":  dict of global means/stds across scenes,
            "per_scene": list of dicts (one per scene, frame-averaged),
            "per_frame": list of dicts (one per evaluated frame; useful for debugging)
        }
        """
        import numpy as np
        per_scene = []
        per_frame = []
        scale_ratios = []

        # which metrics we average at the scene level and then across scenes
        keys = [
            "bbox_center_L2_m",
            "bbox_center_L2_vox",
            "bbox_scale_rel_err",
            "bbox_iou",
            "vox_iou",
            "vox_precision",
            "vox_recall",
            "vox_f1",
            "pred_vox",
            "gt_vox",
            "bbox_scale_ratio",
        ]

        def mean_of(lst, key):
            vals = np.array([x[key] for x in lst], dtype=np.float32)
            return float(vals.mean()) if vals.size else float("nan")

        def std_of(lst, key):
            vals = np.array([x[key] for x in lst], dtype=np.float32)
            return float(vals.std(ddof=0)) if vals.size else float("nan")
        
        def _nanmean(a): return float(np.nanmean(a)) if a.size else float("nan")
        def _nanstd(a):  return float(np.nanstd(a, ddof=0)) if a.size else float("nan")

        for sid in scene_ids:
            # 1) choose frames for this scene
            fids = scan3r.load_frame_idxs(
                data_dir=os.path.join(self.cfg.data.root_dir, "scenes"),
                scan_id=sid,
            )
            # if len(fids) > 50:
            #     fids = fids[50:]
            if isinstance(n_frames, int):
                if frame_policy == "first":
                    if len(fids) > n_frames:
                        fids = fids[:n_frames]
                elif frame_policy == "random":
                    import random
                    fids = random.sample(fids, min(n_frames, len(fids)))
                else:
                    raise ValueError(f"Unknown frame_policy: {frame_policy}")
            fids = ['000001']
            # 2) evaluate each chosen frame independently
            root_dir = self.cfg.data.root_dir
            scenes_dir = osp.join(root_dir, "scenes")
            mesh = scan3r.load_ply_mesh(data_dir=scenes_dir, scan_id=sid,
                                    label_file_name="labels.instances.annotated.v2.ply")
            annos = scan3r.load_ply_data(data_dir=scenes_dir, scan_id=sid,
                                        label_file_name="labels.instances.annotated.v2.ply")["vertex"]["objectId"]
            objects_info_file = osp.join(self.cfg.data.root_dir, "files", "objects.json")
            all_obj_info = common.load_json(objects_info_file)
            obj_data = next(x for x in all_obj_info["scans"] if x["scan"] == sid)
            
            frames_rows = []
            for fid in fids:
                try:
                    t0 = time.time()
                    out = self._predict_one(sid, fid=fid, thr=thr, mesh=mesh, annos=annos, obj_data=obj_data)
                    print("prediction done for", fid, "takes time ", time.time()-t0)
                    t0 = time.time()
                except Exception as e:
                    # skip bad frames but keep going
                    print(f"[eval] WARNING: {sid}:{fid} failed with {e}")
                    continue

                # --- bbox metrics ---
                mu_pred, s_pred = out["mu_pred"], out["s_pred"]
                mu_gt,   s_gt   = out["mu_gt"],   out["s_gt"]

                center_err_m = float(np.linalg.norm(mu_pred - mu_gt))  # meters
                scale_rel    = float(abs(s_pred - s_gt) / (s_gt + 1e-8))
                scale_ratio = float(s_pred / (s_gt + 1e-8))
                bbox_iou     = float(self._bbox_iou_can(mu_pred, s_pred, mu_gt, s_gt))

                # center error in predicted-voxel units (optional but handy)
                err_vox = float(
                    np.linalg.norm(((mu_pred - mu_gt) / (2.0 * s_pred + 1e-8)) * self.G)
                )

                # --- voxel metrics (pred canonical) ---
                vox = self._voxel_iou_from_indices(out["pr_idx"], out["gt_idx_pred"])
                    
                row = dict(
                    scene=sid,
                    frame=fid,
                    bbox_center_L2_m=center_err_m,
                    bbox_center_L2_vox=err_vox,
                    bbox_scale_rel_err=scale_rel,
                    bbox_iou=bbox_iou,
                    bbox_scale_ratio = scale_ratio,
                    vox_iou=vox["iou"],
                    vox_precision=vox["precision"],
                    vox_recall=vox["recall"],
                    vox_tp=vox["tp"],
                    vox_fp=vox["fp"],
                    vox_fn=vox["fn"],
                    vox_f1=vox["f1"],
                    pred_vox=len(out["pr_idx"]),
                    gt_vox=len(out["gt_idx_pred"]),
                )
                frames_rows.append(row)
                per_frame.append(row)
                scale_ratios.append(scale_ratio)

            if not frames_rows:
                # No valid frames for this scene; put NaNs so aggregation still works
                per_scene.append(dict(scene=sid, num_frames=0, **{k: float("nan") for k in keys}))
                continue

            # 3) aggregate per scene (mean over frames)
            scene_row = {"scene": sid, "num_frames": len(frames_rows)}
            for k in keys:
                vals = np.array([r[k] for r in frames_rows], dtype=np.float32)
                # scene_row[k] = mean_of(frames_rows, k)
                scene_row[f"{k}_mean"] = _nanmean(vals)
                scene_row[f"{k}_std"]  = _nanstd(vals)
            per_scene.append(scene_row)

        # 4) aggregate across scenes (means/stds of per-scene means)
        def agg_across_scenes(fn):
            return {f"{k}_{fn.__name__}": fn(per_scene, k) for k in keys}

        # summary = {}
        # summary.update(agg_across_scenes(mean_of))
        # summary.update(agg_across_scenes(std_of))
        # summary["num_scenes"] = len(per_scene)
        # summary["thr"] = float(thr)
        # summary["n_frames"] = n_frames
        # summary["frame_policy"] = frame_policy
        summary = {
            "num_scenes": len(per_scene),
            "thr": float(thr),
            "n_frames": n_frames,
            
        }
        for k in keys:
            scene_means = np.array([s.get(f"{k}_mean", np.nan) for s in per_scene], dtype=np.float32)
            scene_stds  = np.array([s.get(f"{k}_std",  np.nan) for s in per_scene], dtype=np.float32)

            # Across-scene stats of scene means
            summary[f"{k}_mean_of_scenes"] = _nanmean(scene_means)
            summary[f"{k}_std_of_scenes"]  = _nanstd(scene_means)

            # Across-scene stats of the within-scene stds (frame variability)
            summary[f"{k}_within_scene_std_mean"] = _nanmean(scene_stds)
            summary[f"{k}_within_scene_std_std"]  = _nanstd(scene_stds)
        print(f"max of scale ratio: {np.max(scale_ratios)}, min: {np.min(scale_ratios)}, mean: {np.mean(scale_ratios)}")
        return dict(summary=summary, per_scene=per_scene, per_frame=per_frame)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train structured latent inference model."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the results."
    )

    parser.add_argument("--scene_id", type=str, default=None, help="Specific scene.")
    parser.add_argument("--split", type=str, default="train", help="Specific split.")
    return parser.parse_known_args()


def main() -> None:
    """Run training."""

    common.init_log(level=logging.INFO)
    args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args, do_ensure_dir=False)
    pipeline = StructureCompletionPipeline(
        cfg, visualize=args.visualize, split=args.split
    )
    pipeline.run(args.scene_id)
    scenes = ["fcf66d88-622d-291c-871f-699b2d063630"]
    # scenes = ['fcf66d9e-622d-291c-84c2-bb23dfe31327',"fcf66d88-622d-291c-871f-699b2d063630", "fcf66d8a-622d-291c-8429-0e1109c6bb26", "e44d238c-52a2-2879-89d9-a29ba04436e0"]
    report = pipeline.evaluate(scenes, thr=0.5, n_frames = 50)
    print("== Aggregate ==")
    for k, v in report["summary"].items():
        if isinstance(v, (int, float)):         # or: numbers.Number / np.floating
            print(f"{k}: {float(v):.4f}")
        else:
            print(f"{k}: {v}")
    print("\n== Per scene ==")
    for row in report["per_scene"]:
        # print(row["scene"],
        #     "bbox_iou=", f'{row["bbox_iou"]:.3f}',
        #     "vox_iou=",  f'{row["vox_iou"]:.3f}',
        #     "vox_f1=" ,  f'{row["vox_f1"]:.3f}',
        #     "center(m)=", f'{row["bbox_center_L2_m"]:.3f}')
        for k in ["bbox_iou","vox_iou","vox_f1","bbox_center_L2_m"]:
            print(f"  {k}: mean={row[f'{k}_mean']:.4f}  std={row[f'{k}_std']:.4f}")


if __name__ == "__main__":
    main()
