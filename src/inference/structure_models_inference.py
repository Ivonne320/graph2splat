import argparse
import logging
import os
import os.path as osp

import numpy as np
import torch

from configs import Config, update_configs
from src.datasets import Scan3RSceneGraphDataset
from src.datasets import Scan3RPatchObjectModifiedDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.representations import Gaussian
from utils import common, torch_util
# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SCRATCH = os.environ.get("SCRATCH", "/scratch")

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
import cv2
import matplotlib.cm as cm
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

_LOGGER = logging.getLogger(__name__)


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
        self.output_dir = osp.join(cfg.data.root_dir, cfg.inference.output_dir)

    def load_model(self):
        model = StructureModel(cfg=self.cfg.autoencoder, device=self.device)
        model.load_state_dict(torch.load('/home/yihan/graph2splat/pretrained/training_structure_models/2025-09-10_19-46-48/snapshots/epoch-1000.pth.tar',  map_location=self.device)["model"])
        model.eval()
        return model
    
    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        base = osp.join(root_dir, "files", "gs_annotations", scene_id, "scene_level")
        p = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
        if not osp.exists(p):
            cands = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_") and f.endswith(".npz")]
            if not cands: raise FileNotFoundError(f"No aligned pack in {base}")
            p = osp.join(base, cands[0])
        return np.load(p)
    
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
    
    @torch.no_grad()
    def sample_rectified_flow(self, flow_module, z_shape, cond_tokens,
                            n_steps: int = 40,
                            method: str = "euler",
                            device: torch.device = torch.device("cuda"),
                            seed: int = 0):
        """
        flow_module: your model.flow (callable as v = flow(z, t, cond_tokens))
        z_shape: (B, zc, D, H, W) latent shape expected by decoder
        cond_tokens: (B, Lctx, C) float16/float32 on device
        """
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)

        B = z_shape[0]
        z = torch.randn(z_shape, generator=g, device=device)

        # t grid from 1.0 -> 0.0
        t_grid = torch.linspace(1.0, 0.0, steps=n_steps + 1, device=device)
        dt = t_grid[1] - t_grid[0]  # negative

        for k in range(n_steps):
            t_k = t_grid[k].expand(B)  # (B,)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                v_k = flow_module(z, t_k, cond_tokens)  # same shape as z

            if method == "euler":
                z = z + dt * v_k

            elif method == "heun":
                z_pred = z + dt * v_k
                t_k1 = t_grid[k+1].expand(B)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    v_k1 = flow_module(z_pred, t_k1, cond_tokens)
                z = z + 0.5 * dt * (v_k + v_k1)

            else:
                raise ValueError(f"Unknown sampler method: {method}")

        return z  # z at t=0
    
    def _build_flow_cond_tokens_from_pack_infer(self, pack: dict, G: int, patch_size: int,
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

    def _infer_latent_shape(self, G: int, pack):
        """
        Get the latent shape expected by decoder.
        We run one encoder pass on occ_vis just to query shape. (Cheap and safe.)
        """
        occ_vis = self._rasterize_idx(pack.get("seed_idx", np.zeros((0,3), np.int32)), G).to(self.device)
        occ_vis = occ_vis.unsqueeze(1)  # (1,1,G,G,G)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            z_tmp = self.model.encoder(occ_vis, sample_posterior=False)  # (1, zc, dz, dh, dw)
        return tuple(z_tmp.shape)

    def inference(self, idx):
        with torch.no_grad():
            pack = self._load_aligned_pack('/home/yihan/3RScan_structure_models_inference/', idx, '000000')
            G = int(pack["G"])
            mean = pack["mean"]
            scale = pack["scale"]
            occ_vis_list = []
            occ_vis = self._rasterize_idx(pack["seed_idx"], G).to(self.device)   # (1,1,G,G,G)
            occ_vis = occ_vis.unsqueeze(1) 
            patch_size = getattr(self.model.flow, "patch_size", 1)
            n_samples = 1
            n_steps = 40
            sampler = 'euler'
            seed = int(0)
            # occ_vis_list.append(occ_vis)
            # occ_vis_tensor = torch.tensor(occ_vis_list, device=self.device)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                z = self.model.encoder(occ_vis, sample_posterior=False)  # (1,zc,8,8,8)
                logits = self.model.decoder(z)                            # (1,1,G,G,G)
                
            # flow-sampling pipeline
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                cond_tokens = self._build_flow_cond_tokens_from_pack_infer(pack, G, patch_size)  # (1,Lctx,C)
                cond_tokens = cond_tokens.to(device=self.device, dtype=torch.float16)
                z_shape = self._infer_latent_shape(G, pack) 
                
                for k in range(n_samples):
                    z0 = self.sample_rectified_flow(
                        self.model.flow, z_shape, cond_tokens,
                        n_steps=40, method=sampler,
                        device=self.device, seed=(seed + k)
                    )
                   
                    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                        logits_flow = self.model.decoder(z0)  # (1,1,G,G,G)
                    
            
        if self.vis:
            self.visualize(logits, logits_flow, pack.get("seed_idx", None), mean, scale, idx)
            
    def visualize(
        self, logits, logits_flow = None, seed_idx = None, mean=None, scale = None, idx = None
    ) -> None:
        
        """
        Visuals for structure inference:
        - Save Pred occupancy as PLY (voxel centers).
        - (Optional) Save a sample from the flow as PLY/PNG for quick sanity.
        """
        
        outdir = "vis/structure_inference"
        os.makedirs(outdir, exist_ok=True)
        
         # ---- unpack ----
        logits_b11ggg: torch.Tensor = logits       # (B,1,G,G,G)
        # gt_b11ggg: torch.Tensor     = output_dict["occ_gt"]        # (B,1,G,G,G) or (B,1,G,G,G) float
        G: int                      = 64
        thr: float                  = 0.5
        B = logits_b11ggg.shape[0]
        
        # sample_logits = output_dict.get("sample_logits", None)  # (B,1,G,G,G) or None
        
        
        # ---- save first few items ----
        max_items = min(4, B)
        for i in range(max_items):
            logits_1 = logits_b11ggg[i,0]               # (G,G,G)
            # gt_1     = gt_b11ggg[i,0]                   # (G,G,G)

            # point clouds
            pr_idx = (logits_1.sigmoid() > thr).nonzero(as_tuple=False)
            # gt_idx = (gt_1 > 0.5).nonzero(as_tuple=False)
            # save_vox_as_ply(gt_idx, G, f"{outdir}/{mode}_gt_struct_{i}_completion.ply")
            save_vox_as_ply(pr_idx, G, f"{outdir}/pred_struct_completion.ply")
            if mean is not None: # paint predicted voxels onto another frame of the same scene
                pts_world = self._idx_to_world(pr_idx, G, mean, scale)
                target_fids =  ["000003","000010"]
                for target_fid in target_fids:
                    extrinsics_all = scan3r.load_frame_poses(data_dir=self.cfg.data.root_dir, scan_id=idx, frame_idxs=[target_fid]) 
                    T_wc = extrinsics_all[target_fid]
                    T_cw = np.linalg.inv(T_wc).astype(np.float32)
                    K = scan3r.load_intrinsics(data_dir=osp.join(self.cfg.data.root_dir, "scenes"), scan_id=idx)["intrinsic_mat"]
                    rgb_path = f"{self.cfg.data.root_dir}/scenes/{idx}/sequence/frame-{target_fid}.color.jpg"
                    rgb = np.array(Image.open(rgb_path).convert("RGB"))
                    Xc = self._world_to_cam(pts_world, T_cw)
                    uv, depth, valid_mask = self._project(K, Xc, rgb.shape[:2])
                    overlay = self._paint_points(rgb, (uv, depth), color=(0,255,0), radius=3)
                    out_path = f"{outdir}/reproj_{idx}_{target_fid}.png"
                    # os.makedirs("vis/structure_inference", exist_ok=True)
                    Image.fromarray(overlay).save(out_path)
                    print(f"Saved reprojected overlay → {out_path}")

            if logits_flow is not None:
                logits_2 = logits_flow[i,0]
                pr_flow_idx = (logits_2.sigmoid() > thr).nonzero(as_tuple=False)
                save_vox_as_ply(pr_flow_idx, G, f"{outdir}/pred_flow_struct_completion.ply")
            # slice mosaic PNG
            # sbs = side_by_side(gt_1, logits_1, max_slices=6)  # (2,H,W)
            # save_image(sbs, f"{outdir}/{mode}_slices_{i}.png", normalize=True)

            # TensorBoard (channels-first: we’ll make it NCHW)
            # self.writer.add_image(
            #     f"{mode}/slices_{i}_gt_pred",
            #     sbs.unsqueeze(1),  # (2,1,H,W)
            #     global_step=epoch,
            #     dataformats="NCHW",
            # )
        if seed_idx is not None:
            in_idx_np = np.asarray(seed_idx)
            if in_idx_np.size > 0:
                # enforce shape (M,3)
                assert in_idx_np.ndim == 2 and in_idx_np.shape[1] == 3, f"seed_idx shape {in_idx_np.shape} must be (M,3)"
                in_idx = torch.from_numpy(in_idx_np).long().cpu()   # <-- convert to tensor
                # (optional) safety clamp to grid
                in_idx = in_idx.clamp_(min=0, max=G-1)
                save_vox_as_ply(in_idx, G, f"{outdir}/input_occ_vis_from_seeds.ply")
        # if sample_logits is not None:
        #     for i in range(min(2, sample_logits.shape[0])):
        #         sm = sample_logits[i,0]  # (G,G,G) logits or probs
        #         sm_idx = (sm.sigmoid() > thr).nonzero(as_tuple=False)
        #         save_vox_as_ply(sm_idx, G, f"{outdir}/{mode}_sample_struct_{i}.ply")
        #         s = slice_mosaic(sm, max_slices=6)  # (1,H,W)
        #         save_image(s, f"{outdir}/{mode}_sample_slices_{i}.png", normalize=True)
        #         self.writer.add_image(
        #             f"{mode}/sample_slices_{i}",
        #             s.unsqueeze(0),   # (1,1,H,W)
        #             global_step=epoch,
        #             dataformats="NCHW",
        #         )
    def run(self, scene_id=None):
        if scene_id is not None:
            self.inference(scene_id)
        else:
            for idx in range(len(self.dataset)):
                self.inference(idx)


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


if __name__ == "__main__":
    main()
