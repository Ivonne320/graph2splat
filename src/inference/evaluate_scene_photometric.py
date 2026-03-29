#!/usr/bin/env python3
"""
Evaluate scene-level photometric metrics (PSNR, SSIM, LPIPS) for checkpoints
trained with src/trainval/train_scene_gs.py.
"""

import argparse
import copy
import json
import math
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tqdm

from gaussian_renderer import render
from scene.cameras import MiniCam, MiniCam2
from argparse import Namespace

from configs import update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.losses.reconstruction import LPIPS
from src.representations.gaussian.gaussian_model import Gaussian
from utils import scan3r, torch_util
from utils.gaussian_splatting import GaussianSplat
from utils.graphics_utils import focal2fov
from utils.loss_utils import ssim


# def psnr_b(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     b.psnr implementation:
#       mse = mean over pixels per image (B,)
#       psnr = 20*log10(1/sqrt(mse)) per image (B,)
#     Assumes inputs are in [0,1]. Returns (psnr_per_img, mse_per_img).
#     """
#     if img1.dim() == 3:
#         img1 = img1.unsqueeze(0)
#     if img2.dim() == 3:
#         img2 = img2.unsqueeze(0)
#     # mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True).clamp_min(eps)
#     mse = F.mse_loss(img1, img2)
#     psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
#     return psnr, mse
def psnr_b(
    img1: torch.Tensor,
    img2: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-image PSNR.
    Inputs: (B,C,H,W) or (C,H,W), assumed in [0,1].
    Returns:
        psnr: (B,)
        mse:  (B,)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    mse = ((img1 - img2) ** 2).flatten(1).mean(dim=1)   # (B,)
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(eps)))
    return psnr, mse
# def masked_psnr_b(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
#     """
#     pred, gt: [B,3,H,W] in [0,1]
#     mask: [B,1,H,W] or [B,H,W] (1=valid)
#     returns psnr_per_img [B,1], mse_per_img [B,1]
#     """
#     if pred.dim() == 3: pred = pred.unsqueeze(0)
#     if gt.dim() == 3: gt = gt.unsqueeze(0)
#     if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
#     if mask.dim() == 3: mask = mask.unsqueeze(1)
#     mask = mask.to(dtype=pred.dtype, device=pred.device)

#     diff2 = (pred - gt).pow(2)                    # [B,3,H,W]
#     m = mask.expand_as(diff2)                     # [B,3,H,W]
#     denom = m.sum(dim=(1,2,3), keepdim=True).clamp_min(1.0)
#     mse = (diff2 * m).sum(dim=(1,2,3), keepdim=True) / denom
#     mse = mse.clamp_min(eps)
#     psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
#     return psnr, mse

def apply_scene_alignment(reconstruction: Gaussian, translation: torch.Tensor, scale: torch.Tensor) -> None:
        """Mirror the alignment logic used at inference so training and eval share coordinate transforms."""
        device = reconstruction.get_xyz.device

        reconstruction.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
        reconstruction.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))
        # self.logger.info(f"scale:{scale}")
        # self.logger.info(f"translation:{translation}")
        reconstruction.rescale(scale)
        reconstruction.translate(translation)

def append_log_scale_to_sparse(
        x: "sp.SparseTensor",
        scales: torch.Tensor,
        eps: float = 1e-8,
    ) -> "sp.SparseTensor":
        """
        Append per-scene log(scale) (3 dims) to every sparse feature row for that scene.
        Assumes:
         - x.feats is [M, C]
          - x.layout is list-like with per-batch row selectors (slice/int/tensor/list)
          - scales is [B,3] or [B,1] on CPU/GPU
        """
        feats = x.feats
        device = feats.device
        dtype = feats.dtype

        s = scales.to(device=device, dtype=torch.float32).view(scales.shape[0], -1)
        if s.shape[1] == 1:
            s = s.repeat(1, 3)
        elif s.shape[1] > 3:
            s = s[:, :3]
        log_s = torch.log(torch.clamp(s, min=eps)).to(dtype=dtype)  # [B,3]

        extra = torch.empty((feats.shape[0], 3), device=device, dtype=dtype)

        # fill per-batch rows
        for b, rows in enumerate(x.layout):
            if isinstance(rows, slice):
                extra[rows, :] = log_s[b]
            elif isinstance(rows, int):
                extra[rows:rows+1, :] = log_s[b]
            elif torch.is_tensor(rows):
                extra.index_copy_(0, rows.to(device=device), log_s[b].expand(rows.numel(), 3))
            else:
                idx = torch.as_tensor(list(rows), device=device, dtype=torch.long)
                extra.index_copy_(0, idx, log_s[b].expand(idx.numel(), 3))

        feats2 = torch.cat([feats, extra], dim=1)
        return x.replace(feats2)

def clamp_gaussian_scale(reconstruction: Gaussian, bbox_scale: torch.Tensor) -> None:
    """Clamp each Gaussian's scale so it does not exceed the physical voxel size."""
    device = reconstruction.get_xyz.device
    dtype = reconstruction.get_xyz.dtype
    if not torch.is_tensor(bbox_scale):
        bbox_scale = torch.tensor(bbox_scale, device=device, dtype=dtype)
    bbox_scale = bbox_scale.to(device=device, dtype=dtype).flatten()
    if bbox_scale.numel() == 0:
        return
    if bbox_scale.numel() == 1:
        bbox_scale = bbox_scale.repeat(3)
    elif bbox_scale.numel() > 3:
        bbox_scale = bbox_scale[:3]
    max_scale = 1.2 * (bbox_scale / 128.0).clamp_min(1e-7)
    # self.logger.info(f"max_scale:{max_scale}")
    # max_scale = torch.nan_to_num(max_scale, nan=1e-3, posinf=1e-3, neginf=1e-3)
    min_allowed = 2e-4 + 1e-6
    current_scale = reconstruction.get_scaling
    current_scale = torch.nan_to_num(current_scale, nan=1e-3, posinf=1e-3, neginf=1e-3)
    
    clamped = torch.minimum(current_scale, max_scale.view(1, 3))
    # self.logger.info(f"clamped before min limitation: min{clamped.min()}, max{clamped.max()}")
    clamped = clamped.clamp_min(1e-7)  # scales should not go <= 0
    # self.logger.info(f"clamped after min limitation: min{clamped.min()}, max{clamped.max()}")
    clamped = clamped.clamp_min(min_allowed)
    reconstruction.from_scaling(clamped)


def gaussians_to_splat(gaussians: List[Gaussian]) -> GaussianSplat:
    if not gaussians:
        raise ValueError("No gaussians to convert.")
    xyz = torch.cat([g.get_xyz for g in gaussians], dim=0)
    features_dc = torch.cat([g._features_dc for g in gaussians], dim=0)
    features_rest: Optional[torch.Tensor] = None
    opacity = torch.cat(
        [torch.logit(g.get_opacity.clamp(1e-5, 1.0 - 1e-5)) for g in gaussians],
        dim=0,
    )
    scaling = torch.cat(
        [torch.log(g.get_scaling.clamp_min(1e-6)) for g in gaussians],
        dim=0,
    )
    rotation = torch.cat([g.get_rotation for g in gaussians], dim=0)
    return GaussianSplat(
        xyz=xyz,
        features_dc=features_dc,
        opacity=opacity,
        scaling=scaling,
        rotation=rotation,
        features_rest=features_rest,
    )


def load_image_tensor(root_dir: str, scene_id: str, frame_id: int, device: torch.device) -> Optional[torch.Tensor]:
    img_path = osp.join(root_dir, "scenes", scene_id, "sequence", f"frame-{frame_id}.color.jpg")
    if not osp.isfile(img_path):
        return None
    with Image.open(img_path) as pil_img:
        img = torch.from_numpy(np.array(pil_img, copy=False))
    img = img.permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Scene-level photometric evaluation.")
    parser.add_argument("--config", required=True, help="Path to config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file from train_scene_gs.")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate.")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--scene_ids", nargs="+", default=None, help="Optional subset of scene ids to evaluate.")
    parser.add_argument("--max_frames", type=int, default=10, help="Max frames per scene.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def accumulate(acc: Dict[str, float], mse: float, ssim_val: float, lpips_val: float) -> None:
    acc["mse"] += mse
    acc["ssim"] += ssim_val
    acc["lpips"] += lpips_val
    acc["count"] += 1


def summarize(acc: Dict[str, float]) -> Dict[str, float]:
    if acc["count"] == 0:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "frames": 0}
    mse_mean = acc["mse"] / acc["count"]
    psnr = 10.0 * math.log10(1.0 / max(mse_mean, 1e-8))
    return {
        "psnr": psnr,
        "ssim": acc["ssim"] / acc["count"],
        "lpips": acc["lpips"] / acc["count"],
        "frames": acc["count"],
    }


def main() -> None:
    args = parse_args()
    cfg = update_configs(args.config, [], do_ensure_dir=False)
    # Scene-level encoder expects raw gs_annotations splats (1024-dim features).
    cfg.data.preload_slat = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Scan3RSceneBatchDataset(cfg, split=args.split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = LatentAutoencoder(cfg.autoencoder, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    lpips_metric = LPIPS(device="cuda" if device.type == "cuda" else "cpu").eval()
    pipe_cfg = Namespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)

    target_scene_ids = set(args.scene_ids) if args.scene_ids else None
    # per_scene_stats = defaultdict(lambda: {"mse": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0})
    # overall_stats = {"mse": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
    per_scene_stats = defaultdict(lambda: {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0})
    overall_stats = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}
 
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating scenes"):
            if not batch:
                continue
            data_dict = torch_util.to_cuda(batch) if device.type == "cuda" else batch
            scene_graphs = data_dict["scene_graphs"]
            scales = scene_graphs["scale_obj_splat"]
            # ref_ids = scene_graphs['ref_ids']
            embedding = model.encode(data_dict)
            # embedding = data_dict["scene_graphs"]["tot_obj_splat"] 
            # embedding = append_log_scale_to_sparse(embedding, scales)
            reconstruction = model.decode(embedding)

            # scene_graphs = data_dict["scene_graphs"]
            raw_scene_ids = scene_graphs["scene_ids"]
            translations = scene_graphs["mean_obj_splat"]
            # scales = scene_graphs["scale_obj_splat"]
            R_cans = scene_graphs["R_cans"]
            intrinsics_map = scene_graphs["obj_intrinsics"]
            obj_2D_masks = scene_graphs['obj_2D_masks']
            image_frames = scene_graphs.get("image_frames", {})

            for idx, raw_sid in enumerate(raw_scene_ids):
                sid = raw_sid[0] if isinstance(raw_sid, (list, np.ndarray)) else raw_sid
                # ref_id = ref_ids[idx]
                if target_scene_ids and sid not in target_scene_ids:
                    continue

                frames = image_frames.get(sid, [])
                if not frames:
                    continue
                if args.max_frames > 0:
                    frames = frames[: args.max_frames]

                recon = reconstruction[idx]
                apply_scene_alignment(recon, translations[idx], scales[idx])
                clamp_gaussian_scale(recon, scales[idx])
                gauss = gaussians_to_splat([copy.deepcopy(recon)])

                intr = intrinsics_map[sid]
                width, height = int(intr["width"]), int(intr["height"])
                fx, fy = intr["intrinsic_mat"][0, 0], intr["intrinsic_mat"][1, 1]
                fovx = focal2fov(fx, width)
                fovy = focal2fov(fy, height)
                extrinsics_frames = scan3r.load_frame_poses(
                    cfg.data.root_dir, sid, tuple(frames)
                )

                R_h = np.eye(4, dtype=np.float32)
                R_can = R_cans[idx].detach().cpu().numpy()
                if R_can.ndim == 2:
                    R_h[:3, :3] = R_can
                elif R_can.ndim == 3:
                    R_h[:3, :3] = R_can[0]

                for fid in frames:
                    image = load_image_tensor(cfg.data.root_dir, sid, fid, device)
                    if image is None:
                        continue

                    # extr = R_h @ extrinsics_frames[fid]
                    extr = extrinsics_frames[fid]
                    pose = np.linalg.inv(extr)
                    cam = MiniCam2(
                        width=width,
                        height=height,
                        fovy=fovy,
                        fovx=fovx,
                        znear=0.01,
                        zfar=100.0,
                        R=pose[:3, :3].T,
                        T=pose[:3, 3],
                        K = intr["intrinsic_mat"]
                    )
                    render_out = render(
                        cam,
                        gauss,
                        pipe=pipe_cfg,
                        bg_color=torch.tensor((0.0, 0.0, 0.0), device=device),
                    )
                    predicted = render_out["render"]
                    mask_np = obj_2D_masks[sid][fid]
                    mask = torch.as_tensor(mask_np, device=image.device)
                    if mask.ndim == 2:                          # make it CxHxW for broadcasting
                        mask = mask.unsqueeze(0)
                    mask = mask.to(image.dtype) 
                    # print(f"predicted range {predicted.min()},{predicted.max()}") 
                    predicted = predicted * mask
                    image = image * mask
                    if predicted.dim() == 3:
                        predicted = predicted.unsqueeze(0)
                    predicted = predicted.clamp(0.0, 1.0)
                    

                    # mse_val = F.mse_loss(predicted, image, reduction="mean").item()
                    # ssim_val = ssim(predicted, image).item()
                    # lpips_val = lpips_metric(predicted, image).item()

                    # accumulate(per_scene_stats[sid], mse_val, ssim_val, lpips_val)
                    # accumulate(overall_stats, mse_val, ssim_val, lpips_val)
                    # --- b.psnr style: per-image PSNR then average over frames ---
                    psnr_t, mse_t = psnr_b(predicted, image)  # each is shape [B,1]
                    # psnr_t, mse_t = masked_psnr_b(predicted, image, mask)
                    psnr_val = psnr_t.mean().item()
                    mse_val = mse_t.mean().item()
                    ssim_val = ssim(predicted, image).item()
                    lpips_val = lpips_metric(predicted, image).item()

                    # accumulate psnr directly (mean of per-image psnr)
                    per_scene_stats[sid]["psnr"] += psnr_val
                    per_scene_stats[sid]["mse"] += mse_val
                    per_scene_stats[sid]["ssim"] += ssim_val
                    per_scene_stats[sid]["lpips"] += lpips_val
                    per_scene_stats[sid]["count"] += 1

                    overall_stats["psnr"] += psnr_val
                    overall_stats["mse"] += mse_val
                    overall_stats["ssim"] += ssim_val
                    overall_stats["lpips"] += lpips_val
                    overall_stats["count"] += 1
    # scene_results = {sid: summarize(stats) for sid, stats in per_scene_stats.items() if stats["count"] > 0}
    # overall_result = summarize(overall_stats)
    def summarize_b(acc: Dict[str, float]) -> Dict[str, float]:
        if acc["count"] == 0:
            return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "frames": 0}
        return {
            "psnr": acc["psnr"] / acc["count"],
            "ssim": acc["ssim"] / acc["count"],
            "lpips": acc["lpips"] / acc["count"],
            "frames": acc["count"],
            # optional: keep mean MSE for debugging
            "mse": acc["mse"] / acc["count"],
        }

    scene_results = {sid: summarize_b(stats) for sid, stats in per_scene_stats.items() if stats["count"] > 0}
    overall_result = summarize_b(overall_stats)

    print("Scene-level Photometric Metrics")
    for sid, metrics in scene_results.items():
        print(
            f"{sid}: PSNR={metrics['psnr']:.3f} dB, SSIM={metrics['ssim']:.4f}, "
            f"LPIPS={metrics['lpips']:.4f} over {metrics['frames']} frame(s)"
        )
    print(
        f"Overall: PSNR={overall_result['psnr']:.3f} dB, SSIM={overall_result['ssim']:.4f}, "
        f"LPIPS={overall_result['lpips']:.4f} over {overall_result['frames']} frame(s)"
    )

    if args.output:
        payload = {
            "overall": overall_result,
            "scenes": scene_results,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "split": args.split,
        }
        out_dir = osp.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
