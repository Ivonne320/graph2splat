import argparse
from argparse import Namespace
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision.utils import save_image

from configs import Config, update_configs
from gaussian_renderer import render
from scene.cameras import MiniCam

from src.datasets import Scan3RSceneBatchDataset
from src.datasets.loaders import get_train_val_data_loader
from src.engine import EpochBasedTrainer
from src.models.scene_gaussian_unet import SceneGaussianUNet
from src.models.backbones.sparse_decoder import _REPRESENTATION_CONFIG
from src.models.losses.reconstruction import LPIPS
from src.representations.gaussian.gaussian_model import Gaussian
from utils import common, scan3r
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: Optional[argparse.ArgumentParser] = None) -> None:
        super().__init__(cfg, parser)
        self.cfg = cfg
        self.cfg.data.preload_slat = False
        self.cfg.data.use_student_structure = True
        loss_cfg = getattr(cfg.train, "loss", None)
        # if not hasattr(self.cfg.data, "single_view_supervision_frames"):
        #     frames = getattr(loss_cfg, "single_view_supervision_frames", 0)
        self.cfg.data.single_view_supervision_frames = 40
        self.root_dir = cfg.data.root_dir

        # Grid/unet settings
        self.G = getattr(loss_cfg, "grid_size", 64)
        self.gauss_per_voxel = getattr(loss_cfg, "gaussians_per_voxel", 16)
        self.param_dim = getattr(loss_cfg, "param_dim", 11)
        self.dino_feat_dim = getattr(loss_cfg, "dino_feat_dim", 1024)
        self.dino_grid_channels = getattr(loss_cfg, "dino_grid_channels", 16)
        self.use_dino = self.dino_grid_channels > 0
        self.occ_weight = getattr(loss_cfg, "occ_weight", 1.0)
        self.decoder_weight = getattr(loss_cfg, "decoder_weight", 1.0)
        self.occ_warmup_epochs = getattr(loss_cfg, "occ_warmup_epochs", 0)
        warmup_default = 10.0 * self.occ_weight
        self.occ_warmup_loss_weight = getattr(
            loss_cfg, "occ_warmup_loss_weight", warmup_default
        )
        self.skip_render_during_warmup = getattr(
            loss_cfg, "skip_render_during_warmup", True
        )
        self.param_layout = {
            "delta": (0, 3),
            "scale": (3, 6),
            "rgb": (6, 9),
            "opacity": (9, 10),
        }
        occ_start = self.param_layout["opacity"][1]
        occ_end = occ_start + 1
        assert occ_end <= self.param_dim, (
            f"param_dim={self.param_dim} is insufficient for separate occupancy channel; "
            "please increase loss.param_dim."
        )
        self.param_layout["occ"] = (occ_start, occ_end)
        assert self.param_layout["occ"][1] <= self.param_dim, "invalid parameter layout"
        self._center_cache: dict[int, torch.Tensor] = {}
        base_rep_cfg = deepcopy(_REPRESENTATION_CONFIG)
        rep_cfg_override = getattr(loss_cfg, "representation_config", None)
        if rep_cfg_override:
            for key, value in rep_cfg_override.items():
                if isinstance(value, dict) and key in base_rep_cfg:
                    merged = dict(base_rep_cfg[key])
                    merged.update(value)
                    base_rep_cfg[key] = merged
                else:
                    base_rep_cfg[key] = value
        base_rep_cfg.setdefault("sh_degree", 0)
        base_rep_cfg.setdefault("3d_filter_kernel_size", 9e-4)
        base_rep_cfg.setdefault("scaling_bias", 4e-3)
        base_rep_cfg.setdefault("opacity_bias", 0.1)
        base_rep_cfg.setdefault("scaling_activation", "softplus")
        self.rep_config = base_rep_cfg

        # Dataloaders
        start = time.time()
        train_loader, val_loader = get_train_val_data_loader(cfg, dataset=Scan3RSceneBatchDataset)
        self.logger.info(f"Data loader created: {time.time() - start} collapsed." )
        self.register_loader(train_loader, val_loader)

        # Model
        model = self.create_model()
        self.register_model(model)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self.register_optimizer(optimizer)

        scheduler = None
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
        if scheduler is not None:
            self.register_scheduler(scheduler)

        self.perceptual_loss = LPIPS()
        self.logger.info("UNet trainer initialised")
    def create_model(self) -> SceneGaussianUNet:
        in_channels = 2 + (self.dino_grid_channels if self.use_dino else 0)
        base_ch = getattr(getattr(self.cfg.train, "model", None), "base_channels", 32)
        model = SceneGaussianUNet(
            in_channels=in_channels,
            base_channels=base_ch,
            dino_feat_dim=self.dino_feat_dim,
            dino_grid_channels=self.dino_grid_channels,
            gaussians_per_voxel=self.gauss_per_voxel,
            param_dim=self.param_dim,
        ).to(self.device)
        # ckpt_path = getattr(self.cfg.train, "checkpoint_path", None)
        # if ckpt_path and os.path.isfile(ckpt_path):
        #     state = torch.load(ckpt_path, map_location=self.device)
        #     weights = state.get("model", state)
        #     missing, unexpected = model.load_state_dict(weights, strict=False)
        #     self.logger.info(
        #         f"Loaded UNet checkpoint from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
        #     )
        # elif ckpt_path:
        #     self.logger.warning("Checkpoint path %s not found; training from scratch.", ckpt_path)
        return model

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _voxel_centers(self, resolution: int, device: torch.device) -> torch.Tensor:
        key = (resolution, device)
        if key not in self._center_cache:
            coords = torch.arange(resolution, device=device, dtype=torch.float32)
            grid = torch.stack(
                torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1
            )  # (R,R,R,3)
            # canonical cube in [0,1]^3 to match latent auto-decoder convention
            centers = (grid + 0.5) / resolution
            self._center_cache[key] = centers.view(-1, 3)
        return self._center_cache[key]

    def _in_warmup(self, epoch: int) -> bool:
        return self.occ_warmup_epochs > 0 and epoch <= self.occ_warmup_epochs

    def _sanitize_indices(self, idx: torch.Tensor, name: str, scene: str) -> torch.Tensor:
        if idx.numel() == 0:
            return idx
        if idx.dtype != torch.long:
            idx = idx.long()
        invalid = (idx < 0) | (idx >= self.G)
        if invalid.any():
            bad = idx[invalid]
            self.logger.warning(
                "[data-check] %s has %d invalid voxels for %s (min=%d, max=%d); clamping.",
                name,
                bad.shape[0],
                scene,
                int(bad.min().item()),
                int(bad.max().item()),
            )
            idx = idx.clone()
            idx[invalid] = bad.clamp(0, self.G - 1)
        return idx

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, scene: str) -> torch.Tensor:
        if tensor is None or tensor.numel() == 0:
            return tensor
        if torch.isfinite(tensor).all():
            return tensor
        self.logger.warning(
            f"[data-check] Non-finite values in {name} for {scene}; clamping."
        )
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sanitize_gaussian(self, gauss: Gaussian, scene: str) -> None:
        fields = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]
        for field in fields:
            value = getattr(gauss, field, None)
            if isinstance(value, torch.Tensor):
                cleaned = self._sanitize_tensor(value, f"{field}", scene)
                setattr(gauss, field, cleaned)

    def _prepare_unet_inputs(
        self, scene_graphs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt_indices = scene_graphs["student_gt_indices"]
        seed_indices = scene_graphs["student_seed_indices"]
        feats = scene_graphs["student_aligned_feats"]
        resolutions = scene_graphs["student_grid_resolution"]
        scene_ids = [sid[0] for sid in scene_graphs.get("scene_ids", [])]

        grid_feats: List[torch.Tensor] = []
        seed_grids: List[torch.Tensor] = []
        gt_grids: List[torch.Tensor] = []
        feat_channels = self.dino_grid_channels if self.use_dino else 0

        for idx, (gt_idx, seed_idx, feat_vec, res) in enumerate(
            zip(gt_indices, seed_indices, feats, resolutions)
        ):
            scene_name = scene_ids[idx] if idx < len(scene_ids) else f"scene-{idx}"
            res_val = int(res)
            if res_val != self.G:
                raise ValueError(f"Grid size mismatch: expected {self.G}, got {res_val}")
            gt_idx_t = self._sanitize_indices(gt_idx.to(device=self.device), "student_gt_indices", scene_name)
            seed_idx_t = self._sanitize_indices(seed_idx.to(device=self.device), "student_seed_indices", scene_name)
            feat_vec_t = feat_vec.to(device=self.device, dtype=torch.float32)
            if feat_vec_t.numel() > 0 and not torch.isfinite(feat_vec_t).all():
                self.logger.warning(
                    "[data-check] Non-finite DINO feats detected for %s; sanitising input.",
                    scene_name,
                )
                feat_vec_t = torch.nan_to_num(feat_vec_t, nan=0.0, posinf=0.0, neginf=0.0)
            if feat_vec_t.numel() == 0:
                feat_proj = torch.zeros((0, feat_channels), device=self.device)
            elif self.use_dino:
                feat_proj = self.model.project_dino_feats(feat_vec_t.to(self.device))
                if not torch.isfinite(feat_proj).all():
                    self.logger.warning(
                       f"[data-check] Non-finite projected DINO feats for {scene_name}; clamping."
                    )
                    feat_proj = torch.nan_to_num(feat_proj, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                feat_proj = torch.zeros((0, 0), device=self.device)

            if feat_channels > 0:
                grid_feat = torch.zeros(
                    feat_channels, self.G, self.G, self.G, device=self.device
                )
                if feat_proj.numel() > 0 and gt_idx_t.numel() > 0:
                    grid_feat[:, gt_idx_t[:, 0], gt_idx_t[:, 1], gt_idx_t[:, 2]] = (
                        feat_proj.transpose(0, 1)
                    )
            else:
                grid_feat = torch.zeros(
                    0, self.G, self.G, self.G, device=self.device
                )
            grid_feats.append(grid_feat)

            seed_grid = torch.zeros(1, self.G, self.G, self.G, device=self.device)
            if seed_idx_t.numel() > 0:
                seed_grid[0, seed_idx_t[:, 0], seed_idx_t[:, 1], seed_idx_t[:, 2]] = 1.0
            seed_grids.append(seed_grid)

            gt_grid = torch.zeros(1, self.G, self.G, self.G, device=self.device)
            if gt_idx_t.numel() > 0:
                gt_grid[0, gt_idx_t[:, 0], gt_idx_t[:, 1], gt_idx_t[:, 2]] = 1.0
            gt_grids.append(gt_grid)

        feat_tensor = None
        if feat_channels > 0:
            feat_tensor = torch.stack(grid_feats, dim=0)
        seed_tensor = torch.stack(seed_grids, dim=0)
        gt_tensor = torch.stack(gt_grids, dim=0)
        gt_tensor = gt_tensor.clamp_(0.0, 1.0)
        return feat_tensor, seed_tensor, gt_tensor

    def _params_to_gaussians(
        self,
        params: torch.Tensor,
        gt_occ: torch.Tensor,
        scene_ids: List[str],
    ) -> List[Gaussian]:
        """
        params: (B, num_gauss, param_dim, G, G, G)
        gt_occ: (B, 1, G, G, G)
        """
        B, num_gauss, _, G, _, _ = params.shape
        gaussians: List[Gaussian] = []
        base_centers = self._voxel_centers(G, self.device)  # (G^3, 3)
        voxel_size = 1.0 / G
        rep_cfg = self.rep_config
        sh_degree = rep_cfg.get("sh_degree", 0)
        scaling_bias = rep_cfg.get("scaling_bias", 4e-3)
        opacity_bias = rep_cfg.get("opacity_bias", 0.1)
        scaling_activation = rep_cfg.get("scaling_activation", "softplus")
        kernel_size = rep_cfg.get("3d_filter_kernel_size", 9e-4)
        for b in range(B):
            scene_name = scene_ids[b] if b < len(scene_ids) else f"scene-{b}"
            occ_mask = gt_occ[b, 0].reshape(-1) > 0.5
            valid_idx = occ_mask.nonzero(as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                valid_idx = torch.arange(base_centers.shape[0], device=self.device)
            centers = base_centers[valid_idx]  # (M,3)

            delta = params[b, :, self.param_layout["delta"][0] : self.param_layout["delta"][1], ...]
            delta = torch.tanh(delta.reshape(num_gauss, 3, -1)[:, :, valid_idx])
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            delta = (delta.permute(0, 2, 1) / G)

            centers_norm = centers.unsqueeze(0) + delta
            centers_norm = centers_norm.clamp(0.0, 1.0)

            scale_param = params[b, :, self.param_layout["scale"][0] : self.param_layout["scale"][1], ...]
            scale_param = scale_param.reshape(num_gauss, 3, -1)[:, :, valid_idx]
            scale_param = scale_param.permute(0, 2, 1).reshape(-1, 3)
            scale_lr = self.rep_config.get("lr", {}).get("_scaling", 1.0)
            if isinstance(scale_lr, (list, tuple)):
                scale_lr = torch.tensor(scale_lr, device=self.device, dtype=scale_param.dtype)
            scale_param = scale_param * scale_lr
            scale_param = torch.nan_to_num(scale_param, nan=0.0, posinf=1e3, neginf=-1e3)
            scale_param = torch.clamp(scale_param, min=-20.0, max=1.0)
            scale_param = self._sanitize_tensor(scale_param, "gaussian_scale_logits", scene_name)

            rgb = params[b, :, self.param_layout["rgb"][0] : self.param_layout["rgb"][1], ...]
            rgb = torch.sigmoid(rgb.reshape(num_gauss, 3, -1)[:, :, valid_idx])
            rgb = torch.nan_to_num(rgb, nan=0.5, posinf=1.0, neginf=0.0)
            rgb_flat = rgb.permute(0, 2, 1).reshape(-1, 3)

            opacity_param = params[b, :, self.param_layout["opacity"][0] : self.param_layout["opacity"][1], ...]
            opacity_param = opacity_param.reshape(num_gauss, 1, -1)[:, :, valid_idx]
            opacity_param = opacity_param.permute(0, 2, 1).reshape(-1, 1)
            opacity_lr = self.rep_config.get("lr", {}).get("_opacity", 1.0)
            if isinstance(opacity_lr, (list, tuple)):
                opacity_lr = torch.tensor(opacity_lr, device=self.device, dtype=opacity_param.dtype)
            opacity_param = opacity_param * opacity_lr
            opacity_param = torch.clamp(opacity_param, -20.0, 20.0)
            opacity_param = torch.nan_to_num(opacity_param, nan=0.0, posinf=20.0, neginf=-20.0)

            xyz_flat = centers_norm.reshape(-1, 3)
            xyz_flat = self._sanitize_tensor(xyz_flat, "gaussian_xyz", scene_name)

            gauss = Gaussian(
                sh_degree=sh_degree,
                aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                mininum_kernel_size=kernel_size,
                scaling_bias=scaling_bias,
                opacity_bias=opacity_bias,
                scaling_activation=scaling_activation,
                device=self.device,
            )
            gauss.from_xyz(xyz_flat)
            gauss._scaling = scale_param
            rot = torch.zeros((xyz_flat.shape[0], 4), device=self.device)
            rot[:, 0] = 1.0
            gauss.from_rotation(rot)
            gauss._opacity = opacity_param
            gauss.from_features(rgb_flat.unsqueeze(1))
            self._sanitize_gaussian(gauss, scene_name)
            gaussians.append(gauss)
        return gaussians

    # ------------------------------------------------------------------ #
    # Training / validation
    # ------------------------------------------------------------------ #
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        scene_graphs = data_dict["scene_graphs"]
        scene_ids = [sid[0] for sid in scene_graphs["scene_ids"]]
        intrinsic = scene_graphs["obj_intrinsics"]
        translations = scene_graphs["mean_obj_splat"].to(self.device)
        scales = scene_graphs["scale_obj_splat"].to(self.device)
        R_cans = scene_graphs["R_cans"]
        image_frames = scene_graphs["image_frames"]
        obj_2D_masks = scene_graphs["obj_2D_masks"]
        warmup_phase = self._in_warmup(epoch)

        feat_grid, seed_occ, gt_occ = self._prepare_unet_inputs(scene_graphs)
        mask = (seed_occ > 0).float()
        inputs = [seed_occ, mask]
        if feat_grid is not None:
            inputs.append(feat_grid)
        x_in = torch.cat(inputs, dim=1)

        params_raw = self.model(x_in, mask)
        B = params_raw.shape[0]
        params = params_raw.view(
            B, self.gauss_per_voxel, self.param_dim, self.G, self.G, self.G
        )
        occ_logits = params[:, :, self.param_layout["occ"][0] : self.param_layout["occ"][1], ...]
        occ_probs = torch.sigmoid(occ_logits)
        occ_probs = torch.clamp(occ_probs, min=1e-4, max=1 - 1e-4)
        occ_pred = occ_probs.mean(dim=1)
        occ_loss = F.binary_cross_entropy(occ_pred, gt_occ)

        if warmup_phase and self.skip_render_during_warmup:
            warmup_total = self.occ_warmup_loss_weight * occ_loss
            loss = warmup_total * self.decoder_weight
            zero = occ_loss.detach() * 0.0
            loss_dict = {
                "loss": loss * 10,
                "l1_loss": zero,
                "volume_loss": zero,
                "opacity_loss": zero,
                "perceptual_loss": zero,
                "occ_loss": occ_loss.detach(),
                "warmup_active": torch.tensor(1.0, device=self.device),
            }
            empty_image = torch.empty((0, 3, 1, 1))
            output_dict = {
                "reconstruction": [],
                "predicted_images": empty_image,
                "ground_truth_images": empty_image,
            }
            return output_dict, loss_dict

        gaussians = self._params_to_gaussians(params, gt_occ, scene_ids)

        embedding = gaussians  # for logging compatibility
        reconstruction = gaussians

        l1_terms: List[torch.Tensor] = []
        ssim_terms: List[torch.Tensor] = []
        perceptual_terms: List[torch.Tensor] = []
        preview_preds: List[torch.Tensor] = []
        preview_gts: List[torch.Tensor] = []
        max_preview = 4
        frames_processed = 0

        for b, sid in enumerate(scene_ids):
            recon_b = reconstruction[b]
            recon_b.rescale(torch.tensor([2, 2, 2], device=recon_b.get_xyz.device))
            recon_b.translate(-torch.tensor([1, 1, 1], device=recon_b.get_xyz.device))
            recon_b.rescale(scales[b])
            recon_b.translate(translations[b])
            self._sanitize_gaussian(recon_b, f"{sid}-world")

            R_can = R_cans[b].detach().cpu().numpy()
            fids = image_frames.get(sid, [])
            if len(fids) == 0:
                continue

            extrinsics_frames = scan3r.load_frame_poses(
                self.cfg.data.root_dir,
                sid,
                tuple(fids),
            )

            intrinsics = intrinsic[sid]
            H, W = int(intrinsics["height"]), int(intrinsics["width"])
            fx, fy = intrinsics["intrinsic_mat"][0, 0], intrinsics["intrinsic_mat"][1, 1]

            fovx = focal2fov(fx, W)
            fovy = focal2fov(fy, H)

            R_h = np.eye(4, dtype=np.float32)
            R_h[:3, :3] = R_can

            for fid in fids:
                img_path = f"{self.cfg.data.root_dir}/scenes/{sid}/sequence/frame-{fid}.color.jpg"
                image = Image.open(img_path)
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                image = image.to(self.device, non_blocking=True)

                extrinsics = extrinsics_frames[fid]
                extrinsics =(R_h) @ extrinsics
                pose_camera_to_world = np.linalg.inv(extrinsics)

                camera = MiniCam(
                    width=W,
                    height=H,
                    fovy=fovy,
                    fovx=fovx,
                    znear=0.01,
                    zfar=100.0,
                    R=pose_camera_to_world[:3, :3].T,
                    T=pose_camera_to_world[:3, 3],
                )

                pipe_cfg = Namespace(
                    debug=False,
                    compute_cov3D_python=False,
                    convert_SHs_python=False,
                )

                rendered = render(
                    camera,
                    recon_b,
                    pipe=pipe_cfg,
                    bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
                )["render"]

                mask_np = obj_2D_masks[sid][fid]
                mask_2d = torch.as_tensor(mask_np, device=image.device).unsqueeze(0)
                rendered = rendered * mask_2d
                image = image * mask_2d

                l1_terms.append(l1_loss(rendered, image))
                ssim_terms.append(ssim(rendered, image))
                perceptual_terms.append(self.perceptual_loss(rendered, image))
                frames_processed += 1

                if len(preview_preds) < max_preview:
                    preview_preds.append(rendered.detach().cpu())
                    preview_gts.append(image.detach().cpu())

        if frames_processed == 0:
            return {}, {}

        l1_mean = torch.stack(l1_terms).mean()
        ssim_mean = torch.stack(ssim_terms).mean()
        perceptual_loss = torch.stack(perceptual_terms).mean()
        protometric_loss = 0.8 * l1_mean + 0.2 * (1.0 - ssim_mean)

        volume_loss = torch.stack(
            [recon.get_scaling.prod(dim=-1).mean() for recon in reconstruction]
        ).mean()
        opacity_loss = torch.stack(
            [((1 - recon.get_opacity) ** 2).mean() for recon in reconstruction]
        ).mean()

        if warmup_phase:
            total_loss = self.occ_warmup_loss_weight * occ_loss
        else:
            total_loss = (
                protometric_loss
                + 1 * volume_loss
                + 0.1 * opacity_loss
                + perceptual_loss
                # + 1 * self.occ_weight * occ_loss
            )
        loss = total_loss * self.decoder_weight

        loss_dict = {
            "loss": loss * 10,
            "l1_loss": protometric_loss.detach(),
            "volume_loss": volume_loss.detach(),
            "opacity_loss": opacity_loss.detach(),
            "perceptual_loss": perceptual_loss.detach(),
            "occ_loss": occ_loss.detach(),
            "warmup_active": torch.tensor(1.0 if warmup_phase else 0.0, device=self.device),
        }

        predicted_images_cpu = (
            torch.stack(preview_preds, dim=0) if preview_preds else torch.empty((0, 3, 1, 1))
        )
        ground_truth_images_cpu = (
            torch.stack(preview_gts, dim=0) if preview_gts else torch.empty((0, 3, 1, 1))
        )

        output_dict = {
            "reconstruction": [g.to("cpu") for g in reconstruction],
            "predicted_images": predicted_images_cpu,
            "ground_truth_images": ground_truth_images_cpu,
        }
        return output_dict, loss_dict

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            return self.train_step(epoch, iteration, data_dict)

    def visualize(self, output_dict: Dict[str, Any], epoch: int, mode: str = "train") -> None:
        predicted_images = output_dict["predicted_images"]
        ground_truth_images = output_dict["ground_truth_images"]
        reconstructions = output_dict["reconstruction"]

        for i in range(min(len(reconstructions), 4)):
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )

        if isinstance(predicted_images, torch.Tensor) and predicted_images.ndim == 4 and predicted_images.size(0) > 0:
            save_image(
                predicted_images[:4],
                f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
            )
            save_image(
                ground_truth_images[:4],
                f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
            )

    # No-op hooks
    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        pass

    def set_eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        self.perceptual_loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        self.training = True
        self.model.train()
        self.perceptual_loss.train()
        torch.set_grad_enabled(True)


def parse_args(
    parser: Optional[argparse.ArgumentParser] = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--load_encoder", default=None, type=str)
    args, unknown = parser.parse_known_args()
    return parser, args, unknown


def main() -> None:
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
