import argparse
from argparse import Namespace
import logging
import os
import time
import matplotlib.pyplot as plt
import os.path as osp
from typing import Any, Dict, List, Tuple, Optional

from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixFromIntrinsics

from src.datasets import Scan3RPatchObjectModifiedDataset
from src.datasets import Scan3RSceneBatchDataset, Scan3RObjectDataset, ScanNetSceneBatchDataset
from src.representations.gaussian.gaussian_model import Gaussian
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov
import torch.nn.functional as F

# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
try:
    from src.models.latent_autoencoder_rgb_skip import LatentAutoencoderRgbSkip
    _LATENT_TYPES = (LatentAutoencoder, LatentAutoencoderRgbSkip)
except Exception:
    _LATENT_TYPES = (LatentAutoencoder,)
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r
from utils.gaussian_splatting import GaussianSplat
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getProjectionMatrix
from utils.slat_to_scene import revoxelize_to_fixed_scene_slat, revoxelize_to_fixed_scene_slat_with_aggregation, visualize_slat_alignment, revoxelize_scene_via_normalized_coords
from utils.visualisation import visualize_object_embeddings

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        self.root_dir = cfg.data.root_dir
        self.cfg = cfg
        # self.object_level = getattr(self.cfg.train, "object_level", False)
        self.object_level = False
        if self.object_level:
            self.cfg.data.preload_masks = True
        self.cfg.data.preload_slat = False
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules

        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight
        self.volume_loss_weight: float = getattr(
            # cfg.train.loss, "volume_loss_weight", 10000
            cfg.train.loss, "volume_loss_weight", 10000
            # cfg.train.loss, "volume_loss_weight", 1
        )
        #original: 1000, 0.01
        self.opacity_loss_weight: float = getattr(
            cfg.train.loss, "opacity_loss_weight", 0.001
            # cfg.train.loss, "opacity_loss_weight", 0.1
            # cfg.train.loss, "opacity_loss_weight", 0.001
        )
        
        self.opacity_vis_threshold: Optional[float] = getattr(
            cfg.train, "opacity_vis_threshold", None
        )

        # Dataloader
        start_time: float = time.time()

        dataset_cls = Scan3RObjectDataset if self.object_level else Scan3RSceneBatchDataset
        train_loader, val_loader = get_train_val_data_loader(           
            cfg, dataset = dataset_cls
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)
        # self.freeze_encoder()

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
        # self.scans2ref = self._build_scan_to_ref()
        # self.scan_transforms = scan3r.read_transform_mat(osp.join(self.cfg.data.root_dir, "files", "3RScan.json"))
        # self.logger.info("Initialisation Complete")
    def init_sh_weights(self, model):
        layout = model.decoder.layout  # or decoder_complex
        weight = model.decoder.out_layer.weight
        bias = model.decoder.out_layer.bias

        if "_features_rest" not in layout:
            print("No SH degree > 0, skipping SH weight init")
            return

        start, end = layout["_features_rest"]["range"]
        nn.init.normal_(weight[start:end], mean=0.0, std=0.01)
        nn.init.constant_(bias[start:end], 0.0)
   
    def create_model(self) -> LatentAutoencoder:
        if self.cfg.autoencoder.guidance:
            from src.guidance.text_guidance import TextGuidance

            self.text_guidance = TextGuidance(device=self.device)
            model = LatentAutoencoder(
                cfg=self.cfg.autoencoder,
                device=self.device,
                text_guidance=self.text_guidance,
            )
        else:
            model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)

        # model.load_state_dict(
        #     torch.load(
        #         "/cluster/scratch/wangyih/overfitting_dataset/pretrained/slat_pretrained.pth.tar", map_location=self.device
        #     )["model"]
        # )
        # model.load_state_dict(
        #     torch.load(
        #         "/cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/10-clean/test-128-no-dilation/snapshots/epoch-120.pth.tar", map_location=self.device
        #     )["model"]
        # )
        # model.load_state_dict(
        #     torch.load(
        #         "/cluster/scratch/wangyih/overfitting_dataset/pretrained/slat_pretrained.pth.tar", map_location=self.device
        #     )["model"]
        # )
       
        self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        # model.eval()
        
        return model

    def freeze_encoder(self) -> None:
        assert self.model is not None and isinstance(self.model, _LATENT_TYPES)
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.logger.info('frozen encoder')

    def _append_log_scale_to_sparse(
        self,
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

    def _apply_scene_alignment(self, reconstruction: Gaussian, translation: torch.Tensor, scale: torch.Tensor) -> None:
        """Mirror the alignment logic used at inference so training and eval share coordinate transforms."""
        device = reconstruction.get_xyz.device

        reconstruction.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
        reconstruction.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))
        # self.logger.info(f"scale:{scale}")
        # self.logger.info(f"translation:{translation}")
        reconstruction.rescale(scale)
        reconstruction.translate(translation)
    # def _apply_scene_alignment(self, reconstruction, translation, scale):
    #     device = reconstruction.get_xyz.device

    #     scale_vec = scale.flatten()
    #     if scale_vec.numel() == 1:
    #         scale_vec = scale_vec.repeat(3)
    #     scale_vec = scale_vec[:3]

    #     translation_vec = translation.flatten()
    #     if translation_vec.numel() == 1:
    #         translation_vec = translation_vec.repeat(3)
    #     translation_vec = translation_vec[:3]

    #     # match voxelization: voxel in [-0.5, 0.5]
    #     reconstruction.translate(torch.tensor([-0.5, -0.5, -0.5], device=device))

    #     # world = voxel*(2*scale) + mean
    #     reconstruction.rescale((2.0 * scale_vec).to(device))
    #     reconstruction.translate(translation_vec.to(device))

    def _clamp_gaussian_scale(self, reconstruction: Gaussian, bbox_scale: torch.Tensor) -> None:
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
        max_scale = 1.2 * (bbox_scale / 256.0).clamp_min(1e-7)
        # self.logger.info(f"max_scale:{max_scale}")
        # max_scale = torch.nan_to_num(max_scale, nan=1e-3, posinf=1e-3, neginf=1e-3)
        min_allowed = 2e-4 + 1e-6
        current_scale = reconstruction.get_scaling
        current_scale = torch.nan_to_num(current_scale, nan=1e-3, posinf=1e-3, neginf=1e-3)
        # clamped = current_scale
        clamped = torch.minimum(current_scale, max_scale.view(1, 3))
        # self.logger.info(f"clamped before min limitation: min{clamped.min()}, max{clamped.max()}")
        clamped = clamped.clamp_min(1e-7)  # scales should not go <= 0
        # self.logger.info(f"clamped after min limitation: min{clamped.min()}, max{clamped.max()}")
        clamped = clamped.clamp_min(min_allowed)
        reconstruction.from_scaling(clamped)
            
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        return self._train_step_scene(epoch, iteration, data_dict)

    def _train_step_scene(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, _LATENT_TYPES)


        # img_poses = data_dict["scene_graphs"]["obj_img_poses"]
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        # self.logger.info(f"scene_ids before: {scene_ids}")
        scene_ids = [sid[0] for sid in scene_ids]
        # ref_ids = data_dict["scene_graphs"].get("ref_ids", None)
        # self.logger.info(f"scene_ids after: {scene_ids}")
        intrinsic = data_dict["scene_graphs"]["obj_intrinsics"]
        # self.logger.info(f"intrinsic: {intrinsic}")
       
        translations = data_dict["scene_graphs"]["mean_obj_splat"]
        scales = data_dict["scene_graphs"]["scale_obj_splat"]
        # R_can = data_dict["scene_graphs"]["R_can"]
        R_cans = data_dict["scene_graphs"]["R_cans"]
        image_frames = data_dict["scene_graphs"]["image_frames"] 
        obj_2D_masks = data_dict["scene_graphs"]['obj_2D_masks']
        # with torch.no_grad():
        # embedding = self.model.encode(data_dict)
        embedding = data_dict["scene_graphs"]["tot_obj_splat"]
        # embedding = self._append_log_scale_to_sparse(embedding, scales)
        reconstruction = self.model.decode(embedding)
        # for recon in reconstruction:
        #     clean = torch.nan_to_num(recon.get_scaling, nan=1e-3, posinf=1e-3, neginf=1e-3)
        #     recon.from_scaling(clean)

        l1_terms: List[torch.Tensor] = []
        ssim_terms: List[torch.Tensor] = []
        perceptual_terms: List[torch.Tensor] = []
        preview_preds: List[torch.Tensor] = []
        preview_gts: List[torch.Tensor] = []
        max_preview = 4
        frames_processed = 0

        # Regularize canonical Gaussians before per-scene transforms so scaling/translation metadata
        # do not skew the penalties. Mimic TRELLIS volume loss by averaging per-Gaussian volumes.
       
        opacity_loss = torch.cat(
            [((1 - recon.get_opacity) ** 2) for recon in reconstruction], dim=0
        ).mean()
        all_scales_canon = torch.cat([recon.get_scaling for recon in reconstruction], dim=0)
        volume_loss = torch.prod(all_scales_canon, dim=1).mean()
        eps = 1e-6
        all_alphas = torch.cat([recon.get_opacity for recon in reconstruction], dim=0)
        alpha = all_alphas.clamp(eps, 1.0 - eps)
        # opacity_loss = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha)).mean()
        # --- loop over scenes in the batch ---
        for b, sid in enumerate(scene_ids):
            # ref_id = self.scans2ref.get(sid, sid)
            
            self.logger.info(f"sid: {sid}")
            recon_b = reconstruction[b]
            self._apply_scene_alignment(reconstruction[b], translations[b], scales[b])
            
            # ###########   sanity geometry check ############# #
            # ###########   sanity geometry check ############# #


            
            self._clamp_gaussian_scale(reconstruction[b], scales[b])
            # R_can = R_cans[b]  # (3,3)
            R_can = R_cans[b].detach().cpu().numpy()

            # Use the frames already picked by the sampler for THIS scene:
            fids = image_frames.get(sid, [])
            if len(fids) == 0:
                # nothing for this scene in this batch
                continue

            # Load all extrinsics for the required frames (once)
            scenes_dir = osp.join(self.cfg.data.root_dir, "scenes")
            # extrinsics_frames = scan3r.load_frame_poses(
            #     self.cfg.data.root_dir,  # same root used to build scenes_dir
            #     sid,
            #     tuple(fids),
            # )
            extrinsics_frames = scan3r.load_frame_poses(
                self.cfg.data.root_dir,
                sid,              # <-- was sid
                tuple(fids),
            )

            intrinsics = intrinsic[sid]
            scenes_dir = osp.join(self.cfg.data.root_dir, "scenes")
            # intrinsics = scan3r.load_intrinsics(scenes_dir, scan_id=ref_id)
            H, W = int(intrinsics["height"]), int(intrinsics["width"])
            fx, fy = intrinsics["intrinsic_mat"][0, 0], intrinsics["intrinsic_mat"][1, 1]
            K = intrinsics["intrinsic_mat"]

            fovx = focal2fov(fx, W)
            fovy = focal2fov(fy, H)

            # Pre-build the 4x4 “apply R_can” matrix
            R_h = np.eye(4, dtype=np.float32)
            # R_h[:3, :3] = R_can
            # self.logger.info(f"computed ref_id: {ref_id}")
            for fid in fids:
                # --- load GT image ---
                img_path = f"{self.cfg.data.root_dir}/scenes/{sid}/sequence/frame-{fid}.color.jpg"
                image = Image.open(img_path)
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # (3,H,W)
                image = image.to(self.device, non_blocking=True)
                
                

                # --- pose & camera ---
                extrinsics = extrinsics_frames[fid]               # (4,4) cam->world or world->cam as your util defines
                # extrinsics =  R_h @ extrinsics                   # apply canonical rot in world
                pose_camera_to_world = np.linalg.inv(extrinsics)  # adapt to your convention
              
                xyz = reconstruction[b].get_xyz
                scaling = reconstruction[b].get_scaling
                opacity = reconstruction[b].get_opacity
                xyz_ok = torch.isfinite(xyz).all().item()
                scaling_ok = torch.isfinite(scaling).all().item()
                opacity_ok = torch.isfinite(opacity).all().item()
                if not (xyz_ok and scaling_ok and opacity_ok):
                    bad_fields = []
                    if not xyz_ok:
                        bad_fields.append("xyz")
                    if not scaling_ok:
                        bad_fields.append("scaling")
                    if not opacity_ok:
                        bad_fields.append("opacity")
                    self.logger.error(
                        f"NaN/Inf detected for {sid}, frame {fid}; skipping render. Bad tensors: {', '.join(bad_fields)}"
                    )
                    continue
               
                C2W = extrinsics_frames[fid]        # camera -> world
                R_cw = C2W[:3, :3]
                t_cw = C2W[:3, 3]

                t_wc = -R_cw.T @ t_cw

                world_view_transform = torch.tensor(getWorld2View2(R_cw, t_wc)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=focal2fov(intrinsics["intrinsic_mat"][0, 0], 
                                                                                               intrinsics["width"]), fovY=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"])).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                
                viewpoint_camera = MiniCam(
                    width=int(intrinsics["width"]),
                    height=int(intrinsics["height"]),
                    fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                    fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
                    znear=0.01,
                    zfar=100.0,
                    world_view_transform = world_view_transform,
                    full_proj_transform = full_proj_transform
                   
                    )

                # pipe_cfg = Namespace(
                #     debug=False,
                #     compute_cov3D_python=False,
                #     convert_SHs_python=False
                # )
                pipe_cfg = Namespace(
                    debug=False,
                    compute_cov3D_python=False,
                    convert_SHs_python=False
                )

                rendered = render(
                    viewpoint_camera,
                    reconstruction[b],
                    pipe=pipe_cfg,
                    bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
                )["render"]

                # -----------------------apply mask -----------------------------------#
                mask_np = obj_2D_masks[sid][fid]           # e.g. HxW numpy/bool
                mask = torch.as_tensor(mask_np, device=image.device)  # move to CUDA
                if mask.ndim == 2:                          # make it CxHxW for broadcasting
                    mask = mask.unsqueeze(0)
                mask = mask.to(image.dtype)  
                rendered = rendered * mask
                image = image * mask

               
                l1_terms.append(l1_loss(rendered, image))
                # ssim_terms.append(ssim(rendered, image)/(mask.mean() + 1e-6))
                ssim_terms.append(ssim(rendered, image))
                # # LPIPS expects batched inputs in [-1, 1]
                rendered_lpips = rendered.clamp(0.0, 1.0) * 2.0 - 1.0
                # # rendered_lpips = rendered
                image_lpips = image.clamp(0.0, 1.0) * 2.0 - 1.0
                # # image_lpips = image
                perceptual_terms.append(
                    self.perceptual_loss(
                        rendered_lpips.unsqueeze(0), image_lpips.unsqueeze(0)
                    )
                )
               
                frames_processed += 1

                if len(preview_preds) < max_preview:
                    preview_preds.append(rendered)
                    preview_gts.append(image)

        # Nothing in batch (shouldn't happen, but guard)
        if frames_processed == 0:
            return {}, {}
        # all_scales = torch.cat([recon.get_scaling for recon in reconstruction], dim=0)
        # volume_loss = torch.prod(all_scales, dim=1).mean()
        # volume_loss = torch.log(all_scales).sum(dim=1).mean()

        l1_mean = torch.stack(l1_terms).mean()
        ssim_mean = torch.stack(ssim_terms).mean()
        perceptual_loss = torch.stack(perceptual_terms).mean()
        protometric_loss = 0.8 * l1_mean + 0.2 * (1.0 - ssim_mean)
        # protometric_loss = l1_mean + 0.2 * (1.0 - ssim_mean)

        loss = (
            protometric_loss 
            + self.volume_loss_weight * volume_loss
            + self.opacity_loss_weight * opacity_loss
            +  perceptual_loss
        ) * self.cfg.train.loss.decoder_weight
        # loss = protometric_loss + perceptual_loss

        loss_dict = {
            "loss": loss * 100   ,
            "l1_loss": protometric_loss.detach(),
            "volume_loss": volume_loss.detach(),
            "opacity_loss": opacity_loss.detach(),
            "perceptual_loss": perceptual_loss.detach(),
        }

        # reconstruction_cpu = [recon.to("cpu") for recon in reconstruction]
        reconstruction_cpu = reconstruction

        if preview_preds:
            predicted_images_cpu = torch.stack(preview_preds, dim=0)
            ground_truth_images_cpu = torch.stack(preview_gts, dim=0)
        else:
            predicted_images_cpu = torch.empty((0, 3, 1, 1))
            ground_truth_images_cpu = torch.empty((0, 3, 1, 1))

        output_dict = {
            "reconstruction": reconstruction_cpu,
            "predicted_images": predicted_images_cpu,
            "ground_truth_images": ground_truth_images_cpu,
        }
        return output_dict, loss_dict



    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        # self._save_embeddings(epoch, iteration, data_dict, output_dict)
        pass
    
    def _save_embeddings(self, epoch, iteration, data_dict, output_dict):
        if "embeddings" not in output_dict:
            return
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
        # with torch.no_grad():
        #     return self.train_step(epoch, iteration, data_dict)
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        # self._save_embeddings(epoch, iteration, data_dict, output_dict)
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

    def visualize(
        self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"
    ) -> None:
        # pass
        predicted_images = output_dict["predicted_images"]
        ground_truth_images = output_dict["ground_truth_images"]
        reconstructions = output_dict["reconstruction"]

        for i in range(len(reconstructions[:4])):
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )

        if (
            isinstance(predicted_images, torch.Tensor)
            and predicted_images.ndim == 4
            and predicted_images.size(0) > 0
        ):
            save_image(
                predicted_images[:4],
                f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
            )
            save_image(
                ground_truth_images[:4],
                f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
            )

            side_by_side_images = torch.concat(
                [ground_truth_images, predicted_images],
                dim=-1,
            )[:4]
            scale = 0.3
            sbs_small = F.interpolate(
                side_by_side_images,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)

            self.writer.add_image(
                f"{mode}/reconstructions",
                sbs_small,
                global_step=epoch,
                dataformats="NCHW",
            )


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
