import argparse
from argparse import Namespace
import logging
import os
import time
import matplotlib.pyplot as plt
import os.path as osp
from typing import Any, Dict, List, Tuple

from gaussian_renderer import render
from scene.cameras import MiniCam

from src.datasets import Scan3RPatchObjectModifiedDataset
from src.datasets import Scan3RSceneBatchDataset
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
        self.cfg.data.preload_slat = False
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules

        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight

        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(           
            # cfg, dataset = Scan3RPatchObjectModifiedDataset
            cfg, dataset = Scan3RSceneBatchDataset
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
        # optimizer.step()   # dummy step to init states
        # optimizer.zero_grad()

        # Now cast states to FP32
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.float()
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
        #         "/mnt/hdd4tb/trainings/training_scene_decoder_single_frame/2025-10-04_17-08-36_frame_aligned_mesh/snapshots/epoch-11000.pth.tar", map_location=self.device
        #     )["model"]
        # )
        model.load_state_dict(
            torch.load(
                "pretrained/slat_pretrained.pth.tar", map_location=self.device
            )["model"]
        )
       
        self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        # model.eval()
        
        return model

    def freeze_encoder(self) -> None:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)
        for param in self.model.encoder.parameters():
            param.requires_grad = False
            
    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)
        

        # img_poses = data_dict["scene_graphs"]["obj_img_poses"]
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        scene_ids = [sid[0] for sid in scene_ids]
        intrinsic = data_dict["scene_graphs"]["obj_intrinsics"]
       
        translations = data_dict["scene_graphs"]["mean_obj_splat"]
        scales = data_dict["scene_graphs"]["scale_obj_splat"]
        # R_can = data_dict["scene_graphs"]["R_can"]
        R_cans = data_dict["scene_graphs"]["R_cans"]
        image_frames = data_dict["scene_graphs"]["image_frames"] 
        
        embedding = self.model.encode(data_dict)
        reconstruction = self.model.decode(embedding)
        predicted_images = []
        ground_truth_images = []
        ground_truths = [] 
        
       # --- loop over scenes in the batch ---
        for b, sid in enumerate(scene_ids):
            recon_b = reconstruction[b]

            # Apply canonical pre-transform ONCE per scene (before rendering all its frames)
            recon_b.rescale(torch.tensor([2, 2, 2], device=recon_b.get_xyz.device))
            recon_b.translate(-torch.tensor([1, 1, 1], device=recon_b.get_xyz.device))
            # Scene-specific normalization
            # handle scalar or vec3 scales robustly
            scl = scales[b]
            recon_b.rescale(scl)
            recon_b.translate(translations[b])

            R_can = R_cans[b]  # (3,3)

            # Use the frames already picked by the sampler for THIS scene:
            fids = image_frames.get(sid, [])
            if len(fids) == 0:
                # nothing for this scene in this batch
                continue

            # Load all extrinsics for the required frames (once)
            scenes_dir = osp.join(self.cfg.data.root_dir, "scenes")
            extrinsics_frames = scan3r.load_frame_poses(
                self.cfg.data.root_dir,  # same root used to build scenes_dir
                sid,
                tuple(fids),
            )

            intrinsics = intrinsic[sid]
            H, W = int(intrinsics["height"]), int(intrinsics["width"])
            fx, fy = intrinsics["intrinsic_mat"][0, 0], intrinsics["intrinsic_mat"][1, 1]

            fovx = focal2fov(fx, W)
            fovy = focal2fov(fy, H)

            # Pre-build the 4x4 “apply R_can” matrix
            R_h = np.eye(4, dtype=np.float32)
            R_h[:3, :3] = R_can

            for fid in fids:
                # --- load GT image ---
                img_path = f"{self.cfg.data.root_dir}/scenes/{sid}/sequence/frame-{fid}.color.jpg"
                image = Image.open(img_path)
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # (3,H,W)

                # --- pose & camera ---
                extrinsics = extrinsics_frames[fid]               # (4,4) cam->world or world->cam as your util defines
                extrinsics = R_h @ extrinsics                     # apply canonical rot in world
                pose_camera_to_world = np.linalg.inv(extrinsics)  # adapt to your convention

                viewpoint_camera = MiniCam(
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
                    convert_SHs_python=False
                )

                rendered = render(
                    viewpoint_camera,
                    recon_b,
                    pipe=pipe_cfg,
                    bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
                )["render"]

                predicted_images.append(rendered)
                ground_truth_images.append(image.cuda())

        # Nothing in batch (shouldn't happen, but guard)
        if len(predicted_images) == 0:
            return {}, {}

        predicted_images = torch.stack(predicted_images).squeeze(1)  # (N,3,H,W)
        ground_truth_images = torch.stack(ground_truth_images).squeeze(1)  # (N,3,H,W)

        protometric_loss = 0.8 * l1_loss(predicted_images, ground_truth_images) \
                        + 0.2 * (1.0 - ssim(predicted_images, ground_truth_images))
        perceptual_loss = self.perceptual_loss(predicted_images, ground_truth_images)

        # volume/opacity from all reconstructions (average across scenes)
        volume_loss = torch.stack(
            [recon.get_scaling.prod(dim=-1).mean() for recon in reconstruction]
        ).mean()
        opacity_loss = torch.stack(
            [((1 - recon.get_opacity) ** 2).mean() for recon in reconstruction]
        ).mean()

        loss = (protometric_loss + volume_loss + 0.01 * opacity_loss + perceptual_loss) \
            * self.cfg.train.loss.decoder_weight

        loss_dict = {
            "loss": loss * 100,
            "l1_loss": protometric_loss,
            "volume_loss": volume_loss,
            "opacity_loss": opacity_loss,
            "perceptual_loss": perceptual_loss,
        }

        output_dict = {
            "reconstruction": reconstruction,
            "gt": ground_truths,
            "predicted_images": predicted_images,
            "ground_truth_images": ground_truth_images,
            "embeddings": embedding,
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
        predicted_images = output_dict["predicted_images"]
        ground_truth_images = output_dict["ground_truth_images"]
        reconstructions = output_dict["reconstruction"]

        for i in range(len(reconstructions[:4])):
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )

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
                    side_by_side_images, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True
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
