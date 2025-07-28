import argparse
from argparse import Namespace
import logging
import os
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

from gaussian_renderer import render
from scene.cameras import MiniCam

from src.datasets import Scan3RPatchObjectDataset
from src.datasets import Scan3RPatchObjectModifiedDataset
from src.datasets import Scan3RObjectDataset
from src.representations.gaussian.gaussian_model import Gaussian
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov

# set cuda launch blocking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
import torch.optim as optim
import cv2
import utils3d
from PIL import Image
from torchvision.utils import save_image
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from configs import Config, update_configs
from src.datasets.loaders import get_train_val_data_loader, get_val_dataloader
from src.engine import EpochBasedTrainer
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.dual_latent_autoencoder import DualDecoderAutoencoder
from src.models.losses.reconstruction import LPIPS
from utils import common, scan3r
from utils.gaussian_splatting import GaussianSplat
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getProjectionMatrix
from utils.slat_to_scene import revoxelize_to_fixed_scene_slat, revoxelize_to_fixed_scene_slat_with_aggregation, visualize_slat_alignment, revoxelize_scene_via_normalized_coords
from utils.visualisation import visualize_object_embeddings
from preprocessing.voxel_anno.voxelise_features_scene import _project_to_image


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
            # cfg, dataset=Scan3RPatchObjectDataset
            # cfg, dataset = Scan3RObjectDataset
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

    def create_model(self) -> DualDecoderAutoencoder:
        if self.cfg.autoencoder.guidance:
            from src.guidance.text_guidance import TextGuidance

            self.text_guidance = TextGuidance(device=self.device)
            # model = LatentAutoencoder(
            #     cfg=self.cfg.autoencoder,
            #     device=self.device,
            #     text_guidance=self.text_guidance,
            # )
            model = DualDecoderAutoencoder(
                cfg=self.cfg.autoencoder,
                device=self.device,
                text_guidance=self.text_guidance,
            )
        else:
            # model = LatentAutoencoder(cfg=self.cfg.autoencoder, device=self.device)
            model = DualDecoderAutoencoder(cfg=self.cfg.autoencoder, device=self.device)

        state_dict = torch.load("pretrained/slat_pretrained.pth.tar", map_location=self.device)["model"]

        # Load encoder
        encoder_dict = {
            k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder_dict, strict=True)

        # load decoder into both easy & complex
        decoder_dict = {
            k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")
        }
        model.decoder_easy.load_state_dict(decoder_dict, strict=True)
        if self.cfg.train.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False

        self.perceptual_loss = LPIPS()
        self.logger.info("DualDecoderAutoencoder created with frozen encoder" if self.cfg.train.freeze_encoder else "with trainable encoder")
        if self.cfg.train.checkpoint_path is not None and os.path.exists(self.cfg.train.checkpoint_path):
            print(f"Loading checkpoint from: {self.cfg.train.checkpoint_path }")
            checkpoint = torch.load(self.cfg.train.checkpoint_path  , map_location=self.device)

            # You must ensure this matches how the model was saved
            model.load_state_dict(checkpoint["model"], strict=False)

            if hasattr(self, "optimizer") and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if hasattr(self, "scheduler") and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
            else:
                self.start_epoch = 0
        return model

    def freeze_encoder(self) -> None:
        assert self.model is not None and isinstance(self.model, DualDecoderAutoencoder)
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

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, LatentAutoencoder)

        img_poses = data_dict["scene_graphs"]["obj_img_poses"]
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        intrinsic = data_dict["scene_graphs"]["obj_intrinsics"]
        frames = data_dict["scene_graphs"]["obj_img_top_frames"]
        masks = data_dict["scene_graphs"]["obj_annos"]
        translations = data_dict["scene_graphs"]["mean_obj_splat"]
        scales = data_dict["scene_graphs"]["scale_obj_splat"]
        held_out_idxs = data_dict["held_out_idxs"]
        # data_dict["scene_graphs"]["tot_obj_splat"] =  data_dict["scene_graphs"]["tot_obj_splat"][0]
        with torch.no_grad():
            embedding = self.model.encode(data_dict)
        embedding = embedding[0]
        # sample based on entropy
        coords = data_dict["scene_graphs"]["tot_obj_splat"][0].coords[:, 1:].float() 
        resolution = 64
        coords_normalized = ((coords + 0.5) / resolution) * 2 - 1
        coords_world = coords_normalized * scales[0] + translations[0]
        N = coords_world.shape[0]
        entropy_sum = torch.zeros(N, device=coords.device)
        entropy_count = torch.zeros(N, device=coords.device)      
        scene_id = scene_ids[0][0] 
        for i in range(len(held_out_idxs[scene_id])):
            # scene_id = scene_ids[0][0]
            # obj_id = obj_ids[i]
            intrinsics = intrinsic[scene_id]
            # pose_idx = np.random.randint(0, len(img_poses[scene_id][obj_id]))
            # frame_id = frames[scene_id][obj_id][pose_idx]
            frame_id = held_out_idxs[scene_id][i]   
            # extrinsics = img_poses[scene_id][obj_id][pose_idx]
            extrinsics = self.get_extrinsics_by_frame_id(scene_id=scene_id, frame_id=frame_id, frames = frames, img_poses=img_poses)
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            # get projection uv

            uv = utils3d.torch.project_cv(
                # coords_world.float(),  # world-space voxel positions
                coords_world.float(),
                torch.from_numpy(np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))).float().to("cuda"),
                torch.from_numpy(intrinsics["intrinsic_mat"]).float().to("cuda")
            )[0]
            uv = uv.round().long()
            H, W = int(intrinsics["height"]), int(intrinsics["width"])
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

            # check if normalized to [-1, 1]
            gray_image = image.convert("L")
            gray_np = np.array(gray_image)
            entropy_map = entropy(gray_np, disk(5))
            # entropy_maps.append(entropy_map)
            entropy_tensor = torch.tensor(entropy_map, dtype=torch.float32, device="cuda")
            # Sample entropy values
            sampled_entropy = torch.zeros(N, device="cuda")
            valid_uv = uv[valid].long()
            sampled_entropy[valid] = entropy_tensor[valid_uv[:, 1], valid_uv[:, 0]]

            # Accumulate
            entropy_sum += sampled_entropy
            entropy_count += valid.float()
        splat_entropy = entropy_sum / (entropy_count + 1e-6)
        use_complex = (splat_entropy >= 5)  # torch.bool mask, shape [N]
        print(embedding.shape)
        print(scene_ids)
        scene_splat = embedding
        reconstruction = self.model.decode(scene_splat, use_complex_mask=use_complex)
        # reconstruction = self.model.decode(scene_splat)     
        # reconstruction = self.model.decode(scene_splat)
        predicted_images = []
        ground_truth_images = []
        ground_truths = [] 

        
        
        reconstruction[0].rescale(
                torch.tensor([2, 2, 2], device=reconstruction[0].get_xyz.device)
            )
        reconstruction[0].translate(
                -torch.tensor([1, 1, 1], device=reconstruction[0].get_xyz.device)
            )
        reconstruction[0].rescale(scales[0])
        reconstruction[0].translate(translations[0])

        
        
        
        
        for i in range(len(held_out_idxs[scene_id])):
            # scene_id = scene_ids[0][0]
            # obj_id = obj_ids[i]
            intrinsics = intrinsic[scene_id]
            # pose_idx = np.random.randint(0, len(img_poses[scene_id][obj_id]))
            # frame_id = frames[scene_id][obj_id][pose_idx]
            frame_id = held_out_idxs[scene_id][i]   
            # extrinsics = img_poses[scene_id][obj_id][pose_idx]
            extrinsics = self.get_extrinsics_by_frame_id(scene_id=scene_id, frame_id=frame_id, frames = frames, img_poses=img_poses)
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            
            self.logger.info(f"i: {i}, frame_id: {frame_id}")
            image = (
                torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )  # SHape: (3, H, W)            
            pose_camera_to_world = np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))

            world_to_camera = pose_quatmat_to_rotmat(extrinsics)
            # world_to_camera = extrinsics
            world_view_transform = torch.tensor(world_to_camera, device="cuda", dtype=torch.float32)
            fovx = focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"])
            fovy = focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"])
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
            
            viewpoint_camera = MiniCam(
                width=int(intrinsics["width"]),
                height=int(intrinsics["height"]),
                fovy=focal2fov(intrinsics["intrinsic_mat"][1, 1], intrinsics["height"]),
                fovx=focal2fov(intrinsics["intrinsic_mat"][0, 0], intrinsics["width"]),
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

            rendered_image = render(
                viewpoint_camera,
                reconstruction[0],
                pipe = pipe_cfg,
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            predicted_images.append(rendered_image)
            ground_truth_images.append(image)

        if len(predicted_images) == 0:
            return {}, {}

        predicted_images = torch.stack(predicted_images).squeeze(1)
        ground_truth_images = torch.stack(ground_truth_images).squeeze(1).cuda().float()
        
        protometric_loss = 0.8 * l1_loss(
            predicted_images, ground_truth_images
        ) + 0.2 * (1.0 - ssim(predicted_images, ground_truth_images))
        perceptual_loss = self.perceptual_loss(predicted_images, ground_truth_images)

        reconstruction: List[Gaussian]
        volume_loss = torch.tensor(
            [recon.get_scaling.prod(dim=-1).mean() for recon in reconstruction]
        ).mean()
        opacity_loss = torch.tensor(
            [((1 - recon.get_opacity) ** 2).mean() for recon in reconstruction]
        ).mean()

        loss = (
            protometric_loss + volume_loss + 0.001 * opacity_loss + perceptual_loss
        ) * self.cfg.train.loss.decoder_weight
        # loss = (
        #     protometric_loss + perceptual_loss
        # ) * self.cfg.train.loss.decoder_weight
        
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
        )
        self.writer.add_image(
            f"{mode}/reconstructions",
            side_by_side_images[:4],
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
