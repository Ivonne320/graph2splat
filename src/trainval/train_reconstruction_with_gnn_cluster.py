import argparse
from argparse import Namespace
import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from gaussian_renderer import render
from PIL import Image
from scene.cameras import MiniCam
from torch import nn
from torchvision.utils import save_image

from configs import Config, update_configs
from src.datasets import Scan3RPatchObjectDataset
from src.datasets.loaders import get_train_val_data_loader
from src.representations.gaussian.gaussian_model import Gaussian
from src.engine import EpochBasedTrainer
from src.models.autoencoder import AutoEncoder
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.losses.reconstruction import LPIPS
from src.models.gnn_refine import SceneGraphRefiner
from src.models.losses.reconstruction import mse_mask_loss
from src.modules.sparse.basic import SparseTensor
from utils import common
from utils.geometry import pose_quatmat_to_rotmat
from utils.graphics_utils import focal2fov
from utils.graphics_utils import getProjectionMatrix
from utils.gaussian_splatting import GaussianSplat
from utils.loss_utils import l1_loss, ssim

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


_FILL = 0.0
_REPRESENTATION_CONFIG = {
    "perturb_offset": True,
    "voxel_size": 1.5,
    "num_gaussians": 32,
    "2d_filter_kernel_size": 0.1,
    "3d_filter_kernel_size": 9e-4,
    "scaling_bias": 4e-3,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)

        # Model Specific params
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.modules: list = cfg.autoencoder.encoder.modules
        self.rep_config = _REPRESENTATION_CONFIG

        # Loss params
        self.zoom: float = cfg.train.loss.zoom
        self.weight_align_loss: float = cfg.train.loss.alignment_loss_weight
        self.weight_contrastive_loss: float = cfg.train.loss.constrastive_loss_weight
        self.threshold = 0.5

        # Dataloader
        start_time: float = time.time()

        train_loader, val_loader = get_train_val_data_loader(
            cfg, dataset=Scan3RPatchObjectDataset
        )

        loading_time: float = time.time() - start_time
        message: str = "Data loader created: {:.3f}s collapsed.".format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)

        # self.loss = nn.MSELoss()
        # self.masked_loss = mse_mask_loss

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

    

    def create_model(self) -> SceneGraphRefiner:
        model = SceneGraphRefiner(input_dim=9, hidden_dim=4, output_dim = 9, heads = 2, vol_shape = (64,64,64))
        model.to(self.device)
        self.latent_autoencoder = LatentAutoencoder(
            cfg=self.cfg.autoencoder, device=self.device
        )
        self.autoencoder = AutoEncoder(
            cfg=self.cfg.autoencoder, device=self.device
        )
        self.latent_autoencoder.load_state_dict(
            torch.load(
                self.cfg.autoencoder.encoder.voxel.pretrained, map_location=self.device
            )["model"]
        )
        print("loading from ",self.cfg.autoencoder.encoder.pretrained  )
        self.autoencoder.load_state_dict(
            torch.load(self.cfg.autoencoder.encoder.pretrained, map_location=self.device
            )["model"],
            strict=False,
        )
        self.latent_autoencoder.eval()
        self.autoencoder.eval()
        self.latent_autoencoder.to(self.device)
        # self.latent_autoencoder.to("cpu")
        self.autoencoder.to(self.device)
        for param in self.latent_autoencoder.parameters():
            param.requires_grad = False
        for param in self.latent_autoencoder.decoder.parameters():
            param.requires_grad = False
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.perceptual_loss = LPIPS()
        message: str = "Model created"
        self.logger.info(message)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")

        return model

    
    # Add edge_index construction utility
    def build_edge_index(self, obj_positions, threshold=1.5):
        """Construct edge_index for GNN based on pairwise distances."""
        N = obj_positions.shape[0]
        edge_list = []
        for i in range(N):
            for j in range(N):
                if i != j and torch.norm(obj_positions[i] - obj_positions[j]) < threshold:
                    edge_list.append([i, j])
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long, device=obj_positions.device).t().contiguous()


    def _densify(self, sparse_splat):
        sparse_splat_dense = sparse_splat.dense()
        dense_splat = torch.full(
            (
                sparse_splat_dense.shape[0],
                1,
                sparse_splat_dense.shape[2],
                sparse_splat_dense.shape[3],
                sparse_splat_dense.shape[4],
            ),
            _FILL,
            device=sparse_splat.device,
        )
        dense_splat[
            sparse_splat.coords[:, 0],
            :,
            sparse_splat.coords[:, 1],
            sparse_splat.coords[:, 2],
            sparse_splat.coords[:, 3],
        ] = 1.0
        return torch.cat((sparse_splat_dense, dense_splat), dim=1)

    def _sparsify(self, dense_splat):
        mask = dense_splat[:, -1] > self.threshold
        coords = torch.nonzero(mask, as_tuple=False)

        if len(coords) == 0:
            return None

        feats = dense_splat[coords[:, 0], :-1, coords[:, 1], coords[:, 2], coords[:, 3]]
        return SparseTensor(coords=coords.int(), feats=feats)

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.model is not None and isinstance(self.model, SceneGraphRefiner)
        torch.cuda.empty_cache()
        print(f"At the beginning of trainer: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # AVOID OVERFLOW CUDA
        if data_dict["scene_graphs"]["tot_obj_splat"].shape[0] > 100:
            self.logger.info(
                f"Skipping too big of a scene - {data_dict['scene_graphs']['tot_obj_splat'].shape[0]}"
            )
            return {}, {"loss": torch.tensor(0.0, device=self.device)}

        if not self.cfg.data.preload_slat:
            with torch.no_grad():
                sparse_splat = self.latent_autoencoder.encode(data_dict)
        else:
            sparse_splat = data_dict["scene_graphs"]["tot_obj_splat"]
        print(f"After latent_autoencoder.encode: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        data_dict["scene_graphs"]["tot_obj_dense_splat"] = self._densify(sparse_splat)
        del sparse_splat
        torch.cuda.empty_cache()
        with torch.no_grad():
            embedding = self.autoencoder.encode(data_dict)
        print("encode() output shape:", embedding.shape[-1])   
        print(f"After autoencoder.encode: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")     
        with torch.no_grad():
            reconstruction = self.autoencoder.decode(embedding)
        print(f"After autoencoder.decode: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        del embedding
        torch.cuda.empty_cache()
        # 2. Global feature pooling for GNN nodes
        obj_feats = F.adaptive_avg_pool3d(reconstruction, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # (N_obj, C)
        obj_feats = F.normalize(obj_feats, p=2, dim=1)
        edge_index = self.build_edge_index(data_dict["scene_graphs"]["mean_obj_splat"])
        refined_reconstruction_dense, gnn_out, proj_out = self.model(obj_feats, edge_index, reconstruction)
        print(f"After model: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        refined_reconstruction_sparse = self._sparsify(refined_reconstruction_dense)
        # refined_reconstruction_sparse =self._sparsify(reconstruction)
        # del reconstruction
        torch.cuda.empty_cache()
        # with torch.no_grad():
        print(f"Before latent_autoencoder.decode: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # with torch.no_grad():
        if refined_reconstruction_sparse is None:
            self.logger.info("Skipping scene: refined_reconstruction_sparse is None")
            return {}, {"loss": torch.tensor(0.0, device=self.device),}
        refined_reconstruction = self.latent_autoencoder.decode(refined_reconstruction_sparse)
        print(f"After latent_autoencoder.decode: [GPU] Mem Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[GPU] Mem Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # refined_reconstruction = refined_reconstruction_sparse
        # del  refined_reconstruction_sparse
        torch.cuda.empty_cache()
        ## assemble objects into scene
        
        scene_ids = data_dict["scene_graphs"]["scene_ids"]
        obj_ids = data_dict["scene_graphs"]["obj_ids"]
        obj_count = data_dict["scene_graphs"]["graph_per_obj_count"]
        intrinsic = data_dict["scene_graphs"]["obj_intrinsics"]
        frames = data_dict["scene_graphs"]["obj_img_top_frames"]
        # masks = data_dict["scene_graphs"]["obj_annos"]
        translations = data_dict["scene_graphs"]["mean_obj_splat"]
        scales = data_dict["scene_graphs"]["scale_obj_splat"]
        scene_id = scene_ids[0][0]
        intrinsics = intrinsic[scene_id]
        img_poses = data_dict["scene_graphs"]["obj_img_poses"][scene_id]

        def _scale(x):
            i, splat = x
            if splat._xyz.numel() <= 0:
                return splat
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 1 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(
                torch.tensor([2, 2, 2], device=refined_reconstruction[i].get_xyz.device)
            )
            assert (
                splat._xyz.min() >= -1e-2 and splat._xyz.max() <= 2.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.translate(
                -torch.tensor([1, 1, 1], device=refined_reconstruction[i].get_xyz.device)
            )
            assert (
                splat._xyz.min() >= -1.0 - 1e-2 and splat._xyz.max() <= 1.0 + 1e-2
            ), f"{splat._xyz.min()} {splat._xyz.max()}"
            splat.rescale(scales[i])
            splat.translate(translations[i])
            return splat
        
        refined_reconstruction = list(map(lambda x: _scale(x), enumerate(refined_reconstruction))) 
        # del reconstruction
        torch.cuda.empty_cache()

        assembled_scene = Gaussian(
            sh_degree=0,
            aabb=[-0.0, -0.0, -0.0, 1.0, 1.0, 1.0],
            mininum_kernel_size=self.rep_config["3d_filter_kernel_size"],
            scaling_bias=self.rep_config["scaling_bias"],
            opacity_bias=self.rep_config["opacity_bias"],
            scaling_activation=self.rep_config["scaling_activation"],
        )
        
        assembled_scene._xyz = torch.concatenate(
            [refined_reconstruction[i]._xyz for i in range(len(refined_reconstruction))]
        )
        assembled_scene._features_dc = torch.concatenate(
            [refined_reconstruction[i]._features_dc for i in range(len(refined_reconstruction))]
        )
        assembled_scene._opacity = torch.concatenate(
            [refined_reconstruction[i]._opacity for i in range(len(refined_reconstruction))]
        )
        assembled_scene._scaling = torch.concatenate(
            [refined_reconstruction[i]._scaling for i in range(len(refined_reconstruction))]
        )
        assembled_scene._rotation = torch.concatenate(
            [refined_reconstruction[i]._rotation for i in range(len(refined_reconstruction))]
        )
             
        rendered_images, gt_images = [], []

        
        for i, frame_id in enumerate(sorted(img_poses.keys())):
            # extrinsics = img_poses[frame_id][1]
            obj_id = obj_ids[i]
            pose_idx = np.random.randint(0, len(img_poses[obj_id]))
            if obj_id not in frames[scene_id]:
                self.logger.warning(f"Skipping obj_id {obj_id} in scene {scene_id} — not in frames.")
                continue
            frame_id = frames[scene_id][obj_id][pose_idx]
            extrinsics = img_poses[obj_id][pose_idx]
            image = Image.open(
                f"{self.cfg.data.root_dir}/scenes/{scene_id}/sequence/frame-{frame_id}.color.jpg"
            )
            image = (
                torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            )
            image = image.to(self.device)
            pose_camera_to_world = np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))
            world_to_camera = np.linalg.inv(pose_camera_to_world)
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
                world_view_transform=world_view_transform,
                full_proj_transform=full_proj_transform,
    
            )
            rendered_image = render(
                viewpoint_camera,
                assembled_scene,
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]
            rendered_images.append(rendered_image)
            gt_images.append(image)


        rendered_images = torch.stack(rendered_images).squeeze(1)
        # print(rendered_images.requires_grad)
        gt_images = torch.stack(gt_images).squeeze(1)
        photometric_loss = (
        0.8 * l1_loss(rendered_images, gt_images) +
        0.2 * (1.0 - ssim(rendered_images, gt_images))
        )
        perceptual_loss = self.perceptual_loss(rendered_images, gt_images)
        
        volume_loss = assembled_scene.get_scaling.prod(dim=-1).mean()
        opacity_loss = ((1 - assembled_scene.get_opacity) ** 2).mean()
        # loss = (
        #     photometric_loss + volume_loss + 0.001 * opacity_loss + perceptual_loss
        # ) * self.cfg.train.loss.decoder_weight
        def compute_sparsity(vol, threshold = 0.1):
            mask = (vol[:, -1] > threshold).float()
            return mask.mean() 


        # sparsity_before = compute_sparsity(reconstruction)
        # sparsity_after = compute_sparsity(refined_reconstruction_dense)
        gnn_energy = gnn_out.pow(2).mean()
        proj_energy = proj_out.abs().mean()
        occupancy = refined_reconstruction_dense[:, -1]
        occupancy_before = reconstruction[:, -1]
        diff = occupancy - occupancy_before
         
        
        # sparsity = (refined_reconstruction_dense.abs() > 0.1).float().mean()
        # sparsity  = occupancy.sigmoid().mean()
        
        # sparsity = torch.relu(sparsity_after - sparsity_before) + sparsity_after
        # sparsity = torch.relu(diff).mean()
        sparsity = F.mse_loss(occupancy, occupancy_before)
        loss = (
                photometric_loss
                + 1e-4 * gnn_energy
                + 1e-4 * proj_energy
                + 1e-3 * sparsity
            )
        
        loss_dict = {
            # "loss": photometric_loss
            "loss": loss * 100,
            "l1_loss": photometric_loss,
            "gnn_energy": gnn_energy,
            "proj_energy": proj_energy,
            "sparsity": sparsity
        
        }

        output_dict = {
            "reconstruction": refined_reconstruction_dense,
            "scene_reconstruction": assembled_scene,
            "ground_truth": data_dict["scene_graphs"]["tot_obj_dense_splat"],
        }
        output_dict.update(data_dict)
        return output_dict, loss_dict

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            assert self.model is not None and isinstance(self.model, SceneGraphRefiner)
            return self.train_step(epoch, iteration, data_dict)

    def set_eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        # self.loss.eval()
        self.perceptual_loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self) -> None:
        self.training = True
        self.model.train()
        # self.loss.train()
        self.perceptual_loss.train()
        torch.set_grad_enabled(True)

    def visualize(
        self, output_dict: Dict[str, Any], epoch: int, mode: str = "train"
    ) -> None:
        ground_truth_sparse = self._sparsify(output_dict["ground_truth"][:4])
        ground_truths = self.latent_autoencoder.decoder(ground_truth_sparse)

        reconstruction_sparse = self._sparsify(output_dict["reconstruction"][:4])
        if reconstruction_sparse is None:
            return
        reconstructions = self.latent_autoencoder.decoder(reconstruction_sparse)

        scene_ids = output_dict["scene_graphs"]["scene_ids"]
        obj_ids = output_dict["scene_graphs"]["obj_ids"]
        obj_count = output_dict["scene_graphs"]["graph_per_obj_count"]
        intrinsic = output_dict["scene_graphs"]["obj_intrinsics"]
        img_poses = output_dict["scene_graphs"]["obj_img_poses"]
        translations = output_dict["scene_graphs"]["mean_obj_splat"]
        scales = output_dict["scene_graphs"]["scale_obj_splat"]
        predicted_images = []
        ground_truth_images = []

        count = 0
        scene_idx = 0
        for i in range(len(reconstructions)):
            count += 1
            if count > obj_count[scene_idx]:
                scene_idx += 1
                count = 0
            scene_id = scene_ids[scene_idx][0]
            intrinsics = intrinsic[scene_id]
            extrinsics = img_poses[scene_id][obj_ids[i]][
                np.random.randint(0, len(img_poses[scene_id][obj_ids[i]]))
            ]

            pose_camera_to_world = np.linalg.inv(pose_quatmat_to_rotmat(extrinsics))
            R = pose_camera_to_world[:3, :3].T
            T = pose_camera_to_world[:3, 3]
            viewmat = np.eye(4)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            viewmat = np.linalg.inv(viewmat)

            world_view_transform = torch.tensor(viewmat, device="cuda", dtype=torch.float32)
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
                # R=pose_camera_to_world[:3, :3].T,
                # T=pose_camera_to_world[:3, 3],
                world_view_transform=world_view_transform,
                full_proj_transform=full_proj_transform,
            )
            reconstructions[i].rescale(
                torch.tensor([2, 2, 2], device=reconstructions[i].get_xyz.device)
            )
            reconstructions[i].translate(
                -torch.tensor([1, 1, 1], device=reconstructions[i].get_xyz.device)
            )
            reconstructions[i].rescale(scales[i])
            reconstructions[i].translate(translations[i])

            ground_truths[i].rescale(
                torch.tensor([2, 2, 2], device=reconstructions[i].get_xyz.device)
            )
            ground_truths[i].translate(
                -torch.tensor([1, 1, 1], device=reconstructions[i].get_xyz.device)
            )
            ground_truths[i].rescale(scales[i])
            ground_truths[i].translate(translations[i])

            rendered_image = render(
                viewpoint_camera,
                reconstructions[i],
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]

            ground_truth_image = render(
                viewpoint_camera,
                ground_truths[i],
                pipe={
                    "debug": False,
                    "compute_cov3D_python": False,
                    "convert_SHs_python": False,
                },
                bg_color=torch.tensor((0.0, 0.0, 0.0), device="cuda"),
            )["render"]
            predicted_images.append(rendered_image)
            ground_truth_images.append(ground_truth_image)
            reconstructions[i].save_ply(
                f"{self.cfg.output_dir}/events/{mode}_reconstruction_{i}.ply"
            )
            ground_truths[i].save_ply(f"{self.cfg.output_dir}/events/{mode}_gt_{i}.ply")

        predicted_images = torch.stack(predicted_images)
        ground_truth_images = torch.stack(ground_truth_images)

        save_image(
            predicted_images,
            f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
        )
        save_image(
            ground_truth_images,
            f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
        )

        side_by_side_images = torch.concat(
            [ground_truth_images, predicted_images],
            dim=-1,
        )
        self.writer.add_image(
            f"{mode}/reconstructions",
            side_by_side_images,
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
