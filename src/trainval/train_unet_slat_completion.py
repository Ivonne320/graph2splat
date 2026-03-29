import argparse
import logging
import itertools
import os
import os.path as osp
import time
import random
from typing import Any, Dict, List, Optional, Tuple
from argparse import Namespace
import numpy as np
from torchvision.utils import save_image

from utils import common, scan3r
import torch
import types
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from gaussian_renderer import render
from scene.cameras import MiniCam2
from utils.graphics_utils import focal2fov
from PIL import Image

from configs import Config, update_configs
from src.datasets import Scan3RSceneBatchDataset
from src.engine import EpochBasedTrainer
from src.models.latent_autoencoder import LatentAutoencoder
from src.models.unet3d_completion import UNetCompletionModel
from src.modules.sparse.basic import sparse_batch_cat, sparse_cat
from src.modules.sparse.basic import SparseTensor
from utils import common
from utils.visualisation import save_vox_as_ply, side_by_side

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg: Config, parser: argparse.ArgumentParser = None) -> None:
        super().__init__(cfg, parser)
        self.cfg = cfg
        encoder_cfg = getattr(cfg.autoencoder, "encoder", None)
        self.G = getattr(encoder_cfg, "resolution", 128)
        # self.latent_dim = getattr(encoder_cfg, "latent_channels", 16)
        self.latent_dim = 16
        self.recon_weight = getattr(cfg.train.loss, "recon_weight", 1.0)
        self.lambda_dice = getattr(cfg.train.loss, "dice_weight", 1.0)
        self.latent_weight = getattr(cfg.train.loss, "latent_weight", 1.0)
        self.latent_pred_weight = getattr(cfg.train.loss, "latent_pred_weight", 0.5)
        self.latent_pred_max = int(getattr(cfg.train.loss, "latent_pred_max", 300000))
        self.latent_pred_teacher_max = int(getattr(cfg.train.loss, "latent_pred_teacher_max", self.latent_pred_max))
        self.latent_pred_radius = float(getattr(cfg.train.loss, "latent_pred_radius", 1.5))
        self.latent_pred_iou_threshold = float(
            getattr(cfg.train.loss, "latent_pred_iou_threshold", 0.5)
        )
        self.latent_pred_occ_threshold = float(
            getattr(cfg.train.loss, "latent_pred_occ_threshold", 0.5)
        )
        self.teacher_scene_use = bool(getattr(cfg.train, "teacher_scene_use", True))
        self.photometric = bool(getattr(cfg.train, "photometric", True))
        self.teacher_scene_subdir = getattr(
            cfg.data, "teacher_scene_subdir", "scene_level_dinov2_128_no_dilation_clean"
        )
        self.teacher_scene_suffix = "_dense" if cfg.data.from_gt else ""
        self.scans_files_dir = osp.join(cfg.data.root_dir, "files")
        self.teacher_overlap_log_every = int(getattr(cfg.train, "teacher_overlap_log_every", 2))
        self.teacher_overlap_axis_diag = bool(
            getattr(cfg.train, "teacher_overlap_axis_diag", True)
        )
        self.teacher_overlap_axis_diag_max = int(
            getattr(cfg.train, "teacher_overlap_axis_diag_max", 12)
        )
        self._teacher_overlap_step = 0
        device_obj = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        self.use_amp = torch.cuda.is_available() and device_obj.type == "cuda"
        default_pack_root = "/cluster/scratch/wangyih/3RScan"
        self.student_pack_root = getattr(cfg.data, "student_pack_root", default_pack_root)
        self._pipe_cfg = Namespace(
            debug=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        subdir_cfg = getattr(cfg.data, "student_pack_subdir", "scene_level_structure_no_dilation_128")
        if isinstance(subdir_cfg, (list, tuple)):
            self.student_pack_subdirs = list(subdir_cfg)
        else:
            self.student_pack_subdirs = [subdir_cfg]

        start = time.time()
        self.train_dataset_ref, train_loader = self._build_loader(
            cfg, split="train", batch_size=cfg.train.batch_size, shuffle=True
        )
        self.val_dataset_ref, val_loader = self._build_loader(
            cfg, split="val", batch_size=cfg.val.batch_size, shuffle=False
        )
        self._wrap_dataset_splats(self.train_dataset_ref)
        self._wrap_dataset_splats(self.val_dataset_ref)
        self.logger.info(f"Data loader built in {time.time() - start:.2f}s")
        self.register_loader(train_loader, val_loader)

        self.seed_feat_dim = getattr(cfg.data, "student_feat_dim", None)
        if self.seed_feat_dim is None:
            attr_dim = getattr(self.train_dataset_ref, "student_pack_feat_dim", None)
            inferred_dim = self._infer_seed_feat_dim(self.train_dataset_ref)
            if attr_dim is not None and attr_dim != inferred_dim:
                self.logger.warning(
                    f"Dataset student_pack_feat_dim={attr_dim} differs from inferred pack dim {inferred_dim}; using inferred value."
                )
            self.seed_feat_dim = inferred_dim if inferred_dim is not None else (attr_dim or 1024)
        else:
            inferred_dim = self._infer_seed_feat_dim(self.train_dataset_ref)
            if inferred_dim is not None and inferred_dim != self.seed_feat_dim:
                self.logger.warning(
                    f"Configured student_feat_dim={self.seed_feat_dim} differs from pack dim {inferred_dim}; using configured value."
                )
        self.logger.info(f"Seed feature dimension: {self.seed_feat_dim}")

        self.teacher_encoder, self.teacher_decoder, self.teacher = self._build_teacher()
        teacher_latent_dim = self._infer_teacher_latent_dim(self.teacher_encoder)
        # teacher_latent_dim = 1024
        if teacher_latent_dim is not None and teacher_latent_dim != self.latent_dim:
            self.logger.warning(
                f"Teacher latent dim {teacher_latent_dim} differs from UNet config {self.latent_dim}; aligning to teacher."
            )
            self.latent_dim = teacher_latent_dim

        model = self.create_model()
        self.register_model(model)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optim.lr,
            eps=1e-8,
            weight_decay=cfg.train.optim.weight_decay,
        )
        self._latent_mismatch_logged = False
        self.register_optimizer(optimizer)
        self.logger.info("Initialization complete.")

    def _build_loader(self, cfg: Config, split: str, batch_size: int, shuffle: bool) -> Tuple[Scan3RSceneBatchDataset, DataLoader]:
        dataset = Scan3RSceneBatchDataset(cfg, split=split)
        num_workers = cfg.train.num_workers if shuffle else cfg.val.num_workers
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._raw_collate,
            pin_memory=True,
            drop_last=False,
        )
        return dataset, loader

    @staticmethod
    def _raw_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"samples": batch}

    @staticmethod
    def _splat_reduce_vec3(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t
        if t.ndim == 0:
            return t.new_full((3,), t.item())
        flat = t.view(-1, t.shape[-1])
        if flat.shape[1] < 3:
            pad = torch.zeros(flat.shape[0], 3 - flat.shape[1], device=t.device, dtype=t.dtype)
            flat = torch.cat([flat, pad], dim=1)
        return flat[:, :3].mean(dim=0)

    def _wrap_dataset_splats(self, dataset: Scan3RSceneBatchDataset) -> None:
        if dataset is None:
            return
        def normalize(output: dict) -> dict:
            mean = output.get("mean_obj_splat")
            if isinstance(mean, torch.Tensor):
                output["mean_obj_splat"] = self._splat_reduce_vec3(mean.float())
            scale = output.get("scale_obj_splat")
            if isinstance(scale, torch.Tensor):
                output["scale_obj_splat"] = self._splat_reduce_vec3(scale.float())
            rot = output.get("R_cans")
            if isinstance(rot, torch.Tensor):
                output["R_cans"] = self._normalize_rotation(rot.float())
            return output

        if hasattr(dataset, "_build_student_pack_splat"):
            orig = dataset._build_student_pack_splat

            def wrapped_build(self_ds, *args, **kwargs):
                out = orig(*args, **kwargs)
                return normalize(out)

            dataset._build_student_pack_splat = types.MethodType(wrapped_build, dataset)

        if hasattr(dataset, "_load_splats"):
            orig_load = dataset._load_splats

            def wrapped_load(self_ds, *args, **kwargs):
                out = orig_load(*args, **kwargs)
                return normalize(out)

            dataset._load_splats = types.MethodType(wrapped_load, dataset)

    @staticmethod
    def _normalize_rotation(rot: torch.Tensor) -> torch.Tensor:
        if rot.ndim == 3 and rot.shape[0] > 0:
            rot = rot[0]
        elif rot.ndim == 1 and rot.numel() == 9:
            rot = rot.view(3, 3)
        elif rot.ndim != 2:
            device, dtype = rot.device, rot.dtype
            rot = torch.eye(3, device=device, dtype=dtype)
        return rot[:3, :3]

    @staticmethod
    def _idx_to_centers(idx: torch.Tensor, G: int) -> torch.Tensor:
        if idx.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=idx.device)
        return (idx.float() + 0.5) / G - 0.5

    @staticmethod
    def _centers_to_idx(centers: torch.Tensor, G: int) -> torch.Tensor:
        if centers.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.int32, device=centers.device)
        idx = torch.floor((centers + 0.5) * G).long()
        return torch.clamp(idx, 0, G - 1)

    def _remap_seed_idx_with_bbox(
        self,
        seed_idx: torch.Tensor,
        mean_src: torch.Tensor,
        scale_src: torch.Tensor,
        mean_dst: torch.Tensor,
        scale_dst: torch.Tensor,
        G: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if seed_idx.numel() == 0:
            return seed_idx.new_zeros((0, 3))
        c_seed = self._idx_to_centers(seed_idx, G)
        s_src = scale_src.view(-1)
        if s_src.numel() == 1:
            s_src = s_src.repeat(3)
        elif s_src.numel() > 3:
            s_src = s_src[:3]
        s_src = torch.clamp(s_src.view(1, 3), min=eps)
        m_src = mean_src.view(1, 3)
        world = c_seed * (2.0 * s_src) + m_src
        s_dst = scale_dst.view(-1)
        if s_dst.numel() == 1:
            s_dst = s_dst.repeat(3)
        elif s_dst.numel() > 3:
            s_dst = s_dst[:3]
        s_dst = torch.clamp(s_dst.view(1, 3), min=eps)
        m_dst = mean_dst.view(1, 3)
        c_dst = (world - m_dst) / (2.0 * s_dst)
        return self._centers_to_idx(c_dst, G)

    def scatter_voxel_mean(self, idx_t: torch.Tensor, feat_t: torch.Tensor, G: int):
        if idx_t.numel() == 0 or feat_t.numel() == 0:
            C = feat_t.shape[-1] if feat_t.ndim == 2 else self.seed_feat_dim
            grid_feats = torch.zeros(1, C, G, G, G, device=self.device)
            seed_occ = torch.zeros(1, 1, G, G, G, device=self.device)
            return grid_feats, seed_occ
        M, C = feat_t.shape
        lin = (idx_t[:, 0] * G * G + idx_t[:, 1] * G + idx_t[:, 2]).long()
        Csum = torch.zeros(C, G * G * G, device=self.device)
        cnt = torch.zeros(G * G * G, device=self.device)
        Csum.index_add_(1, lin, feat_t.T)
        cnt.index_add_(0, lin, torch.ones(M, device=self.device))
        mask = cnt > 0
        Csum[:, mask] = Csum[:, mask] / cnt[mask]
        grid_feats = Csum.view(C, G, G, G).unsqueeze(0)
        seed_occ = torch.zeros(1, 1, G, G, G, device=self.device)
        seed_occ.view(1, 1, -1)[0, 0, torch.unique(lin)] = 1.0
        return grid_feats, seed_occ

    def _build_teacher(self):
        self.logger.info(f"building teacher")
        teacher = LatentAutoencoder(self.cfg.autoencoder, device=self.device)
        self.logger.info(f"teacher initialized")
        self.logger.info(teacher.encoder )
        # snapshot = getattr(self.cfg.train, "teacher_snapshot", None)
        # snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/10-clean/test-256-random-1/snapshots/epoch-200.pth.tar'
        # snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/teacher/500scenes-128/snapshots/epoch-27.pth.tar'
        # snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/teacher/200scenes-128/snapshots/epoch-138.pth.tar'
        snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/debug_gs/teacher/200scenes-128/snapshots/epoch-264.pth.tar'
        if snapshot:
            if osp.isfile(snapshot):
                state = torch.load(snapshot, map_location=self.device)
                state_dict = state.get("model", state)
                missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
                self.logger.info(
                    f"Loaded teacher checkpoint {snapshot} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            else:
                self.logger.warning(f"Teacher snapshot not found at {snapshot}, using randomly initialized weights.")
        teacher.encoder.eval()
        teacher.decoder.eval()

        for param in teacher.encoder.parameters():
            param.requires_grad = False
        for param in teacher.decoder.parameters():
            param.requires_grad = False
        self.logger.info("Initialized frozen teacher encoder for slat supervision")
        return teacher.encoder, teacher.decoder, teacher

    @staticmethod
    def _infer_teacher_latent_dim(teacher_encoder) -> int:
        out_layer = getattr(teacher_encoder, "out_layer", None)
        if out_layer is not None:
            out_features = getattr(out_layer, "out_features", None)
            if isinstance(out_features, int) and out_features > 0:
                return out_features // 2
        return None

    def _infer_seed_feat_dim(self, dataset: Scan3RSceneBatchDataset) -> int:
        fallback = 1024
        if dataset is None:
            return fallback
        def _find_pack_path(scene_id: str, frame_id: str) -> Optional[str]:
            fid = str(frame_id).zfill(6)
            for subdir in self.student_pack_subdirs:
                base = osp.join(
                    self.student_pack_root,
                    "files",
                    "gs_annotations",
                    scene_id,
                    subdir,
                )
                candidate = osp.join(base, f"student_pack_aligned_{fid}.npz")
                if osp.isfile(candidate):
                    return candidate
                if osp.isdir(base):
                    cands = [
                        f
                        for f in os.listdir(base)
                        if f.startswith("student_pack_aligned_") and f.endswith(".npz")
                    ]
                    if cands:
                        return osp.join(base, sorted(cands)[0])
            return None
        for item in dataset.data_items:
            sid = item["scan_id"]
            fid = str(item["frame_idx"]).zfill(6)
            pack_path = _find_pack_path(sid, fid)
            if pack_path is None:
                continue
            tokens_path = osp.join(osp.dirname(pack_path), f"dino_tokens_{fid}.npy")
            if osp.isfile(tokens_path):
                arr = np.load(tokens_path, mmap_mode="r")
                return int(arr.shape[0])
            with np.load(pack_path, allow_pickle=False) as pack:
                feats = pack.get("feats")
                if feats is not None and feats.ndim == 2 and feats.shape[0] > 0:
                    return feats.shape[1]
        return fallback

    def create_model(self) -> UNetCompletionModel:
        out_channels = 1 + self.latent_dim
        model = UNetCompletionModel(
            feat_in=self.seed_feat_dim,
            out_channels=out_channels,
        ).to(self.device)
        # snapshot = getattr(self.cfg.train, "unet_completion_snapshot", None)
        # snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_slat_completion/student/200scenes-debug/snapshots/epoch-15.pth.tar'
        snapshot = '/cluster/scratch/wangyih/overfitting_dataset/pretrained/training_unet_slat_completion/student/10scenes-debug-with-photometric/snapshots/epoch-37.pth.tar'
        if snapshot and osp.exists(snapshot):
            state = torch.load(snapshot, map_location=self.device)
            model_state = state.get("model", state)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            self.logger.info(
                f"Loaded UNet snapshot {snapshot} (missing={missing}, unexpected={unexpected})"
            )
        return model

    def _load_aligned_pack(self, root_dir: str, scene_id: str, frame_id: str):
        for subdir in self.student_pack_subdirs:
            base = osp.join(root_dir, "files", "gs_annotations", scene_id, subdir)
            p = osp.join(base, f"student_pack_aligned_{frame_id}.npz")
            if osp.exists(p):
                return np.load(p, allow_pickle=False)
            if osp.isdir(base):
                cands = [f for f in os.listdir(base) if f.startswith("student_pack_aligned_")]
                if cands:
                    return np.load(osp.join(base, cands[0]), allow_pickle=False)
        raise FileNotFoundError(
            f"No aligned pack found for scene {scene_id}/{frame_id} in {self.student_pack_subdirs}"
        )

    def _make_batch(self, scene_graphs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt_list = scene_graphs.get("student_gt_indices")
        seed_raw_list = scene_graphs.get("student_seed_indices_raw")
        seed_feat_raw_list = scene_graphs.get("student_seed_feats_raw")
        seed_mean_list = scene_graphs.get("student_seed_mean")
        seed_scale_list = scene_graphs.get("student_seed_scale")
        feat_list = scene_graphs.get("student_aligned_feats")
        mean_gt_list = scene_graphs.get("student_mean_gt")
        scale_gt_list = scene_graphs.get("student_scale_gt")
        grid_res_list = scene_graphs.get("student_grid_resolution")
        if not (
            gt_list
            and seed_raw_list
            and seed_feat_raw_list
            and seed_mean_list
            and seed_scale_list
            and mean_gt_list
            and scale_gt_list
            and feat_list
            and grid_res_list
        ):
            raise RuntimeError(
                "Missing student structure fields. "
                "Ensure cfg.data.use_student_structure=True so packs are loaded."
            )

        occ_gt_list: List[torch.Tensor] = []
        occ_vis_list: List[torch.Tensor] = []
        x_list: List[torch.Tensor] = []

        for b in range(len(gt_list)):
            grid_res_val = grid_res_list[b]
            if isinstance(grid_res_val, torch.Tensor):
                G = int(grid_res_val.item())
            else:
                G = int(grid_res_val)
            if G != self.G:
                raise ValueError(f"Grid resolution mismatch: expected {self.G}, got {G}")

            gt_idx = gt_list[b].to(self.device).long()
            seed_idx_raw = seed_raw_list[b].to(self.device).long()
            seed_feats_raw = seed_feat_raw_list[b].to(self.device).float()
            mean_seed = seed_mean_list[b].to(self.device).float()
            scale_seed = seed_scale_list[b].to(self.device).float()
            feats = feat_list[b].to(self.device).float()

            occ_gt = torch.zeros(1, 1, G, G, G, device=self.device)
            if gt_idx.numel():
                occ_gt[0, 0, gt_idx[:, 0], gt_idx[:, 1], gt_idx[:, 2]] = 1.0
            occ_gt_list.append(occ_gt)

            occ_vis_list.append(torch.zeros(1, 1, G, G, G, device=self.device))

            mean_gt = mean_gt_list[b].to(self.device).float()
            scale_gt = scale_gt_list[b].to(self.device).float()

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                feats_comp = self.model.feature_compressor(seed_feats_raw).float()

            idx_dst = self._remap_seed_idx_with_bbox(
                seed_idx_raw,
                mean_seed,
                scale_seed,
                mean_gt,
                scale_gt,
                G,
            )
            grid_feats, seed_occ = self.scatter_voxel_mean(idx_dst.int(), feats_comp, G)
            occ_vis_list[-1] = seed_occ
            x_list.append(torch.cat([seed_occ, grid_feats], dim=1))

        occ_gt = torch.cat(occ_gt_list, dim=0)
        occ_vis = torch.cat(occ_vis_list, dim=0)
        x_in = torch.cat(x_list, dim=0)
        # self.logger.info(f"occ_gt: {occ_gt}")
        # self.logger.info(f"occ_vis: {occ_vis}")
        # self.logger.info(f"x_in: {x_in}")
        return occ_gt, occ_vis, x_in

    def _build_scene_graphs_from_samples(
        self, samples: List[Dict[str, Any]], dataset: Scan3RSceneBatchDataset
    ) -> Dict[str, Any]:
        splat_dicts: List[Dict[str, Any]] = []
        scene_ids = []
        frame_ids = []
        for sample in samples:
            sid = sample["scan_id"]
            fid = sample["frame_idx"]
            scene_ids.append([sid])
            frame_ids.append(fid)
            splat_dicts.append(dataset._build_student_pack_splat(sid, fid))

        scene_graphs: Dict[str, Any] = {}
        scene_graphs["scene_ids"] = np.array(scene_ids)
        scene_graphs["frame_ids"] = frame_ids
        splat = sparse_cat(
            [d["tot_obj_splat"] for d in splat_dicts]
        ).float()
        scene_graphs["tot_obj_splat"] = self._sanitize_sparse_coords(splat)
        scene_graphs["student_gt_indices"] = [d["student_gt_indices"] for d in splat_dicts]
        scene_graphs["student_seed_indices"] = [d["student_seed_indices"] for d in splat_dicts]
        scene_graphs["student_seed_indices_raw"] = [d["student_seed_indices_raw"] for d in splat_dicts]
        scene_graphs["student_seed_feats_raw"] = [d["student_seed_feats_raw"] for d in splat_dicts]
        scene_graphs["student_seed_mean"] = [d["student_seed_mean"] for d in splat_dicts]
        scene_graphs["student_seed_scale"] = [d["student_seed_scale"] for d in splat_dicts]
        scene_graphs["student_seed_feats"] = [d["student_seed_feats"] for d in splat_dicts]
        scene_graphs["student_mean_gt"] = [d["student_mean_gt"] for d in splat_dicts]
        scene_graphs["student_scale_gt"] = [d["student_scale_gt"] for d in splat_dicts]
        scene_graphs["student_aligned_feats"] = [d["student_aligned_feats"] for d in splat_dicts]
        scene_graphs["student_grid_resolution"] = [d["student_grid_resolution"] for d in splat_dicts]
        scene_graphs["mean_obj_splat"] = torch.stack(
            [d["mean_obj_splat"].float() for d in splat_dicts]
        )
        scene_graphs["scale_obj_splat"] = torch.stack(
            [d["scale_obj_splat"].float() for d in splat_dicts]
        )
        scene_graphs["R_cans"] = torch.stack(
            [self._normalize_rotation(d["R_cans"].float()) for d in splat_dicts]
        )
        # self.logger.info(f"scene_graphs: {scene_graphs}")
        # self.logger.info(f"student_aligned_feats mean: {scene_graphs['student_aligned_feats'][0].mean()} ")
        return scene_graphs

    def _sanitize_sparse_coords(self, splat: SparseTensor) -> SparseTensor:
        coords = splat.coords
        if coords.numel() == 0 or coords.shape[1] < 4:
            return splat
        xyz = coords[:, 1:].clone()
        if not ((xyz >= 0).all() and (xyz < self.G).all()):
            minv = xyz.min(dim=0).values
            maxv = xyz.max(dim=0).values
            if (minv >= -(self.G // 2)).all() and (maxv <= (self.G // 2)).all():
                xyz = xyz + (self.G // 2)
            xyz = torch.clamp(xyz, 0, self.G - 1)
            coords = torch.cat([coords[:, :1], xyz], dim=1)
            splat = SparseTensor(feats=splat.feats, coords=coords.int())
        return splat

    def _load_teacher_scene_voxels(self, scan_id: str) -> dict[str, Any]:
        # cached = self._teacher_scene_cache.get(scan_id)
        # if cached is not None:
        #     return cached
        self.logger.info(f"scan_id: {scan_id} ")

        base = osp.join(
            self.scans_files_dir,
            "gs_annotations",
            scan_id,
            self.teacher_scene_subdir,
        )
        voxel_path = osp.join(base, f"voxel_output{self.teacher_scene_suffix}.npz")
        mean_scale_path = osp.join(base, f"mean_scale{self.teacher_scene_suffix}.npz")
        if not osp.exists(voxel_path):
            self.logger.warning(f"Teacher voxel file missing for {scan_id} ({voxel_path}), skipping scene.")
            return None
        try:
            with np.load(voxel_path, mmap_mode="r", allow_pickle=False) as data:
                arr = data["arr_0"]
        except FileNotFoundError:
            self.logger.warning(f"Teacher voxel file missing for {scan_id} ({voxel_path}).")
            return {
                "coords": torch.zeros((0, 3), dtype=torch.int32),
                "feats": torch.zeros((0, 1024), dtype=torch.float32),
                "mean": torch.zeros((3,), dtype=torch.float32),
                "scale": torch.ones((3,), dtype=torch.float32),
                "G": self.G,
            }
        coords = torch.from_numpy(arr[:, :3].astype(np.int32))
        feats = torch.from_numpy(arr[:, 3:].astype(np.float32))
        G_scene = self.G

        if osp.exists(mean_scale_path):
            with np.load(mean_scale_path, allow_pickle=False) as ms:
                mean = torch.from_numpy(ms["mean"]).float().view(-1)
                scale = torch.from_numpy(ms["scale"]).float().view(-1)
        else:
            mean = torch.zeros((3,), dtype=torch.float32)
            scale = torch.ones((1,), dtype=torch.float32)

        if mean.numel() != 3:
            mean = mean.view(-1)[:3]
            if mean.numel() < 3:
                mean = torch.zeros((3,), dtype=torch.float32)
        if scale.numel() == 1:
            scale = scale.repeat(3)
        elif scale.numel() >= 3:
            scale = scale[:3]

        return {
            "coords": coords,
            "feats": feats,
            "mean": mean,
            "scale": scale,
            "G": G_scene,
        }

    def _compute_teacher_latents(self, scene_graphs: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.teacher_scene_use:
            sparse = scene_graphs["tot_obj_splat"].to(self.device)
            with torch.no_grad():
                latents_sparse = self.teacher_encoder(sparse, sample_posterior=False)
                # latents_sparse = sparse
            return latents_sparse.coords, latents_sparse.feats

        scene_ids = scene_graphs["scene_ids"]
        mean_gt_list = scene_graphs["student_mean_gt"]
        scale_gt_list = scene_graphs["student_scale_gt"]
        R_cans = scene_graphs["R_cans"]
        # self.logger.info(f"R_cans:{R_cans}")
        gt_idx_list = scene_graphs.get("student_gt_indices", [])

        splats = []
        overlap_num = torch.zeros((), device=self.device)
        overlap_den = torch.zeros((), device=self.device)
        overlap_gt = torch.zeros((), device=self.device)
        do_axis_diag = (
            self.teacher_overlap_axis_diag
            and self.teacher_overlap_log_every > 0
            and (self._teacher_overlap_step + 1) % self.teacher_overlap_log_every == 0
        )
        axis_diag_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for b, sid in enumerate(scene_ids):
            scan_id = sid[0] if isinstance(sid, (list, tuple, np.ndarray)) else sid
            payload = self._load_teacher_scene_voxels(scan_id)
            if payload is None:
                return None, None
            coords = payload["coords"]
            feats = payload["feats"]
            feat_dim = feats.shape[1] if feats.ndim == 2 else 0
            if coords.numel() == 0 or feats.numel() == 0:
                splats.append(
                    SparseTensor(
                        feats=torch.zeros((0, feat_dim), dtype=torch.float32),
                        coords=torch.zeros((0, 3), dtype=torch.int32),
                    )
                )
                continue

            coords_f = coords.float()
            mean_scene = payload["mean"].view(1, 3)
            scale_scene = payload["scale"].view(1, 3)
            G_scene = payload["G"]
            norm_scene = (coords_f + 0.5) / float(G_scene) - 0.5
            world = norm_scene * (2.0 * scale_scene) + mean_scene

            R_can = self._normalize_rotation(R_cans[b].to(world.device))
            # self.logger.info(f"normalized R_can:{R_can}")
            can = (R_can @ world.T).T

            mean_gt = mean_gt_list[b].to(can.device).view(1, 3)
            scale_gt = scale_gt_list[b].to(can.device).view(-1)
            if scale_gt.numel() == 1:
                scale_gt = scale_gt.repeat(3)
            elif scale_gt.numel() > 3:
                scale_gt = scale_gt[:3]
            scale_gt = scale_gt.view(1, 3)

            norm_gt = (can - mean_gt) / (2.0 * scale_gt)
            idx_gt = torch.floor((norm_gt + 0.5) * self.G).long()
            valid = (idx_gt >= 0).all(dim=1) & (idx_gt < self.G).all(dim=1)
            if not valid.any():
                splats.append(
                    SparseTensor(
                        feats=torch.zeros((0, feat_dim), dtype=torch.float32),
                        coords=torch.zeros((0, 3), dtype=torch.int32),
                    )
                )
                continue
            idx_gt = idx_gt[valid]
            feats_keep = feats[valid]
            idx_gt_raw = idx_gt

            if gt_idx_list:
                gt_idx = gt_idx_list[b].to(idx_gt.device).long()
                if gt_idx.numel() > 0:
                    if do_axis_diag:
                        axis_diag_pairs.append((idx_gt_raw, gt_idx))
                    lin_gt = gt_idx[:, 0] * self.G * self.G + gt_idx[:, 1] * self.G + gt_idx[:, 2]
                    lin_idx = idx_gt[:, 0] * self.G * self.G + idx_gt[:, 1] * self.G + idx_gt[:, 2]
                    mask = torch.isin(lin_idx, lin_gt)
                    overlap_num = overlap_num + mask.sum()
                    overlap_den = overlap_den + lin_idx.numel()
                    overlap_gt = overlap_gt + lin_gt.numel()
                    if mask.any():
                        idx_gt = idx_gt[mask]
                        feats_keep = feats_keep[mask]

            if idx_gt.numel() == 0:
                splats.append(
                    SparseTensor(
                        feats=torch.zeros((0, feat_dim), dtype=torch.float32),
                        coords=torch.zeros((0, 3), dtype=torch.int32),
                    )
                )
                continue

            lin_idx = idx_gt[:, 0] * self.G * self.G + idx_gt[:, 1] * self.G + idx_gt[:, 2]
            unique_lin, inv = torch.unique(lin_idx, return_inverse=True)
            feat_accum = torch.zeros((unique_lin.shape[0], feat_dim), device=feats_keep.device)
            counts = torch.zeros(unique_lin.shape[0], device=feats_keep.device)
            feat_accum.index_add_(0, inv, feats_keep)
            counts.index_add_(0, inv, torch.ones_like(inv, dtype=counts.dtype))
            feat_accum = feat_accum / counts.clamp_min(1).unsqueeze(1)

            x = (unique_lin // (self.G * self.G)).long()
            y = ((unique_lin // self.G) % self.G).long()
            z = (unique_lin % self.G).long()
            idx_gt = torch.stack([x, y, z], dim=1)
            splats.append(
                SparseTensor(
                    feats=feat_accum.float(),
                    coords=idx_gt.int(),
                )
            )

        batch_splat = sparse_batch_cat(splats)
        batch_splat = batch_splat.to(self.device)
        with torch.no_grad():
            latents_sparse = self.teacher_encoder(batch_splat, sample_posterior=False)
        return latents_sparse.coords, latents_sparse.feats

    def _sparse_latent_loss(self, latent_pred: torch.Tensor, teacher_coords: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
        coords = teacher_coords
        feats = teacher_feats
        if coords.numel() == 0 or feats.numel() == 0:
            return torch.zeros((), device=latent_pred.device)

        coords = coords.to(device=latent_pred.device).long()
        feats = feats.to(device=latent_pred.device, dtype=latent_pred.dtype)
        # self.logger.info(f"before everything, latent_pred: {latent_pred.shape}")
        # self.logger.info(f"latent_pred: {latent_pred}")
        # self.logger.info(f"before everything, feats: {feats.shape}")
        b = coords[:, 0]
        xyz = coords[:, 1:]
        xyz = xyz.clamp(min=0, max=self.G - 1)
        b = b.clamp(min=0, max=latent_pred.shape[0] - 1)

        pred_sel = latent_pred[b, :, xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        # self.logger.info(f"before shape regu, pred_sel: {pred_sel.shape}")
        # self.logger.info(f"before shape regu, feats: {feats.shape}")
        if pred_sel.shape[1] != feats.shape[1]:
            C = min(pred_sel.shape[1], feats.shape[1])
            pred_sel = pred_sel[:, :C]
            feats = feats[:, :C]
            # self.logger.info(f"after shape regu, pred_sel: {pred_sel.shape}")
            # self.logger.info(f"after shape regu, feats: {feats.shape}")
        return torch.abs(pred_sel - feats).mean()
    
    def _latent_pred_to_teacher_sparse(
        self,
        latent_pred: torch.Tensor,   # [B, C, G, G, G]
        teacher_coords: torch.Tensor,  # [N, 4] = [b, x, y, z]
        max_latents: Optional[int] = None,
    ) -> SparseTensor:
        """
        Build a SparseTensor from latent_pred sampled at teacher_coords.

        Returns:
            SparseTensor with
            feats  = [N, C]
            coords = [N, 4]  (batch, x, y, z)
        """
        device = latent_pred.device
        dtype = latent_pred.dtype

        if teacher_coords is None or teacher_coords.numel() == 0:
            C = latent_pred.shape[1]
            self.logger.warning(f"teacher coords missing")
            return SparseTensor(
                feats=torch.zeros((0, C), device=device, dtype=dtype),
                coords=torch.zeros((0, 4), device=device, dtype=torch.int32),
            )

        coords = teacher_coords.to(device=device).long()
        # self.logger.info(f"coords shape 0: {coords.shape[0]}")
        # optional subsampling
        # if max_latents is not None and coords.shape[0] > max_latents:
        #     sel = torch.randperm(coords.shape[0], device=device)[:max_latents]
        #     coords = coords[sel]

        b = coords[:, 0].clamp_(0, latent_pred.shape[0] - 1)
        x = coords[:, 1].clamp_(0, self.G - 1)
        y = coords[:, 2].clamp_(0, self.G - 1)
        z = coords[:, 3].clamp_(0, self.G - 1)

        # latent_pred[b, :, x, y, z] -> [N, C]
        feats = latent_pred[b, :, x, y, z]

        return SparseTensor(
            feats=feats.contiguous(),
            coords=coords.int(),
        ).to(device)
  
    def _predicted_latent_loss(
        self,
        logits: torch.Tensor,
        latent_pred: torch.Tensor,
        teacher_coords: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_pred_weight <= 0 or self.latent_pred_max <= 0:
            return torch.zeros((), device=latent_pred.device)

        coords = teacher_coords
        feats = teacher_feats
        if coords.numel() == 0 or feats.numel() == 0:
            return torch.zeros((), device=latent_pred.device)

        coords = coords.to(device=latent_pred.device).long()
        feats = feats.to(device=latent_pred.device, dtype=latent_pred.dtype)

        loss_sum = torch.zeros((), device=latent_pred.device)
        count = torch.zeros((), device=latent_pred.device)
        B = latent_pred.shape[0]
        for b in range(B):
            t_mask = coords[:, 0] == b
            if not t_mask.any():
                continue
            t_xyz = coords[t_mask][:, 1:]
            t_feats = feats[t_mask]
            if self.latent_pred_teacher_max > 0 and t_xyz.shape[0] > self.latent_pred_teacher_max:
                sel = torch.randperm(t_xyz.shape[0], device=latent_pred.device)[: self.latent_pred_teacher_max]
                t_xyz = t_xyz[sel]
                t_feats = t_feats[sel]

            logits_b = logits[b, 0]
            thr = torch.tensor(
                self.latent_pred_occ_threshold,
                device=logits_b.device,
                dtype=logits_b.dtype,
            )
            thr = torch.logit(thr.clamp(1e-4, 1.0 - 1e-4))
            mask = logits_b > thr
            p_xyz = mask.nonzero(as_tuple=False)
            scores = logits_b[mask].sigmoid()
            if p_xyz.shape[0] == 0:
                flat = logits_b.flatten()
                k = min(self.latent_pred_max, flat.numel())
                if k == 0:
                    continue
                vals, idxs = torch.topk(flat, k)
                p_xyz = torch.stack(torch.unravel_index(idxs, (self.G, self.G, self.G)), dim=1)
                scores = vals.sigmoid()

            if self.latent_pred_max is not None and p_xyz.shape[0] > self.latent_pred_max:
                vals, order = torch.topk(scores, self.latent_pred_max)
                p_xyz = p_xyz[order]
                scores = vals

            pred_feats = latent_pred[b, :, p_xyz[:, 0], p_xyz[:, 1], p_xyz[:, 2]].T

            dists = torch.cdist(p_xyz.float(), t_xyz.float(), p=2)
            min_dists, nn_idx = dists.min(dim=1)
            if self.latent_pred_radius > 0:
                keep = min_dists <= self.latent_pred_radius
            else:
                keep = torch.ones_like(min_dists, dtype=torch.bool)
            if not keep.any():
                continue

            pred_sel = pred_feats[keep]
            targ_sel = t_feats[nn_idx[keep]]
            C = min(pred_sel.shape[1], targ_sel.shape[1])
            pred_sel = pred_sel[:, :C]
            targ_sel = targ_sel[:, :C]
            loss_sum = loss_sum + torch.abs(pred_sel - targ_sel).sum()
            count = count + pred_sel.numel()

        if count.item() == 0:
            return torch.zeros((), device=latent_pred.device)
        return loss_sum / (count + 1e-6)
    # def _apply_pack_alignment(self, gauss, mean_gt: np.ndarray, scale_gt: np.ndarray) -> None:
    #     device = gauss.get_xyz.device
    #     gauss.rescale(torch.tensor([2.0, 2.0, 2.0], device=device))
    #     gauss.translate(torch.tensor([-1.0, -1.0, -1.0], device=device))

    #     scale_arr = np.asarray(scale_gt, dtype=np.float32)
    #     scale_vec = torch.from_numpy(scale_arr).to(device=device).view(-1)
    #     if scale_vec.numel() == 1:
    #         scale_vec = scale_vec.repeat(3)
    #     elif scale_vec.numel() > 3:
    #         scale_vec = scale_vec[:3]
    #     gauss.rescale(scale_vec)

    #     translation = torch.as_tensor(mean_gt, device=device).view(-1)
    #     if translation.numel() == 1:
    #         translation = translation.repeat(3)
    #     gauss.translate(translation)

    #     self._clamp_gaussian_scale(gauss, scale_gt)
    def _apply_pack_alignment(self, gauss, mean_gt, scale_gt) -> None:
        device = gauss.get_xyz.device
        dtype = gauss.get_xyz.dtype

        gauss.rescale(torch.tensor([2.0, 2.0, 2.0], device=device, dtype=dtype))
        gauss.translate(torch.tensor([-1.0, -1.0, -1.0], device=device, dtype=dtype))

        scale_vec = torch.as_tensor(scale_gt, device=device, dtype=dtype).view(-1)
        if scale_vec.numel() == 1:
            scale_vec = scale_vec.repeat(3)
        elif scale_vec.numel() > 3:
            scale_vec = scale_vec[:3]
        gauss.rescale(scale_vec)

        translation = torch.as_tensor(mean_gt, device=device, dtype=dtype).view(-1)
        if translation.numel() == 1:
            translation = translation.repeat(3)
        elif translation.numel() > 3:
            translation = translation[:3]
        gauss.translate(translation)

        self._clamp_gaussian_scale(gauss, scale_vec)

    # def _clamp_gaussian_scale(self, reconstruction, bbox_scale: np.ndarray) -> None:
    #     device = reconstruction.get_xyz.device
    #     dtype = reconstruction.get_xyz.dtype

    #     if not torch.is_tensor(bbox_scale):
    #         bbox_scale = torch.tensor(bbox_scale, device=device, dtype=dtype)

    #     bbox_scale = bbox_scale.to(device=device, dtype=dtype).flatten()
    #     if bbox_scale.numel() == 0:
    #         return
    #     if bbox_scale.numel() == 1:
    #         bbox_scale = bbox_scale.repeat(3)
    #     elif bbox_scale.numel() > 3:
    #         bbox_scale = bbox_scale[:3]

    #     max_scale = (bbox_scale / 128).clamp_min(1e-5)
    #     current_scale = reconstruction.get_scaling
    #     clamped = torch.minimum(current_scale, max_scale.view(1, 3))
    #     reconstruction.from_scaling(clamped)
    def _clamp_gaussian_scale(self, reconstruction, bbox_scale) -> None:
        device = reconstruction.get_xyz.device
        dtype = reconstruction.get_xyz.dtype

        bbox_scale = torch.as_tensor(bbox_scale, device=device, dtype=dtype).flatten()
        if bbox_scale.numel() == 0:
            return
        if bbox_scale.numel() == 1:
            bbox_scale = bbox_scale.repeat(3)
        elif bbox_scale.numel() > 3:
            bbox_scale = bbox_scale[:3]

        max_scale = (bbox_scale / self.G).clamp_min(1e-5)
        current_scale = reconstruction.get_scaling
        clamped = torch.minimum(current_scale, max_scale.view(1, 3))
        reconstruction.from_scaling(clamped)

    def _render_current_frame(
        self,
        gauss,
        scan_id: str,
        frame_id: str,
        scene_graphs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # intr = scene_graphs["obj_intrinsics"][scan_id]
        root = '/cluster/project/cvg/Shared_datasets/3RScan/'
        intr = scan3r.load_intrinsics(osp.join(root, "scenes"), scan_id)
        H = int(intr["height"])
        W = int(intr["width"])
        fx = float(intr["intrinsic_mat"][0, 0])
        fy = float(intr["intrinsic_mat"][1, 1])
        K = intr["intrinsic_mat"]

        fovx = focal2fov(fx, W)
        fovy = focal2fov(fy, H)

        # extr = scene_graphs["image_poses"][scan_id][frame_id]
        # pose_camera_to_world = np.linalg.inv(extr)
        extr = scan3r.load_frame_poses(root, scan_id, (frame_id,))
        extr = extr[frame_id]
        # self.logger.info(f"extr:{extr}")
        pose_camera_to_world = np.linalg.inv(extr)
        camera = MiniCam2(
            width=W,
            height=H,
            fovy=fovy,
            fovx=fovx,
            znear=0.01,
            zfar=100.0,
            R=pose_camera_to_world[:3, :3].T,
            T=pose_camera_to_world[:3, 3],
            K=K,
        )

        pred = render(
            camera,
            gauss,
            pipe=self._pipe_cfg,
            bg_color=torch.tensor((0.0, 0.0, 0.0), device=self.device),
        )["render"]

        # img_path = osp.join(root,'scenes', 'sequence',scan_id,frame_id,'.color.jpg')
        img_path = os.path.join(
            root,
            "scenes",
            scan_id,
            "sequence",
            f"frame-{frame_id}.color.jpg"
        )
        gt = Image.open(img_path).convert("RGB")
        gt = torch.from_numpy(np.array(gt)).permute(2, 0, 1).float().to(self.device) / 255.0
        return pred, gt

    def _photometric_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            while mask.ndim < pred.ndim:
                mask = mask.unsqueeze(0)
            return (torch.abs(pred - gt) * mask).sum() / (mask.sum() * pred.shape[0] + 1e-6)
        return torch.abs(pred - gt).mean()


    def _forward_batch(
        self,
        scene_graphs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        occ_gt, occ_vis, x_in = self._make_batch(scene_graphs)
        teacher_coords, teacher_feats = self._compute_teacher_latents(scene_graphs)
        if teacher_coords is None:
            self.logger.warning("Skipping batch because teacher voxels were missing.")
            return {}, {}

        logits_full = self.model(x_in)
        logits = logits_full[:, :1]
        latent_pred = logits_full[:, 1:]
        photo_loss = torch.zeros((), device=latent_pred.device)
        if self.photometric:
            # pred_idx = (logits[0,0].sigmoid() > 0.5).nonzero(as_tuple=False)
            sparse_latent = self._latent_pred_to_teacher_sparse(latent_pred, teacher_coords)
            reconstruction = self.teacher.decode(sparse_latent)
            photo_terms = []
            B = len(scene_graphs["scene_ids"])
            for b in range(B):
                sid = scene_graphs["scene_ids"][b][0]
                fid = str(scene_graphs["frame_ids"][b]).zfill(6)
                gauss_b = reconstruction[b]
                # self.logger.info(f"student_mean_gt: {scene_graphs['student_mean_gt'][b]}" )
                # self.logger.info(f"student_scale_gt: {scene_graphs['student_scale_gt'][b]}" )
                self._apply_pack_alignment(
                    gauss_b,
                    scene_graphs["student_mean_gt"][b].to(self.device),
                    scene_graphs["student_scale_gt"][b].to(self.device),
                )

                gauss_params = [
                    gauss_b.get_xyz,
                    gauss_b.get_scaling,
                    gauss_b.get_rotation,
                    gauss_b.get_opacity,
                    gauss_b.get_features,
                ]
                if gauss_b.get_xyz.numel() == 0 or any(not torch.isfinite(p).all() for p in gauss_params):
                    self.logger.warning(f"Skipping photometric loss for {sid}: Gaussians are empty or contain NaN/Inf")
                    continue

                try:
                    pred_img, gt_img = self._render_current_frame(gauss_b, sid, fid, scene_graphs)
                    photo_terms.append(self._photometric_loss(pred_img, gt_img))
                except RuntimeError as e:
                    self.logger.warning(f"Render failed for {sid}/{fid}, skipping photometric loss: {e}")
                frame_ids = scan3r.load_frame_idxs(
                    osp.join(self.cfg.data.root_dir, "scenes"), sid
                )
                # fid_2 = random.choice(frame_ids)
                # pred_img, gt_img = self._render_current_frame(gauss_b, sid, fid_2, scene_graphs)
                # photo_terms.append(self._photometric_loss(pred_img, gt_img))
            photo_loss = torch.stack(photo_terms).mean() if photo_terms else torch.zeros((), device=self.device)
            # return torch.stack(photo_terms).mean()
            # self.logger.info(f"reconstruction shape: {reconstruction.shape}")
        mask_complete = (1 - occ_vis) + 0.3 * (F.max_pool3d(occ_vis, 3, 1, 1) - occ_vis).clamp_min(0)
        mask_complete = mask_complete.clamp_max(1.0)
        bce = F.binary_cross_entropy_with_logits(logits, occ_gt, reduction="none")
        occ_loss = (bce * mask_complete).sum() / (mask_complete.sum() + 1e-6)

        probs_dice = torch.sigmoid(logits)
        gt = (occ_gt > 0.5).float()
        P = probs_dice * mask_complete
        Gt = gt * mask_complete
        inter = (P * Gt).sum(dim=(1, 2, 3, 4))
        pred = P.sum(dim=(1, 2, 3, 4))
        target = Gt.sum(dim=(1, 2, 3, 4))
        dice_loss = (1.0 - (2 * inter + 1e-6) / (pred + target + 1e-6)).mean()

        with torch.no_grad():
            probs = probs_dice.detach()
            iou_metrics = self._compute_iou(probs, occ_gt)
            iou_gate = iou_metrics.get(0.5, torch.tensor(0.0, device=probs.device)).item()

        latent_loss = self._sparse_latent_loss(latent_pred, teacher_coords, teacher_feats)
        if iou_gate >= self.latent_pred_iou_threshold:
            latent_pred_loss = self._predicted_latent_loss(
                logits, latent_pred, teacher_coords, teacher_feats
            )
        else:
            latent_pred_loss = torch.zeros((), device=latent_pred.device)

        loss = (
            self.recon_weight * occ_loss
            + self.lambda_dice * dice_loss
            + self.latent_weight * latent_loss
            + self.latent_pred_weight * latent_pred_loss
            + photo_loss
        )

        loss_dict = {
            "loss": loss,
            "bce": occ_loss,
            "dice": dice_loss,
            "latent_l1": latent_loss,
            "latent_pred_l1": latent_pred_loss,
            "photo_loss": photo_loss,
            "IoU @ 0.3": iou_metrics[0.3].item(),
            "IoU @ 0.5": iou_metrics[0.5].item(),
            "IoU @ 0.7": iou_metrics[0.7].item(),
        }
        if self.photometric:
            output_dict = {
                "logits": logits[: min(x_in.shape[0], 4)].detach().cpu(),
                "occ_gt": occ_gt[: min(x_in.shape[0], 4)].detach().cpu(),
                "occ_vis": occ_vis[: min(x_in.shape[0], 4)].detach().cpu(),
                "predicted_image": pred_img.detach().cpu(),
                "ground_truth_images": gt_img.detach().cpu(),
                "gaussian": gauss_b
            }
        else:  
            output_dict = {
                "logits": logits[: min(x_in.shape[0], 4)].detach().cpu(),
                "occ_gt": occ_gt[: min(x_in.shape[0], 4)].detach().cpu(),
                "occ_vis": occ_vis[: min(x_in.shape[0], 4)].detach().cpu(),
               
            }
        return output_dict, loss_dict

    def train_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        samples = data_dict["samples"]
        scene_graphs = self._build_scene_graphs_from_samples(samples, self.train_dataset_ref)
        return self._forward_batch(scene_graphs)

    def _compute_iou(self, probs: torch.Tensor, gt: torch.Tensor, thresholds=(0.3, 0.5, 0.7)):
        metrics = {}
        gt_bin = (gt > 0.5).float()
        for thr in thresholds:
            pr = (probs >= thr).float()
            inter = (pr * gt_bin).sum(dim=(1, 2, 3, 4))
            union = pr.sum(dim=(1, 2, 3, 4)) + gt_bin.sum(dim=(1, 2, 3, 4)) - inter
            metrics[thr] = (inter / (union + 1e-6)).mean()
        return metrics

    def val_step(
        self, epoch: int, iteration: int, data_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with torch.no_grad():
            samples = data_dict["samples"]
            scene_graphs = self._build_scene_graphs_from_samples(samples, self.val_dataset_ref)
            return self._forward_batch(scene_graphs)

    def visualize(self, output_dict: Dict[str, Any], epoch: int, mode: str = "train") -> None:
        outdir = f"{self.cfg.output_dir}/events"
        os.makedirs(outdir, exist_ok=True)
        logits = output_dict["logits"]
        occ_gt = output_dict["occ_gt"]
        occ_vis = output_dict["occ_vis"]
        B = logits.shape[0]
        thr = 0.5
        if self.photometric:
            predicted_images = output_dict["predicted_image"]
            ground_truth_images = output_dict["ground_truth_images"]
            output_dict["gaussian"].save_ply(
                    f"{self.cfg.output_dir}/events/reconstruction.ply"
                )
            
        for i in range(B):
            pr_idx = (logits[i, 0].sigmoid() > thr).nonzero(as_tuple=False)
            gt_idx = (occ_gt[i, 0] > thr).nonzero(as_tuple=False)
            vis_idx = (occ_vis[i, 0] > thr).nonzero(as_tuple=False)
            save_vox_as_ply(gt_idx, self.G, f"{outdir}/{mode}_gt_{i}.ply")
            save_vox_as_ply(pr_idx, self.G, f"{outdir}/{mode}_pred_{i}.ply")
            save_vox_as_ply(vis_idx, self.G, f"{outdir}/{mode}_input_{i}.ply")
            sbs = side_by_side(occ_gt[i, 0], logits[i, 0], max_slices=6)
            if self.photometric:
                
                sbs_2 = torch.concat(
                    [ground_truth_images, predicted_images],
                    dim=-1,
                )
                sbs_2 = sbs_2.unsqueeze(0)   
                scale = 0.3
                sbs_small = F.interpolate(
                    sbs_2,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).clamp(0, 1)
                save_image(
                    predicted_images,
                    f"{self.cfg.output_dir}/events/{mode}_predicted_images.png",
                )
                save_image(
                    ground_truth_images,
                    f"{self.cfg.output_dir}/events/{mode}_ground_truth_images.png",
                )
            self.writer.add_image(
                f"{mode}/slices_{i}_gt_pred",
                sbs.unsqueeze(1),
                global_step=epoch,
                dataformats="NCHW",
            )
                # self.writer.add_image(
                #     f"{mode}/slices_{i}_gt_pred",
                #     sbs_small,
                #     global_step=epoch,
                #     dataformats="NCHW",
                # )
                


def parse_args(parser: argparse.ArgumentParser = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--load_encoder", default=None, type=str)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown_args = parser.parse_known_args()
    return parser, args, unknown_args


def main() -> None:
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
